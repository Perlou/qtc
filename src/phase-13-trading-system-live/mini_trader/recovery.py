"""
幂等下单、对账与故障恢复（第 10 课）

核心原则：交易所是真相来源；不确定时"宁停勿错"。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable

from .gateway import BaseGateway
from .mock_exchange import DuplicateOrderError, ExchangeDisconnected
from .models import OrderData, OrderRequest, PositionData, Status
from .oms import OMS
from .persistence import record


@dataclass
class SendResult:
    client_order_id: str
    outcome: str  # accepted / rejected / risk_rejected / confirmed_after_timeout / resent / unknown / not_sent
    attempts: int = 1
    detail: str = ""
    order: OrderData | None = None

    @property
    def ok(self) -> bool:
        return self.outcome in ("accepted", "confirmed_after_timeout", "resent")


async def sync_order(gateway: BaseGateway, oms: OMS, remote: OrderData) -> None:
    """用交易所查询结果更新本地订单，并补录它的成交（成交按 trade_id 去重，可重复调用）"""
    oms.on_order(remote)
    for trade in await gateway.query_trades(remote.client_order_id):
        oms.on_trade(trade)


async def send_with_confirm(
    gateway: BaseGateway,
    oms: OMS,
    req: OrderRequest,
    timeout: float = 1.0,
    max_resend: int = 1,
    log: Callable[[str], None] | None = None,
) -> SendResult:
    """
    超时后先查询，再决定是否重发：
        查到订单     → 按其状态更新本地，不重发
        确认不存在   → 用同一个 client_order_id 重发
        查询也失败   → 标记为"未知"，交给上层暂停该标的并告警
    """
    log = log or (lambda msg: None)
    coid = req.client_order_id
    attempts = 0
    while True:
        attempts += 1
        try:
            ack = await asyncio.wait_for(gateway.send_order(req), timeout)
        except DuplicateOrderError:
            log(f"{coid} 交易所提示订单号重复 → 说明之前的请求已到达，改为查询同步")
            remote = await gateway.query_order(coid)
            await sync_order(gateway, oms, remote)
            return SendResult(coid, "confirmed_after_timeout", attempts, "重复订单号", oms.get_order(coid))
        except ExchangeDisconnected as exc:
            if attempts == 1:
                oms.mark_not_sent(coid, f"未发出：{exc}")
                return SendResult(coid, "not_sent", attempts, str(exc), oms.get_order(coid))
            oms.mark_unknown(coid)
            return SendResult(coid, "unknown", attempts, f"重发时断线：{exc}", oms.get_order(coid))
        except TimeoutError:  # asyncio.wait_for 超时或网关报告超时
            log(f"{coid} 第 {attempts} 次发送超时 → 先查询，不直接重发")
            try:
                remote = await asyncio.wait_for(gateway.query_order(coid), timeout)
            except Exception as exc:  # 查询也失败：状态未知
                oms.mark_unknown(coid)
                log(f"{coid} 查询失败（{type(exc).__name__}）→ 标记为未知，暂停并告警")
                return SendResult(coid, "unknown", attempts, repr(exc), oms.get_order(coid))
            if remote is not None:
                await sync_order(gateway, oms, remote)
                log(f"{coid} 查询到订单（{remote.status.name}）→ 订单其实已到达交易所，不重发")
                return SendResult(coid, "confirmed_after_timeout", attempts, "", oms.get_order(coid))
            if attempts > max_resend:
                oms.mark_unknown(coid)
                return SendResult(coid, "unknown", attempts, "多次重发仍超时", oms.get_order(coid))
            log(f"{coid} 交易所无此订单 → 用同一个 client_order_id 重发")
            continue
        oms.on_order(ack)
        if ack.status is Status.REJECTED:
            return SendResult(coid, "rejected", attempts, ack.reason, oms.get_order(coid))
        return SendResult(coid, "resent" if attempts > 1 else "accepted", attempts, "", oms.get_order(coid))


# ==================== 对账 ====================


def positions_to_map(positions: list[PositionData]) -> dict[str, float]:
    return {f"{p.symbol}|{p.direction.name}": p.volume for p in positions if p.volume != 0}


def reconcile_positions(local: dict[str, float], remote: dict[str, float], tol: float = 1e-9) -> list[dict]:
    """返回差异列表；remote（交易所查询结果）为真相"""
    diffs = []
    for key in sorted(set(local) | set(remote)):
        lv, rv = local.get(key, 0.0), remote.get(key, 0.0)
        if abs(lv - rv) > tol:
            diffs.append({"position": key, "local": lv, "remote": rv, "diff": rv - lv})
    return diffs


def reconcile_orders(local: list[OrderData], remote: list[OrderData]) -> dict[str, list[str]]:
    """挂单对账：本地有交易所无 / 交易所有本地无（孤儿）/ 状态不一致"""
    lmap = {o.client_order_id: o for o in local}
    rmap = {o.client_order_id: o for o in remote}
    return {
        "missing_on_exchange": sorted(k for k, o in lmap.items() if o.is_active and k not in rmap),
        "orphans": sorted(k for k in rmap if k not in lmap),
        "status_mismatch": sorted(k for k in lmap.keys() & rmap.keys() if lmap[k].status != rmap[k].status),
    }


@dataclass
class RecoveryReport:
    diffs_before: list[dict] = field(default_factory=list)  # 恢复前：本地 vs 交易所
    diffs_unexplained: list[dict] = field(default_factory=list)  # 补录成交后仍无法解释的差异
    orders_updated: int = 0
    orders_adopted: int = 0
    orders_missing: int = 0
    trades_applied: int = 0

    @property
    def safe_to_resume(self) -> bool:
        return not self.diffs_unexplained


async def recover_from_exchange(oms: OMS, gateway: BaseGateway, event_log=None) -> RecoveryReport:
    """
    启动 / 重连后的恢复流程（可重复执行，结果不变）：
        1. 查询交易所的订单、成交、持仓
        2. 更新本地未完成订单；收编孤儿订单；交易所查不到的未完成订单按失效处理
        3. 补录缺失成交（按 trade_id 去重）
        4. 再对账：仍有差异 → 无法解释，保持暂停等待人工
        5. 以交易所持仓为准覆盖本地；传入 event_log 时把恢复结果也写入日志
    """
    remote_orders = await gateway.query_orders()
    remote_trades = await gateway.query_trades()
    remote_positions = await gateway.query_positions()
    remote_map = positions_to_map(remote_positions)

    report = RecoveryReport(diffs_before=reconcile_positions(oms.positions.to_volume_map(), remote_map))
    remote_coids = {o.client_order_id for o in remote_orders}
    for coid, order in list(oms.orders.items()):
        if order.is_active and coid not in remote_coids:
            oms.mark_missing(coid, "交易所无此订单")
            report.orders_missing += 1
            if event_log is not None:
                record(event_log, "order", oms.orders[coid])
    for remote in remote_orders:
        known = remote.client_order_id in oms.orders
        changed = oms.on_order(remote) if known else oms.adopt(remote) is not None
        report.orders_updated += bool(known and changed)
        report.orders_adopted += not known
        if changed and event_log is not None:
            record(event_log, "order", remote)
    for trade in remote_trades:
        applied = oms.on_trade(trade)
        report.trades_applied += applied
        if applied and event_log is not None:
            record(event_log, "trade", trade)

    report.diffs_unexplained = reconcile_positions(oms.positions.to_volume_map(), remote_map)
    oms.positions.load(remote_positions)
    if event_log is not None:
        record(event_log, "positions_snapshot", remote_positions)
    return report
