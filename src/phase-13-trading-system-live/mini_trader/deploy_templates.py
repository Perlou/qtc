"""第 11 课生成的部署脚本模板：写到 outputs/phase-13/deploy/ 后可独立运行（只依赖标准库）"""

SCHEDULER = '''\
#!/usr/bin/env python3
"""
按交易日历调度（由 11-docker-deployment.py 生成）

用法：
    python scheduler.py --date 2026-11-09     打印某天的任务计划
    python scheduler.py --daemon              常驻运行（容器内使用）

cron 只知道星期几，不知道节假日；这里每分钟醒来一次，由交易日历决定要不要执行。
任务命令 trading.jobs 是占位示例，替换成你的项目入口即可。
"""

import argparse
import json
import subprocess
import time as time_mod
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Shanghai")
HERE = Path(__file__).resolve().parent
CALENDAR_FILE = HERE / "trading_days.csv"
STATE_FILE = HERE / "data" / "scheduler_state.json"
CATCH_UP = timedelta(minutes=30)  # 错过任务后的补跑窗口，超过则告警而不是补跑

DAY_TASKS = [
    (time(8, 30), "pre_market", "盘前：启动、登录、对账、加载策略"),
    (time(15, 5), "post_close", "收盘：撤销未成交单、日终对账"),
    (time(15, 30), "daily_report", "生成日报，发送 P3 通知"),
    (time(18, 0), "data_update", "更新数据、计算因子、生成次日目标持仓"),
]
NIGHT_TASK = (time(20, 30), "pre_night", "夜盘前：启动、登录、对账")
A_SHARE_SESSIONS = [(time(9, 30), time(11, 30)), (time(13, 0), time(15, 0))]


def load_days(path: Path = CALENDAR_FILE) -> set[date]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {date.fromisoformat(s.strip()) for s in lines if s.strip() and not s.startswith("#")}


def has_night_session(day: date, days: set[date]) -> bool:
    """简化规则：长假前一晚没有夜盘；实际以交易所通知为准"""
    if day not in days:
        return False
    return any(day + timedelta(days=n) in days for n in range(1, 4))


def tasks_for(day: date, days: set[date]) -> list[tuple[time, str, str]]:
    if day not in days:
        return []
    tasks = list(DAY_TASKS)
    if has_night_session(day, days):
        tasks.append(NIGHT_TASK)
    return tasks


def in_session(now: datetime, days: set[date]) -> bool:
    now = now.astimezone(TZ)
    return now.date() in days and any(a <= now.time() <= b for a, b in A_SHARE_SESSIONS)


def run_task(name: str) -> None:
    code = subprocess.run(["python", "-m", "trading.jobs", name], check=False).returncode
    if code != 0:
        print(f"[ALERT] 任务 {name} 失败（退出码 {code}），需要人工处理", flush=True)


def check_calendar(days: set[date], today: date) -> None:
    """日历过期是静默故障：所有日期都会被当成非交易日，系统什么也不做"""
    if max(days) < today + timedelta(days=30):
        print(f"[ALERT] 交易日历只覆盖到 {max(days)}，请尽快更新 trading_days.csv", flush=True)


def daemon(days: set[date]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    while True:
        now = datetime.now(TZ)
        today = now.date().isoformat()
        for at, name, _ in tasks_for(now.date(), days):
            start = datetime.combine(now.date(), at, TZ)
            if name in done.get(today, []) or now < start:
                continue
            if now - start <= CATCH_UP:
                run_task(name)
                done.setdefault(today, []).append(name)  # 记录状态：重启后不会重复执行
                STATE_FILE.write_text(json.dumps(done))
            else:
                print(f"[ALERT] 任务 {name} 已错过补跑窗口，需要人工处理", flush=True)
                done.setdefault(today, []).append(name)
        time_mod.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="打印某一天的任务计划")
    parser.add_argument("--daemon", action="store_true")
    args = parser.parse_args()
    days = load_days()
    check_calendar(days, datetime.now(TZ).date())
    if args.daemon:
        daemon(days)
    day = date.fromisoformat(args.date) if args.date else datetime.now(TZ).date()
    tasks = tasks_for(day, days)
    print(f"{day} {'交易日' if tasks else '非交易日：不执行任何任务'}")
    for at, name, desc in tasks:
        print(f"  {at:%H:%M}  {name:<13} {desc}")


if __name__ == "__main__":
    main()
'''

HEALTHCHECK = '''\
#!/usr/bin/env python3
"""
业务级健康检查（由 11-docker-deployment.py 生成）

交易进程定期写 data/heartbeat.json：
    {"ts": 时间戳, "last_tick_ts": 最后一笔行情时间戳, "gateway_connected": true}
退出码 0 = 健康，1 = 不健康；docker 据此标记容器状态。
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from scheduler import TZ, in_session, load_days

MAX_HEARTBEAT_AGE = 60  # 秒
MAX_TICK_AGE = 30  # 秒，只在交易时段检查


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heartbeat", default=str(Path(__file__).resolve().parent / "data" / "heartbeat.json"))
    parser.add_argument("--now", help="ISO 时间，用于测试")
    args = parser.parse_args()
    now = datetime.fromisoformat(args.now).replace(tzinfo=TZ) if args.now else datetime.now(TZ)
    try:
        hb = json.loads(Path(args.heartbeat).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"UNHEALTHY: 无法读取心跳文件（{exc}）")
        return 1
    problems = []
    if now.timestamp() - hb.get("ts", 0) > MAX_HEARTBEAT_AGE:
        problems.append("主循环心跳超时（假活？）")
    if not hb.get("gateway_connected"):
        problems.append("网关未连接")
    if in_session(now, load_days()) and now.timestamp() - hb.get("last_tick_ts", 0) > MAX_TICK_AGE:
        problems.append("交易时段内行情停滞")
    print("UNHEALTHY: " + "；".join(problems) if problems else "HEALTHY")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
'''
