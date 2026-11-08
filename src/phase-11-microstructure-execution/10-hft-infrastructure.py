"""
高频交易基础设施与延迟
======================

学习目标：
    1. 理解 Tick-to-Trade 延迟链路，以及每一段的优化手段（机房托管、Kernel Bypass、FPGA）
    2. 用代码实测 Python 各类操作的延迟数量级：字典查找、JSON 解析、pandas 访问、撮合、网络往返
    3. 理解为什么低延迟系统关心尾延迟（p99 / p99.9）和抖动，而不是平均值
    4. 了解国内程序化交易与高频交易的监管框架，以及个人量化应如何选择"延迟不敏感"的策略

核心概念：
    - Tick-to-Trade：从交易所发出行情到你的订单到达交易所的总时间
    - 尾延迟（Tail Latency）：p99 表示 99% 的请求快于这个值，剩下 1% 更慢
    - 抖动（Jitter）：延迟的波动，来源包括 GC、缓存未命中、线程调度、网络排队
    - 光速极限：光在光纤中约 5 μs/km，物理距离决定延迟下限

前置知识：
    - 01 课（撮合引擎）、04 课（REST 往返耗时）

环境要求：
    - pip install numpy pandas matplotlib ccxt
    - 本地测量不需要网络；外网往返测量在 QTC_OFFLINE=1 或访问失败时自动跳过

⚠️ 测得的数字取决于你的机器、Python 版本和系统负载，每次运行都会不同，关注"数量级"即可。
⚠️ 监管规定会更新，本课涉及的程序化交易、高频交易监管要求以监管与交易所最新规定为准。
"""

import gc
import json
import os
import socket
import sys
import threading
import time
import timeit
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter
import pandas as pd
from dotenv import load_dotenv

from orderbook import BUY, SELL, MatchingEngine
from utils.data import is_offline
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()


def fmt_ns(ns: float) -> str:
    """把纳秒格式化为合适的单位"""
    if ns < 1e3:
        return f"{ns:.0f} ns"
    if ns < 1e6:
        return f"{ns / 1e3:.1f} μs"
    if ns < 1e9:
        return f"{ns / 1e6:.2f} ms"
    return f"{ns / 1e9:.2f} s"


def measure(fn, n=10_000) -> np.ndarray:
    """逐次计时，返回每次调用的耗时（纳秒，含约几十纳秒的计时开销）"""
    lat = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter_ns()
        fn()
        lat[i] = time.perf_counter_ns() - t0
    return lat


def pct_row(name, lat):
    p50, p99, p999 = np.percentile(lat, [50, 99, 99.9])
    return [name, fmt_ns(lat.mean()), fmt_ns(p50), fmt_ns(p99), fmt_ns(p999), fmt_ns(lat.max()), f"{p999 / p50:.1f}x"]


# ==================== 第一部分：Tick-to-Trade 链路 ====================


def latency_chain():
    print("=" * 60)
    print("第一部分：Tick-to-Trade 延迟链路")
    print("=" * 60)
    print("""
    交易所撮合 ──▶ 行情发布 ──网络──▶ 网卡 ──▶ 内核/协议栈 ──▶ 解码
                                                              │
                                                              ▼
    交易所接收 ◀──网络── 网卡 ◀── 内核/协议栈 ◀── 风控检查 ◀── 策略决策

    每一段都可以优化：
      网络：机房托管（Colocation）缩短物理距离；微波链路在空气中比光纤快
      内核：Kernel Bypass（如 DPDK、Solarflare/ef_vi、Onload）绕过操作系统协议栈
      计算：C++ / Rust、CPU 绑核、大页内存、无锁队列；或用 FPGA 直接在硬件里解码和下单
      时钟：PTP / GPS 授时，纳秒级时间戳用于延迟度量与审计
    """)
    table = pd.DataFrame(
        [
            ["CPU L1 缓存访问", "~1 ns"],
            ["主内存访问", "~100 ns"],
            ["光在光纤中传播 1 km", "~5 μs"],
            ["FPGA tick-to-trade", "亚微秒级"],
            ["C++ 软件 tick-to-trade（优化后）", "微秒级"],
            ["Python 处理一条行情（含解析）", "数十 μs 到 ms 级（本课实测）"],
            ["同城机房间网络往返", "百 μs 到 ms 级"],
            ["家用宽带到交易所", "数 ms 到数十 ms"],
            ["跨国网络往返", "数十到数百 ms"],
        ],
        columns=["操作", "数量级"],
    )
    print(table.to_string(index=False))


# ==================== 第二部分：Python 操作的延迟 ====================


def python_latency():
    print("\n" + "=" * 60)
    print("第二部分：实测 Python 各类操作的延迟")
    print("=" * 60)

    overhead = measure(lambda: None, 20_000)
    print(f"    计时本身的开销（调用一个空函数并计时）：中位数 {fmt_ns(np.median(overhead))}")
    print("    → 比这个还快的操作，逐次计时测不准，下面先用批量平均（timeit）测：\n")

    d = {f"k{i}": i for i in range(1000)}
    lst = list(range(1000))
    arr = np.arange(10, dtype=float)
    series = pd.Series(np.arange(1000.0), index=[f"k{i}" for i in range(1000)])
    msg = '{"instId":"BTC-USDT","bids":[["75860.7","0.35","0","3"]],"asks":[["75860.8","1.30","0","5"]],"ts":"1794069711151"}'
    cases = [
        ("整数加法 a + b", "a + b", {"a": 1, "b": 2}),
        ("dict 查找", "d['k500']", {"d": d}),
        ("list 下标访问", "lst[500]", {"lst": lst}),
        ("float('75860.7') 字符串转数字", "float('75860.7')", {}),
        ("json.loads 一条盘口消息", "json.loads(msg)", {"json": json, "msg": msg}),
        ("numpy：10 个元素求均值", "arr.mean()", {"arr": arr}),
        ("pandas：Series.loc 按标签取值", "series.loc['k500']", {"series": series}),
        ("pandas：Series.iat 按位置取值", "series.iat[500]", {"series": series}),
    ]
    rows = []
    for name, stmt, env in cases:
        timer = timeit.Timer(stmt, globals=env)
        n, _ = timer.autorange()
        best = min(timer.repeat(repeat=5, number=n)) / n * 1e9
        rows.append([name, best])
    table = pd.DataFrame(rows, columns=["操作", "ns"])
    base = table.loc[table["操作"].str.startswith("dict"), "ns"].iloc[0]
    table["相对 dict 查找"] = (table["ns"] / base).map(lambda x: f"{x:.0f}x")
    table["耗时"] = table["ns"].map(fmt_ns)
    print(table[["操作", "耗时", "相对 dict 查找"]].to_string(index=False))
    loc_ns = table.loc[table["操作"].str.contains("loc"), "ns"].iloc[0]
    print(f"""
    💡 解读：
       - Python 解释器里，最简单的操作也要数十纳秒（C++ 里是 1 ns 量级）
       - pandas 为向量化计算设计：单个标签取值 .loc 约是 dict 查找的 {loc_ns / base:.0f} 倍
         → 行情热路径（每条消息都要走的代码）里不要逐条操作 DataFrame
       - JSON 解析是行情处理的大头之一：交易所二进制协议、专用解析库都是为了省这一段
    """)
    return msg


def per_call_distribution(msg):
    print("\n" + "=" * 60)
    print("第三部分：逐次计时 —— 中位数不够，要看尾部")
    print("=" * 60)

    eng = MatchingEngine()
    rng = np.random.default_rng(0)
    for _ in range(200):  # 预先放一些挂单
        eng.submit_limit(BUY, 10 - rng.integers(1, 10) * 0.01, 100)
        eng.submit_limit(SELL, 10 + rng.integers(1, 10) * 0.01, 100)
    sides = rng.choice([BUY, SELL], 50_000)
    prices = 10 + rng.integers(-8, 9, 50_000) * 0.01
    it = iter(range(50_000))

    def submit_one():
        i = next(it)
        eng.submit_limit(str(sides[i]), float(prices[i]), 100)

    def parse_and_update():
        m = json.loads(msg)
        bid = float(m["bids"][0][0])
        ask = float(m["asks"][0][0])
        return (bid + ask) / 2

    lat = {
        "json.loads + 计算中间价": measure(parse_and_update, 30_000),
        "撮合引擎提交一笔限价单": measure(submit_one, 30_000),
        "pandas DataFrame 追加一行（反例）": None,
    }
    df_rows = []

    def df_append():
        df_rows.append(pd.DataFrame({"bid": [1.0], "ask": [2.0]}))

    lat["pandas DataFrame 追加一行（反例）"] = measure(df_append, 3_000)
    rows = [pct_row(name, l) for name, l in lat.items()]
    table = pd.DataFrame(rows, columns=["操作", "均值", "p50", "p99", "p99.9", "最大值", "p99.9/p50"])
    print(table.to_string(index=False))
    print("""
    💡 均值和中位数掩盖了尾部：偶发的慢请求（GC、内存分配、操作系统调度）决定了 p99.9 和最大值。
       在高频场景里，最关键的行情跳动往往伴随消息洪峰，系统恰好最忙 —— 尾延迟就是在那时咬人。
    """)

    # GC 对尾延迟的影响
    def alloc_heavy():
        return [{"px": i, "qty": i} for i in range(200)]

    gc.collect()
    with_gc = measure(alloc_heavy, 20_000)
    gc.disable()
    try:
        without_gc = measure(alloc_heavy, 20_000)
    finally:
        gc.enable()
    gc.collect()
    rows = [pct_row("GC 开启", with_gc), pct_row("GC 关闭", without_gc)]
    print("    分配大量小对象（每次 200 个 dict）时，垃圾回收对尾延迟的影响：")
    print("    " + pd.DataFrame(rows, columns=["设置", "均值", "p50", "p99", "p99.9", "最大值", "p99.9/p50"])
          .to_string(index=False).replace("\n", "\n    "))
    print("""
    📌 GC 的影响取决于运行环境，单次测量不一定看得出差别；低延迟系统的通用做法是：
       热路径上避免分配对象（预分配、对象池），在空闲时手动触发 GC，或干脆用无 GC 的语言（C++ / Rust）
    """)
    return lat


# ==================== 第四部分：网络往返 ====================


def echo_server(sock):
    conn, _ = sock.accept()
    with conn:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            conn.sendall(data)


def network_latency():
    print("\n" + "=" * 60)
    print("第四部分：网络往返（RTT）")
    print("=" * 60)
    results = {}

    # 1. 本机回环 TCP：不经过网卡，但要走内核协议栈
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        th = threading.Thread(target=echo_server, args=(server,), daemon=True)
        th.start()
        client = socket.create_connection(server.getsockname(), timeout=5)
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # 关闭 Nagle 算法，小包立即发送
        payload = b"x" * 64

        def rtt():
            client.sendall(payload)
            got = 0
            while got < len(payload):
                got += len(client.recv(1024))

        for _ in range(200):  # 预热
            rtt()
        results["本机回环 TCP（127.0.0.1）"] = measure(rtt, 5_000)
        client.close()
        th.join(timeout=2)
        server.close()
    except OSError as e:
        print(f"    ⚠️ 本机回环测试失败（{type(e).__name__}: {e}），跳过")

    # 2. 外网：交易所公开接口（离线时跳过）
    if is_offline():
        print("    ⏭️ 离线模式（QTC_OFFLINE=1）：跳过外网往返测量")
    else:
        try:
            import ccxt

            proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
            config = {"timeout": 10000}
            if proxy:
                config["proxies"] = {"http": proxy, "https": proxy}
            ex = ccxt.okx(config)
            ex.fetch_time()  # 第一次包含建立 TLS 连接的开销
            lat = measure(ex.fetch_time, 8)
            results["外网 HTTPS（OKX 公开时间接口" + ("，经代理）" if proxy else "）")] = lat
            print("    （外网只读取交易所公开的服务器时间，不涉及任何交易）")
        except Exception as e:
            print(f"    ⚠️ 外网往返测量失败（{type(e).__name__}: {str(e)[:80]}），跳过")

    if results:
        rows = [pct_row(name, l) for name, l in results.items()]
        print(pd.DataFrame(rows, columns=["链路", "均值", "p50", "p99", "p99.9", "最大值", "p99.9/p50"]).to_string(index=False))
        if len(results) > 1:
            print("    （外网只采样了几次，p99 / p99.9 没有统计意义，只看数量级）")
    print("""
    💡 本机回环已经是微秒级 —— 这还没出过网卡；真实链路再加上交换机、光纤和交易所网关。
       家用网络到交易所是毫秒级，比机房托管的 HFT 慢三到四个数量级。
    """)

    print("    物理极限：光在光纤中约 2×10⁵ km/s（真空光速的 2/3），即约 5 μs/km")
    dist = pd.DataFrame(
        [["同一机房内", 0.1], ["上海 ↔ 杭州", 165], ["上海 ↔ 北京", 1070], ["上海 ↔ 深圳", 1210], ["上海 ↔ 新加坡", 3800]],
        columns=["路径（直线距离估算）", "km"],
    )
    dist["单程下限"] = (dist["km"] * 5e3).map(fmt_ns)
    dist["往返下限"] = (dist["km"] * 1e4).map(fmt_ns)
    print(dist.to_string(index=False))
    print("    实际光纤路径比直线更长，再加设备处理时间，真实延迟高于这个下限。")
    return results


# ==================== 第五部分：尾延迟的放大效应 ====================


def tail_amplification(lat):
    print("\n" + "=" * 60)
    print("第五部分：尾延迟为什么重要 —— 扇出放大")
    print("=" * 60)
    print("""
    一个决策如果要等 N 个组件都完成（如同时处理 N 只股票的行情、或请求 N 个服务），
    总延迟由最慢的那个决定。即使每个组件只有 1% 的概率超过 p99：
        P(至少一个超过 p99) = 1 − 0.99^N
    """)
    rows = [[n, 1 - 0.99**n] for n in [1, 10, 50, 100, 500]]
    print(pd.DataFrame(rows, columns=["N", "至少一个慢的概率"]).to_string(index=False, formatters={"至少一个慢的概率": "{:.1%}".format}))

    base = lat["json.loads + 计算中间价"]
    rng = np.random.default_rng(0)
    rows = []
    for n in [1, 10, 100]:
        draws = rng.choice(base, size=(5000, n)).max(axis=1)  # 从实测分布中抽 N 个取最大值
        rows.append([n, fmt_ns(np.median(draws)), fmt_ns(np.percentile(draws, 99))])
    print("\n    用第三部分实测的 json 解析延迟做重抽样，N 个任务中最慢那个的延迟：")
    print("    " + pd.DataFrame(rows, columns=["N", "最慢者的中位数", "最慢者的 p99"]).to_string(index=False).replace("\n", "\n    "))
    single_p99 = np.percentile(base, 99)
    med_100 = np.median(rng.choice(base, size=(5000, 100)).max(axis=1))
    print(f"""
    单个任务的 p99 = {fmt_ns(single_p99)}；N = 100 时"最慢者的中位数" = {fmt_ns(med_100)}
    💡 理论上 N ≈ 69 时 1 − 0.99^N ≈ 50%：单个任务的 p99 会变成扇出系统的"中位数体验"，
       单任务的 p99.9、最大值则决定了扇出系统的 p99 —— 这就是低延迟系统盯着尾部的原因。
    """)


# ==================== 第六部分：监管与个人量化的选择 ====================


def regulation_and_takeaways():
    print("\n" + "=" * 60)
    print("第六部分：国内监管环境与个人量化的选择")
    print("=" * 60)
    print("""
    📋 国内监管框架（以监管与交易所最新规定为准）：
       - 证监会《证券市场程序化交易管理规定（试行）》及沪深北交易所的程序化交易实施细则，
         对程序化交易实行报告制度，对高频交易有额外的报告与监管要求
       - 交易所细则对高频交易的认定标准包括单账户每秒申报、撤单笔数或单日申报、撤单笔数达到一定阈值
         （例如每秒 300 笔以上或单日 20000 笔以上）
       - 交易所对频繁申报撤单、瞬时大额申报、虚假申报等异常交易行为实时监控，并可能对高频交易收取差异化费用
       - 期货市场也有程序化交易报告和穿透式监管要求
       上述阈值和要求会随监管更新而调整，实际执行以监管与交易所最新规定为准；Phase 13 会讨论合规流程

    🧭 个人量化的现实选择：
       机构 HFT：     延迟 纳秒到微秒级；机房托管、FPGA、专线；策略周期 毫秒到秒；优势来自速度、规模、做市资格
       个人 / 小团队：延迟 毫秒到秒级；云服务器 / 家用网络；  策略周期 分钟、日、周；优势来自 alpha 研究、风控、耐心
       → 选择延迟不敏感的策略，延迟预算"够用即可"；把精力放在 alpha、成本建模和风控上。
         本阶段学到的微观结构知识，对中低频策略同样有用：估算成本、设计执行、理解容量。
    """)


# ==================== 可视化 ====================


def plot_all(lat, net):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1.1]})
    fig.subplots_adjust(wspace=0.55)
    ax = axes[0]
    for name, l in lat.items():
        us = l / 1e3
        ax.hist(us, bins=np.logspace(np.log10(us.min()), np.log10(us.max()), 60), histtype="step", lw=1.5, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("逐次延迟分布（对数坐标，长尾一目了然）")
    ax.set_xlabel("延迟（μs）")
    ax.set_ylabel("次数")
    ax.legend(fontsize=8, loc="center right")

    ax = axes[1]
    all_lat = {**lat, **net}
    names = list(all_lat)
    p50 = [np.percentile(all_lat[n], 50) / 1e3 for n in names]
    p99 = [np.percentile(all_lat[n], 99) / 1e3 for n in names]
    y = np.arange(len(names))
    ax.barh(y - 0.2, p50, height=0.4, label="p50")
    ax.barh(y + 0.2, p99, height=0.4, label="p99")
    ax.set_yticks(y, names, fontsize=8)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("延迟（μs，对数坐标）")
    ax.set_title("各类操作延迟的数量级")
    ax.legend()

    path = save_figure(fig, "10_latency.png", subdir="phase-11")
    print(f"\n📈 图表已保存：{path}")


# ==================== 主函数 ====================


def main():
    latency_chain()
    msg = python_latency()
    lat = per_call_distribution(msg)
    net = network_latency()
    tail_amplification(lat)
    regulation_and_takeaways()
    plot_all(lat, net)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 用 orjson 或 msgspec（如已安装）替换 json.loads，比较 p50 与 p99.9（不要为此安装新包，可只做思考题）
    2. 给 01 课的撮合引擎把 min()/max() 换成有序结构后，重新测量提交一笔限价单的 p99
    3. 写一个"行情处理循环"：解析 JSON → 更新本地簿 → 计算微价格 → 生成信号，测量端到端 p50 / p99.9，
       找出最慢的环节
    4. 查阅交易所最新的程序化交易实施细则，整理出高频交易认定标准与报告要求（注明文件名与日期）
    """)


if __name__ == "__main__":
    main()
