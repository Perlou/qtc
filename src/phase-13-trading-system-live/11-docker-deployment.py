"""
Docker 部署与定时任务
====================

学习目标：
    1. 理解容器化对交易系统的价值：环境固定、自动拉起、一键迁移
    2. 生成可用的 Dockerfile、docker-compose.yml，理解每一项配置的取舍
    3. 掌握时区与时钟同步：容器默认 UTC 的坑、带时区的时间、NTP
    4. 实现按交易日历调度：cron 的"周一到周五"不等于交易日
    5. 实现业务级健康检查：不仅检查进程，还要检查行情和网关

核心概念：
    - 交易日历：节假日休市、调休补班的周末也不开市，cron 表达式无法描述
    - 业务级健康检查：进程活着 ≠ 系统健康
    - 带时区的时间（aware datetime）：统一 Asia/Shanghai，避免偏差 8 小时

前置知识：
    - Phase 1 第 07 课（交易日历）
    - 09、10 课
    - 本阶段 CONCEPT.md 第 12 节

环境要求：
    - 生成的文件写到 outputs/phase-13/deploy/，不需要真正运行 docker
    - 交易日历优先用 akshare 获取；离线（QTC_OFFLINE=1）时用工作日近似 + 人为指定的示例休市日
"""

import shutil
import subprocess
import sys
import threading
import time as time_mod
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml
from dotenv import load_dotenv

from mini_trader.deploy_templates import HEALTHCHECK, SCHEDULER
from mini_trader.sessions import TradingCalendar, in_a_share_session
from utils import DATA_DIR, OUTPUT_DIR
from utils.data import is_offline

load_dotenv()

DEPLOY_DIR = OUTPUT_DIR / "phase-13" / "deploy"
CACHE = DATA_DIR / "phase-13" / "trade_calendar.csv"
SHANGHAI = ZoneInfo("Asia/Shanghai")
EXAMPLE_HOLIDAY = date(2026, 11, 11)  # 离线演示用的"示例休市日"，不是真实假期

DOCKERFILE = """\
# 与课程环境一致的 Python 版本；CTP 等库对平台（如 linux/amd64）有要求，以其发布说明为准
FROM python:3.12-slim

ENV TZ=Asia/Shanghai PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends tzdata \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# 先复制依赖清单：代码变更不会使依赖层缓存失效
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# 不以 root 运行：进程被攻破时降低影响
RUN useradd -m trader && chown -R trader /app
USER trader

CMD ["python", "-m", "trading.main"]
"""

COMPOSE = """\
services:
  trader:
    build: .
    restart: unless-stopped          # 崩溃后自动拉起；启动流程必须先恢复、对账（第 10 课）
    stop_grace_period: 30s           # 留时间撤单、落盘
    env_file: .env                   # 密钥放 .env，且 .env 不进代码仓库
    volumes:
      - ./data:/app/data             # 事件日志、数据库持久化到宿主机
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "python", "healthcheck.py"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 60s
    logging:
      driver: json-file
      options:
        max-size: "50m"              # 防止日志写满磁盘
        max-file: "5"

  scheduler:
    build: .
    restart: unless-stopped
    command: ["python", "scheduler.py", "--daemon"]
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
"""

ENV_EXAMPLE = """\
# 复制为 .env 后填写真实值；.env 必须加入 .gitignore 和 .dockerignore
CTP_USER_ID=
CTP_PASSWORD=
CTP_APP_ID=
CTP_AUTH_CODE=
ALERT_WEBHOOK_URL=
"""

DOCKERIGNORE = """\
.env
.git
__pycache__/
data/
logs/
"""

CRONTAB = """\
# ❌ 只用 cron：周一到周五 08:30 启动 —— 节假日也会启动，调休的周末不受影响
# 30 8 * * 1-5  docker compose up -d trader

# ✅ 推荐：调度逻辑放进 scheduler.py，由 docker compose 常驻运行
#    cron 如果要用，也只负责"每分钟唤醒一次"，是否执行由交易日历决定
"""


# ==================== 第一部分：为什么用 Docker ====================


def why_docker():
    print("=" * 60)
    print("第一部分：为什么用 Docker")
    print("=" * 60)
    print(pd.DataFrame(
        [
            ["在我机器上能跑", "镜像固定 Python 版本与依赖"],
            ["CTP 等库只有 Linux/Windows 版", "macOS 上也能运行 Linux 容器（Apple Silicon 注意 linux/amd64 平台与性能）"],
            ["进程崩溃", "restart: unless-stopped 自动拉起"],
            ["多组件部署", "docker compose 一次编排交易进程、调度、监控"],
            ["迁移服务器", "镜像 + 数据卷，快速迁移"],
        ],
        columns=["问题", "Docker 的作用"],
    ).to_string(index=False))
    print("""
    📍 部署位置：家用电脑只适合开发调试（断电、休眠、系统更新）；
       个人中低频策略推荐云服务器；对延迟敏感的策略才考虑券商 / 期货公司机房托管（需单独申请）。
    """)


# ==================== 第二部分：生成部署文件 ====================


def write_deploy_files():
    print("\n" + "=" * 60)
    print("第二部分：生成部署文件")
    print("=" * 60)
    if DEPLOY_DIR.exists():
        shutil.rmtree(DEPLOY_DIR)
    (DEPLOY_DIR / "data").mkdir(parents=True)
    files = {
        "Dockerfile": DOCKERFILE,
        "docker-compose.yml": COMPOSE,
        ".env.example": ENV_EXAMPLE,
        ".dockerignore": DOCKERIGNORE,
        "scheduler.py": SCHEDULER,
        "healthcheck.py": HEALTHCHECK,
        "crontab.example": CRONTAB,
    }
    for name, content in files.items():
        (DEPLOY_DIR / name).write_text(content, encoding="utf-8")
    print(f"   已写入 {DEPLOY_DIR}：")
    for name in files:
        print(f"     {name:<20} {(DEPLOY_DIR / name).stat().st_size:>6} 字节")

    print("\n📄 Dockerfile：\n")
    print(DOCKERFILE)
    print("📄 docker-compose.yml：\n")
    print(COMPOSE)

    compose = yaml.safe_load(COMPOSE)
    compile(SCHEDULER, "scheduler.py", "exec")
    compile(HEALTHCHECK, "healthcheck.py", "exec")
    print(f"   ✅ 校验：compose 解析成功，服务 {list(compose['services'])}；两个 Python 脚本语法正确")
    print("""
    📌 关键细节：
    - restart: unless-stopped 是把双刃剑：进程被拉起时，必须先走第 10 课的恢复流程，
      风控保持暂停，否则崩溃循环可能反复下单
    - trading.main / trading.jobs 是占位入口，换成你自己的项目模块
    - 日志设置 max-size：磁盘写满时系统无法持久化，就不能交易（第 10 课演练表）
    - 镜像里不放密钥：.env 在 .dockerignore 中，运行时通过 env_file 注入
    """)


# ==================== 第三部分：时区与时钟同步 ====================


def timezone_and_ntp():
    print("\n" + "=" * 60)
    print("第三部分：时区与时钟同步")
    print("=" * 60)
    moment = datetime(2026, 11, 9, 1, 30, tzinfo=timezone.utc)  # 同一个时刻
    naive_in_utc_container = moment.replace(tzinfo=None)  # 容器默认 UTC 时 datetime.now() 得到的值
    aware = moment.astimezone(SHANGHAI)
    cal = TradingCalendar([date(2026, 11, 9)])
    print("   同一时刻，容器时区设置不同，判断结果完全相反：")
    print(pd.DataFrame(
        [
            ["容器默认 UTC，naive datetime", naive_in_utc_container.strftime("%H:%M"), in_a_share_session(naive_in_utc_container, cal)],
            ["Asia/Shanghai，aware datetime", aware.strftime("%H:%M"), in_a_share_session(aware.replace(tzinfo=None), cal)],
        ],
        columns=["写法", "读到的时间", "判断是否在 A 股交易时段"],
    ).to_string(index=False))
    print("""
    📌 规则：
    - 镜像里设置 TZ=Asia/Shanghai 并安装 tzdata
    - 代码里用带时区的时间：datetime.now(ZoneInfo("Asia/Shanghai"))，不用 naive 的 datetime.now()
    - 加密市场 7×24、跨时区，内部统一存 UTC，展示时再转换也是好做法 —— 关键是全系统统一
    """)

    def skew_check(local_ms: int, server_ms: int, limit_ms: int = 1000) -> str:
        skew = local_ms - server_ms
        return f"偏差 {skew:+d} ms → {'⚠️ 告警并暂停签名请求' if abs(skew) > limit_ms else '正常'}"

    base = 1_794_000_000_000
    print("   时钟偏差检查（用交易所服务器时间做对比，阈值为示意）：")
    print(f"     本机偏快 120 ms：{skew_check(base + 120, base)}")
    print(f"     本机偏快 3.5 秒：{skew_check(base + 3500, base)}")
    print("""
    📌 宿主机开启 NTP（如 chrony）：容器共用宿主机时钟，容器内不需要单独跑 NTP。
       时钟偏差会导致：签名请求被拒（第 08 课）、日志时序错乱、延迟统计失真。
    """)


# ==================== 第四部分：按交易日历调度 ====================


def fetch_real_calendar() -> pd.DatetimeIndex | None:
    if CACHE.exists():
        return pd.DatetimeIndex(pd.read_csv(CACHE)["trade_date"])
    result: list = []

    def worker():
        try:
            import akshare as ak

            result.append(ak.tool_trade_date_hist_sina())
        except Exception as exc:  # 网络或接口异常：回退到近似日历
            print(f"   ⚠️ 获取交易日历失败（{type(exc).__name__}），使用离线近似日历")

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(15)  # 网络卡住时不能拖住整个课程
    if not result:
        return None
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    result[0].to_csv(CACHE, index=False)
    return pd.DatetimeIndex(pd.to_datetime(result[0]["trade_date"]))


def trading_calendar_schedule():
    print("\n" + "=" * 60)
    print("第四部分：按交易日历调度")
    print("=" * 60)
    year = 2026
    weekdays = pd.bdate_range(f"{year}-01-01", f"{year}-12-31")
    real = None if is_offline() else fetch_real_calendar()
    if real is not None:
        days = real[real.year == year]
        source = f"akshare 交易日历（缓存：{CACHE.name}）"
    else:
        days = weekdays.difference(pd.DatetimeIndex([EXAMPLE_HOLIDAY]))
        source = f"离线近似：工作日 + 人为指定的示例休市日 {EXAMPLE_HOLIDAY}（不是真实假期，不可用于实盘）"

    trade_days = [d.date() for d in days]
    header = f"# source: {source}\n"
    (DEPLOY_DIR / "trading_days.csv").write_text(header + "\n".join(map(str, trade_days)) + "\n", encoding="utf-8")

    closed = weekdays.difference(days)
    print(f"   日历来源：{source}")
    print(f"   {year} 年：周一到周五 {len(weekdays)} 天，交易日 {len(days)} 天，差 {len(closed)} 天")
    print(f"   cron 的 '1-5' 会在这些休市日照常运行：{[d.strftime('%m-%d') for d in closed][:10]}"
          f"{' ...' if len(closed) > 10 else ''}")
    print("""
    ⚠️ 后果：节假日启动系统 → 连不上柜台 → 假告警；更糟的是定时任务按错误的日期生成持仓。
       调休补班的周末，A 股也不开市：工作日历和交易日历是两回事。
    """)

    today = date.today()
    next_day = next((d for d in trade_days if d >= today), trade_days[-1])
    samples = [next_day, next_day + timedelta(days=(5 - next_day.weekday()) % 7 or 7), closed[-1].date()]
    labels = ["最近的交易日", "周六", "最后一个休市的工作日"]
    for label, d in zip(labels, samples):
        out = subprocess.run(
            [sys.executable, "-B", "scheduler.py", "--date", d.isoformat()],
            cwd=DEPLOY_DIR, capture_output=True, text=True, timeout=20,
        )
        print(f"   ▶ {label}：python scheduler.py --date {d}")
        print("     " + out.stdout.strip().replace("\n", "\n     "))
    print("""
    📌 设计要点：
    - 任务完成状态写入 data/scheduler_state.json：调度器重启后不会重复执行
    - 错过的任务有补跑窗口（30 分钟），超过就告警，而不是盲目补跑（盘前任务在午后执行没有意义）
    - 夜盘归属下一交易日（第 06 课），长假前一晚通常没有夜盘，以交易所通知为准
    - 日历过期是静默故障：所有日期都变成非交易日，系统什么也不做。调度器启动时检查覆盖范围并告警
    """)
    return trade_days, next_day


# ==================== 第五部分：业务级健康检查 ====================


def health_checks(next_day: date):
    print("\n" + "=" * 60)
    print("第五部分：业务级健康检查")
    print("=" * 60)
    hb_path = DEPLOY_DIR / "data" / "heartbeat.json"
    in_market = datetime(next_day.year, next_day.month, next_day.day, 10, 0, tzinfo=SHANGHAI)
    lunch = in_market.replace(hour=12)

    def check(label, now, ts_age, tick_age, connected):
        ts = now.timestamp()
        hb_path.write_text(f'{{"ts": {ts - ts_age}, "last_tick_ts": {ts - tick_age}, "gateway_connected": {str(connected).lower()}}}')
        out = subprocess.run(
            [sys.executable, "-B", "healthcheck.py", "--heartbeat", str(hb_path), "--now", now.replace(tzinfo=None).isoformat()],
            cwd=DEPLOY_DIR, capture_output=True, text=True, timeout=20,
        )
        return [label, out.returncode, out.stdout.strip()]

    rows = [
        check("一切正常", in_market, 5, 1, True),
        check("主循环卡死（假活）", in_market, 300, 300, True),
        check("网关断线", in_market, 5, 1, False),
        check("盘中行情停滞", in_market, 5, 120, True),
        check("午休时没有行情", lunch, 5, 1800, True),
    ]
    hb_path.unlink()
    print(pd.DataFrame(rows, columns=["场景", "退出码", "输出"]).to_string(index=False))
    print("""
    💡 只检查"进程是否存在"，上面第 2、3、4 行都会被判为健康。
       健康检查要回答的问题是"系统还能正常交易吗"，并且要知道当前是否在交易时段。
    """)


# ==================== 主函数 ====================


def main():
    t0 = time_mod.perf_counter()
    why_docker()
    write_deploy_files()
    timezone_and_ntp()
    _, next_day = trading_calendar_schedule()
    health_checks(next_day)
    print(f"\n   全部生成文件位于 {DEPLOY_DIR}（耗时 {time_mod.perf_counter() - t0:.1f} 秒）")

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 mini_trader 打包进镜像：补全 requirements.txt 和 trading/main.py，用 docker compose 跑起来
    2. 让 TradingSystem 定期写 heartbeat.json，并在第 09 课的 MonitorService 里使用同样的判断
    3. 给 scheduler.py 增加夜盘归属判断：用 mini_trader.sessions.TradingCalendar 生成当晚任务的日志目录
    4. 在 Apple Silicon 上用 platform: linux/amd64 构建镜像，比较启动时间；思考生产环境为什么不应这样运行
    """)


if __name__ == "__main__":
    main()
