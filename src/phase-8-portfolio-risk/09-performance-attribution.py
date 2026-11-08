"""
绩效归因：Brinson 与因子归因
============================

学习目标：
    1. 用 Brinson（BHB）模型把超额收益拆成配置、选择、交互三部分，并验证可加性
    2. 理解 Brinson-Fachler 改进，以及"组合没持有某行业"时的处理
    3. 用 Carino 方法把多期归因正确地链接起来
    4. 用时间序列回归做因子归因：收益 = α + Σ β_k·f_k + ε
    5. 用截面因子暴露做单期 Barra 式归因，并把跟踪误差拆成因子风险与特异风险

核心概念：
    - 配置效应 (w_p - w_b)·R_b,i；选择效应 w_b·(R_p,i - R_b,i)；交互效应 (w_p - w_b)·(R_p,i - R_b,i)
    - BF 配置效应：(w_p - w_b)·(R_b,i - R_b)
    - 多期：Σ_t (R_p,t - R_b,t) ≠ Π(1+R_p,t) - Π(1+R_b,t)，需要链接系数
    - 主动收益 = 主动暴露 × 因子收益 + 特异收益

前置知识：
    - Phase 7 因子收益与截面回归、03-risk-parity.py 的风险贡献
    - CONCEPT.md 第 11 节

环境要求：
    - pip install numpy pandas statsmodels akshare
    - 第一、二、四部分使用构造数据；第三部分在 QTC_OFFLINE=1 时优先读本地缓存，缓存缺失才改用合成 / 模拟数据，并打印 ⚠️
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv

from portfolio_utils import TRADING_DAYS, load_etf_returns, load_stock_returns
from utils.data import is_synthetic

load_dotenv()
pd.set_option("display.width", 200)


def brinson(df: pd.DataFrame, fachler: bool = False) -> pd.DataFrame:
    """df 列：w_p, w_b, r_p, r_b（行业内收益），index 为行业"""
    Rb = (df.w_b * df.r_b).sum()
    base = df.r_b - Rb if fachler else df.r_b
    out = pd.DataFrame({
        "配置": (df.w_p - df.w_b) * base,
        "选择": df.w_b * (df.r_p - df.r_b),
        "交互": (df.w_p - df.w_b) * (df.r_p - df.r_b),
    })
    out["合计"] = out.sum(axis=1)
    out.loc["总计"] = out.sum()
    return out


# ==================== 第一部分：单期 Brinson ====================


def single_period_brinson():
    print("=" * 60)
    print("第一部分：单期 Brinson 归因（BHB 1986）")
    print("=" * 60)

    df = pd.DataFrame({
        "w_p": [0.15, 0.25, 0.20, 0.10, 0.30],
        "w_b": [0.25, 0.20, 0.15, 0.20, 0.20],
        "r_p": [0.020, 0.080, -0.020, 0.050, 0.100],
        "r_b": [0.030, 0.060, -0.040, 0.040, 0.070],
    }, index=pd.Index(["银行", "食品饮料", "医药", "周期资源", "科技制造"], name="行业"))
    Rp, Rb = (df.w_p * df.r_p).sum(), (df.w_b * df.r_b).sum()
    print("构造示例（单季度）：w = 行业权重，r = 行业内收益")
    print(df.map(lambda x: f"{x:.1%}").to_string())
    print(f"\n组合收益 R_p = {Rp:.3%}，基准收益 R_b = {Rb:.3%}，超额 = {Rp - Rb:+.3%}")

    print("""
                        基准行业收益 R_b,i      组合行业收益 R_p,i
                     ┌────────────────────┬────────────────────┐
      基准权重 w_b,i │  Q1 = Σ w_b·r_b    │  Q2 = Σ w_b·r_p    │
                     ├────────────────────┼────────────────────┤
      组合权重 w_p,i │  Q3 = Σ w_p·r_b    │  Q4 = Σ w_p·r_p    │
                     └────────────────────┴────────────────────┘
         配置 = Q3 - Q1     选择 = Q2 - Q1     交互 = Q4 - Q3 - Q2 + Q1
    """)
    bhb = brinson(df)
    print("BHB 归因（单位：%）：")
    print((bhb * 100).round(3).to_string())
    Q1, Q2 = (df.w_b * df.r_b).sum(), (df.w_b * df.r_p).sum()
    Q3, Q4 = (df.w_p * df.r_b).sum(), (df.w_p * df.r_p).sum()
    tot = bhb.loc["总计"]
    print(f"""
    ✅ 校验：配置 + 选择 + 交互 = {tot['合计']:+.4%}，R_p - R_b = {Rp - Rb:+.4%}
    ✅ 四象限：Q3-Q1 = {Q3 - Q1:+.4%}（配置），Q2-Q1 = {Q2 - Q1:+.4%}（选择），Q4-Q3-Q2+Q1 = {Q4 - Q3 - Q2 + Q1:+.4%}（交互）
    """)

    bf = brinson(df, fachler=True)
    cmp = pd.DataFrame({"主动权重%": (df.w_p - df.w_b) * 100, "行业收益-R_b %": (df.r_b - Rb) * 100,
                        "BHB 配置%": bhb["配置"].drop("总计") * 100, "BF 配置%": bf["配置"].drop("总计") * 100}).round(3)
    print("Brinson-Fachler：配置效应改用 (w_p - w_b)·(R_b,i - R_b)")
    print(cmp.to_string())
    print(f"   配置效应总和：BHB {bhb.loc['总计', '配置']:+.3%}，BF {bf.loc['总计', '配置']:+.3%}（因为 Σ(w_p - w_b) = 0，总和相同）")
    flip = cmp[np.sign(cmp["BHB 配置%"]) != np.sign(cmp["BF 配置%"])]
    if len(flip):
        name = flip.index[0]
        row = df.loc[name]
        print(f"""
    差别在行业层面的解释，例如{name}：主动权重 {row.w_p - row.w_b:+.0%}，行业收益 {row.r_b:+.1%}，
    {'跑赢' if row.r_b > Rb else '跑输'}基准整体 {Rb:.2%} —— BHB 记为{'正' if flip.loc[name, 'BHB 配置%'] > 0 else '负'}贡献，BF 记为{'正' if flip.loc[name, 'BF 配置%'] > 0 else '负'}贡献。""")
    print("""
    BF 的逻辑：低配一个跑输整体基准的行业是对的、超配一个跑输整体基准的行业才算失误，
    而不看该行业收益本身的正负。实践中 BF 更常用。
    """)
    return df


# ==================== 第二部分：缺失行业与多期链接 ====================


def multi_period_linking(seed: int = 3):
    print("\n" + "=" * 60)
    print("第二部分：组合未持有的行业 & 多期链接（Carino）")
    print("=" * 60)

    print("""
    ① 组合没有持有某行业（w_p,i = 0）时，组合行业收益 R_p,i 没有定义。
       常用做法：用基准行业收益填充，选择效应、交互效应记为 0，该行业只剩配置效应。
    """)
    rng = np.random.default_rng(seed)
    industries = ["银行", "食品饮料", "医药", "周期资源", "科技制造"]
    w_b = np.array([0.25, 0.20, 0.15, 0.20, 0.20])
    periods = []
    for q in range(4):
        w_p = np.clip(w_b + rng.normal(0, 0.05, 5), 0, None)
        w_p[2] = 0.0 if q == 1 else w_p[2]           # 第二季度完全不持有医药
        w_p /= w_p.sum()
        r_b = rng.normal(0.02, 0.06, 5)
        r_p = r_b + rng.normal(0.005, 0.02, 5)
        r_p = np.where(w_p == 0, r_b, r_p)            # 未持有 → 用基准收益填充
        periods.append(pd.DataFrame({"w_p": w_p, "w_b": w_b, "r_p": r_p, "r_b": r_b}, index=industries))
    q2 = brinson(periods[1], fachler=True)
    print("第二季度（医药权重为 0）的 BF 归因（%）：")
    print((q2 * 100).round(3).to_string())

    Rp = np.array([(d.w_p * d.r_p).sum() for d in periods])
    Rb = np.array([(d.w_b * d.r_b).sum() for d in periods])
    effects = pd.DataFrame([brinson(d, fachler=True).loc["总计", ["配置", "选择", "交互"]] for d in periods],
                           index=[f"Q{i + 1}" for i in range(4)])
    R_total, Rb_total = np.prod(1 + Rp) - 1, np.prod(1 + Rb) - 1

    # Carino 系数：k_t = [ln(1+R_p,t) - ln(1+R_b,t)] / (R_p,t - R_b,t)，k 为全期同理
    k_t = (np.log1p(Rp) - np.log1p(Rb)) / (Rp - Rb)
    k = (np.log1p(R_total) - np.log1p(Rb_total)) / (R_total - Rb_total)
    linked = effects.mul(k_t / k, axis=0)
    print(f"""
    ② 多期链接：四个季度
       各季度超额：{np.round((Rp - Rb) * 100, 3)}%
       简单相加 Σ(R_p,t - R_b,t)             = {np.sum(Rp - Rb):+.4%}
       真实全年超额 Π(1+R_p) - Π(1+R_b)     = {R_total - Rb_total:+.4%}   ← 两者不相等
    """)
    print("Carino 链接后的全年归因（%）：")
    out = pd.DataFrame({"简单相加": effects.sum(), "Carino 链接": linked.sum()}) * 100
    out.loc["合计"] = out.sum()
    print(out.round(4).to_string())
    print(f"   ✅ Carino 合计 = {linked.to_numpy().sum():+.4%}，等于真实全年超额 {R_total - Rb_total:+.4%}")
    print("   原理：用对数收益（可加）做桥梁，k_t/k 把每期的算术效应缩放到可以相加的尺度")


# ==================== 第三部分：时间序列回归的因子归因 ====================


def regression_attribution():
    print("\n" + "=" * 60)
    print("第三部分：时间序列回归的因子归因")
    print("=" * 60)

    stock_ret, _ = load_stock_returns()
    etf = load_etf_returns(verbose=False)
    if is_synthetic(stock_ret) or is_synthetic(etf):
        print("⚠️ 本部分用到了合成 / 模拟数据，回归结果没有实际含义，只用于理解方法")
    port = stock_ret.mean(axis=1).rename("组合")               # 20 只股票等权（每日再平衡）
    factors = pd.DataFrame({
        "市场": etf["沪深300"],
        "规模(中证500-沪深300)": etf["中证500"] - etf["沪深300"],
        "成长(创业板-沪深300)": etf["创业板"] - etf["沪深300"],
    })
    data = pd.concat([port, factors], axis=1, join="inner").dropna()
    y, X = data["组合"], sm.add_constant(data[factors.columns])
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    print(f"组合：个股池 20 只等权；因子：用 ETF 收益差构造的简化因子；样本 {data.index[0].date()} ~ {data.index[-1].date()}，{len(data)} 天")
    print(f"回归：r_p = α + Σ β_k·f_k + ε（HAC 标准误），R² = {res.rsquared:.3f}\n")

    mean_f = data[factors.columns].mean()
    contrib = res.params[factors.columns] * mean_f * TRADING_DAYS
    tbl = pd.DataFrame({
        "β": res.params[factors.columns].round(3),
        "t 值": res.tvalues[factors.columns].round(2),
        "因子年化收益": (mean_f * TRADING_DAYS).map("{:+.2%}".format),
        "贡献(年化)": contrib.map("{:+.2%}".format),
    })
    tbl = tbl.astype(object)
    tbl.loc["α（特异）"] = ["—", round(res.tvalues["const"], 2), "—", f"{res.params['const'] * TRADING_DAYS:+.2%}"]
    print(tbl.to_string())
    total = contrib.sum() + res.params["const"] * TRADING_DAYS
    print(f"""
    ✅ 校验：各因子贡献 + α = {total:+.4%}，组合年化算术收益 = {y.mean() * TRADING_DAYS:+.4%}（OLS 带截距时恒等）

    💡 读法：
       - β 是组合对每个因子的平均暴露；贡献 = β × 因子平均收益
       - α 的 t 值才说明"选股能力"是否显著；|t| < 2 时，α 很可能只是噪声
       - 这里的因子是 ETF 收益差构造的粗糙代理，换一套因子定义，"α"可能被重新解释为某个因子的贡献
       ⚠️ 这个个股池是"今天"挑出来的各行业龙头，带有幸存者 / 事后选择偏差，
          回归出的 α 很可能被高估——这正是 Phase 1 第 10 课讲过的偏差在归因中的体现
    """)


# ==================== 第四部分：截面因子归因（Barra 式） ====================


def cross_sectional_attribution(seed: int = 11):
    print("\n" + "=" * 60)
    print("第四部分：单期截面因子归因与风险归因（构造数据）")
    print("=" * 60)

    rng = np.random.default_rng(seed)
    n = 30
    ind = np.repeat(["金融", "消费", "科技"], 10)
    B = pd.DataFrame(0.0, index=[f"S{i:02d}" for i in range(n)], columns=["金融", "消费", "科技", "规模", "动量"])
    for i, g in enumerate(ind):
        B.iloc[i, ["金融", "消费", "科技"].index(g)] = 1.0
    B["规模"] = rng.standard_normal(n)          # 已标准化的风格暴露（期初值）
    B["动量"] = rng.standard_normal(n)
    w_b = np.full(n, 1 / n)
    tilt = -0.4 * B["规模"].to_numpy() + 0.3 * B["动量"].to_numpy()   # 偏小盘、偏高动量
    w_p = np.clip(w_b * (1 + tilt), 0, None)
    w_p /= w_p.sum()
    f = pd.Series({"金融": 0.010, "消费": -0.015, "科技": 0.030, "规模": -0.020, "动量": 0.008})  # 当期因子收益
    u = rng.normal(0, 0.03, n)                  # 特异收益
    r = B.to_numpy() @ f.to_numpy() + u

    a = w_p - w_b
    x_active = B.T @ a
    contrib = x_active * f
    total_active = a @ r
    out = pd.DataFrame({"主动暴露": x_active.round(3), "因子收益": f.map("{:+.2%}".format), "贡献": contrib.map("{:+.3%}".format)})
    print("组合相对等权基准：偏小盘（规模暴露为负）、偏高动量；暴露用期初值")
    print(out.to_string())
    print(f"""
       因子贡献合计 {contrib.sum():+.3%}
       特异收益     {total_active - contrib.sum():+.3%}   = Σ a_i·u_i = {a @ u:+.3%}
       主动收益合计 {total_active:+.3%}   ✅ 因子 + 特异 = 主动
    """)

    # 风险归因：TE² = x'Fx + a'Da，再用 Euler 分解到每个因子
    F = np.diag([0.04, 0.05, 0.08, 0.03, 0.02]) ** 2 * 12   # 年化因子协方差（示例：对角）
    F[3, 4] = F[4, 3] = 0.3 * np.sqrt(F[3, 3] * F[4, 4])
    D = np.full(n, (0.03 * np.sqrt(12)) ** 2)
    x = x_active.to_numpy()
    var_factor, var_spec = x @ F @ x, a @ (D * a)
    te = np.sqrt(var_factor + var_spec)
    rc = x * (F @ x) / te
    risk = pd.Series(rc, index=B.columns)
    risk["特异"] = var_spec / te
    print(f"事前跟踪误差 TE = {te:.2%}（因子部分方差占比 {var_factor / te**2:.1%}，特异部分 {var_spec / te**2:.1%}）")
    print("TE 的 Euler 分解（Σ = TE）：")
    print(pd.DataFrame({"风险贡献": risk.map("{:.3%}".format), "占比": (risk / te).map("{:.1%}".format)}).to_string())
    print(f"   ✅ Σ 风险贡献 = {risk.sum():.4%}，TE = {te:.4%}")
    print("""
    💡 收益归因回答"钱从哪来"，风险归因回答"波动从哪来"。
       理想的选股策略：特异收益稳定为正，且特异风险占 TE 的大头；
       如果规模因子贡献了大部分收益和风险，策略本质上是"小盘 beta"。
    """)


# ==================== 主函数 ====================


def main():
    single_period_brinson()
    multi_period_linking()
    regression_attribution()
    cross_sectional_attribution()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 用申万一级行业指数（akshare: index_hist_sw）做基准行业收益，对你的 Phase 7 组合做月度 Brinson 归因
    2. 实现 Menchero 链接方法，与 Carino 比较各期效应的分配有何不同
    3. 第三部分改用滚动 250 天回归，画出 β 和 α 随时间的变化——暴露稳定吗？
    4. 第四部分如果用期末暴露（B 加上一点随机扰动）做归因，特异收益会被怎样错误地解释？
    """)


if __name__ == "__main__":
    main()
