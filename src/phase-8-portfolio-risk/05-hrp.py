"""
层次风险平价 HRP
================

学习目标：
    1. 理解 HRP 的三步：树聚类 → 准对角化 → 递归二分
    2. 手写相关距离、层次聚类、准对角化和递归二分，打印每一步的中间结果
    3. 与 riskfolio-lib 的 HRP 实现逐位对比，确认手写版本正确
    4. 把 HRP 与逆方差、ERC、GMV 放在一起做样本外比较

核心概念：
    - 相关距离 d_ij = √(½(1 - ρ_ij))：ρ = 1 → d = 0，ρ = -1 → d = 1
    - 准对角化：按聚类树的叶子顺序重排协方差，相似资产相邻
    - 递归二分：每次把列表一分为二，按子簇的逆方差组合方差分配权重
      α = 1 - V_L / (V_L + V_R)，左边乘 α，右边乘 1 - α
    - 全程不需要对协方差矩阵求逆

前置知识：
    - 03-risk-parity.py；Phase 3 相关系数
    - CONCEPT.md 第 7 节

环境要求：
    - pip install numpy pandas scipy cvxpy riskfolio-lib matplotlib
    - QTC_OFFLINE=1 时优先读本地缓存；缓存缺失才改用合成 / 模拟数据，并打印 ⚠️
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform

from portfolio_utils import (
    ETF_UNIVERSE, TRADING_DAYS, erc_weights, gmv_weights, load_etf_returns,
    load_stock_returns, perf_table, walk_forward,
)
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()
setup_chinese_font()
pd.set_option("display.width", 200)


# ==================== HRP 手写实现 ====================


def correlation_distance(cov: pd.DataFrame) -> pd.DataFrame:
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return np.sqrt(np.clip(0.5 * (1 - corr), 0.0, 1.0))


def cluster_order(cov: pd.DataFrame, method: str = "single", optimal_ordering: bool = False):
    """Step 1 + 2：层次聚类，返回 (linkage 矩阵, 准对角化后的资产顺序)"""
    dist = correlation_distance(cov)
    link = linkage(squareform(dist.to_numpy(), checks=False), method=method, optimal_ordering=optimal_ordering)
    return link, cov.index[leaves_list(link)].tolist()


def _cluster_var(cov: pd.DataFrame, items: list) -> float:
    """子簇内部用逆方差加权得到的簇方差"""
    sub = cov.loc[items, items].to_numpy()
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_weights(cov: pd.DataFrame, method: str = "single", optimal_ordering: bool = False,
                verbose: bool = False) -> pd.Series:
    """Step 3：递归二分（按列表中点切开，López de Prado 2016 原版）"""
    _, order = cluster_order(cov, method, optimal_ordering)
    w = pd.Series(1.0, index=order)
    clusters = [order]
    level = 0
    while clusters:
        clusters = [c[s:e] for c in clusters if len(c) > 1
                    for s, e in ((0, len(c) // 2), (len(c) // 2, len(c)))]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            v_l, v_r = _cluster_var(cov, left), _cluster_var(cov, right)
            alpha = 1 - v_l / (v_l + v_r)
            w[left] *= alpha
            w[right] *= 1 - alpha
            if verbose:
                print(f"   第{level + 1}层  {str(left):38s} | {str(right):38s}  "
                      f"σ_L={np.sqrt(v_l * TRADING_DAYS):6.2%}  σ_R={np.sqrt(v_r * TRADING_DAYS):6.2%}  α={alpha:.3f}")
        level += 1
    return w.reindex(cov.index)


def ivp_weights(cov: np.ndarray) -> np.ndarray:
    ivp = 1 / np.diag(cov)
    return ivp / ivp.sum()


def gmv_long_only(cov: np.ndarray) -> np.ndarray:
    w = cp.Variable(len(cov), nonneg=True)
    cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov * 1e4))), [cp.sum(w) == 1]).solve(solver=cp.CLARABEL)
    return np.clip(w.value, 0, None) / np.clip(w.value, 0, None).sum()


# ==================== 第一部分：相关距离与树聚类 ====================


def tree_clustering(stock_ret: pd.DataFrame, industry: pd.Series):
    print("=" * 60)
    print("第一部分：相关距离与层次聚类")
    print("=" * 60)
    print("""
    比喻：分家产。不让每个亲戚和所有亲戚比较（MV 要对整个 Σ 求逆），
    而是先按家族分支分（大房、二房），再在每一房内部往下分。

    相关距离 d = √(½(1-ρ))：
        ρ =  1.0 → d = 0.000（完全同涨同跌）
        ρ =  0.5 → d = 0.500
        ρ =  0.0 → d = 0.707
        ρ = -1.0 → d = 1.000
    """)
    cov = stock_ret.cov()
    link, order = cluster_order(cov)
    print(f"个股池 {stock_ret.shape[1]} 只，{stock_ret.index[0].date()} ~ {stock_ret.index[-1].date()}，single linkage 聚类")
    print("\n前 6 次合并（scipy linkage 的每一行 = 一次合并）：")
    labels = list(cov.index)
    members = {i: [labels[i]] for i in range(len(labels))}
    for k, (a, b, d, cnt) in enumerate(link[:6]):
        members[len(labels) + k] = members[int(a)] + members[int(b)]
        print(f"   合并 {members[int(a)]} + {members[int(b)]}  距离 {d:.3f}（对应相关 {1 - 2 * d**2:+.2f}）")
    for k in range(6, len(link)):
        a, b = int(link[k, 0]), int(link[k, 1])
        members[len(labels) + k] = members[a] + members[b]

    print("\n准对角化后的顺序（叶子顺序）与行业：")
    print("   " + " → ".join(f"{s}({industry[s]})" for s in order))
    same = sum(industry[a] == industry[b] for a, b in zip(order[:-1], order[1:]))
    sizes = industry.value_counts()
    n = len(order)
    expected = (n - 1) * float((sizes * (sizes - 1)).sum()) / (n * (n - 1))  # 随机排列下相邻同行业的期望对数
    print(f"   相邻两只属于同一行业：{same}/{n - 1} 对（随机排列的期望约 {expected:.1f} 对）")
    if same >= 2 * expected:
        print("   聚类只用了收益相关性，并没有告诉它行业标签——相关结构本身就包含了行业信息")
    else:
        print("   相邻同行业的对数与随机排列接近：这组收益里看不出行业结构（例如合成数据彼此独立）")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    dendrogram(link, labels=[f"{s}\n{industry[s]}" for s in labels], ax=ax, color_threshold=0, above_threshold_color="#2a78d6")
    ax.set_ylabel("相关距离")
    ax.set_title("个股层次聚类树（single linkage）")
    ax.tick_params(axis="x", labelsize=8)
    path = save_figure(fig, "05_dendrogram.png", subdir="phase-8")
    print(f"🖼  聚类树已保存：{path}")

    corr = stock_ret.corr()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, mat, title in [(axes[0], corr, "原始顺序"), (axes[1], corr.loc[order, order], "准对角化后")]:
        im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(mat)), mat.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(mat)), mat.index, fontsize=7)
        ax.set_title(f"相关矩阵（{title}）")
    fig.colorbar(im, ax=axes, shrink=0.8)
    path = save_figure(fig, "05_quasi_diag.png", subdir="phase-8")
    print(f"🖼  准对角化前后的相关矩阵已保存：{path}")


# ==================== 第二部分：递归二分 ====================


def recursive_bisection_demo(etf_ret: pd.DataFrame) -> pd.Series:
    print("\n" + "=" * 60)
    print("第二部分：递归二分 —— 一步步看权重怎么分下去")
    print("=" * 60)
    cov = etf_ret.cov()
    print(f"\n数据：{etf_ret.shape[1]} 只 ETF（含 2019 年底上市的豆粕），{etf_ret.index[0].date()} ~ {etf_ret.index[-1].date()}")
    print("\n相关距离矩阵：")
    print(correlation_distance(cov).round(3).to_string())
    _, order = cluster_order(cov)
    print(f"\n准对角化顺序：{order}")
    print("\n递归二分过程（σ 为子簇逆方差组合的年化波动；α 是分给左边的比例）：")
    w = hrp_weights(cov, verbose=True)
    dist = correlation_distance(cov)
    left, right = order[: len(order) // 2], order[len(order) // 2 :]
    cross = dist.loc[left, right].stack()
    (a, b), d_ab = cross.idxmin(), cross.min()
    all_pairs = np.sort(squareform(dist.to_numpy(), checks=False))
    rank = int(np.searchsorted(all_pairs, d_ab - 1e-12)) + 1
    print(f"\n   注意第 1 层的切口：跨越切口最相似的一对是 {a} 与 {b}（距离 {d_ab:.3f}，"
          f"在全部 {len(all_pairs)} 对中排第 {rank} 小）")
    print(f"\nHRP 权重（权重和 = {w.sum():.6f}）：")
    print((w * 100).round(2).to_string())
    print("""
    💡 每一层只比较"两个子簇"的风险，风险小的子簇分得多，永远不需要求逆。
       原版按列表中点二分，可能会把聚类树上本来属于同一簇的资产切开（HERC 等改进版按树结构切）。
    """)
    return w


# ==================== 第三部分：与 riskfolio-lib 对比 ====================


def compare_riskfolio(etf_ret: pd.DataFrame, w_hand: pd.Series):
    print("\n" + "=" * 60)
    print("第三部分：与 riskfolio-lib 的 HRP 对比")
    print("=" * 60)
    try:
        import riskfolio as rp
    except ImportError:
        print("   ⚠️ 未安装 riskfolio-lib，跳过对比")
        return

    rows = {}
    for leaf_order in [False, True]:
        try:
            port = rp.HCPortfolio(returns=etf_ret)
            w_rf = port.optimization(model="HRP", codependence="pearson", rm="MV", rf=0,
                                     linkage="single", leaf_order=leaf_order)["weights"]
        except Exception as e:
            print(f"   ⚠️ riskfolio 调用失败（{type(e).__name__}: {e}），跳过")
            return
        w_mine = w_hand if not leaf_order else hrp_weights(etf_ret.cov(), optimal_ordering=True)
        rows[f"手写(optimal_ordering={leaf_order})"] = w_mine
        rows[f"riskfolio(leaf_order={leaf_order})"] = w_rf.reindex(w_mine.index)
        print(f"   leaf_order={leaf_order!s:5s}：手写 vs riskfolio 最大权重差 = {np.abs(w_mine - w_rf.reindex(w_mine.index)).max():.1e}")
    print("\n权重 %：")
    print((pd.DataFrame(rows) * 100).round(2).to_string())
    print("""
    💡 riskfolio 的 HCPortfolio.optimization(model="HRP") 默认 leaf_order=True，
       即 scipy linkage 的 optimal_ordering：叶子顺序会被重排以让相邻叶子更相似。
       排序变了，"按中点二分"的切法就变了，权重随之不同——HRP 对这类实现细节是敏感的。
       对比两个库或复现论文时，务必对齐 codependence、linkage、leaf_order 这些参数。
    """)


# ==================== 第四部分：方法对比与样本外检验 ====================


def compare_methods(etf_ret: pd.DataFrame, stock_ret: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：HRP vs 逆方差 / ERC / GMV")
    print("=" * 60)

    cov = etf_ret.cov()
    C = cov.to_numpy() * TRADING_DAYS
    sols = {
        "逆方差": ivp_weights(C),
        "ERC": erc_weights(C),
        "GMV(多头)": gmv_long_only(C),
        "HRP": hrp_weights(cov).to_numpy(),
    }
    tbl = pd.DataFrame({k: v * 100 for k, v in sols.items()}, index=etf_ret.columns)
    print("ETF 权重 %（全样本协方差）：")
    print(tbl.round(1).to_string())
    for k, v in sols.items():
        print(f"   {k:8s} 年化波动 {np.sqrt(v @ C @ v):.2%}，有效资产数 1/Σw² = {1 / np.sum(v**2):.1f}")

    print(f"""
    样本外检验（个股池 {stock_ret.shape[1]} 只）：每月末用过去 120 天估计协方差（n/T = {stock_ret.shape[1] / 120:.2f}），持有一个月
    """)
    strategies = {
        "1/N": lambda x: np.ones(x.shape[1]) / x.shape[1],
        "逆方差": lambda x: ivp_weights(x.cov().to_numpy()),
        "GMV 闭式解(可做空)": lambda x: gmv_weights(x.cov().to_numpy()),
        "GMV(多头)": lambda x: gmv_long_only(x.cov().to_numpy() * TRADING_DAYS),
        "ERC": lambda x: erc_weights(x.cov().to_numpy() * TRADING_DAYS),
        "HRP": lambda x: hrp_weights(x.cov()).to_numpy(),
    }
    rets, extra = {}, {}
    for label, fn in strategies.items():
        r, W, to = walk_forward(stock_ret, fn, lookback=120)
        rets[label] = r
        extra[label] = {"年化换手": f"{to:.0%}", "平均最大单仓": f"{W.abs().max(axis=1).mean():.1%}"}
    print(perf_table(rets, extra).to_string())
    print("""
    💡 读表：
       - GMV 在"样本内"方差最小，但依赖 Σ⁻¹，样本外常伴随更高的换手和集中度
       - HRP 不求逆，集中度通常介于逆方差与 GMV 之间；聚类结构随窗口变化会让权重跳动，
         所以换手未必比逆方差低——看上表的"年化换手"列
       - López de Prado 的"HRP 样本外优于 MV"主要来自蒙特卡洛实验；
         在真实数据上谁更好取决于样本期，结论要靠你自己的滚动检验确认
    """)


# ==================== 主函数 ====================


def main():
    stock_ret, industry = load_stock_returns()
    etf_ret = load_etf_returns(list(ETF_UNIVERSE))
    tree_clustering(stock_ret, industry)
    w_hand = recursive_bisection_demo(etf_ret)
    compare_riskfolio(etf_ret, w_hand)
    compare_methods(etf_ret, stock_ret)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 linkage 从 single 改为 ward、average，聚类树和 HRP 权重怎么变？哪种更能还原行业？
    2. 实现原论文的"距离的距离"：d̃_ij = √Σ_k (d_ki - d_kj)²，再做聚类，和直接用 d 比较
    3. 用 riskfolio 的 model="HERC" 跑同样的数据，比较它与 HRP 的权重差异，读源码理解"按树二分"
    4. 复现 López de Prado 的蒙特卡洛实验：生成有块状相关结构的收益，比较 HRP、IVP、GMV 的样本外方差
    """)


if __name__ == "__main__":
    main()
