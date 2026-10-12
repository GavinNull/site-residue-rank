#!/usr/bin/env python3
"""第三个截尾检测器：**密度/混合模型型**，堆积的位置与宽度都从数据估，没有固定窗口。

为什么需要第三个（这是本仓库第五次静默工具失效的直接后果）
--------------------------------------------------------------
- 第一版（`ligand_specific_rank.py`：数「恰好等于最小值」的格子）在
  PD-CoV 上静默报 0.0%，真值约 46%（docs/13 §3）。
- 第二版（`subspace_lib.censor_fraction` = max(精确并列, `min+0.25` 窗口)）
  是**固定窗口**，docs/19 §2.2 的双向标定证明它两头都不对：
  σ ≈ 0 时偏高 +7..+10pp（窗口除了堆积还吃掉下限上方 0.25 以内的连续质量），
  σ ≳ w/3.5 ≈ 0.07 时窗口整段落在堆积下方 ——**塌回 ~0，又是静默欠检**。
  docs/19 §2.2 的 σ×w 二维表证明固定窗口**没有任何一个 w** 能同时对两端正确。

机制 → 估计量（这是设计的全部依据，不是经验公式）
--------------------------------------------------------------
真实机制（docs/19 §2.2 的合成网格按它写的，本模块逐字复用那个网格）：
真值低于检测下限 c 的格子被**压平到 c**，再叠一层 SD = σ 的测量噪声。
于是观测分布是一个**两成分混合**：

  * **堆积**：质量 q，分布 ≈ N(c, σ)；
  * **连续成分**：质量 1−q，**在 c 以下几乎没有质量**
    （只有被噪声漏下去的 O(0.4·σ·f(c)) 一点）。

这给出两个**互相独立**的 q 估计量，而且它们必须一致 —— 这一点就是自检：

  (I)  **反射法**：c 以下几乎只有堆积，而堆积关于 c 对称，所以
         q ≈ 2·P(X < c) + P(X = c) − (c 以下的硬赋原子质量)
       第二项处理 σ = 0 的原子（Starr ACE2 系列把非结合变体赋成
       5.000/6.000）；第三项防止「原子被当成对称堆积的下半」而**双计**。
  (II) **截断拟合法**：取最低 m 个观测，它们是 N(c, σ) 右截断在
       T = x_(m) 的样本。右截断正态的 (均值, 方差) 只依赖
       t = (T−c)/σ 与 σ：E[T−X] = σ(t+λ), Var = σ²(1−tλ−λ²)，λ = φ(t)/Φ(t)。
       所以 **CV² = Var/E² 只是 t 的函数且严格单调**（见 `_CV2` 表），
       一次二分就得到 t̂，继而 σ̂ = m̂1/(t̂+λ)、ĉ = T − σ̂t̂、
         q ≈ m / (n · Φ(t̂))。
       这是矩估计，没有优化器，不会像 MLE 那样在这类数据上发散。

**选 m 的判据不是调参，是 (I) 与 (II) 的一致性**：在 m 落在「纯堆积」区间时
两个估计量都无偏；m 太大（吃进连续成分）或太小（样本不够）时它们分叉。
所以扫 m，取 |q_II − q_I| / q_I 最小的那个 m。这同时是一个
**每个矩阵自带的自检量**（`resid`），在输出里逐个报。

存在性闸（既要命中也要不误伤 —— CLAUDE.md §10.2）
--------------------------------------------------------------
`σ̂ ≤ γ · s_up`，其中 `s_up` 是**上半分布**的鲁棒尺度
（由 Q60/Q75/Q90/Q95 两对给出，取大者）。
为什么用上半：压平只压低端，上半尺度不受截尾影响，是唯一干净的参照。
机制上这条闸就是假设 (A3)「堆积相对整体尺度是窄的」。
无截尾的单峰分布里拟合出来的「堆积」就是分布本体，σ̂ ≈ s_up，比值 ≈ 1；
真堆积的比值在本项目里是 0.02–0.25。两者差一个量级，所以 γ 不是敏感参数
（`censor_density_check.py` 对 γ ∈ {0.20, 0.25, 0.35, 0.50} 做扫描）。

假设（不满足时本检测器不适用，必须写明）
--------------------------------------------------------------
(A1) 机制是「压平到下限 + 叠噪声」，所以堆积关于 c 近似对称、近似高斯。
     若机制是「下限以下直接记成 NA」，数值里根本没有堆积，本检测器
     正确地报 ~0 —— 那种截尾要用**缺失率**度量（docs/15 的口径），
     是另一个量，不能和本量混用。
(A2) 连续成分在 c 以下没有实质质量。
(A3) σ ≪ s_up。σ ~ s_up 时堆积与分布本体不可分，本检测器报 ~0，
     而且**应该**报 ~0：那时「截尾比例」本身没有定义。
(A4) n >= 200 且堆积质量 >= ~2%，否则最低 m 个点不够估两个矩。

自检：`python3 analysis/censor_density_check.py`（复用 docs/19 §2.2 的
真实机制 σ×q 网格 + 本仓库的已知真实锚点，双向判据，不通过即中止）。
"""
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- 标定后固定的常数（敏感性扫描见 censor_density_check.py） ----
GAMMA = 0.35       # 闸 1：σ̂ / s_up 的上限（标定见 check 的 [S] 扫描）
CT_FRAC = 0.70     # 闸 2：σ̂ <= CT_FRAC * IQR/1.349（堆积不能比中心尺度还宽）
EV_LO, EV_HI = 0.6, 1.35  # 闸 3：(ĉ−min)/σ̂ 落在极值期望 z* 的这个倍数带内
TIE_FAC, TIE_ABS = 2.0, 0.01   # 「残差并列」的判定：resid <= TIE_FAC*min + TIE_ABS
SUB_FRAC = 0.80    # 子抽样比例（报区间用；不放回，不造人工并列）
N_SUB = 12         # 子抽样次数
M_MIN = 40         # 最低 m（A4：两个矩需要的最小样本）
M_MAX_FRAC = 0.60  # m 的上限（取到 60% 分位；再往上混合模型必然失效）
N_M = 160          # 扫 m 的格点数（几何等距）
ATOM_THR = 0.005   # 「硬赋原子」判为原子的最小质量
P_MIN = 0.02       # Φ(t) 下界：再小则 q_II 被 1/Φ 放大到无意义
TIE_TOL = 1e-6

# ---- CV^2(t) 查表 + 单调反演（替代逐点二分，快 ~50x，等价性由 check 核对） ----
_T_GRID = np.linspace(-6.0, 8.0, 2801)


def _lam(t):
    return norm.pdf(t) / np.maximum(norm.cdf(t), 1e-300)


def _cv2(t):
    l = _lam(t)
    return (1.0 - t * l - l * l) / np.maximum((t + l) ** 2, 1e-300)


_CV2 = _cv2(_T_GRID)                     # 严格单调递减（check 项 [1] 核对）
_T_OF_CV2 = (_CV2[::-1], _T_GRID[::-1])  # 升序，供 np.interp


def solve_t(c2):
    """由 CV² 反解 t（单调插值）。超出表范围时夹到端点。"""
    return float(np.interp(c2, _T_OF_CV2[0], _T_OF_CV2[1]))


def upper_scale(a):
    """上半分布的鲁棒尺度：压平只压低端，所以只能用上半做参照。

    取**三对上分位各自给出的尺度的最小值**，而不是最大值：
    取最大会被右侧长尾撑大（lehner/PSD95-PDZ3 的 (Q95−Q75)/0.971 = 8.9，
    而它的中心尺度只有 0.94），于是 σ̂/s_up 这道闸形同虚设 —— 实测中
    PSD95-PDZ3 正是因此被误判成 10.8% 截尾（已知事实：ddPCA 推出的
    ΔΔG，无下限赋值，必须 < 10%）。取最小值使这道闸**更难**通过，
    偏向不误伤那一侧。
    三对都避开 Q50 以下，所以 q <= 0.5 的堆积不会污染它们。
    """
    q = np.percentile(a, [55, 60, 75, 90, 95])
    return float(min((q[2] - q[0]) / 0.5483,      # (Q75-Q55)
                     (q[3] - q[1]) / 1.0283,      # (Q90-Q60)
                     (q[4] - q[2]) / 0.9709))     # (Q95-Q75)


def central_scale(a):
    """中心鲁棒尺度 IQR/1.349。只用于一条量级常识闸：
    **堆积不可能比分布自己的中心尺度还宽**。"""
    return float(np.subtract(*np.percentile(a, [75, 25]))) / 1.349


def _atom_masses(a, thr=ATOM_THR):
    u, cnt = np.unique(np.round(a, 9), return_counts=True)
    f = cnt / len(a)
    k = f >= thr
    return u[k], f[k]


def censor_fraction_density(raw, gamma=GAMMA, m_min=M_MIN, n_m=N_M,
                            atom_thr=ATOM_THR, detail=False):
    """密度/混合模型型截尾比例。

    detail=True 时返回 (q, info)，info 含 ĉ（以 min 为原点）、σ̂、σ̂/s_up、
    选中的 m、两个估计量 q_I / q_II 及其相对残差 —— 用于逐个数字审计。
    """
    a = np.asarray(raw, float)
    a = np.sort(a[np.isfinite(a)])
    n = len(a)
    info = {"n": int(n), "gate": None, "why": "", "c_rel": None,
            "sigma_hat": None, "sigma_over_s": None, "m": None,
            "q_atom": 0.0, "q_refl": None, "q_trunc": None, "resid": None,
            "s_up": None, "s_ct": None, "ev_over_zstar": None}
    if n < 200:
        info["gate"] = "n 太少"
        info["why"] = "n < 200（假设 A4）"
        return (0.0, info) if detail else 0.0
    lo = float(a[0])
    q_atom = float((a <= lo + TIE_TOL).mean())
    info["q_atom"] = q_atom
    s_up = upper_scale(a)
    s_ct = central_scale(a)
    info["s_up"] = s_up
    info["s_ct"] = s_ct
    if s_up <= 0:
        info["gate"] = "尺度为 0"
        return (q_atom, info) if detail else q_atom

    av, af = _atom_masses(a, atom_thr)
    cands = []
    ms = np.unique(np.round(np.geomspace(m_min, max(m_min + 1,
                                                    int(M_MAX_FRAC * n)),
                                         n_m)).astype(int))
    for m in ms:
        if m >= n:
            break
        T = a[m]
        d = T - a[:m]
        m1 = float(d.mean())
        if m1 <= 0:
            continue
        t = solve_t(float(d.var(ddof=1)) / (m1 * m1))
        l = float(_lam(t))
        if (t + l) <= 0:
            continue
        sig = m1 / (t + l)
        c = T - sig * t
        P = float(norm.cdf(t))
        if (sig <= 0 or sig > gamma * s_up or sig > CT_FRAC * s_ct
                or P < P_MIN):
            continue
        q_II = m / (n * P)
        dbl = float(af[av < c - TIE_TOL].sum())   # 防止硬赋原子被双计
        q_I = (2.0 * float((a < c - TIE_TOL).mean())
               + float((np.abs(a - c) <= TIE_TOL).mean()) - dbl)
        if q_I <= 0:
            continue
        # 闸 3（极值一致性）：若堆积真有 q_I·n 个点、宽 σ̂，则样本最小值应落在
        # ĉ 以下约 z* = Φ^{-1}(1 − 1/(q_I·n + 1)) 个 σ̂ 处。
        # 这条闸是把 GB1 / PSD95-PDZ3 这类「尖峰 + 长右尾」的**分布本体**
        # 从「下限堆积」里分出来的关键：它们的 (ĉ−min)/σ̂ 高达 5，
        # 而真堆积是 2.5–3.4。少了这一条，GB1 会被误判成 67% 截尾。
        z_star = float(norm.ppf(1.0 - 1.0 / (max(q_I * n, 4.0) + 1.0)))
        ev = (c - lo) / sig
        if not (EV_LO * z_star <= ev <= EV_HI * z_star):
            continue
        resid = abs(q_II - q_I) / max(q_I, 1e-9)
        cands.append((resid, int(m), float(c), float(sig), float(q_II),
                      float(q_I), float(ev / z_star)))
    if not cands:
        info["gate"] = "无堆积"
        info["why"] = (f"扫遍 m 都没有同时满足 σ̂/s_up <= {gamma}、"
                       f"σ̂ <= IQR/1.349 与 Φ(t) >= {P_MIN} 的拟合："
                       f"低端没有窄堆积")
        return (q_atom, info) if detail else q_atom
    # **残差并列必须取平均，不能取 argmin**：实测里常有一个小 m 的
    # 「部分堆积」拟合与正确的大 m 拟合残差几乎相等（0.0018 vs 0.0026），
    # argmin 于是变成抛硬币 —— 合成网格上约 1/8 的重复会因此塌到真值的
    # 1/4 以下。那正是本检测器要修掉的静默欠检，所以这里对所有
    # 「残差与最小值并列」的候选取中位数。
    r0 = min(z[0] for z in cands)
    sel = [z for z in cands if z[0] <= TIE_FAC * r0 + TIE_ABS]
    q_I = float(np.median([z[5] for z in sel]))
    resid = r0
    m = int(np.median([z[1] for z in sel]))
    c = float(np.median([z[2] for z in sel]))
    sig = float(np.median([z[3] for z in sel]))
    q_II = float(np.median([z[4] for z in sel]))
    evr = float(np.median([z[6] for z in sel]))
    info["n_cand"] = len(cands)
    info["n_tie"] = len(sel)
    info.update({"gate": "通过", "why": "检出下限堆积", "m": m,
                 "c_rel": c - lo, "sigma_hat": sig,
                 "sigma_over_s": sig / s_up, "q_refl": q_I,
                 "q_trunc": q_II, "resid": resid, "ev_over_zstar": evr})
    q = float(max(q_atom, min(q_I, 1.0)))
    return (q, info) if detail else q


def censor_fraction_density_ci(raw, n_sub=N_SUB, frac=SUB_FRAC, seed=0,
                               **kw):
    """点估计 + 子抽样区间（CLAUDE.md §5：点估计不单独汇报）。

    **为什么必须带区间**：逐矩阵的点估计方差大（合成网格上单次重复的
    跨度可达 ±15pp，偶发会掉到真值的 1/4 以下）。带上区间，
    「这一次拟合塌了」就**看得见**，而不是又一次静默欠检 ——
    这正是本检测器要修掉的失效模式，所以区间不是装饰。
    子抽样用**不放回**（80%），不用有放回 bootstrap：有放回会制造
    人工并列，污染原子分支。

    返回 {"q": 点估计, "lo"/"hi": 10–90% 子抽样区间, "sub": 全部子抽样值,
          "info": 点估计的审计信息}
    """
    a = np.asarray(raw, float)
    a = a[np.isfinite(a)]
    q, info = censor_fraction_density(a, detail=True, **kw)
    rng = np.random.default_rng(seed)
    k = max(200, int(frac * len(a)))
    sub = []
    if len(a) > k:
        for _ in range(n_sub):
            idx = rng.choice(len(a), size=k, replace=False)
            sub.append(censor_fraction_density(a[idx], **kw))
    out = {"q": float(q), "info": info,
           "sub": [float(x) for x in sub]}
    if sub:
        out["lo"] = float(np.percentile(sub, 10))
        out["hi"] = float(np.percentile(sub, 90))
        out["sub_med"] = float(np.median(sub))
    else:
        out["lo"] = out["hi"] = out["sub_med"] = float(q)
    return out


# ---------------------------------------------------------------- 合成机制
def synth_censored(rng, q, sigma, nr=200, nc=20):
    """docs/19 §2.2 的真实机制网格，**逐字复用**（不发明更容易的网格）。

    与 `censor_recheck._synth_censored` 必须逐元素相同；由
    `censor_density_check.py` 的 [0] 项用同一种子核对，不核对就不许用。
    """
    a = rng.normal(0, 0.8, nr)
    b = rng.normal(0, 0.6, nc)
    v = (a[:, None] + b[None, :] + rng.normal(0, 0.3, (nr, nc))
         + 7.0).ravel()
    if q > 0:
        thr = np.quantile(v, q)
        v = np.where(v < thr, thr, v)
    if sigma > 0:
        v = v + rng.normal(0, sigma, v.shape)
    return v


def raw_values(d):
    """矩阵记录 -> 原生数值全集（含下限堆积）。

    `subspace_lib._starr_matrix` 在建矩阵时把恰好等于最小值的格子剔掉了，
    所以**不能**用 M[mask] 当原始数值（sarbecovirus 的重度截尾列会丢掉
    83–93% 的堆积）。那里现在额外存了 `raw`。
    """
    if "raw" in d:
        return np.asarray(d["raw"], float)
    return d["M"][d["mask"]]
