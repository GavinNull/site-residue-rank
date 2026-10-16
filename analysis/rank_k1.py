#!/usr/bin/env python3
"""去偏定秩程序 K1（联合偏置拟合）的生产模块 + 双向自检。

来源与动机（docs/43 §8、D-090、D-091）
----------------------------------------
现行程序 `rank_core.cv_rank_noleak` 在每折**训练格子**上先估主效应（行/列偏置），再对残差拟合低秩项。
缺失使行/列上的耦合均值在训练折里不为零，主效应因此吸收一部分耦合，在留出格子上留下类似额外低秩成分的偏移；
耦合/噪声比高时定秩偏高（rank_core.selfcheck C6b；docs/43 把它量化成一条悬崖：s/σ ≳ 2.5–3 时真秩 1 几乎总被报 2）。
K1 去掉这一步：每折在训练格子上**联合**拟合  r_i + c_j + U Vᵀ，r、c 是不受岭惩罚的偏置，岭只作用于 U、V。
没有「先估主效应」的两步，也就没有被吸收的那一部分。K1 不依赖 σ（不引入任何噪声水平的估计）。

与 `cv_rank_noleak` 的接口约定
-----------------------------
`cv_rank_k1(M, mask, max_rank, K, restarts, seed, iters, ridge)`：参数名、默认值、返回值逐项相同，
返回 {rank: (rmse_full, rmse_trim95, rmse_median)}；主指标 rmse_trim95（裁掉最差 5%，与头条一致）。
所以可以直接传给 `rank_core.best_rank` / `rank_core.gains`。

关于 als_guarded（与任务书的一处偏离，如实记录）
-----------------------------------------------
任务书要求「复用 als_guarded，不用未保护的 als_complete」。K1 的模型（偏置列不受岭惩罚、与因子联合更新）
**不是** als_guarded 能表达的：als_guarded 拟合的是已去中心化的残差上的纯低秩项。所以 K1 不能字面调用 als_guarded，
而是用一个**保护语义相同**的联合版本 `als_joint`：训练观测数 < rank+2 的行/列（因子 + 偏置共 rank+1 个参数，
比 als_guarded 的 rank 个参数多一个，故阈值是 rank+2 而非 rank+1），因子置 0、偏置取观测均值——即欠定行列退化为加性预测，
与 als_guarded「预测去中心均值」的保护语义一致。该保护由自检 K-guard 直接验证（欠定行不爆、预测等于加性预测）。
并且 `als_joint` 与 docs/43 的原型 `overshoot_lib.als_joint` 逐位一致（自检 K-proto，D-023 式「新实现 == 旧实现」核对）。
**未使用** `rank_core.als_complete` / `rank_power.als_complete`。

留出格子无泄露
--------------
每折：训练 mask = mask 去掉本折格子；偏置初值 `center_fit(M, trn)` 只看训练格子；ALS 只看训练格子；
训练误差（用来在重启间择优）只在训练格子上算；留出格子的真值只在算误差的最后一步出现。
自检 K-leak 用已知事实核对：改变一个格子的值，该格子被留出时的预测不变。

CLAUDE.md §10：本模块任何调用方必须先调 `selfcheck()`，不通过则抛 AssertionError 中止。
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rank_core as RC                                   # noqa: E402

# 头条口径的 CV 设置（与 overshoot_lib.CV、independence_audit.CV 一致）
CV_HEADLINE = dict(K=5, restarts=2, iters=150, ridge=0.1)
SEEDS_HEADLINE = (1, 2, 3)
MAXR_HEADLINE = 3


# ======================================================================
# 联合拟合
# ======================================================================
def als_joint(Y, mask, rank, rng, iters=150, ridge=0.1, r0=None, c0=None):
    """联合拟合 Y ≈ r_i + c_j + U Vᵀ，只在 mask 内。返回完整预测 P = r + c + U Vᵀ（全矩阵）。

    r、c 不受岭惩罚（正则矩阵里偏置那一维只有 1e-8 的数值稳定项）。
    欠定行/列保护：观测数 < rank+2 的行/列，因子置 0、偏置取其观测均值。
    """
    nr, nc = Y.shape
    k = rank
    W = mask.astype(float)
    r = np.zeros(nr) if r0 is None else r0.copy()
    c = np.zeros(nc) if c0 is None else c0.copy()
    U = rng.normal(0, 0.1, (nr, k))
    V = rng.normal(0, 0.1, (nc, k))
    Ym = np.where(mask, Y, 0.0)
    Ireg = np.diag([max(ridge, 1e-8)] * k + [1e-8]) if k > 0 else np.array([[1e-8]])
    cnt_r = mask.sum(1)
    cnt_c = mask.sum(0)
    ok_r = cnt_r >= k + 2
    ok_c = cnt_c >= k + 2
    for _ in range(iters):
        # 行步：θ_i=[u_i, r_i]，回归量 [v_j, 1]，目标 y − c_j
        Z = np.hstack([V, np.ones((nc, 1))])
        ZZ = (Z[:, :, None] * Z[:, None, :]).reshape(nc, -1)
        A = (W @ ZZ).reshape(nr, k + 1, k + 1) + Ireg
        b = (W * (Ym - c[None, :])) @ Z
        th = np.linalg.solve(A, b[:, :, None])[:, :, 0]
        U = th[:, :k]
        r = th[:, k]
        if (~ok_r).any():
            U[~ok_r] = 0.0
            tot = (W * (Ym - c[None, :])).sum(1)
            r[~ok_r] = np.where(cnt_r[~ok_r] > 0, tot[~ok_r] / np.maximum(cnt_r[~ok_r], 1), 0.0)
        # 列步：θ_j=[v_j, c_j]，回归量 [u_i, 1]，目标 y − r_i
        Z2 = np.hstack([U, np.ones((nr, 1))])
        ZZ2 = (Z2[:, :, None] * Z2[:, None, :]).reshape(nr, -1)
        A2 = (W.T @ ZZ2).reshape(nc, k + 1, k + 1) + Ireg
        b2 = (W * (Ym - r[:, None])).T @ Z2
        th2 = np.linalg.solve(A2, b2[:, :, None])[:, :, 0]
        V = th2[:, :k]
        c = th2[:, k]
        if (~ok_c).any():
            V[~ok_c] = 0.0
            tot = (W * (Ym - r[:, None])).sum(0)
            c[~ok_c] = np.where(cnt_c[~ok_c] > 0, tot[~ok_c] / np.maximum(cnt_c[~ok_c], 1), 0.0)
    return r[:, None] + c[None, :] + (U @ V.T if k > 0 else 0.0)


# ======================================================================
# 定秩（接口与 rank_core.cv_rank_noleak 相同）
# ======================================================================
def _folds(mask, K, seed):
    rng = np.random.default_rng(seed)
    idx = np.argwhere(mask)
    rng.shuffle(idx)
    return rng, idx, np.array_split(idx, K)


def _fold_errors(Mf, mask, folds, rng, rank, restarts, iters, ridge):
    """返回与 np.concatenate(folds) 同序的留出平方误差。"""
    errs = []
    for f in folds:
        trn = mask.copy()
        for i, j in f:
            trn[i, j] = False
        r0, c0 = RC.center_fit(Mf, trn)                 # 只看训练格子，仅作初值
        best, bestP = np.inf, None
        for _ in range(restarts):
            P = als_joint(Mf, trn, rank, rng, iters=iters, ridge=ridge, r0=r0, c0=c0)
            tr = np.mean((Mf[trn] - P[trn]) ** 2)        # 按**训练**误差选重启
            if tr < best:
                best, bestP = tr, P
        for i, j in f:
            errs.append((Mf[i, j] - bestP[i, j]) ** 2)
    return np.array(errs)


def cv_rank_k1(M, mask, max_rank=4, K=5, restarts=3, seed=3, iters=250, ridge=0.1):
    """K1 去偏定秩。返回 {rank: (rmse_full, rmse_trim95, rmse_median)}。"""
    Mf = np.where(mask, M, 0.0)
    rng, _, folds = _folds(mask, K, seed)
    out = {}
    for rank in range(max_rank + 1):
        e = _fold_errors(Mf, mask, folds, rng, rank, restarts, iters, ridge)
        out[rank] = (math.sqrt(e.mean()), math.sqrt(np.quantile(e, 0.95)),
                     math.sqrt(np.median(e)))
    return out


def cv_cell_errors_k1(M, mask, max_rank=3, K=5, restarts=2, seed=3, iters=150, ridge=0.1):
    """逐格留出平方误差（供位点聚类 bootstrap）。返回 (cells, {rank: err})，
    cells = np.argwhere(mask)，err 与 cells 同序。"""
    Mf = np.where(mask, M, 0.0)
    rng, idx, folds = _folds(mask, K, seed)
    key = {(int(i), int(j)): n for n, (i, j) in enumerate(np.argwhere(mask))}
    order = np.array([key[(int(i), int(j))] for i, j in idx])
    out = {}
    for rank in range(max_rank + 1):
        e = _fold_errors(Mf, mask, folds, rng, rank, restarts, iters, ridge)
        full = np.empty(len(idx))
        full[order] = e
        out[rank] = full
    return np.argwhere(mask), out


def cv_cell_errors_noleak(M, mask, max_rank=3, K=5, restarts=2, seed=3, iters=150, ridge=0.1):
    """现行程序的逐格误差，同一约定（供「旧 vs 新」并列）。"""
    Mf = np.where(mask, M, 0.0)
    rng, idx, folds = _folds(mask, K, seed)
    key = {(int(i), int(j)): n for n, (i, j) in enumerate(np.argwhere(mask))}
    order = np.array([key[(int(i), int(j))] for i, j in idx])
    out = {}
    for rank in range(max_rank + 1):
        errs = []
        for f in folds:
            trn = mask.copy()
            for i, j in f:
                trn[i, j] = False
            r, c = RC.center_fit(Mf, trn)
            R = Mf - r[:, None] - c[None, :]
            Yt = np.where(trn, R, 0.0)
            best, bestP = np.inf, None
            for _ in range(restarts):
                P = RC.als(Yt, trn, rank, rng, iters=iters, ridge=ridge)
                tr = np.mean((R[trn] - P[trn]) ** 2)
                if tr < best:
                    best, bestP = tr, P
            for i, j in f:
                errs.append((R[i, j] - bestP[i, j]) ** 2)
        full = np.empty(len(idx))
        full[order] = np.array(errs)
        out[rank] = full
    return np.argwhere(mask), out


def trim95(e):
    return float(math.sqrt(np.quantile(e, 0.95)))


# ======================================================================
# 头条口径（3 种子均值曲线的 argmin，并列取小秩）
# ======================================================================
def headline(M, mask, procedure="k1", seeds=SEEDS_HEADLINE, max_rank=MAXR_HEADLINE,
             cv=None):
    """返回 dict(rank=headline argmin, mean_curve={r: trim95 均值}, curves=[每种子 {r: trim95}],
    seed_bests=[每种子 argmin], mode_rank=种子众数)。procedure ∈ {'k1','noleak'}。"""
    cv = dict(CV_HEADLINE if cv is None else cv)
    fn = {"k1": cv_rank_k1, "noleak": RC.cv_rank_noleak}[procedure]
    Mf = np.where(mask, M, 0.0)
    curves, bests = [], []
    for sd in seeds:
        e = fn(Mf, mask, max_rank=max_rank, seed=sd, **cv)
        c = {r: float(e[r][1]) for r in e}
        curves.append(c)
        bests.append(_argmin(c))
    mc = {r: float(np.mean([c[r] for c in curves])) for r in curves[0]}
    vals, cnt = np.unique(bests, return_counts=True)
    return {"rank": _argmin(mc), "mean_curve": mc, "curves": curves, "seed_bests": bests,
            "mode_rank": int(vals[np.argmax(cnt)]), "stab": float(cnt.max() / len(bests))}


def _argmin(curve):
    m = min(curve.values())
    return min(r for r in curve if curve[r] <= m + 1e-15)


# ======================================================================
# 自检（CLAUDE.md §10：已知事实；既测命中，也测不误伤；并证明校正确实起作用）
# ======================================================================
def selfcheck(verbose=True):
    """全部断言用已知事实。失败抛 AssertionError。返回日志行列表。"""
    log = []
    rng0 = np.random.default_rng(2026)

    # K-fact：无噪声完备，联合拟合精确还原 主效应 + 秩 k（k = 0, 1, 2）
    for k in (0, 1, 2):
        a = rng0.normal(0, 1.5, 25)
        b = rng0.normal(0, 1.5, 18)
        T = a[:, None] + b[None, :]
        if k:
            T = T + rng0.normal(0, 1, (25, k)) @ rng0.normal(0, 1, (18, k)).T
        mk = np.ones((25, 18), bool)
        P = als_joint(T, mk, k, np.random.default_rng(1), iters=800, ridge=1e-8)
        rel = np.linalg.norm(T - P) / np.linalg.norm(T)
        assert rel < 1e-4, f"als_joint 无噪声秩{k}未还原 rel={rel:.2e}"
    log.append("K-fact 无噪声完备：联合拟合精确还原 主效应+秩 0/1/2（rel<1e-4）  OK")

    # K-add：秩 0 的联合拟合 == 缺失下收敛的加性最小二乘（center_fit 迭代到收敛）。已知事实：同一个最小二乘问题
    A = rng0.normal(0, 1, (40, 14)) + rng0.normal(0, 1.5, 40)[:, None] + rng0.normal(0, 1, 14)[None, :]
    mk = rng0.random((40, 14)) > 0.3
    Pj = als_joint(np.where(mk, A, 0.0), mk, 0, np.random.default_rng(0), iters=400, ridge=0.1)
    r_, c_ = RC.center_fit(np.where(mk, A, 0.0), mk, iters=2000)
    d = np.abs((Pj - (r_[:, None] + c_[None, :]))[mk]).max()
    assert d < 1e-6, f"秩 0 联合拟合与收敛的加性拟合不一致 {d:.2e}"
    log.append(f"K-add 秩 0 联合拟合 == 收敛的加性最小二乘（观测格 max|Δ|={d:.1e}）  OK")

    # K-guard：欠定行/列保护。训练观测数 < rank+2 的行：因子置 0，预测 = 加性预测，有限且量级同数据
    T = rng0.normal(0, 1, (30, 12)) + 3.0
    mk = rng0.random((30, 12)) > 0.4
    mk[4] = False                      # 整行缺失
    mk[:, 5] = False                   # 整列缺失
    mk[7, :] = False
    mk[7, :2] = True                   # 只有 2 个观测（< rank+2 = 4）
    P = als_joint(np.where(mk, T, 0.0), mk, 2, np.random.default_rng(2), iters=60, ridge=0.1)
    assert np.isfinite(P).all() and np.abs(P).max() < 50, "欠定行列下联合拟合爆了"
    # 已知事实：第 4 行整行缺失 => r4=0、U4=0 => P[4,:] = c；第 7 行只有 2 个观测(<4) => U7=0 => P[7,:] = r7 + c
    # 所以 P[7,:] − P[4,:] 在所有列上是同一个常数 r7（欠定行退化为加性预测）。
    dd = P[7] - P[4]
    assert dd.max() - dd.min() < 1e-9, f"欠定行未退化为加性预测 (spread {dd.max()-dd.min():.2e})"
    log.append("K-guard 整行/整列缺失、欠定行(2 个观测)下联合拟合有限、量级同数据  OK")

    # K-leak：留出格子的预测与它自己的值无关（泄露哨兵，已知事实）
    M0, mk0 = RC._synth(40, 12, 1, 0.5, np.random.default_rng(3))
    mk0 = mk0 & (np.random.default_rng(4).random(M0.shape) > 0.2)
    cells, E = cv_cell_errors_k1(M0, mk0, max_rank=2, seed=5, restarts=2, iters=60)
    for pick in (0, len(cells) // 2, len(cells) - 1):
        i, j = cells[pick]
        M1 = M0.copy()
        M1[i, j] += 50.0
        _, E1 = cv_cell_errors_k1(M1, mk0, max_rank=2, seed=5, restarts=2, iters=60)
        for r in (1, 2):
            e0, e1 = E[r][pick], E1[r][pick]
            # 预测 P 不变 => e1 = (±sqrt(e0) + 50)^2 之一
            ok = min(abs(e1 - (math.sqrt(e0) + 50) ** 2), abs(e1 - (50 - math.sqrt(e0)) ** 2))
            assert ok < 1e-6 * max(1.0, e1), f"留出预测依赖于被留出格子自己的值（泄露）: cell {pick} 秩{r}"
    log.append("K-leak 改变一个格子的值，该格子被留出时的预测不变（3 格 × 秩1,2）  OK")

    # K-proto：生产实现 == docs/43 原型 overshoot_lib（逐位；防 D-023 式静默偏差）
    import overshoot_lib as OL
    Mp, mkp = RC._synth(45, 14, 2, 0.4, np.random.default_rng(9))
    mkp = mkp & (np.random.default_rng(10).random(Mp.shape) > 0.25)
    cv = dict(restarts=2, iters=150, ridge=0.1)
    e_new = cv_rank_k1(Mp, mkp, max_rank=3, K=5, seed=3, **cv)
    c_old = OL.cv_curve(np.where(mkp, Mp, 0.0), mkp, range(4), seed=3, joint=True, **cv)
    for r in range(4):
        assert abs(e_new[r][1] - c_old[r]) < 1e-12, (r, e_new[r][1], c_old[r])
    cells_n, En = cv_cell_errors_k1(Mp, mkp, max_rank=3, seed=3, **cv)
    cells_o, Eo = OL.cv_cell_errs(np.where(mkp, Mp, 0.0), mkp, range(4), 3, joint=True)
    assert np.array_equal(cells_n, cells_o)
    for r in range(4):
        assert np.abs(En[r] - Eo[r]).max() < 1e-12
    # 现行程序的逐格版本也与原型一致，且与 rank_core.cv_rank_noleak 的 trim95 一致
    _, Enl = cv_cell_errors_noleak(Mp, mkp, max_rank=3, seed=3, **cv)
    enl = RC.cv_rank_noleak(Mp, mkp, max_rank=3, K=5, seed=3, **cv)
    for r in range(4):
        assert abs(trim95(Enl[r]) - enl[r][1]) < 1e-12
    log.append("K-proto 生产 K1 == docs/43 原型（trim95 曲线与逐格误差逐位 <1e-12）；"
               "现行程序逐格版 == rank_core.cv_rank_noleak  OK")

    # K-trim：裁尾指标的已知值
    assert abs(trim95(np.arange(101.0) ** 2) - math.sqrt(95.0 ** 2)) < 1e-9
    log.append("K-trim trim95 已知值  OK")

    # ---- 命中 / 不误伤 / 校正确实起作用（植入数据；完备 60×20，rank_core._synth 的设计点）
    def both(rank, sigma, seed, miss=0.0):
        M, mk = RC._synth(60, 20, rank, sigma, np.random.default_rng(seed))
        if miss:
            mk = np.random.default_rng(seed + 1).random(M.shape) > miss
        k1 = headline(M, mk, "k1")["rank"]
        old = headline(M, mk, "noleak")["rank"]
        return k1, old

    seeds3 = (7, 17, 27)
    # 命中：植入秩 2，中等与高信噪比，K1 必须给 2
    t2 = {sg: [both(2, sg, s) for s in seeds3] for sg in (0.64, 0.15)}
    for sg, res in t2.items():
        assert all(k1 == 2 for k1, _ in res), f"K1 未命中植入秩 2 @σ={sg}: {res}"
    # 不误伤：植入秩 1，同样的信噪比（含现行程序失败的高信噪比），K1 必须不报 2，且恰为 1
    t1 = {sg: [both(1, sg, s) for s in seeds3] for sg in (0.64, 0.3, 0.15)}
    for sg, res in t1.items():
        assert all(k1 == 1 for k1, _ in res), f"K1 把植入秩 1 报成别的 @σ={sg}: {res}"
    # 带缺失的高信噪比秩 1（真实矩阵有 ~20–30% 缺失；C6b 的偏差正来自缺失）
    tm = [both(1, 0.15, s, miss=0.25) for s in seeds3]
    assert all(k1 == 1 for k1, _ in tm), f"带缺失高信噪比秩 1: K1 {tm}"
    # 结构缺失 -> 0
    t0 = [both(0, 0.3, s)[0] for s in seeds3]
    assert all(k == 0 for k in t0), f"结构缺失未给 0: {t0}"
    # 回归：现行程序在这些高信噪比秩 1 上过冲（校正确实起作用，不是原本就对）
    over = [old for sg in (0.3, 0.15) for _, old in t1[sg]]
    assert all(o >= 2 for o in over), f"现行程序没有复现高信噪比过冲（要么被修了，要么设置变了）: {over}"
    over_m = [old for _, old in tm]
    assert sum(o >= 2 for o in over_m) >= 2, f"带缺失时现行程序过冲 <2/3: {over_m}"
    # 回归：现行程序对植入秩 2 在高信噪比被报 3（docs/43：真 2 -> 3）
    assert all(old >= 3 for _, old in t2[0.15]), f"现行程序高信噪比下秩 2 未被抬到 3: {t2[0.15]}"
    # 现行程序在中等信噪比是对的（说明回归是 SNR 依赖的，不是无条件的）
    assert all(old == 1 for _, old in t1[0.64]) and all(old == 2 for _, old in t2[0.64])
    log.append("K-hit 植入秩 2 @σ=0.64/0.15 -> K1=2（3/3 种子各）  OK")
    log.append("K-nofool 植入秩 1 @σ=0.64/0.3/0.15 -> K1=1（3/3 各）；带 25% 缺失的高信噪比秩 1 -> K1=1（3/3）  OK")
    log.append("K-null 结构缺失 -> K1=0（3/3）  OK")
    log.append("K-regress 现行程序同一批矩阵：秩 1 @σ=0.3/0.15 -> "
               + str([o for sg in (0.3, 0.15) for _, o in t1[sg]])
               + "（过冲）；带缺失 -> " + str(over_m) + "；秩 2 @σ=0.15 -> "
               + str([o for _, o in t2[0.15]]) + "（被抬到 3）；σ=0.64 处两者一致  OK")
    if verbose:
        for s in log:
            print("  " + s, flush=True)
    return log


if __name__ == "__main__":
    print("rank_k1 自检（先过 rank_core.selfcheck）")
    RC.selfcheck(verbose=False)
    print("  rank_core.selfcheck 通过")
    selfcheck()
    print("全部通过")
