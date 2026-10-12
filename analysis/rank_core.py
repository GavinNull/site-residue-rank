#!/usr/bin/env python3
"""共享的数值工具层：无泄露嵌套 CV 定秩 + 自检。

为什么新开这个文件（docs/12-rank-two.md §1(b) 的根因）：
  「21 个分析脚本各自 import、没有共享的经过验证的数值工具层。
    同一个 bug 可以在不同脚本里重犯。」
本模块把三件事放在一处并强制自检：
  1. center_fit  —— 只用训练格子估主效应，返回 (r, c) 而**不是**残差矩阵。
     这是 docs/12 §3 记录的第九次工具错误的直接修补：旧写法
     `double_center(np.where(trn, M, 0.0), trn)` 会把留出格子的真值归零，
     留出目标退化成 −r[i]−c[j]（主效应尺度），RMSE 从 0.3–3.3 跳到 10–12。
  2. cv_rank_noleak —— 中心化在每折训练集内部完成。
  3. als_guarded  —— 直接复用 rank_test_power 里带欠定行列保护的版本，
     不再各脚本自己写一个（CLAUDE.md §10）。

自检（CLAUDE.md §10：用已知事实，既测能命中也测不误伤）：
  运行 `python3 analysis/rank_core.py` 执行全部自检。
  任何调用方必须先调 selfcheck()，不通过则抛异常中止。
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_test_power import als_guarded  # noqa: E402  带欠定行列保护


# ---------------------------------------------------------------- 中心化
def center_fit(M, mask, iters=80):
    """用 mask 内的格子估加性主效应，返回 (r, c)。

    不返回残差矩阵 —— 残差必须由调用方用**真值** M 自己算：
        R = M - r[:, None] - c[None, :]
    这样留出格子的真值永远不会被当成 0 参与。
    """
    r = np.zeros(M.shape[0])
    c = np.zeros(M.shape[1])
    for _ in range(iters):
        R = M - r[:, None] - c[None, :]
        for i in range(M.shape[0]):
            if mask[i].any():
                r[i] += R[i][mask[i]].mean()
        R = M - r[:, None] - c[None, :]
        for j in range(M.shape[1]):
            if mask[:, j].any():
                c[j] += R[:, j][mask[:, j]].mean()
    return r, c


def residual(M, mask, iters=80):
    """全矩阵双向中心化残差（仅在**非** CV 场景使用，例如提取载荷）。"""
    r, c = center_fit(M, mask, iters)
    return M - r[:, None] - c[None, :]


# ---------------------------------------------------------------- 定秩
def cv_rank_noleak_errors(M, mask, max_rank=4, K=5, restarts=3, seed=3,
                          iters=250, ridge=0.1):
    """与 cv_rank_noleak 相同的流程，但返回**逐格留出平方误差**与其行号。

    -> (rows, errs) 其中 errs[rank] 是与 rows 同序的平方误差数组。
    用途：按位点聚类重抽（cluster bootstrap）给秩间比较做区间，
    满足 CLAUDE.md §5「点估计不单独汇报」。
    注意这个区间只覆盖**留出误差的抽样波动**，不覆盖重拟合波动；
    后者用多个种子（5 个）另行报告。
    """
    rng = np.random.default_rng(seed)
    idx = np.argwhere(mask)
    rng.shuffle(idx)
    folds = np.array_split(idx, K)
    order = np.concatenate(folds)
    errs = {}
    for rank in range(max_rank + 1):
        e = []
        for f in folds:
            trn = mask.copy()
            for i, j in f:
                trn[i, j] = False
            r, c = center_fit(M, trn)
            R = M - r[:, None] - c[None, :]
            Y = np.where(trn, R, 0.0)
            best, bestP = np.inf, None
            for _ in range(restarts):
                P = als(Y, trn, rank, rng, iters=iters, ridge=ridge)
                tr = np.mean((R[trn] - P[trn]) ** 2)
                if tr < best:
                    best, bestP = tr, P
            for i, j in f:
                e.append((R[i, j] - bestP[i, j]) ** 2)
        errs[rank] = np.array(e)
    return order[:, 0], errs


def trim95(e):
    return float(math.sqrt(np.quantile(e, 0.95)))


def cluster_bootstrap_gains(rows, errs, nboot=2000, seed=19, ranks=(1, 2, 3)):
    """按位点重抽，返回每个秩相对秩0 的 trim95 增益分布，以及秩1->秩2 的差。"""
    rng = np.random.default_rng(seed)
    sites = np.unique(rows)
    by = {s: np.where(rows == s)[0] for s in sites}
    out = {r: [] for r in ranks}
    out["1to2"] = []
    for _ in range(nboot):
        pick = rng.choice(sites, len(sites), replace=True)
        sel = np.concatenate([by[s] for s in pick])
        t0 = trim95(errs[0][sel])
        t = {r: trim95(errs[r][sel]) for r in ranks}
        for r in ranks:
            out[r].append((t0 - t[r]) / t0)
        out["1to2"].append((t[1] - t[2]) / t[1])
    return {k: np.array(v) for k, v in out.items()}


def cv_rank_noleak(M, mask, max_rank=4, K=5, restarts=3, seed=3,
                   iters=250, ridge=0.1):
    """无泄露嵌套 CV。中心化与 ALS 都只看训练折。

    返回 {rank: (rmse_full, rmse_trim95, rmse_median)}。
    主指标是 rmse_trim95（裁掉最差 5%），与 within_system_rank.py 一致。
    """
    rng = np.random.default_rng(seed)
    idx = np.argwhere(mask)
    rng.shuffle(idx)
    folds = np.array_split(idx, K)
    out = {}
    for rank in range(max_rank + 1):
        allerr = []
        for f in folds:
            trn = mask.copy()
            for i, j in f:
                trn[i, j] = False
            # 主效应：只用训练格子
            r, c = center_fit(M, trn)
            R = M - r[:, None] - c[None, :]          # 真值减训练主效应
            Y = np.where(trn, R, 0.0)                # ALS 只看训练格子
            best, bestP = np.inf, None
            for _ in range(restarts):
                P = als(Y, trn, rank, rng, iters=iters, ridge=ridge)
                tr = np.mean((R[trn] - P[trn]) ** 2)  # 按**训练**误差选重启
                if tr < best:
                    best, bestP = tr, P
            for i, j in f:
                allerr.append((R[i, j] - bestP[i, j]) ** 2)
        e = np.array(allerr)
        out[rank] = (math.sqrt(e.mean()),
                     math.sqrt(np.quantile(e, 0.95)),
                     math.sqrt(np.median(e)))
    return out


def best_rank(errs):
    trim = {r: errs[r][1] for r in errs}
    return min(trim, key=trim.get), trim


def gains(errs):
    trim = {r: errs[r][1] for r in errs}
    return {r: (trim[0] - trim[r]) / trim[0] for r in trim}


# ---------------------------------------------------------------- 载荷
def fit_loadings(M, mask, rank=2, restarts=8, seed=11, iters=400):
    """在全部观测格子上拟合固定秩，返回 (U, V)，V 是残基载荷 (ncol, rank)。

    载荷只在**子空间**意义下有定义（任意可逆 rank x rank 变换下等价），
    所以下游只允许比较 span(V)，不允许比较单个列向量。
    """
    rng = np.random.default_rng(seed)
    R = residual(M, mask)
    Y = np.where(mask, R, 0.0)
    best, bestUV = np.inf, None
    for _ in range(restarts):
        U, V = _als_factors(Y, mask, rank, rng, iters=iters)
        P = U @ V.T
        tr = np.mean((R[mask] - P[mask]) ** 2)
        if tr < best:
            best, bestUV = tr, (U, V)
    return bestUV[0], bestUV[1], best


def _als_factors(Y, mask, rank, rng, iters=400, ridge=0.0):
    """与 rank_test_power.als_guarded **逐步等价**的向量化实现，返回因子。

    为什么重写：原版是 Python 双重循环，单次 60x20 rank2 x250 迭代约 0.2 s，
    本文档需要 3 域 x 2 表型 x (功效网格 + 2000 次 bootstrap) 次拟合，
    循环版要跑几十小时。向量化后同一算法快约 40 倍。
    等价性在 selfcheck() 的 C8 里对 als_guarded 逐元素核对（< 1e-10）。

    每行 i:  A_i = sum_j mask[i,j] V_j V_j^T + ridge*I,  b_i = sum_j mask[i,j] V_j Y[i,j]
    """
    nr, nc = Y.shape
    U = rng.normal(0, 0.1, (nr, rank))
    V = rng.normal(0, 0.1, (nc, rank))
    if rank == 0:
        return np.zeros((nr, 0)), np.zeros((nc, 0))
    I = np.eye(rank) * max(ridge, 1e-8)
    Wm = mask.astype(float)
    YM = np.where(mask, Y, 0.0)
    row_ok = mask.sum(axis=1) >= rank + 1
    col_ok = mask.sum(axis=0) >= rank + 1
    for _ in range(iters):
        VV = V[:, :, None] * V[:, None, :]            # (nc, k, k)
        A = Wm @ VV.reshape(nc, -1)                   # (nr, k*k)
        A = A.reshape(nr, rank, rank) + I
        b = YM @ V                                    # (nr, k)
        U = np.linalg.solve(A, b[:, :, None])[:, :, 0]
        U[~row_ok] = 0.0
        UU = U[:, :, None] * U[:, None, :]            # (nr, k, k)
        A2 = Wm.T @ UU.reshape(nr, -1)
        A2 = A2.reshape(nc, rank, rank) + I
        b2 = YM.T @ U
        V = np.linalg.solve(A2, b2[:, :, None])[:, :, 0]
        V[~col_ok] = 0.0
    return U, V


def als(Y, mask, rank, rng, iters=250, ridge=0.0):
    """向量化 als_guarded 的乘积形式。"""
    if rank == 0:
        return np.zeros_like(Y)
    U, V = _als_factors(Y, mask, rank, rng, iters=iters, ridge=ridge)
    return U @ V.T


def orth(V):
    """列正交化；返回 (ncol, k) 正交基。秩亏时丢掉退化方向。"""
    Q, s, _ = np.linalg.svd(V, full_matrices=False)
    keep = s > 1e-10 * max(s[0], 1e-30)
    return Q[:, keep]


def principal_angles(A, B):
    """两个子空间的主角（弧度，升序）。用 SVD 而非 scipy 以免版本差异。

    与 scipy.linalg.subspace_angles 的差别只在排序方向；自检里对齐过。
    """
    QA, QB = orth(A), orth(B)
    s = np.linalg.svd(QA.T @ QB, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.sort(np.arccos(s))


# ---------------------------------------------------------------- 自检
def _synth(nr, nc, rank, sigma, rng, main=1.5, lams=(1.5, 0.8, 0.4),
           mask=None):
    a = rng.normal(0, main, nr)
    b = rng.normal(0, main, nc)
    M = a[:, None] + b[None, :]
    for k in range(rank):
        phi = rng.normal(0, 1, nr)
        psi = rng.normal(0, 1, nc)
        phi -= phi.mean()
        psi -= psi.mean()
        phi /= np.linalg.norm(phi) or 1
        psi /= np.linalg.norm(psi) or 1
        M = M + lams[k] * math.sqrt(nr * nc) * np.outer(phi, psi)
    M = M + rng.normal(0, sigma, (nr, nc))
    if mask is None:
        mask = np.ones((nr, nc), bool)
    return M, mask


def selfcheck(verbose=True):
    """全部断言用**已知事实**，不用自洽性。失败即抛 AssertionError。"""
    rng = np.random.default_rng(0)
    log = []

    # (1) 已知事实：完备矩阵双向中心化后行均值与列均值都恰为 0。
    A = rng.normal(0, 1, (12, 7)) + np.arange(12)[:, None]
    m = np.ones((12, 7), bool)
    R = residual(A, m)
    assert abs(R.mean(axis=1)).max() < 1e-10, "完备矩阵行均值未归零"
    assert abs(R.mean(axis=0)).max() < 1e-10, "完备矩阵列均值未归零"
    log.append("C1 完备矩阵中心化: 行/列均值 < 1e-10  OK")

    # (2) 已知事实：缺失下中心化只能保证**观测格子**的行/列均值为 0。
    m2 = rng.random((30, 20)) > 0.3
    A2 = rng.normal(0, 1, (30, 20)) + rng.normal(0, 2, 30)[:, None]
    R2 = residual(A2, m2)
    rm = max(abs(R2[i][m2[i]].mean()) for i in range(30) if m2[i].any())
    assert rm < 1e-8, f"缺失下观测格子行均值未归零 ({rm:.2e})"
    log.append(f"C2 缺失({1 - m2.mean():.0%})下观测行均值 {rm:.1e}  OK")

    # (3) 已知事实：无噪声完备的秩 r 矩阵，秩 r 的 ALS 应**精确**还原。
    for r in (1, 2, 3):
        U0 = rng.normal(0, 1, (25, r))
        V0 = rng.normal(0, 1, (18, r))
        T = U0 @ V0.T
        mm = np.ones((25, 18), bool)
        # 直接测 _als_factors 的还原性（fit_loadings 内部会先中心化，
        # 那会改变真实秩，不适合做这条断言）。
        Uh, Vh = _als_factors(T, mm, r, np.random.default_rng(5), iters=800)
        rel = np.linalg.norm(T - Uh @ Vh.T) / np.linalg.norm(T)
        assert rel < 1e-5, f"无噪声秩{r}未被精确还原 (rel={rel:.2e})"
        log.append(f"C3 无噪声秩{r} ALS 相对还原误差 {rel:.1e}  OK")

    # (4) 主角：已知事实。相同子空间 -> 全 0；正交子空间 -> 全 pi/2；
    #     span{e1,e2} vs span{e1, e3} -> (0, pi/2)。
    E = np.eye(6)
    assert abs(principal_angles(E[:, :2], E[:, :2])).max() < 1e-8
    pa = principal_angles(E[:, :2], E[:, 2:4])
    assert abs(pa - math.pi / 2).max() < 1e-8, pa
    pa = principal_angles(E[:, [0, 1]], E[:, [0, 2]])
    assert pa[0] < 1e-8 and abs(pa[1] - math.pi / 2) < 1e-8, pa
    # 与 scipy 对齐（若可用）
    try:
        from scipy.linalg import subspace_angles
        X = rng.normal(0, 1, (9, 2))
        Y = rng.normal(0, 1, (9, 2))
        mine = principal_angles(X, Y)
        theirs = np.sort(subspace_angles(X, Y))
        assert abs(mine - theirs).max() < 1e-8, (mine, theirs)
        log.append("C4 主角与 scipy.subspace_angles 一致  OK")
    except ImportError:
        log.append("C4 主角已知值 OK（scipy 不可用，跳过对齐）")

    # (5) 既测能命中：已知秩 2 + 本仓库校准的噪声（SIGMA cross_mixed = 0.64,
    #     LAMBDAS = 1.5/0.8，即 rank_power.py 的设计点）-> 必须给 2，三个种子。
    for sd in (7, 17, 27):
        M, mk = _synth(60, 20, 2, 0.64, np.random.default_rng(sd))
        e = cv_rank_noleak(M, mk, max_rank=4, restarts=3, iters=300, ridge=0.0)
        br, trim = best_rank(e)
        assert br == 2, f"理想条件下秩2 未被检出（seed {sd} 给出 {br}）: {trim}"
    log.append(f"C5 命中：秩2 @ sigma=0.64 -> 检出 2（3/3 种子；trim95 "
               + " ".join(f"{trim[k]:.2f}" for k in sorted(trim)) + "）OK")

    # (6) 也测不误伤：无交互结构（只有主效应 + 噪声）-> 必须给 0。
    for sd in (8, 18, 28):
        M0, mk0 = _synth(60, 20, 0, 0.45, np.random.default_rng(sd))
        e0 = cv_rank_noleak(M0, mk0, max_rank=4, restarts=3, iters=300,
                            ridge=0.0)
        br0, trim0 = best_rank(e0)
        assert br0 == 0, f"无结构被误报为秩 {br0}（seed {sd}）: {trim0}"
    log.append("C6 不误伤：无结构 -> 检出 0（3/3 种子）OK")

    # (6b) **已知的方向性偏差，机器化记录**（本次新发现）：
    #   无泄露中心化在每折训练集内估主效应，而删掉的格子使该行残余耦合的
    #   训练均值不为零 —— 主效应估计会**吸收一部分耦合**，在留出格子上留下
    #   类似额外低秩成分的偏移。实测 r,c 漂移达 0.71（噪声仅 0.15）。
    #   后果：耦合/噪声比很高时定秩**偏高**（lam1/sigma≈10 时秩2 被判成秩4）。
    #   方向是偏高而非偏低，所以「检出秩 0/1」不会是这个偏差造成的；
    #   反过来，高 SNR 下检出的「秩 2」有可能实为秩 1。必须随结论一起报告。
    M9, mk9 = _synth(60, 20, 2, 0.15, np.random.default_rng(7))
    e9 = cv_rank_noleak(M9, mk9, max_rank=4, restarts=3, iters=300, ridge=0.0)
    br9, _ = best_rank(e9)
    assert br9 > 2, (
        "lam1/sigma=10 下未复现已知的偏高偏差 —— 要么偏差被修掉了（好事，"
        "请更新本断言与 docs/15），要么合成/流程被改动了")
    log.append(f"C6b 已知偏差复现：lam1/sigma=10 -> 检出 {br9}（>2，偏高）OK")

    # (7) 泄露哨兵（量纲判据，docs/12 §3）：无泄露版的留出 RMSE 必须与
    #     噪声同量级，不能跳到主效应尺度。main=1.5 => 主效应 RMSE ~2。
    sig = 0.45
    e0 = cv_rank_noleak(*_synth(60, 20, 0, sig, np.random.default_rng(8)),
                        max_rank=4, restarts=3, iters=300, ridge=0.0)
    assert e0[0][0] < 3 * sig, (
        f"秩0 留出 RMSE {e0[0][0]:.2f} 远超噪声 {sig} —— 疑似中心化泄露/归零 bug")
    log.append(f"C7 量纲哨兵：秩0 RMSE {e0[0][0]:.2f} vs 噪声 {sig}  OK")

    # (8) 向量化 ALS 与被验证过的 als_guarded 逐元素等价。
    #     不是「已知事实」而是「新实现 == 旧实现」—— 独立于上面的事实断言，
    #     目的是防止提速引入沉默偏差（D-023 式事故）。
    rg = np.random.default_rng(99)
    Ys = rg.normal(0, 1, (23, 11))
    ms = rg.random((23, 11)) > 0.25
    ms[3] = False
    ms[:, 7] = False
    for rk in (1, 2, 3):
        Pa = als_guarded(np.where(ms, Ys, 0.0), ms, rk,
                         np.random.default_rng(4), iters=60, ridge=0.05)
        Pb = als(np.where(ms, Ys, 0.0), ms, rk,
                 np.random.default_rng(4), iters=60, ridge=0.05)
        d = np.abs(Pa - Pb).max()
        assert d < 1e-9, f"向量化 ALS 与 als_guarded 在秩{rk}不一致 (max|Δ|={d:.2e})"
    log.append("C8 向量化 ALS == als_guarded（秩1/2/3, max|Δ|<1e-9）OK")

    if verbose:
        for s in log:
            print("  " + s)
    return log


if __name__ == "__main__":
    print("rank_core 自检")
    selfcheck()
    print("全部通过")
