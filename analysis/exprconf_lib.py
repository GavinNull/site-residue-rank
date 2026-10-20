#!/usr/bin/env python3
"""docs/47：结合交互里的表达量（展示）结构 —— 共享层。

预先登记：cache/docs47_prereg.md，sha256 锁在 cache/docs47_prereg_lock.json（入口 verify_lock47 断言）。
复用，不重写：u2split_lib（分半、_protect）、subspace_lib（extract_subspace、align、dc_fast、haar_1d_null）、
d46_lib（RR 世界的 row_permute）、d46_partb_lib（表达量矩阵构建、centered_orth、right_basis）、
rank_k1（als_joint、cv_cell_errors_k1）、overshoot_lib.Geometry。**未使用 als_complete**。

三种校正（预先登记 §2）：
  ia  全局斜率（主）      ib 逐残基斜率      ii 子空间投影（位点逐行最小二乘；分半时交叉拟合）
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import hashlib
import json
import math
import os
import pickle
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

import u2split_lib as U                      # noqa: E402
import subspace_lib as L                     # noqa: E402
import rank_core as RC                       # noqa: E402
import rank_k1 as K1                         # noqa: E402
import d46_lib as D                          # noqa: E402
import d46_partb_lib as B                    # noqa: E402

LOCK47 = os.path.join(ROOT, "cache", "docs47_prereg_lock.json")
OUT = os.path.join(ROOT, "cache", "exprconf")
os.makedirs(OUT, exist_ok=True)
AA20 = L.AA20

# ---- 预先登记 §5 的固定数
S_REAL = 250
R_W = 36
S_W = 5
NBOOT = 3000
NB_GAIN = 2000
CV_SEEDS = (11, 12, 13)
HAAR_N = 4000
CORRS = ("none", "ia", "ib", "ii")
CORRS_A = ("none", "ia", "ic")      # 修订 A1 层
EXTS = ("EM", "K1")
PT_IDX = (0, 3, 6)
GAMMAS = (0.25, 0.5, 1.0, 2.0)

S1_BG = ("Wuhan-Hu-1", "Eta", "Omicron_BA2")
LIGS = ("deer", "hamster", "bat", "cat", "human_monomer")
BG_SHORT = {"Wuhan-Hu-1": "WH1", "Eta": "Eta", "Omicron_BA2": "BA2", "PDCoV": "PDCoV"}
PARTNER = {"Wuhan-Hu-1": "WH1_v2", "Eta": "Eta", "Omicron_BA2": "BA2", "PDCoV": "PDCoV"}
SRC_RANK2_OK = {"WH1_v1": True, "WH1_v2": True, "Eta": True, "BA2": False, "PDCoV": False}   # 取自 docs/46 §6.2


def verify_lock47(verbose=True):
    lk = json.load(open(LOCK47))
    h = hashlib.sha256(open(os.path.join(ROOT, lk["file"]), "rb").read()).hexdigest()
    assert h == lk["sha256"], (f"docs/47 预先登记与锁不一致 —— 按 CLAUDE.md §4 中止\n  锁定 {lk['sha256']}\n  现在 {h}")
    if verbose:
        print(f"  预先登记锁 {lk['file']} sha256 {h[:16]}… 一致（锁定于 {lk['locked_at']}）  OK", flush=True)
    return {"sha256": h, "locked_at": lk["locked_at"]}


def verify_lock47a(verbose=True):
    lk = json.load(open(os.path.join(ROOT, "cache", "docs47a_prereg_lock.json")))
    h = hashlib.sha256(open(os.path.join(ROOT, lk["file"]), "rb").read()).hexdigest()
    assert h == lk["sha256"], f"docs/47 修订 A1 与锁不一致 —— 中止\n  锁定 {lk['sha256']}\n  现在 {h}"
    if verbose:
        print(f"  修订 A1 锁 {lk['file']} sha256 {h[:16]}… 一致（锁定于 {lk['locked_at']}）  OK", flush=True)
    return {"sha256": h, "locked_at": lk["locked_at"]}


def verify_lock47b(verbose=True):
    lk = json.load(open(os.path.join(ROOT, "cache", "docs47b_prereg_lock.json")))
    h = hashlib.sha256(open(os.path.join(ROOT, lk["file"]), "rb").read()).hexdigest()
    assert h == lk["sha256"], f"docs/47 修订 A2 与锁不一致 —— 中止\n  锁定 {lk['sha256']}\n  现在 {h}"
    if verbose:
        print(f"  修订 A2 锁 {lk['file']} sha256 {h[:16]}… 一致（锁定于 {lk['locked_at']}）  OK", flush=True)
    return {"sha256": h, "locked_at": lk["locked_at"]}


def seed47(*parts):
    return zlib.crc32(("ec47|" + "|".join(str(p) for p in parts)).encode()) & 0x7FFFFFFF


# ======================================================================
# 抽取器（秩可选）
# ======================================================================
def X(ext, M, mask, rank, seed=0):
    """返回 (B, cols) 或 None。EM = subspace_lib.extract_subspace；K1 = rank_k1.als_joint 联合拟合（与 U.extract_k1 同，秩可选）。"""
    M = np.asarray(M, float)
    mask = np.asarray(mask, bool)
    if ext == "EM":
        Bq, cols, _ = L.extract_subspace(np.where(mask, M, 0.0), mask, rank=rank, method="emsvd", seed=seed)
        return (Bq, cols) if Bq is not None else None
    pr = U._protect(M, mask)
    if pr is None:
        return None
    Ms, ms, cols = pr
    Mf = np.where(ms, Ms, 0.0)
    rng = np.random.default_rng(seed)
    best, bestP = np.inf, None
    for _ in range(2):
        P = K1.als_joint(Mf, ms, rank, rng, iters=150, ridge=0.1)
        tr = float(np.mean((Mf[ms] - P[ms]) ** 2))
        if tr < best:
            best, bestP = tr, P
    Q = bestP - bestP.mean(1, keepdims=True) - bestP.mean(0, keepdims=True) + bestP.mean()
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    Bq, _ = np.linalg.qr(Vt[:rank].T)
    return Bq, cols


def cenu(v):
    v = np.asarray(v, float)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def axis_cos(ra, rb):
    """一维 |cos|：ra、rb = (B 或向量, cols)；用 B[:,0] 或给定向量；公共残基、去均值。返回 (|cos|, d) 或 None。"""
    if ra is None or rb is None:
        return None
    a = np.asarray(ra[0], float)
    b = np.asarray(rb[0], float)
    a = a[:, 0] if a.ndim == 2 else a
    b = b[:, 0] if b.ndim == 2 else b
    ca, cb = np.asarray(ra[1]), np.asarray(rb[1])
    com = np.intersect1d(ca, cb)
    if len(com) < 8:
        return None
    va = cenu(a[np.searchsorted(ca, com)])
    vb = cenu(b[np.searchsorted(cb, com)])
    return abs(float(va @ vb)), len(com)


FULL20 = np.arange(20)


def tr(v):
    """20 维向量打包成 (v, cols)。"""
    return (np.asarray(v, float), FULL20)


_H1 = {}


def haar1_p95(d):
    d = int(d)
    if d not in _H1:
        _H1[d] = float(np.quantile(L.haar_1d_null(d, n=HAAR_N, seed=d), 0.95))
    return _H1[d]


# ======================================================================
# 校正
# ======================================================================
def dcK(M, K):
    return L.dc_fast(np.where(K, M, 0.0), K)


def expr_subspace(Me, ke, k, seed=0):
    """表达量矩阵的残基载荷子空间 (V, cols)，秩 k，EM。"""
    r = X("EM", Me, ke, k, seed)
    return r


def correct(Mb, kb, Me, ke, kind, Vexp=None):
    """返回 (Mc, Kc, info)。kind ∈ none/ia/ib/ii；Mc 在 Kc 外为 0。
    Vexp = (V, cols) 仅 ii 用（交叉拟合时由调用方传入另一半的表达量子空间）。"""
    K = kb & ke
    Mbk = np.where(K, Mb, 0.0)
    if kind == "none":
        return Mbk, K, {"share": 0.0, "beta": 0.0}
    Rb = dcK(Mb, K)
    Re = dcK(Me, K)
    nb = float(np.sqrt(np.mean(Rb[K] ** 2)))
    if kind == "ia":
        den = float((Re[K] ** 2).sum())
        beta = float((Rb[K] * Re[K]).sum() / den) if den > 0 else 0.0
        F = beta * Re
        return np.where(K, Mb - F, 0.0), K, {"beta": beta, "share": float(np.sqrt(np.mean(F[K] ** 2)) / nb)}
    if kind == "ib":
        den_g = float((Re[K] ** 2).sum())
        beta_g = float((Rb[K] * Re[K]).sum() / den_g) if den_g > 0 else 0.0
        bj = np.full(Mb.shape[1], beta_g)
        for j in range(Mb.shape[1]):
            kj = K[:, j]
            if kj.sum() >= 20:
                d_ = float((Re[kj, j] ** 2).sum())
                if d_ > 0:
                    bj[j] = float((Rb[kj, j] * Re[kj, j]).sum() / d_)
        F = Re * bj[None, :]
        return np.where(K, Mb - F, 0.0), K, {"beta": beta_g, "share": float(np.sqrt(np.mean(F[K] ** 2)) / nb)}
    if kind == "ic":                              # 修订 A1：低秩部分与余项两个斜率；Vexp 此处携带秩 k
        k = int(Vexp)
        Lr = lowrank_part(Re, K, k)
        Rm_ = Re - Lr
        A = np.stack([Lr[K], Rm_[K]], axis=1)
        G = A.T @ A
        beta = np.linalg.solve(G + 1e-12 * np.eye(2), A.T @ Rb[K])
        F = beta[0] * Lr + beta[1] * Rm_
        return np.where(K, Mb - F, 0.0), K, {"beta": float(beta[0]), "beta2": float(beta[1]),
                                              "share": float(np.sqrt(np.mean(F[K] ** 2)) / nb)}
    if kind in ("ii", "dst"):                       # dst = 牙齿：投影掉结合自己的第一轴（必须被门抓住）
        V, cols = Vexp
        F = np.zeros_like(Rb)
        Kc = K.copy()
        pos = {int(c): n for n, c in enumerate(cols)}
        for i in range(Mb.shape[0]):
            obs = np.where(K[i])[0]
            if len(obs) == 0:
                continue
            sel = [j for j in obs if int(j) in pos]
            if len(obs) < 8 or len(sel) < V.shape[1] + 3:
                Kc[i] = False
                continue
            A = V[[pos[int(j)] for j in sel], :]
            coef, *_ = np.linalg.lstsq(A, Rb[i, sel], rcond=None)
            F[i, sel] = A @ coef
        return np.where(Kc, Mb - F, 0.0), Kc, {"share": float(np.sqrt(np.mean(F[Kc] ** 2)) / nb) if Kc.any() else 0.0}
    raise ValueError(kind)


def lowrank_part(Rm, K, k, iters=40):
    """K 上缺失的双向去均值矩阵的秩 k 低秩部分（EM 硬填补，与 subspace_lib.extract_subspace 同一迭代）。"""
    X = np.where(K, Rm, 0.0)
    for _ in range(iters):
        U_, s_, Vt = np.linalg.svd(X, full_matrices=False)
        Lr = (U_[:, :k] * s_[:k]) @ Vt[:k]
        X = np.where(K, Rm, Lr)
        X = X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()
    U_, s_, Vt = np.linalg.svd(X, full_matrices=False)
    return (U_[:, :k] * s_[:k]) @ Vt[:k]


def ii_rank(src):
    return 2 if SRC_RANK2_OK[src] else 1


# ======================================================================
# 一个实例的统计
# ======================================================================
def _vexp_for(corr, Me, ke, src, seed, Mb=None, kb=None):
    if corr == "dst":
        return X("EM", Mb, kb & ke, 1, seed)
    if corr == "ic":
        return ii_rank(src)
    if corr != "ii":
        return None
    return expr_subspace(Me, ke, ii_rank(src), seed)


def split_stats(Mb, kb, Me, ke, corr, ext, src, S, seed, do2=True):
    """S 个位点不相交分半：每半各自校正（ia/ib 在半内估斜率；ii 的 V_e 来自另一半=交叉拟合）。
    返回 (S,3) [第一轴 |cos|, cos θ2(秩 2), d]。"""
    rng = np.random.default_rng(seed)
    out = np.full((S, 3), np.nan)
    for s in range(S):
        ia, ib = U.split_sites(Mb.shape[0], rng)
        sd = int(rng.integers(1 << 30))
        halves = []
        for h, (H, O) in enumerate(((ia, ib), (ib, ia))):
            Ve = _vexp_for(corr, Me[O], ke[O], src, sd + 5 + h)        # 另一半的表达量子空间
            Mc, Kc, _ = correct(Mb[H], kb[H], Me[H], ke[H], corr, Ve)
            r1 = X(ext, Mc, Kc, 1, sd + h)
            r2 = X(ext, Mc, Kc, 2, sd + 2 + h) if do2 else None
            halves.append((r1, r2))
        a1 = axis_cos(halves[0][0], halves[1][0])
        if a1 is not None:
            out[s, 0], out[s, 2] = a1
        if do2:
            c = U.compare(halves[0][1], halves[1][1])
            if c is not None:
                out[s, 1] = c[1]
    return out


# ======================================================================
# 自检（预先登记 §3 末；既测命中，也测不误伤）
# ======================================================================
def selfcheck47(verbose=True):
    log = []
    rng = np.random.default_rng(4747)
    nr, nc = 160, 20
    mask = rng.random((nr, nc)) < 0.9
    # 已知结构：表达量 = 位点载荷 × 残基方向 Ve1；结合特异 = 独立残基方向 Vb1
    Ue = rng.normal(size=nr)
    Ub = rng.normal(size=nr)
    allv = B.centered_orth(nc, rng, k=3)
    Ve1, Vb1, Vx = allv[:, 0], allv[:, 1], allv[:, 2]
    r, c = rng.normal(0, 1, nr), rng.normal(0, .5, nc)
    Me = r[:, None] + c[None, :] + 4.0 * np.outer(Ue, Ve1)
    Mp = r[:, None] * 0.7 + c[None, :] + 2.0 * np.outer(Ub, Vb1)             # 结合特异（无表达量）
    beta_true = 0.6
    Mb = Mp + beta_true * 4.0 * np.outer(Ue, Ve1)
    Me_ = np.where(mask, Me, 0.0)
    # T1 命中：(a) 结合 = 主效应 + β·表达量交互 时，ia 精确还原 β（无噪声）；(b) 加上独立的结合特异结构，β̂ 只差有限样本的交叉项，
    #   校正后的轴是结合特异方向而不是表达量方向
    Mpure = r[:, None] * 0.7 + c[None, :] + beta_true * 4.0 * np.outer(Ue, Ve1)
    _, _, infop = correct(Mpure, mask, Me_, mask, "ia")
    assert abs(infop["beta"] - beta_true) < 1e-8, f"T1a β̂={infop['beta']:.8f} ≠ {beta_true}"
    Mc, Kc, info = correct(Mb, mask, Me_, mask, "ia")
    assert abs(info["beta"] - beta_true) < 0.03, f"T1b β̂={info['beta']:.6f} ≠ {beta_true}"
    ax = X("EM", Mc, Kc, 1)
    cb, ce = axis_cos(ax, tr(Vb1))[0], axis_cos(ax, tr(Ve1))[0]
    ax0 = X("EM", np.where(mask, Mb, 0.0), mask, 1)
    ce0 = axis_cos(ax0, tr(Ve1))[0]
    assert cb > 0.99 and ce < 0.15 and ce0 > 0.5, f"T1b 校正后轴 cos(Vb1)={cb:.3f} cos(Ve1)={ce:.3f} 未校正 cos(Ve1)={ce0:.3f}"
    log.append(f"T1 ia：纯表达量依赖时 β̂ = {infop['beta']:.6f}（真 {beta_true}，精确）；混入独立结合特异结构后 β̂ = {info['beta']:.3f}；"
               f"校正后轴 |cos(Vb1)| {cb:.4f}、|cos(Ve1)| {ce:.3f}（未校正 |cos(Ve1)| {ce0:.3f}）  OK")
    # T2 不误伤：结合里没有表达量成分 -> β̂ ≈ 0，且结合轴不变
    Mc2, Kc2, info2 = correct(Mp, mask, Me_, mask, "ia")
    assert abs(info2["beta"]) < 0.03, f"T2 无表达量成分时 β̂ = {info2['beta']:.3f}"
    a_before = axis_cos(X("EM", np.where(mask, Mp, 0), mask, 1), tr(Vb1))[0]
    a_after = axis_cos(X("EM", Mc2, Kc2, 1), tr(Vb1))[0]
    assert a_after > 0.98 and abs(a_after - a_before) < 0.01
    log.append(f"T2 ia 无表达量成分：β̂ = {info2['beta']:.4f}；结合轴 |cos(真 Vb1)| {a_before:.4f} -> {a_after:.4f}  OK")
    # T3 ii 命中：投影后每个位点的残差在观测格上与 V_e 正交（精确）；不误伤：与 V_e 正交的结构保留
    Vexp = X("EM", Me_, mask, 1)
    Mc3, Kc3, info3 = correct(Mb, mask, Me_, mask, "ii", Vexp)
    Rc3 = dcK(Mb, mask) - (Mb - Mc3)         # 校正时的逐行残差（用校正时的 Rb，不再重新双向去均值）
    V, cols = Vexp
    pos = {int(cc): n for n, cc in enumerate(cols)}
    worst = 0.0
    for i in range(nr):
        sel = [j for j in np.where(Kc3[i])[0] if int(j) in pos]
        if len(sel) > 5:
            worst = max(worst, abs(float(Rc3[i, sel] @ V[[pos[int(j)] for j in sel], 0])))
    assert worst < 1e-8, f"T3 ii 校正后残差与 V_e 不正交 {worst:.2e}"
    a3 = axis_cos(X("EM", Mc3, Kc3, 1), tr(Vb1))[0]
    assert a3 > 0.95, f"T3 ii 误伤了与 V_e 正交的结构 |cos|={a3:.3f}"
    log.append(f"T3 ii：校正后残差对 V_e 正交（max {worst:.1e}）；与 V_e 正交的结合特异轴保留 |cos| = {a3:.4f}  OK")
    # T4 ii 的负面已知事实：结合特异方向若与 V_e 共享，则被删除（by design）——这条是「校正会删什么」的声明，不是缺陷
    Mshare = r[:, None] * 0.7 + c[None, :] + 2.0 * np.outer(Ub, Ve1)
    Mc4, Kc4, _ = correct(Mshare, mask, Me_, mask, "ii", Vexp)
    en0 = float((dcK(Mshare, mask)[mask] ** 2).sum())
    en1 = float((dcK(Mc4, Kc4)[Kc4] ** 2).sum())
    a4 = en1 / en0
    assert a4 < 0.1, f"T4 ii 没有删除与 V_e 共享的方向（剩余能量比 {a4:.3f}）"
    Mc4a, Kc4a, _ = correct(Mshare, mask, Me_, mask, "ia")
    a4a = axis_cos(X("EM", Mc4a, Kc4a, 1), tr(Ve1))[0]
    assert a4a > 0.95, f"T4 ia 误删了位点载荷独立的共享残基方向 |cos|={a4a:.3f}"
    log.append(f"T4 残基方向共享但位点载荷独立：ii 删除（剩余交互能量比 {a4:.3f}，by design），ia 保留（|cos(轴,Ve1)| {a4a:.3f}）  OK")
    # T5 交叉拟合不泄露：改动本半位点的表达量，ii 的结果（V_e 来自另一半）逐位不变
    ia_, ib_ = U.split_sites(nr, np.random.default_rng(3))
    Ve_other = expr_subspace(Me_[ib_], mask[ib_], 1, 5)
    Mx = Me_.copy()
    Mx[ia_[0]] += 30.0 * np.random.default_rng(1).normal(size=nc)
    a_ = correct(Mb[ia_], mask[ia_], Me_[ia_], mask[ia_], "ii", Ve_other)[0]
    b_ = correct(Mb[ia_], mask[ia_], Mx[ia_], mask[ia_], "ii", Ve_other)[0]
    assert np.array_equal(a_, b_), "T5 ii 校正受本半表达量影响 —— 交叉拟合泄露"
    Ve_other2 = expr_subspace(Mx[ib_], mask[ib_], 1, 5)
    assert np.array_equal(Ve_other[0], Ve_other2[0])
    # 对照：ia 确实用本半格子的表达量（非泄露，是设计）；该测试证明自检能看见变化
    a2_ = correct(Mb[ia_], mask[ia_], Me_[ia_], mask[ia_], "ia")[0]
    b2_ = correct(Mb[ia_], mask[ia_], Mx[ia_], mask[ia_], "ia")[0]
    assert not np.array_equal(a2_, b2_)
    log.append("T5 ii 交叉拟合：改动本半表达量不改变校正结果；ia 对同一改动有响应（自检可见变化）  OK")
    # T6 K1 秩参数：秩 2 时与 U.extract_k1 逐位一致；秩 1 给出 20 维一列
    Y = np.where(mask, Mb, 0.0)
    r2 = X("K1", Y, mask, 2, 9)
    ref = U.extract_k1(Y, mask, 9)
    assert np.array_equal(r2[0], ref[0]) and np.array_equal(r2[1], ref[1]), "T6 X(K1,rank=2) 与 U.extract_k1 不同"
    r1 = X("K1", Y, mask, 1, 9)
    assert r1[0].shape[1] == 1
    log.append("T6 X('K1', rank=2) == U.extract_k1（逐位）；rank=1 形状正确  OK")
    # T9/T10 (i-c)：协变量 = 低秩部分 + 非低秩余项；结合依赖两者的不同斜率时精确还原（无噪声）；只依赖低秩部分时 β₂≈0
    Lrank = 3.0 * np.outer(Ue, Ve1)
    Rrest = 1.2 * rng.normal(size=(nr, nc))
    Me2 = np.where(mask, r[:, None] + c[None, :] + Lrank + Rrest, 0.0)
    Re2 = dcK(Me2, mask)
    Lr2 = lowrank_part(Re2, mask, 1)
    Rm2 = Re2 - Lr2
    b1, b2 = 0.8, 0.3
    Mbc = r[:, None] * 0.5 + c[None, :] + b1 * Lr2 + b2 * Rm2
    _, _, inf_c = correct(Mbc, mask, Me2, mask, "ic", 1)
    assert abs(inf_c["beta"] - b1) < 0.01 and abs(inf_c["beta2"] - b2) < 0.01, f"T9 (β₁,β₂) = ({inf_c['beta']:.5f}, {inf_c['beta2']:.5f}) ≠ ({b1}, {b2})"
    Mbl = r[:, None] * 0.5 + c[None, :] + b1 * Lr2 + 2.0 * np.outer(Ub, Vb1)
    Mcl, Kcl, inf_l = correct(Mbl, mask, Me2, mask, "ic", 1)
    cbl = axis_cos(X("EM", Mcl, Kcl, 1), tr(Vb1))[0]
    _, _, inf_a = correct(Mbl, mask, Me2, mask, "ia")
    assert abs(inf_l["beta"] - b1) < 0.06 and abs(inf_l["beta2"]) < 0.06 and cbl > 0.98, (inf_l, cbl)
    log.append(f"T9 ic 无噪声还原两个斜率（容差 0.01：缺失下 Lr、Rm 并非严格双向去均值，再去均值带来 <0.5% 的偏差） ({inf_c['beta']:.5f}, {inf_c['beta2']:.5f}) = ({b1}, {b2})；T10 结合只依赖低秩部分时 ic 得 β₁ = {inf_l['beta']:.3f}（真 {b1}）、β₂ = {inf_l['beta2']:.3f}（真 0）、校正后轴 |cos(Vb1)| {cbl:.3f}；"
               f"同一数据 ia 的单斜率 β̂ = {inf_a['beta']:.3f}（被稀释，这正是诊断的机制）  OK")
    assert inf_a["beta"] < 0.8 * b1, "T10 ia 没有被非低秩余项稀释 —— 诊断机制在玩具上不成立"
    # T7 零分布对 d 敏感（D-036 / docs/23 的错）
    assert haar1_p95(20) < haar1_p95(10) - 0.05
    log.append(f"T7 一维 Haar p95：d=20 {haar1_p95(20):.3f} < d=10 {haar1_p95(10):.3f}  OK")
    if verbose:
        for s in log:
            print("  " + s, flush=True)
    return log


# ======================================================================
# 数据：配对
# ======================================================================
_GEOMS = {}


def load_all():
    """所有 (结合, 表达量) 配对。返回 list[dict]。结合几何来自 docs/45/46 同一缓存；表达量来自 docs/46 Part B 缓存。"""
    if "pairs" in _GEOMS:
        return _GEOMS["pairs"]
    geoms = pickle.load(open(os.path.join(ROOT, "cache", "overshoot_geoms.pkl"), "rb"))
    noise = json.load(open(os.path.join(ROOT, "cache", "overshoot_noise.json")))["matrices"]
    Ed = pickle.load(open(os.path.join(ROOT, "cache", "d46", "partb", "data.pkl"), "rb"))
    fid = json.load(open(os.path.join(ROOT, "cache", "d46", "fidelity.json")))
    pairs = []

    def rr_faithful(n):
        if n not in fid:
            return None
        pts = {k: v for k, v in fid[n]["pts"].items() if k.startswith("RR|")}
        return bool(pts) and all(v["pass_am"] for v in pts.values())

    for bg in S1_BG:
        for lig in LIGS:
            n = f"animalACE2/{bg}/{lig}"
            src = PARTNER[bg]
            gb = geoms[n]
            e = Ed["expr"][src]
            path = noise[n]["path_k2"] if (n in noise and noise[n].get("path_k2")) else []
            pairs.append(dict(id=f"{BG_SHORT[bg]}/{lig}", cls="SAME-LIB", unit=BG_SHORT[bg], name=n, src=src, gb=gb,
                              Mb=gb.M, kb=gb.mask, Me=e["M"], ke=e["mask"], ge=e["geom"], path=path,
                              faithful=rr_faithful(n), stratum="S1"))
    sites_cache = {}
    for src, bgname in (("WH1_v1", "WH1"), ("WH1_v2", "WH1"), ("Eta", "Eta"), ("BA2", "BA2")):
        e = Ed["expr"][src]
        Mb, kb = Ed["bindH"][src]["M"], Ed["bindH"][src]["mask"]
        fn, tgt = {"WH1_v1": (B.OMI, "Wuhan-Hu-1_v1"), "WH1_v2": (B.OMI, "Wuhan-Hu-1_v2"), "Eta": (B.OMI, "Eta"),
                   "BA2": (B.OMI, "Omicron_BA2")}[src]
        Dm, Nrep, _ = B.build_reps_arrays(fn, tgt, ["bind_rep1", "bind_rep2", "bind_rep3"], Mb, kb, e["sites"])
        gb = B.make_geom(f"HACE2/{src}", Mb, kb, Dm, Nrep)
        path = gb.path(2, gb.sig["sigma_cell"])
        pairs.append(dict(id=f"hACE2({src})", cls="SAME-EXP", unit=bgname, name=f"hACE2/{src}", src=src, gb=gb,
                          Mb=gb.M, kb=gb.mask, Me=e["M"], ke=e["mask"], ge=e["geom"], path=path, faithful=None, stratum="SAME-EXP"))
    n = "PDCoV/PDCoV/gAPN"
    e = Ed["expr"]["PDCoV"]
    pairs.append(dict(id="gAPN", cls="CROSS-FAMILY-S2", unit="PDCoV", name=n, src="PDCoV", gb=geoms[n], Mb=geoms[n].M,
                      kb=geoms[n].mask, Me=e["M"], ke=e["mask"], ge=e["geom"], path=noise[n]["path_k2"], faithful=rr_faithful(n),
                      stratum="S2"))
    _GEOMS["pairs"] = pairs
    _GEOMS["Ed"] = Ed
    return pairs


def eligible_for_worlds(pairs):
    """世界与判决用的配对：有路径且 RR 世界忠实（SAME-EXP 的忠实性在 fidelity_bindH 阶段另测）。"""
    return [p for p in pairs if len(p["path"]) == 7 and (p["faithful"] is True or p["cls"] == "SAME-EXP" and p.get("faithful_bindH"))]


# ======================================================================
# 对照世界（预先登记 §3）
# ======================================================================
SCEN = {
    "BS": dict(gamma=0.0, rho=0.0, bs=True, par=None),
    "GAM0.25": dict(gamma=0.25, rho=0.0, bs=True, par=None),
    "GAM0.5": dict(gamma=0.5, rho=0.0, bs=True, par=None),
    "GAM1": dict(gamma=1.0, rho=0.0, bs=True, par=None),
    "GAM2": dict(gamma=2.0, rho=0.0, bs=True, par=None),
    "PURE2": dict(gamma=2.0, rho=0.0, bs=False, par=None),
    "CORR0.08": dict(gamma=0.0, rho=0.08, bs=True, par=None),
    "CORR0.30": dict(gamma=0.0, rho=0.30, bs=True, par=None),
    "PARRES": dict(gamma=0.0, rho=0.0, bs=True, par="RES"),
    "PARCELL": dict(gamma=0.0, rho=0.0, bs=True, par="CELL"),
    "GAMF1": dict(gamma=1.0, rho=0.0, bs=True, par=None, full=True),          # 修订 A1：结合响应表达量的全部交互（含非低秩部分）
    "GAMF2": dict(gamma=2.0, rho=0.0, bs=True, par=None, full=True),
    "PUREF2": dict(gamma=2.0, rho=0.0, bs=False, par=None, full=True),
}


POWER = {f"F{f}": dict(gamma=0.0, rho=0.0, bs=(f > 0), par=None, bsf=f) for f in (0.0, 0.1, 0.2, 0.3, 0.5)}   # §10.3 还原性诊断：结合特异振幅 = f·s
SCEN_ALL = {**SCEN, **POWER}


def build_world(pair, scen, p, rng):
    """返回 (Y, truth)。Y 在结合掩码外为 0。truth = dict(Vb(nc,2), Ve(nc,2), parts)。"""
    gb, ge = pair["gb"], pair["ge"]
    m = gb.mask
    ke = pair["ke"]
    pt = pair["path"][p]
    s, sig = pt["s"], pt["sigma"]
    sc = SCEN_ALL[scen]
    P = gb.fit[2]["P"]
    Uu, S, _ = np.linalg.svd(P, full_matrices=False)
    Pe = ge.fit[2]["P"]
    Ve = B.right_basis(Pe, 2)
    nc = gb.nc
    if sc["par"] == "RES":
        v2 = B.centered_orth(nc, rng, against=Ve[:, :1], k=1)
        Vb = np.concatenate([Ve[:, :1], v2], axis=1)
    else:
        Vb = B.centered_orth(nc, rng, against=Ve, k=2)
    Ib = gb._scale_to((Uu[:, :2] * S[:2]) @ Vb.T, s * sc.get("bsf", 1.0)) if sc["bs"] else np.zeros((gb.nr, nc))
    Ie = np.zeros((gb.nr, nc))
    if sc["gamma"] > 0:
        src_ = RC.residual(ge.M, ke) if sc.get("full") else Pe
        Ie = sc["gamma"] * gb._scale_to(np.where(ke, src_, 0.0), s)
    if sc["par"] == "CELL":
        Ue_, Se_, Vt_ = np.linalg.svd(Pe, full_matrices=False)
        Ie = gb._scale_to(Se_[0] * np.outer(Ue_[:, 0], Vt_[0]), 0.5 * s)
    Dd = RC.residual(gb.M, m) - P
    E = D.row_permute(Dd, m, rng) * (sig / gb.fit[2]["sigma"])
    if sc["rho"] > 0:
        K = m & ke
        De = RC.residual(ge.M, ke) - Pe
        ze = np.where(K, De, 0.0) / float(De[K].std())
        zb = E / float(E[m].std()) * 1.0
        E = sig * (sc["rho"] * ze + math.sqrt(1 - sc["rho"] ** 2) * zb)
    Y = gb.r[:, None] + gb.c[None, :] + Ib + Ie + E
    return np.where(m, Y, 0.0), {"Vb": Vb, "Ve": Ve, "Ib": Ib, "Ie": Ie}


_OBS = {}


def obs_expr_axis(pair):
    key = pair["id"]
    if key not in _OBS:
        K = pair["kb"] & pair["ke"]
        _OBS[key] = {ext: X(ext, pair["Me"], K, 1, 3) for ext in EXTS}
    return _OBS[key]


def gains(Mc, Kc, seeds=CV_SEEDS):
    """K1 的 g01、g12、g23（各 CV 种子 trim95 曲线取平均后）。"""
    cur = []
    for sd in seeds:
        _, E = K1.cv_cell_errors_k1(Mc, Kc, max_rank=3, seed=sd, **{k: v for k, v in K1.CV_HEADLINE.items() if k != "K"})
        cur.append({r: K1.trim95(E[r]) for r in range(4)})
    t = {r: float(np.mean([c[r] for c in cur])) for r in range(4)}
    return {"g01": (t[0] - t[1]) / t[0], "g12": (t[1] - t[2]) / t[1], "g23": (t[2] - t[3]) / t[2]}


def eval_instance(pair, Y, truth, seed, corrs=CORRS, exts=EXTS, split_ext=("EM",), with_gain=False, S=S_W):
    """一个实例（世界或真实矩阵）在各校正 × 抽取器下的统计。truth=None 表示真实数据。"""
    kb, Me, ke, src = pair["kb"], pair["Me"], pair["ke"], pair["src"]
    oa = obs_expr_axis(pair)
    out = {}
    for corr in corrs:
        Vexp = _vexp_for(corr, Me, ke, src, seed + 7, Y, kb)
        Mc, Kc, info = correct(Y, kb, Me, ke, corr, Vexp)
        for ext in exts:
            r1 = X(ext, Mc, Kc, 1, seed)
            r2 = X(ext, Mc, Kc, 2, seed + 1)
            rec = {"share": info["share"], "beta": info.get("beta", 0.0), "n": int(Kc.sum())}
            if r1 is not None:
                a = axis_cos(r1, oa[ext])
                rec["ce1obs"] = a[0] if a else np.nan
                if truth is not None:
                    rec["cb1"] = axis_cos(r1, tr(truth["Vb"][:, 0]))[0]
                    rec["ce1"] = axis_cos(r1, tr(truth["Ve"][:, 0]))[0]
            if truth is not None and r2 is not None:
                al = L.align(r2[0], r2[1], truth["Vb"], FULL20)
                rec["sub_b"] = al["cos"] if al else [np.nan, np.nan]
            if ext in split_ext and corr != "dst":
                sp = split_stats(Y, kb, Me, ke, corr, ext, src, S, seed + 11)
                rec["split_ax"] = float(np.nanmedian(sp[:, 0]))
                rec["split_c2"] = float(np.nanmedian(sp[:, 1]))
            if with_gain and ext == "K1":
                rec.update(gains(Mc, Kc))
            out[(corr, ext)] = rec
    return out
