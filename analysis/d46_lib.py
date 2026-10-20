#!/usr/bin/env python3
"""docs/46：忠实对照世界下的第二成分复现性（Part A）与它是否结合特异（Part B）—— 共享层。

预先登记：`cache/docs46_prereg.md`，sha256 锁在 `cache/docs46_prereg_lock.json`。
任何依赖本模块做结论的脚本必须先调 `verify_lock46()`（CLAUDE.md §4）。

复用，不重写：u2split_lib（抽取器、分半、零分布、植入世界 make_world）、overshoot_lib.Geometry、
subspace_lib、rank_core、rank_k1。**未使用 als_complete**。
本模块新增：
  * 新种子命名空间 `d46`
  * 三个对照世界：G（无截断高斯，主）、RR（残差重采样，次）、GCLIP（旧截断世界，仅作缺陷对照）
  * 保真统计量 F1/F2/F3 与判据（预先登记 §3）
  * 自检（含 FT 系列的辅助函数）
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import hashlib
import json
import math
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

import u2split_lib as U                      # noqa: E402
import subspace_lib as L                     # noqa: E402
import rank_core as RC                       # noqa: E402

LOCK46 = os.path.join(ROOT, "cache", "docs46_prereg_lock.json")
LOCK46A = os.path.join(ROOT, "cache", "docs46a_prereg_lock.json")      # 事后修订 A1（F1 -> F1a）
EXTRACTORS = U.EXTRACTORS

# ---- 预先登记 §3/§4/§5 的固定数
S_REAL = 300
S_WORLD = 8
R_WORLD = 60
N_PERM_MAT = 50
S_PERM = 8
NBOOT = 2000
N_FID_DRAW = 20
F1_MAX, F2_MAX, F3_MAX = 0.01, 0.10, 0.40
QS = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
WORLDS = ("G", "RR")


def verify_lock46(verbose=True):
    out = {}
    for tag, path in (("prereg", LOCK46), ("amend_A1", LOCK46A)):
        lk = json.load(open(path))
        h = hashlib.sha256(open(os.path.join(ROOT, lk["file"]), "rb").read()).hexdigest()
        assert h == lk["sha256"], (
            f"docs/46 {tag} 与锁不一致 —— 按 CLAUDE.md §4 中止\n"
            f"  锁定 {lk['sha256']}\n  现在 {h}")
        if verbose:
            print(f"  预先登记锁 {lk['file']} sha256 {h[:16]}… 一致（锁定于 {lk['locked_at']}）  OK", flush=True)
        out[tag] = {"sha256": h, "locked_at": lk["locked_at"]}
    return out


def seed46(*parts):
    return zlib.crc32(("d46|" + "|".join(str(p) for p in parts)).encode()) & 0x7FFFFFFF


# ======================================================================
# 世界
# ======================================================================
def row_permute(D, mask, rng):
    """每行内对观测格子的值随机置换（保留位点的噪声水平与边缘分布，摧毁残基结构）。"""
    E = np.zeros_like(D)
    for i in range(D.shape[0]):
        obs = np.where(mask[i])[0]
        if len(obs):
            E[i, obs] = D[i, obs][rng.permutation(len(obs))]
    return E


def make_world46(g, world, kind, rng, point=None, infl=1.0):
    """world ∈ {G, RR, GCLIP, REF}；kind ∈ {R0, R1, R2}。返回 Y（mask 外为 0）。
    G：r + c + I(k,s) + N(0,σ²)，不截断。 RR：r + c + I(k,s) + (σ/σ_fit(k))·行内置换的秩 k 拟合残差，不截断。
    GCLIP：G 的 clip=True 版本（旧缺陷）。 REF：r + c + P_k + 行内置换的拟合残差（已知忠实的参照，σ=σ_fit(k) 端，尺度 1）。
    infl ≠ 1：把双向中心化部分（交互+噪声）整体放大（FT4 的方差失配缺陷）。"""
    m = g.mask
    if kind == "R0":
        if world in ("G", "GCLIP"):
            Y, _ = g.make(0, "real", 0.0, g.sigma0(), "gauss", rng, clip=(world == "GCLIP"))
            return Y
        if world == "RR":
            Rf = RC.residual(g.M, m)
            Y = g.r[:, None] + g.c[None, :] + row_permute(Rf, m, rng)
            return np.where(m, Y, 0.0)
        raise ValueError(world)
    k = 1 if kind == "R1" else 2
    if world in ("G", "GCLIP"):
        Y, _ = g.make(k, "real", point["s"], point["sigma"], "gauss", rng, clip=(world == "GCLIP"))
    elif world == "RR":
        P = g.fit[k]["P"]
        D = RC.residual(g.M, m) - P
        E = row_permute(D, m, rng)
        I = g._scale_to(P.copy(), point["s"])
        Y = g.r[:, None] + g.c[None, :] + I + (point["sigma"] / g.fit[k]["sigma"]) * E
        Y = np.where(m, Y, 0.0)
    elif world == "REF":
        P = g.fit[k]["P"]
        D = RC.residual(g.M, m) - P
        E = row_permute(D, m, rng)
        Y = np.where(m, g.r[:, None] + g.c[None, :] + P + E, 0.0)
    else:
        raise ValueError(world)
    if infl != 1.0:
        base = g.r[:, None] + g.c[None, :]
        Y = np.where(m, base + infl * (Y - base), 0.0)
    return Y


# ======================================================================
# 保真统计量（预先登记 §3）
# ======================================================================
def real_ref(g):
    """真实矩阵的 R（观测格，排序）与 cso_f（同一个 dc_fast 口径）。"""
    R = L.dc_fast(np.where(g.mask, g.M, 0.0), g.mask)[g.mask]
    Rs = np.sort(R)
    return {"Rs": Rs, "cso": float(R.std()), "q": np.quantile(R, QS), "lo": g.lo, "hi": g.hi}


def fid_stats(g, Y, ref):
    m = g.mask
    v = Y[m]
    f1 = float(np.mean((v <= ref["lo"] + 1e-9) | (v >= ref["hi"] - 1e-9)))      # 锁定判据：在端点或之外
    f1a = float(np.mean((np.abs(v - ref["lo"]) < 1e-9) | (np.abs(v - ref["hi"]) < 1e-9)))   # 修订 A1：堆积于端点
    f1b = float(np.mean((v < ref["lo"] - 1e-9) | (v > ref["hi"] + 1e-9)))       # 越界（只报告）
    R = L.dc_fast(Y, m)[m]
    Rs = np.sort(R)
    f2 = float(np.mean(np.abs(Rs - ref["Rs"])) / ref["cso"])
    f3 = float(np.max(np.abs(np.quantile(R, QS) - ref["q"])) / ref["cso"])
    return f1, f2, f3, f1a, f1b


def fid_point(g, ref, world, kind, point, tag, ndraw=N_FID_DRAW, infl=1.0):
    """20 个新鲜实例的均值。返回 dict(f1,f2,f3,pass_f1,pass_f2,pass_f3,pass)。"""
    out = []
    for d in range(ndraw):
        rng = np.random.default_rng(seed46("fid", tag, g.name, world, kind, d))
        Y = make_world46(g, world, kind, rng, point=point, infl=infl)
        out.append(fid_stats(g, Y, ref))
    f1, f2, f3, f1a, f1b = np.mean(out, axis=0)
    p1, p2, p3, p1a = f1 <= F1_MAX, f2 <= F2_MAX, f3 <= F3_MAX, f1a <= F1_MAX
    return {"f1": float(f1), "f2": float(f2), "f3": float(f3), "f1a": float(f1a), "f1b": float(f1b),
            "pass_f1": bool(p1), "pass_f2": bool(p2), "pass_f3": bool(p3), "pass_f1a": bool(p1a),
            "pass": bool(p1 and p2 and p3),            # 锁定判据
            "pass_am": bool(p1a and p2 and p3)}        # 修订 A1 判据


# ======================================================================
# 自检
# ======================================================================
def selfcheck46(geoms, verbose=True):
    """先决：dc_fast 与 RC.residual 等价（D-023 式）、世界生成器的已知事实、保真统计量的已知值。"""
    log = []
    rng = np.random.default_rng(46)
    names = list(geoms)[:2]
    for nm in names:
        g = geoms[nm]
        d = np.abs(L.dc_fast(g.M, g.mask) - RC.residual(g.M, g.mask))[g.mask].max()
        assert d < 1e-8, f"dc_fast 与 RC.residual 不等价 max|Δ|={d:.2e}"
        log.append(f"E1 {nm}: dc_fast ≡ RC.residual（max|Δ| = {d:.1e}）  OK")
        ref = real_ref(g)
        # 已知值：真实矩阵对自己，F2 = F3 = 0，F1 = 值域端点的格子数 / n
        f1, f2, f3, f1a, f1b = fid_stats(g, g.M, ref)
        assert f2 < 1e-12 and f3 < 1e-12, "真实对真实的 F2/F3 不为 0"
        nb = int(((g.M[g.mask] <= g.lo + 1e-9) | (g.M[g.mask] >= g.hi - 1e-9)).sum())
        assert abs(f1 - nb / g.n) < 1e-12 and nb >= 2
        assert f1 <= F1_MAX, f"真实矩阵自己的 F1 = {f1:.4f} > 判据 {F1_MAX}（判据与真实矩阵不符）"
        log.append(f"E2 {nm}: 真实对真实 F1 = {f1:.4f}（{nb} 个格子在端点）、F2 = F3 = 0  OK")
        # 已知事实：clip=True 的世界把越界的格子堆在端点上，被截比例 = F1 减去本来在端点上的格子
        pt = {"sigma": g.fit[1]["sigma"], "s": 0.0}
        # 人为：σ 很大的世界必须被截，F1 > 5%
        big = {"sigma": 3.0 * g.cso, "s": 0.0}
        Yc = make_world46(g, "GCLIP", "R1", np.random.default_rng(1), point=big)
        Yg = make_world46(g, "G", "R1", np.random.default_rng(1), point=big)
        assert fid_stats(g, Yc, ref)[0] > 0.05, "大噪声截断世界没有被 F1 抓住"
        assert fid_stats(g, Yc, ref)[3] > 0.05, "大噪声截断世界没有被 F1a（堆积于端点）抓住"
        assert fid_stats(g, Yg, ref)[3] < 0.002, "无截断世界的 F1a 应 ≈ 0（连续值不会恰在端点）"
        assert fid_stats(g, Yg, ref)[4] > 0.05, "极端噪声的无截断世界应有大量越界（F1b）"
        log.append(f"E3 {nm}: 极端噪声的截断世界 F1 = {fid_stats(g, Yc, ref)[0]:.3f} > 5%；无截断世界 F1 = {fid_stats(g, Yg, ref)[0]:.3f}（F1 数的是「在边界或之外」，截断与否越界比例相同：被截的格子堆在边界上）  OK")
        # 无截断世界不会改动值（clip=False 与 clip=True 在不越界的值上相同）
        Yc2 = make_world46(g, "GCLIP", "R0", np.random.default_rng(2))
        Yg2 = make_world46(g, "G", "R0", np.random.default_rng(2))
        inside = (Yg2 > g.lo) & (Yg2 < g.hi) & g.mask
        assert np.array_equal(Yc2[inside], Yg2[inside]), "截断世界在未越界处与无截断世界不同"
        assert np.all(Yc2[g.mask] >= g.lo) and np.all(Yc2[g.mask] <= g.hi)
        log.append(f"E4 {nm}: 截断世界仅在越界处不同，且被限在 [lo, hi]  OK")
        # RR 世界的行内置换保持每行的值集合
        Rf = RC.residual(g.M, g.mask)
        D = Rf - g.fit[1]["P"]
        E = row_permute(D, g.mask, rng)
        for i in range(0, g.nr, 17):
            a = np.sort(D[i][g.mask[i]]); b = np.sort(E[i][g.mask[i]])
            assert np.allclose(a, b)
        assert (E[~g.mask] == 0).all()
        log.append(f"E5 {nm}: 残差行内置换保留每行值集合、未观测处为 0  OK")
    if verbose:
        for s in log:
            print("  " + s, flush=True)
    return log


def boot_ci_frac(x, nboot=NBOOT, seed=0):
    return U.boot_median(x, nboot, seed)
