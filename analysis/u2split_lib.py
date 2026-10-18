#!/usr/bin/env python3
"""docs/45：第二成分的复现性在「位点不共享」时还存在吗 —— 共享层。

预先登记：`cache/docs45_prereg.md`，sha256 锁在 `cache/docs45_prereg_lock.json`。
任何依赖本模块做结论的脚本必须先调 `verify_lock()`（CLAUDE.md §4）。

本模块：
  * 位点分半（不相交）+ 两个子空间抽取器（E-EM = subspace_lib.extract_subspace；E-K1 = rank_k1.als_joint 联合拟合）
  * 主角余弦比较（subspace_lib.align）+ 该比较自己的 d 上的 Haar 零分布
  * 植入世界（overshoot_lib.Geometry 的真实几何；R0/R1/R2/R1P/R2P/R2λ）
  * 方差占比（标准拟合与 K1 拟合）
  * 工具自检（预先登记 §8 T1–T4）
数值层复用，不重写：subspace_lib.extract_subspace / align / haar_pair_null / permute_within_sites，
rank_k1.als_joint，overshoot_lib.Geometry。**未使用 als_complete。**
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

import subspace_lib as L                       # noqa: E402
import rank_k1 as K1                           # noqa: E402
import rank_core as RC                         # noqa: E402

LOCK = os.path.join(ROOT, "cache", "docs45_prereg_lock.json")
AA20 = L.AA20
EXTRACTORS = ("EM", "K1")

# ---- 预先登记 §3/§7 的固定数
S_REAL = 200
S_WORLD = 8
R_WORLD = 60
N_PERM_MAT = 50
S_PERM = 8
NBOOT = 2000
HAAR_N = 4000
MIN_PER_ROW, MIN_PER_COL = L.MIN_PER_ROW, L.MIN_PER_COL


def verify_lock(verbose=True):
    lk = json.load(open(LOCK))
    h = hashlib.sha256(open(os.path.join(ROOT, lk["file"]), "rb").read()).hexdigest()
    assert h == lk["sha256"], (
        "docs/45 预先登记与锁不一致 —— 按 CLAUDE.md §4 中止\n"
        f"  锁定 {lk['sha256']}\n  现在 {h}")
    if verbose:
        print(f"  预先登记锁 {lk['file']} sha256 {h[:16]}… 一致（锁定于 {lk['locked_at']}）  OK", flush=True)
    return {"prereg_sha256": h, "locked_at": lk["locked_at"]}


def seed_of(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


# ======================================================================
# 抽取器
# ======================================================================
def _protect(M, mask):
    """与 subspace_lib.extract_subspace 相同的行/列保护。返回 (Ms, ms, cols) 或 None。"""
    keep_r = mask.sum(axis=1) >= MIN_PER_ROW
    keep_c = mask.sum(axis=0) >= MIN_PER_COL
    if keep_r.sum() < 4 or keep_c.sum() < 4:
        return None
    cols = np.where(keep_c)[0]
    Ms = M[np.ix_(keep_r, cols)]
    ms = mask[np.ix_(keep_r, cols)]
    for _ in range(3):
        kr = ms.sum(axis=1) >= MIN_PER_ROW
        kc = ms.sum(axis=0) >= MIN_PER_COL
        if kr.all() and kc.all():
            break
        Ms = Ms[np.ix_(kr, kc)]
        ms = ms[np.ix_(kr, kc)]
        cols = cols[kc]
    if Ms.shape[0] < 4 or Ms.shape[1] < 4:
        return None
    return Ms, ms, cols


def extract_em(M, mask, seed=0):
    B, cols, _ = L.extract_subspace(M, mask, rank=2, method="emsvd", seed=seed)
    return (B, cols) if B is not None else None


def extract_k1(M, mask, seed=0):
    """E-K1：联合拟合 r_i + c_j + U Vᵀ（rank_k1.als_joint，秩 2，ridge 0.1，iters 150，2 次重启按训练误差择优），
    对完整预测 P 做全矩阵双向去均值，取前 2 个右奇异向量。"""
    pr = _protect(np.asarray(M, float), np.asarray(mask, bool))
    if pr is None:
        return None
    Ms, ms, cols = pr
    Mf = np.where(ms, Ms, 0.0)
    rng = np.random.default_rng(seed)
    best, bestP = np.inf, None
    for _ in range(2):
        P = K1.als_joint(Mf, ms, 2, rng, iters=150, ridge=0.1)
        tr = float(np.mean((Mf[ms] - P[ms]) ** 2))
        if tr < best:
            best, bestP = tr, P
    Q = bestP - bestP.mean(1, keepdims=True) - bestP.mean(0, keepdims=True) + bestP.mean()
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    B, _ = np.linalg.qr(Vt[:2].T)
    return B, cols


def extract(ext, M, mask, seed=0):
    return extract_em(M, mask, seed) if ext == "EM" else extract_k1(M, mask, seed)


def compare(ra, rb):
    """ra, rb = (B, cols)；返回 (cos1, cos2, d) 或 None。"""
    if ra is None or rb is None:
        return None
    a = L.align(ra[0], ra[1], rb[0], rb[1])
    if a is None:
        return None
    return float(a["cos"][0]), float(a["cos"][1]), int(a["n_res"])


# ======================================================================
# 分半
# ======================================================================
def split_sites(nr, rng):
    """随机不相交两半，各 nr//2。返回 (ia, ib)；断言不相交。"""
    p = rng.permutation(nr)
    h = nr // 2
    ia, ib = np.sort(p[:h]), np.sort(p[h:2 * h])
    assert len(np.intersect1d(ia, ib)) == 0, "两半位点相交 —— 分半实现有错"
    return ia, ib


def split_pair_cos(M, mask, ia, ib, ext, seed=0):
    ra = extract(ext, M[ia], mask[ia], seed)
    rb = extract(ext, M[ib], mask[ib], seed + 1)
    return compare(ra, rb)


def many_splits(M, mask, ext, S, seed):
    """S 个随机分半 -> array (S,3) [cos1, cos2, d]；失败的行记 NaN。"""
    rng = np.random.default_rng(seed)
    out = np.full((S, 3), np.nan)
    for s in range(S):
        ia, ib = split_sites(M.shape[0], rng)
        r = split_pair_cos(M, mask, ia, ib, ext, seed=int(rng.integers(1 << 30)))
        if r is not None:
            out[s] = r
    return out


_HAAR = {}


def haar_null(d):
    """该比较自己的环境维数 d 上的 Haar 零分布（预先登记 §5；docs/23 的 d 错误）。"""
    d = int(d)
    if d not in _HAAR:
        _HAAR[d] = L.haar_pair_null(d, rank=2, n=HAAR_N, seed=d)
    return _HAAR[d]


def perm_null(M, mask, ext, nmat, S, seed):
    """位点内残基置换矩阵上的分半 -> array (nmat*S, 3)。"""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(nmat):
        Mp, mp = L.permute_within_sites(np.where(mask, M, np.nan), mask, rng)
        rows.append(many_splits(np.where(mp, Mp, 0.0), mp, ext, S, int(rng.integers(1 << 30))))
    return np.concatenate(rows)


def boot_median(x, nboot=NBOOT, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (nboot, len(x)))
    med = np.median(x[idx], axis=1)
    return float(np.median(x)), float(np.quantile(med, 0.025)), float(np.quantile(med, 0.975))


# ======================================================================
# 植入世界（在矩阵自己的几何上）
# ======================================================================
def make_world(g, kind, rng, point=None, lam=None, clip=True):
    """返回 Y（mask 外为 0）。g = overshoot_lib.Geometry。
    kind ∈ {'R0','R1','R2','R1P','R2P','R2L'}；point = {'sigma','s'}（R1/R2/R2L 用）。"""
    nr, nc = g.nr, g.nc
    if kind == "R0":
        sg = g.sigma0()
        Y, _ = g.make(0, "real", 0.0, sg, "gauss", rng, clip=True)
        return Y
    if kind in ("R1", "R2"):
        k = 1 if kind == "R1" else 2
        Y, _ = g.make(k, "real", point["s"], point["sigma"], "gauss", rng, clip=clip)
        return Y
    if kind in ("R1P", "R2P"):
        k = 1 if kind == "R1P" else 2
        P = g.fit[k]["P"]
        Rf = RC.residual(g.M, g.mask)
        E = np.zeros((nr, nc))
        D = Rf - P
        for i in range(nr):
            obs = np.where(g.mask[i])[0]
            if len(obs):
                E[i, obs] = D[i, obs][rng.permutation(len(obs))]
        Y = g.r[:, None] + g.c[None, :] + P + E
        Y = np.clip(Y, g.lo, g.hi)
        return np.where(g.mask, Y, 0.0)
    if kind == "R2L":
        P2 = g.fit[2]["P"]
        U, s, Vt = np.linalg.svd(P2, full_matrices=False)
        c1 = s[0] * np.outer(U[:, 0], Vt[0])
        c2 = s[1] * np.outer(U[:, 1], Vt[1])
        rms2 = math.sqrt(float((P2[g.mask] ** 2).mean()))
        f = point["s"] / rms2
        Y = g.r[:, None] + g.c[None, :] + f * (c1 + lam * c2) + rng.normal(0, point["sigma"], (nr, nc))
        Y = np.clip(Y, g.lo, g.hi)
        return np.where(g.mask, Y, 0.0)
    raise ValueError(kind)


def planted_share2(g, point, lam):
    """R2L 世界里第二项的真实占比（观测格上，f·λ·c2 对 f·(c1+λ c2) 两项平方和之比）。"""
    P2 = g.fit[2]["P"]
    U, s, Vt = np.linalg.svd(P2, full_matrices=False)
    c1 = s[0] * np.outer(U[:, 0], Vt[0])
    c2 = lam * s[1] * np.outer(U[:, 1], Vt[1])
    a = float((c1[g.mask] ** 2).sum())
    b = float((c2[g.mask] ** 2).sum())
    return b / (a + b)


# ======================================================================
# 方差占比（预先登记 §10）
# ======================================================================
def comps_share(P, mask, R):
    """P = 拟合的交互（全矩阵）。SVD 取前两项，在观测格上的平方和。
    返回 dict(share2 = 第二项/(第一+第二)，tot1 = 第一项/总残差，tot2 = 第二项/总残差)。"""
    U, s, Vt = np.linalg.svd(P, full_matrices=False)
    q = []
    for k in (0, 1):
        ck = s[k] * np.outer(U[:, k], Vt[k])
        q.append(float((ck[mask] ** 2).sum()))
    tot = float((R[mask] ** 2).sum())
    return {"share2": q[1] / (q[0] + q[1]), "tot1": q[0] / tot, "tot2": q[1] / tot,
            "s": [float(s[0]), float(s[1])]}


def share_standard(M, mask, seed=7):
    """docs/14 同口径：双向去均值 + als_guarded 秩 2（ridge 0.05，restarts 6，iters 400）。"""
    import site_loadings as SL
    Mm = np.where(mask, M, np.nan)
    L_, V_, s_, Rm, P, info = SL.fit_rank2(Mm, mask, seed=seed)
    out = comps_share(P, mask, Rm)
    out["rank2_expl"] = info["rank2_expl"]
    return out


def share_k1(M, mask, seed=3):
    Mf = np.where(mask, M, 0.0)
    rng = np.random.default_rng(seed)
    best, bestP = np.inf, None
    for _ in range(2):
        P = K1.als_joint(Mf, mask, 2, rng, iters=150, ridge=0.1)
        tr = float(np.mean((Mf[mask] - P[mask]) ** 2))
        if tr < best:
            best, bestP = tr, P
    # 去掉可加部分（全矩阵双向去均值）-> 交互；总残差取数据的双向去均值残差（观测格）
    Q = bestP - bestP.mean(1, keepdims=True) - bestP.mean(0, keepdims=True) + bestP.mean()
    R = RC.residual(Mf, mask)
    return comps_share(Q, mask, R)


# ======================================================================
# 自检（预先登记 §8；既测命中，也测不误伤）
# ======================================================================
def selfcheck(geoms, verbose=True):
    """geoms = {name: Geometry}（至少 2 个）。失败抛 AssertionError。"""
    log = []

    # T1 已知事实：Haar 2 维子空间对，Σcos² 均值 = 4/(d-1)
    for d in (20, 12):
        h = L.haar_pair_null(d, rank=2, n=3000, seed=1)
        m = float(h["ssc"].mean())
        want = 4.0 / (d - 1)
        assert abs(m - want) < 0.012 * (20 / d), f"T1 Haar 零分布 d={d} Σcos² 均值 {m:.4f} ≠ {want:.4f}"
        log.append(f"T1 Haar 零分布 d={d}：Σcos² 均值 {m:.4f} vs 解析 4/(d-1) = {want:.4f}  OK")
    # T1b：d 敏感（docs/23 的错误方向）：d=20 与 d=10 的 cos θ₂ 中位必须明显不同
    h20 = np.median(L.haar_pair_null(20, n=2000, seed=2)["cos2"])
    h10 = np.median(L.haar_pair_null(10, n=2000, seed=2)["cos2"])
    assert h10 > h20 + 0.05, "T1b 零分布对 d 不敏感"
    log.append(f"T1b 零分布随 d 变化：cos θ₂ 中位 d=20 {h20:.3f} < d=10 {h10:.3f}（用错 d 会错）  OK")

    names = list(geoms)[:2]
    rng = np.random.default_rng(45)
    for nm in names:
        g = geoms[nm]
        # T2 理想条件还原：无噪声秩 2 —— 命中
        P2 = g.fit[2]["P"]
        for ext in EXTRACTORS:
            Y = np.where(g.mask, g.r[:, None] + g.c[None, :] + P2, 0.0)
            c = many_splits(Y, g.mask, ext, 6, seed_of("T2a", nm, ext))
            m2 = float(np.nanmedian(c[:, 1]))
            assert m2 >= 0.98, f"T2 {nm} {ext} 无噪声秩 2 分半 cos θ₂ 中位 {m2:.3f} < 0.98（抽取器还原不出已知结构）"
            log.append(f"T2a {nm} [{ext}] 无噪声秩 2：分半 cos θ₁ {np.nanmedian(c[:,0]):.4f} cos θ₂ {m2:.4f}  OK")
        # T2b：无噪声秩 1 —— 第一成分命中（cos θ₁ ≥ 0.98）
        P1 = g.fit[1]["P"]
        for ext in EXTRACTORS:
            Y = np.where(g.mask, g.r[:, None] + g.c[None, :] + P1, 0.0)
            c = many_splits(Y, g.mask, ext, 6, seed_of("T2b", nm, ext))
            m1 = float(np.nanmedian(c[:, 0]))
            assert m1 >= 0.98, f"T2b {nm} {ext} 无噪声秩 1 分半 cos θ₁ {m1:.3f} < 0.98"
            log.append(f"T2b {nm} [{ext}] 无噪声秩 1：分半 cos θ₁ {m1:.4f}（cos θ₂ {np.nanmedian(c[:,1]):.3f} 此处无意义）  OK")
        # T3 不误伤：R0 无结构世界的 cos θ₂ 中位不超过同几何 N-perm 的 p95
        for ext in EXTRACTORS:
            Y0 = make_world(g, "R0", np.random.default_rng(seed_of("T3", nm, ext)))
            c0 = many_splits(Y0, g.mask, ext, 24, seed_of("T3b", nm, ext))
            pn = perm_null(g.M, g.mask, ext, 6, 4, seed_of("T3c", nm, ext))
            p95 = float(np.nanquantile(pn[:, 1], 0.95))
            med = float(np.nanmedian(c0[:, 1]))
            assert med <= p95, f"T3 {nm} {ext} 无结构世界 cos θ₂ 中位 {med:.3f} > 置换零 p95 {p95:.3f}（假警报）"
            log.append(f"T3 {nm} [{ext}] 无结构世界 cos θ₂ 中位 {med:.3f} ≤ 置换零 p95 {p95:.3f}  OK")
        # T4 分半不泄露：改一个位点的全部数值，另一半的基逐位不变
        M0, mk = g.M.copy(), g.mask
        ia, ib = split_sites(g.nr, np.random.default_rng(5))
        for ext in EXTRACTORS:
            rb0 = extract(ext, M0[ib], mk[ib], 11)
            M1 = M0.copy()
            # 注意：整行加常数会被双向去均值吸收（那是主效应），必须加**非可加**的模式
            M1[ia[0]] += 50.0 * np.random.default_rng(6).normal(size=M1.shape[1])
            ra0 = extract(ext, M0[ia], mk[ia], 10)
            ra1 = extract(ext, M1[ia], mk[ia], 10)
            rb1 = extract(ext, M1[ib], mk[ib], 11)
            assert np.array_equal(rb0[0], rb1[0]), "T4 另一半的基被改动 —— 泄露"
            assert not np.allclose(ra0[0], ra1[0]), "T4 改动没有进入所在半（自检本身失效）"
        log.append(f"T4 {nm} 分半不泄露：改一个位点，另一半基逐位不变；所在半基改变；两半位点不相交  OK")
    if verbose:
        for s in log:
            print("  " + s, flush=True)
    return log
