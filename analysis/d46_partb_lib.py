#!/usr/bin/env python3
"""docs/46 Part B 共享层：表达量矩阵的构建与自检、配对比较、植入配对世界。

预先登记：cache/docs46_prereg.md §6 + 修订 A1（cache/docs46a_prereg.md）+ 修订 A2（cache/docs46b_prereg.md，Part B 的实现细节，
写于任何表达量统计量之前）。入口断言三个锁。
复用，不重写：u2split_lib（抽取器、分半、Haar）、overshoot_lib.Geometry、d46_lib（世界、保真）、subspace_lib.align。
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
import d46_lib as D                                  # noqa: E402
import u2split_lib as U                              # noqa: E402
import subspace_lib as L                             # noqa: E402
import rank_core as RC                               # noqa: E402
import overshoot_lib as OL                           # noqa: E402

DMS = os.path.join(ROOT, "cache", "dms")
AA20 = L.AA20
LOCK46B = os.path.join(ROOT, "cache", "docs46b_prereg_lock.json")
OMI = "starr_omicron_final_variant_scores.csv"
PDC = "starr_PDCoV_RBD_final_variant_scores.csv"
R_B = 40
S_B = 6
S_REAL_B = 300
NPERM_B = 30
PT_IDX = (0, 3, 6)


def verify_locks(verbose=True):
    out = D.verify_lock46(verbose)
    lk = json.load(open(LOCK46B))
    h = hashlib.sha256(open(os.path.join(ROOT, lk["file"]), "rb").read()).hexdigest()
    assert h == lk["sha256"], f"docs/46 A2 与锁不一致 —— 中止\n  锁定 {lk['sha256']}\n  现在 {h}"
    if verbose:
        print(f"  预先登记锁 {lk['file']} sha256 {h[:16]}… 一致（锁定于 {lk['locked_at']}）  OK", flush=True)
    out["amend_A2"] = {"sha256": h, "locked_at": lk["locked_at"]}
    return out


# ======================================================================
# 数据
# ======================================================================
_CACHE = {}


def _df(fn):
    if fn not in _CACHE:
        _CACHE[fn] = pd.read_csv(os.path.join(DMS, fn))
    return _CACHE[fn]


def build_mat(fn, target, col, sites):
    """位点（给定列表，整数）× AA20 的矩阵。格子 = 该列非空且大于该列（该目标内）最小值 + 1e-6
    （与 subspace_lib._starr_matrix 对结合列的惯例相同）。返回 M（mask 外为 0）, mask。"""
    d = _df(fn)
    d = d[d["target"] == target][["position", "mutant", col]].dropna()
    lo = float(d[col].min())
    d = d[d[col] > lo + 1e-6]
    si = {int(s): i for i, s in enumerate(sites)}
    M = np.zeros((len(sites), 20))
    mask = np.zeros((len(sites), 20), bool)
    for pos, mut, v in zip(d["position"].values, d["mutant"].values, d[col].values):
        if int(pos) in si and mut in AA20:
            M[si[int(pos)], AA20.index(mut)] = v
            mask[si[int(pos)], AA20.index(mut)] = True
    return M, mask


def build_reps_arrays(fn, target, cols, M, mask, sites, tol=1e-4):
    """cols = 重复列名列表。核对：矩阵每个观测格子 == 可用重复的均值（已知事实，不符即中止，D-023 式）。
    返回 (Dm, Nrep)，Dm = 前两个重复之差（仅两个都有的格子）。"""
    d = _df(fn)
    d = d[d["target"] == target].set_index(["position", "mutant"])
    nr = len(sites)
    Dm = np.full((nr, 20), np.nan)
    Nrep = np.zeros((nr, 20), int)
    maxdev = 0.0
    for i, s in enumerate(sites):
        for j, a in enumerate(AA20):
            if not mask[i, j]:
                continue
            row = d.loc[(int(s), a)]
            have = [row[c] for c in cols if np.isfinite(row[c])]
            assert len(have) >= 1, f"{target} ({s},{a}) 无重复值"
            dev = abs(M[i, j] - float(np.mean(have)))
            maxdev = max(maxdev, dev)
            assert dev < tol, f"{target} ({s},{a}) 矩阵值 {M[i, j]} ≠ 重复均值 {np.mean(have)}（列 {cols}）"
            Nrep[i, j] = len(have)
            if len(have) >= 2 and np.isfinite(row[cols[0]]) and np.isfinite(row[cols[1]]):
                Dm[i, j] = row[cols[0]] - row[cols[1]]
    return Dm, Nrep, maxdev


def make_geom(name, M, mask, Dm=None, Nrep=None):
    d = {"name": name, "M": M, "mask": mask, "sites": list(range(M.shape[0]))}
    g = OL.Geometry(d)
    if Dm is not None:
        g.Dm, g.Nrep, g.rep_kind = Dm, Nrep, "measured"
        g.sig = OL.sigma_meas_from(Dm, Nrep, mask)
    return g


# ======================================================================
# 配对比较
# ======================================================================
def pair_split_cos(Mb, kb, Me, ke, ia, ib, ext, seed):
    """结合取半 A、表达量取半 B（与反向）；返回 (cos1, cos2, d) 的两个方向的均值，或 None。"""
    bA = U.extract(ext, Mb[ia], kb[ia], seed)
    bB = U.extract(ext, Mb[ib], kb[ib], seed + 1)
    eA = U.extract(ext, Me[ia], ke[ia], seed + 2)
    eB = U.extract(ext, Me[ib], ke[ib], seed + 3)
    r1 = U.compare(bA, eB)
    r2 = U.compare(bB, eA)
    if r1 is None or r2 is None:
        return None
    return ((r1[0] + r2[0]) / 2, (r1[1] + r2[1]) / 2, (r1[2] + r2[2]) / 2)


def pair_many(Mb, kb, Me, ke, ext, S, seed):
    rng = np.random.default_rng(seed)
    out = np.full((S, 3), np.nan)
    for s in range(S):
        ia, ib = U.split_sites(Mb.shape[0], rng)
        r = pair_split_cos(Mb, kb, Me, ke, ia, ib, ext, int(rng.integers(1 << 30)))
        if r is not None:
            out[s] = r
    return out


def pair_full(Mb, kb, Me, ke, ext, seed):
    r = U.compare(U.extract(ext, Mb, kb, seed), U.extract(ext, Me, ke, seed + 1))
    return np.array(r, float) if r is not None else np.full(3, np.nan)


def pair_perm_null(Mb, kb, Me, ke, ext, nmat, S, seed):
    """表达量矩阵位点内残基置换后与（真实）结合矩阵比较。返回 (nmat*S, 3)。"""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(nmat):
        Mp, mp = L.permute_within_sites(np.where(ke, Me, np.nan), ke, rng)
        rows.append(pair_many(Mb, kb, np.where(mp, Mp, 0.0), mp, ext, S, int(rng.integers(1 << 30))))
    return np.concatenate(rows)


def haar_null_2k(d, k, n=3000, seed=0):
    """2 维对 k 维子空间的主角余弦零分布（中心化 d-1 维环境）。返回 dict(cos1, cos2)。"""
    from scipy.linalg import subspace_angles
    rng = np.random.default_rng(seed)
    one = np.ones((d, 1)) / math.sqrt(d)
    Pm = np.eye(d) - one @ one.T
    c1, c2 = [], []
    for _ in range(n):
        A, _ = np.linalg.qr(Pm @ rng.normal(size=(d, 2)))
        B, _ = np.linalg.qr(Pm @ rng.normal(size=(d, k)))
        cs = np.sort(np.cos(subspace_angles(A, B)))[::-1]
        c1.append(cs[0]); c2.append(cs[1])
    return {"cos1": np.array(c1), "cos2": np.array(c2)}


# ======================================================================
# 植入配对世界
# ======================================================================
def right_basis(P, k=2):
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    return Vt[:k].T


def centered_orth(nc, rng, against=None, k=1):
    """k 个中心化、彼此正交（并与 against 列正交）的随机单位向量。"""
    cols = [] if against is None else [against[:, j] for j in range(against.shape[1])]
    out = []
    for _ in range(k):
        v = rng.normal(size=nc)
        v -= v.mean()
        for q in cols + out:
            v -= (v @ q) * q
        v /= np.linalg.norm(v)
        out.append(v)
    return np.stack(out, axis=1)


def expr_world(ge, V, rng, pt, noise):
    """表达量侧：交互 = U_e S_e Vᵀ（表达量自己的位点载荷与谱，残基基换成 V），按 pt 的 s 缩放；
    noise = 'RR'（自己秩 2 拟合残差行内置换，按 σ/σ_fit 缩放）或 'G'（无截断高斯）。"""
    P = ge.fit[2]["P"]
    Uu, S, _ = np.linalg.svd(P, full_matrices=False)
    Pn = (Uu[:, :2] * S[:2]) @ V.T
    I = ge._scale_to(Pn, pt["s"])
    m = ge.mask
    base = ge.r[:, None] + ge.c[None, :] + I
    if noise == "RR":
        Dd = RC.residual(ge.M, m) - P
        E = row_permute(Dd, m, rng)
        Y = base + (pt["sigma"] / ge.fit[2]["sigma"]) * E
    else:
        Y = base + rng.normal(0, pt["sigma"], ge.M.shape)
    return np.where(m, Y, 0.0)


def row_permute(D_, mask, rng):
    return D.row_permute(D_, mask, rng)


def planted_pair(gb, ge, mode, rng, ptb, pte, noise):
    """返回 (Yb, Ye)。mode ∈ {SHARED, PARTIAL, INDEP}。结合侧永远是它自己的真实秩 2 结构（V_b）。"""
    Yb = D.make_world46(gb, noise if noise == "RR" else "G", "R2", rng, point=ptb)
    Vb = right_basis(gb.fit[2]["P"], 2)
    nc = Vb.shape[0]
    if mode == "SHARED":
        V = Vb
    elif mode == "PARTIAL":
        v2 = centered_orth(nc, rng, against=Vb[:, :1], k=1)
        V = np.concatenate([Vb[:, :1], v2], axis=1)
    elif mode == "INDEP":
        V = centered_orth(nc, rng, k=2)
    else:
        raise ValueError(mode)
    Ye = expr_world(ge, V, rng, pte, noise)
    return Yb, Ye
