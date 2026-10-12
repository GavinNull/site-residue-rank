#!/usr/bin/env python3
"""闸标定的共享层：把 docs/23 §5 的秩 1 上下限机器**推广到秩 r**。

为什么是「推广」而不是「第三个副本」
------------------------------------
docs/23 的 `third_family_power.py` 已经有一套
「同 mask 几何 + 纯噪声 = 下限 / 同 mask 几何 + 已知真结构 = 上限」的机器，
但它只做秩 1（`rank1_expl` / `boot_stab_1d`）。
docs/23 §5.3、§9.2 明确记下：**G4 实际用的是秩 2 的 `stab cos2`，
而那个量的纯噪声下限从没标定过**。
本模块把同一套机器参数化到 rank，使 rank=1 时**逐字复现**
`third_family_power` 的数值（这是 CLAUDE.md §10 要求的双向自检之一：
如果复现不出来，就说明我的 harness 和 docs/23 的不是同一个，必须停）。

数值层全部复用：`subspace_lib`（`extract_subspace` / `align` / `dc_fast` /
`haar_pair_null` / `haar_1d_null` / `ci`）、
`third_family_rank1`（`align_1d` / `rank1_dir`）、
`rank_test_power.als_guarded`（经 `extract_subspace(method="als")`；
**没有**用未保护的 `als_complete`）。
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import (MIN_PER_ROW, ROOT, align, ci,  # noqa: E402
                          dc_fast, extract_subspace, haar_1d_null,
                          haar_pair_null)
from third_family_rank1 import align_1d, rank1_dir  # noqa: E402

# docs/21 §4.1 的证据集（9 个成员）。名字必须与 subspace_lib 的 name 字段一致。
COV_REPS = ["starr2020/WH1/huACE2", "animalACE2/Wuhan-Hu-1/bat",
            "PDCoV/PDCoV/gAPN",
            "sarbecovirus/SARS-CoV-1_Urban/Ra9479ACE2",
            "animalACE2/Omicron_BA2/human_monomer"]
OMT = ["OMTKY3/chymotrypsin/1CHO", "OMTKY3/HLE-elastase/1PPF",
       "OMTKY3/subtilisin-Carlsberg/1R0R", "OMTKY3/SGPB/3SGB"]
EVIDENCE9 = OMT + COV_REPS

SIGMAS = [0.15, 0.3, 0.5, 0.8, 1.2, 2.0]


# ----------------------------------------------------------------------
# 1. 秩 r 对**交互**方差的解释率（docs/23 `rank1_expl` 的 rank 参数化版）
# ----------------------------------------------------------------------
def rankr_expl(M, mask, rank=2):
    """秩 r 对双向中心化残差（= 交互）方差的样本内解释率。

    rank=1 时与 `third_family_power.rank1_expl` 逐字同算法。
    """
    B, cols, err = extract_subspace(M, mask, rank=rank)
    if B is None:
        return None, err
    keep_r = mask.sum(axis=1) >= MIN_PER_ROW
    Ms, mk = M[np.ix_(keep_r, cols)], mask[np.ix_(keep_r, cols)]
    R = dc_fast(np.where(mk, Ms, 0.0), mk)
    X = np.where(mk, R, 0.0)
    for _ in range(40):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        L = (U[:, :rank] * s[:rank]) @ Vt[:rank]
        X = np.where(mk, R, L)
        X = X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    L = (U[:, :rank] * s[:rank]) @ Vt[:rank]
    denom = (R[mk] ** 2).sum()
    if denom <= 0:
        return None, "交互方差为 0"
    expl = 1.0 - ((R[mk] - L[mk]) ** 2).sum() / denom
    return float(expl), None


# ----------------------------------------------------------------------
# 2. bootstrap 稳定性（闸实际在用的量），秩 r 版
# ----------------------------------------------------------------------
def stab_boot(M, mask, rank=2, nboot=200, seed=7):
    """返回 nboot 个「bootstrap 子空间 vs 全数据子空间」的对齐值。

    rank = 1 -> `|cos|`（与 `third_family_power.boot_stab_1d` 逐字一致，
                 同一套 rng 调用顺序，所以数值可逐位复现）。
    rank >= 2 -> `cos θ_rank`（rank = 2 即 `cos θ2`，**这就是 G4 的那个量**）。

    口径注意：这是 bootstrap 相对**全数据**子空间的稳定性，不是点估计的
    置信区间。docs/13 §6.2：按位点有放回重抽使对齐系统性偏低
    （bootstrap 中位/点估计 ≈ 0.65），所以这里的下界**不是**置信下界。
    """
    if rank == 1:
        w0, c0, _ = rank1_dir(M, mask)
        if w0 is None:
            return []
    else:
        B0, c0, _ = extract_subspace(M, mask, rank=rank)
        if B0 is None:
            return []
    rng = np.random.default_rng(seed)
    nr = M.shape[0]
    out = []
    for _ in range(nboot):
        idx = rng.integers(0, nr, nr)
        sd = int(rng.integers(1e9))
        if rank == 1:
            wb, kb, _ = rank1_dir(M[idx], mask[idx], seed=sd)
            if wb is None:
                continue
            r = align_1d(w0, c0, wb, kb)
            if r:
                out.append(r["cos"])
        else:
            Bb, kb, _ = extract_subspace(M[idx], mask[idx], rank=rank,
                                         seed=sd)
            if Bb is None:
                continue
            r = align(B0, c0, Bb, kb)
            if r:
                out.append(r["cos"][rank - 1])
    return out


# ----------------------------------------------------------------------
# 3. 合成：纯噪声（无任何低秩结构） / 已知秩 r 真结构
# ----------------------------------------------------------------------
def synth_noise(shape, rng, sigma=1.0):
    """**没有任何低秩结构**的矩阵。这是下限的定义。"""
    return rng.normal(0.0, sigma, shape)


def synth_rank(shape, rank, sigma, rng):
    """已知秩 rank 的真结构 + 各向同性噪声。残基侧的 rank 条方向
    取中心化后正交（与观测量一致：双向中心化后全 1 方向不可见）。"""
    nr, nc = shape
    V = rng.normal(size=(nc, rank))
    V -= V.mean(axis=0, keepdims=True)
    V, _ = np.linalg.qr(V)
    A = rng.normal(0.0, 1.0, (nr, rank))
    return A @ V.T + rng.normal(0.0, sigma, shape)


# ----------------------------------------------------------------------
# 4. 下限 / 上限（种子方案与 docs/23 §5.2 完全一致，便于逐位复现）
# ----------------------------------------------------------------------
def floor_dist(shape, mask, rank=2, n_real=24, nboot=200,
               seed_noise=300, seed_boot=50):
    """纯噪声下限：**每次噪声实现的中位数**的分布。

    docs/23 §5.2 的口径声明照抄：闸用的是每个矩阵的中位数，所以零分布
    必须是「每次实现的中位数」的分布，**不是**把全部 bootstrap 抽样混在
    一起。混在一起会给出 [0.04, 0.97] 这种宽到无意义的区间，
    看起来像「实测落在区间内」。
    """
    meds = []
    for t in range(n_real):
        rng = np.random.default_rng(seed_noise + t)
        X = synth_noise(shape, rng)
        v = stab_boot(X, mask, rank=rank, nboot=nboot, seed=seed_boot + t)
        if v:
            meds.append(float(np.median(v)))
    return meds


def match_sigma(shape, mask, rank, target_expl, sigmas=SIGMAS, n_try=4,
                seed=700):
    """挑一个 sigma，使同几何、已知秩 rank 真结构的**解释率**最接近实测。"""
    best, sg_best, table = None, None, []
    for sg in sigmas:
        ee = []
        for t in range(n_try):
            rng = np.random.default_rng(seed + t + int(sg * 1e4))
            X = synth_rank(shape, rank, sg, rng)
            x, _ = rankr_expl(X, mask, rank=rank)
            if x is not None:
                ee.append(x)
        if not ee:
            table.append((float(sg), float("nan")))
            continue
        m = float(np.median(ee))
        table.append((float(sg), m))
        gap = abs(m - (target_expl if target_expl is not None else 0.3))
        if best is None or gap < best:
            best, sg_best = gap, sg
    return sg_best, table


def ceiling_dist(shape, mask, rank=2, sigma=None, n_real=24, nboot=200,
                 seed_truth=900, seed_boot=80):
    """已知秩 rank 真结构在匹配 SNR 下的上限：同样是「每次实现的中位数」。"""
    meds = []
    for t in range(n_real):
        rng = np.random.default_rng(seed_truth + t)
        X = synth_rank(shape, rank, sigma, rng)
        v = stab_boot(X, mask, rank=rank, nboot=nboot, seed=seed_boot + t)
        if v:
            meds.append(float(np.median(v)))
    return meds


def hi_snr_check(shape, mask, rank=2, sigma=0.05, n_real=6, nboot=120,
                 seed=1234):
    """§10 双向自检之一：**高 SNR 下已知秩 rank 结构必须被还原**
    （stab 贴近 1）。只测上限不测下限的护栏会被绕过（D-007），
    所以这一项必须和 `floor_dist` 的「结构性矩阵必须落进下限分布」成对出现。
    """
    meds = []
    for t in range(n_real):
        rng = np.random.default_rng(seed + t)
        X = synth_rank(shape, rank, sigma, rng)
        v = stab_boot(X, mask, rank=rank, nboot=nboot, seed=seed + 500 + t)
        if v:
            meds.append(float(np.median(v)))
    return meds


# ----------------------------------------------------------------------
# 5. 对半分（docs/13 §4.3 的保守口径），秩 r 版 —— Part 2 要用
# ----------------------------------------------------------------------
def split_half(M, mask, rank=2, nsplit=400, seed=42, which=0):
    """位点随机对半，各自抽秩 rank 子空间再对齐。返回 cos θ_{which+1} 列表。

    rank = 2、which = 0 即 docs/13 §4.3 表里的「对半分自对齐 cos1」。
    """
    rng = np.random.default_rng(seed)
    nr = M.shape[0]
    out = []
    for _ in range(nsplit):
        p = rng.permutation(nr)
        h1, h2 = p[: nr // 2], p[nr // 2:]
        if rank == 1:
            w1, k1, _ = rank1_dir(M[h1], mask[h1])
            w2, k2, _ = rank1_dir(M[h2], mask[h2])
            if w1 is None or w2 is None:
                continue
            r = align_1d(w1, k1, w2, k2)
            if r:
                out.append(r["cos"])
        else:
            B1, k1, _ = extract_subspace(M[h1], mask[h1], rank=rank)
            B2, k2, _ = extract_subspace(M[h2], mask[h2], rank=rank)
            if B1 is None or B2 is None:
                continue
            r = align(B1, k1, B2, k2)
            if r:
                out.append(r["cos"][which])
    return out


def split_half_ceiling(shape, mask, rank=2, sigma=None, n_real=12,
                       nsplit=400, which=0, seed=0):
    """同几何、已知秩 rank 真结构下对半分统计量的上限（中位 + 区间）。"""
    allv = []
    for t in range(n_real):
        rng = np.random.default_rng(seed + 1000 * t + int(sigma * 1e4))
        X = synth_rank(shape, rank, sigma, rng)
        allv += split_half(X, mask, rank=rank,
                           nsplit=max(40, nsplit // n_real), seed=7 + t,
                           which=which)
    return allv


# ----------------------------------------------------------------------
# 6. 零分布缓存
# ----------------------------------------------------------------------
_NULL2, _NULL1 = {}, {}


def null2(d=20, n=4000, seed=1):
    if d not in _NULL2:
        _NULL2[d] = haar_pair_null(d, 2, n=n, seed=seed)
    return _NULL2[d]


def null1(d=20, n=20000, seed=2):
    if d not in _NULL1:
        _NULL1[d] = haar_1d_null(d, n=n, seed=seed)
    return _NULL1[d]


def pcts(x, qs=(50, 90, 95, 99)):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)])
    if len(x) == 0:
        return {q: float("nan") for q in qs}
    return {q: float(np.percentile(x, q)) for q in qs}


def load_evidence():
    """docs/21 §4.1 的 9 个成员（+ 便于对照的其余矩阵）。"""
    from subspace_lib import (dedup_by_content, load_all_dms,
                              load_independent, load_skempi, pick_primary)
    mats, _ = dedup_by_content(load_independent() + load_skempi()
                               + load_all_dms())
    mats = pick_primary(mats)
    return {d["name"]: d for d in mats}
