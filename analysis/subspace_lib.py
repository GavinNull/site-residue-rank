#!/usr/bin/env python3
"""共享工具层：从各体系抽取 site x residue 交互的**二维残基载荷子空间**，
并用主角度 (principal angles) 做子空间对齐比较。

为什么必须是「子空间」而不是「载荷向量」
--------------------------------------------------
秩2 双线性模型 interaction ~ U S V^T 里，对任意可逆 2x2 矩阵 T，
(U T^{-T}, V T) 给出完全相同的拟合。所以 v_1, v_2 **各自不可辨识**，
把体系 A 的 v_1 去和体系 B 的 v_1 求相关是没有意义的（只会得到噪声）。
可辨识的是二维子空间 span(V) ⊂ R^20（严格说是其中心化 19 维子空间里的一个 2 维面）。
因此一切比较都走 scipy.linalg.subspace_angles。

本模块的所有结论性函数都有自检（见 subspace_selftest.py，CLAUDE.md §10）。
"""
import csv
import json
import math
import os
import sys

import numpy as np
from scipy.linalg import subspace_angles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coupling_structure import double_center          # noqa: E402
from rank_test_power import als_guarded               # noqa: E402  （必须是带保护的版本）

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DMS = os.path.join(ROOT, "cache", "dms")
AA20 = list("ACDEFGHIKLMNPQRSTVWY")

MIN_PER_ROW = 4     # 保护：训练观测 < 4 的位点丢弃（rank2 需要 >= 3）
MIN_PER_COL = 6     # 保护：观测 < 6 的残基列丢弃（载荷会被少数位点主导）


# ----------------------------------------------------------------------
# 1. 子空间抽取
# ----------------------------------------------------------------------
def dc_fast(M, mask, iters=60):
    """coupling_structure.double_center 的向量化等价实现。

    为什么需要：原版是 Python 双层循环（60 x (nr+nc) 次），在 EM 抽取器
    里要被调用上万次，bootstrap 2000 次根本跑不完。
    **等价性由 selftest 的 [G] 用原版逐元素核对**（差 < 1e-9），
    不核对就不许用 —— 这正是 D-023 那一类静默替换事故的来源。
    """
    M = np.asarray(M, float)
    W = mask.astype(float)
    X = np.where(mask, M, 0.0)
    rn = W.sum(1); cn = W.sum(0)
    rn_s = np.where(rn > 0, rn, 1.0); cn_s = np.where(cn > 0, cn, 1.0)
    Sr = X.sum(1); Sc = X.sum(0)
    r = np.zeros(M.shape[0]); c = np.zeros(M.shape[1])
    for _ in range(iters):
        r += np.where(rn > 0, (Sr - r * rn - W @ c) / rn_s, 0.0)
        c += np.where(cn > 0, (Sc - W.T @ r - c * cn) / cn_s, 0.0)
    return M - r[:, None] - c[None, :]


def _center_rows_cols(M, mask):
    if mask.all():
        return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) \
            + M.mean()
    return dc_fast(np.where(mask, M, 0.0), mask)


def extract_subspace(M, mask, rank=2, method="emsvd", iters=40, seed=0,
                     return_full=False):
    """返回残基侧载荷的正交基 (n_res_kept x rank) + 保留的残基列索引。

    method="emsvd": 双向中心化 -> 缺失处用低秩重构迭代填充 -> SVD。
      这是 hard-impute / EM-PCA 的标准做法，比 ALS 快两个数量级，
      bootstrap 2000 次才可行。**必须**与 als_guarded 交叉核验
      （见 selftest 的 cross_check_extractors），因为快不等于对。
    method="als":   直接用 rank_test_power.als_guarded（稀疏矩阵用这个）。

    保护（对应 als_guarded 的欠定行列保护）：先丢弃观测过少的行/列，
    否则该行/列的因子极不稳定，会污染子空间方向。
    """
    M = np.asarray(M, float)
    mask = np.asarray(mask, bool)
    keep_r = mask.sum(axis=1) >= MIN_PER_ROW
    keep_c = mask.sum(axis=0) >= MIN_PER_COL
    if keep_r.sum() < rank + 2 or keep_c.sum() < rank + 2:
        return None, None, "行或列保留太少"
    cols = np.where(keep_c)[0]
    Ms = M[np.ix_(keep_r, cols)]
    ms = mask[np.ix_(keep_r, cols)]
    # 二次保护：丢行丢列后可能又有行/列不够
    for _ in range(3):
        kr = ms.sum(axis=1) >= MIN_PER_ROW
        kc = ms.sum(axis=0) >= MIN_PER_COL
        if kr.all() and kc.all():
            break
        Ms = Ms[np.ix_(kr, kc)]
        ms = ms[np.ix_(kr, kc)]
        cols = cols[kc]
    if Ms.shape[0] < rank + 2 or Ms.shape[1] < rank + 2:
        return None, None, "二次保护后太小"

    R = _center_rows_cols(Ms, ms)
    if method == "als":
        rng = np.random.default_rng(seed)
        Y = np.where(ms, R, 0.0)
        best, bestV = np.inf, None
        for _ in range(4):
            U = rng.normal(0, 0.1, (Ms.shape[0], rank))
            V = rng.normal(0, 0.1, (Ms.shape[1], rank))
            P = als_guarded(Y, ms, rank, rng, iters=250, ridge=0.0)
            tr = np.mean((R[ms] - P[ms]) ** 2)
            if tr < best:
                # 从拟合矩阵里取右奇异向量 —— 与因子分解的尺度不定性无关
                _, _, Vt = np.linalg.svd(P, full_matrices=False)
                best, bestV = tr, Vt[:rank].T
        B = bestV
    else:
        X = np.where(ms, R, 0.0)
        for _ in range(iters):
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            L = (U[:, :rank] * s[:rank]) @ Vt[:rank]
            X = np.where(ms, R, L)
            X = _center_rows_cols(X, np.ones_like(ms, bool))
        _, s, Vt = np.linalg.svd(X, full_matrices=False)
        B = Vt[:rank].T
        if return_full:
            return B, cols, {"sv": s[:rank + 2].tolist()}
    # 正交化（SVD 已给正交基；ALS 分支也经过 SVD）
    Q, _ = np.linalg.qr(B)
    return Q, cols, None


# ----------------------------------------------------------------------
# 2. 子空间比较：共同残基 + 再中心化 + 再正交化
# ----------------------------------------------------------------------
def _recenter_orth(B, rows):
    """取 B 的指定行 -> 在该子集内去掉全 1 方向 -> 正交化。

    为什么要再中心化：原矩阵的双向中心化只保证在**各体系自己的**残基集合上
    列和为零。换到共同残基子集后，全 1 分量重新出现；若不去掉，
    两个子空间会因为共享这个纯粹人造的方向而假性对齐。
    """
    A = np.asarray(B)[rows, :]
    d = A.shape[0]
    one = np.ones((d, 1)) / math.sqrt(d)
    A = A - one @ (one.T @ A)
    Q, R = np.linalg.qr(A)
    # 退化检查：某方向被中心化吃掉
    keep = np.abs(np.diag(R)) > 1e-8 * max(1.0, abs(R[0, 0]))
    if not keep.all():
        return Q[:, keep]
    return Q


def align(Ba, cols_a, Bb, cols_b):
    """返回 {'cos': [cos th1, cos th2], 'n_res': d, 'ssc': sum cos^2}。"""
    common = np.intersect1d(cols_a, cols_b)
    if len(common) < 8:
        return None
    ia = np.searchsorted(cols_a, common)
    ib = np.searchsorted(cols_b, common)
    Qa = _recenter_orth(Ba, ia)
    Qb = _recenter_orth(Bb, ib)
    if Qa.shape[1] < 2 or Qb.shape[1] < 2:
        return None
    th = subspace_angles(Qa, Qb)          # 降序角 => cos 升序
    cos = np.sort(np.cos(th))[::-1]       # cos th1 >= cos th2
    return {"cos": cos.tolist(), "n_res": int(len(common)),
            "ssc": float((cos ** 2).sum()),
            "chordal": float(math.sqrt(max(0.0, 2 - (cos ** 2).sum())))}


def align_deflated(Ba, cols_a, Bb, cols_b, h, cols_h):
    """把已知方向 h（疏水/体积，即成分1 的身份）从两个子空间里投掉之后，
    剩下的一维方向还对齐吗？这是「成分2 是否普适」的直接检验。

    做法：共同残基 -> 再中心化 -> 从各自 2 维子空间里取**与 h 最正交**的
    那一维（即 h 在子空间内投影的正交补），比较这两条一维方向的夹角。
    """
    common = np.intersect1d(np.intersect1d(cols_a, cols_b), cols_h)
    if len(common) < 8:
        return None
    Qa = _recenter_orth(Ba, np.searchsorted(cols_a, common))
    Qb = _recenter_orth(Bb, np.searchsorted(cols_b, common))
    if Qa.shape[1] < 2 or Qb.shape[1] < 2:
        return None
    hv = np.asarray(h)[np.searchsorted(cols_h, common)].astype(float)
    hv = hv - hv.mean()
    if np.linalg.norm(hv) < 1e-12:
        return None
    hv /= np.linalg.norm(hv)

    def residual_dir(Q):
        # h 在 span(Q) 内的坐标
        c = Q.T @ hv
        if np.linalg.norm(c) < 1e-12:
            return None, 0.0
        c = c / np.linalg.norm(c)
        # 2 维内与 c 正交的坐标
        cperp = np.array([-c[1], c[0]])
        return Q @ cperp, float(np.linalg.norm(Q.T @ hv))

    da, fa = residual_dir(Qa)
    db, fb = residual_dir(Qb)
    if da is None or db is None:
        return None
    cos = abs(float(da @ db))
    return {"cos_resid": cos, "n_res": int(len(common)),
            "h_in_a": fa, "h_in_b": fb}


# ----------------------------------------------------------------------
# 3. 零分布
# ----------------------------------------------------------------------
def haar_pair_null(d, rank=2, n=4000, seed=0):
    """同环境维数下两个独立 Haar 随机 rank 维子空间的主角度分布。
    环境是 d 个残基里的**中心化 d-1 维**子空间（与观测量一致）。
    """
    rng = np.random.default_rng(seed)
    one = np.ones((d, 1)) / math.sqrt(d)
    P = np.eye(d) - one @ one.T
    c1, c2, ssc = [], [], []
    for _ in range(n):
        A = P @ rng.normal(size=(d, rank))
        B = P @ rng.normal(size=(d, rank))
        Qa, _ = np.linalg.qr(A)
        Qb, _ = np.linalg.qr(B)
        cos = np.sort(np.cos(subspace_angles(Qa, Qb)))[::-1]
        c1.append(cos[0]); c2.append(cos[1]); ssc.append((cos ** 2).sum())
    return {"cos1": np.array(c1), "cos2": np.array(c2), "ssc": np.array(ssc)}


def haar_1d_null(d, n=4000, seed=0):
    """一维方向的随机零分布（用于 deflated 比较）。"""
    rng = np.random.default_rng(seed)
    one = np.ones(d) / math.sqrt(d)
    out = []
    for _ in range(n):
        a = rng.normal(size=d); a -= one * (one @ a); a /= np.linalg.norm(a)
        b = rng.normal(size=d); b -= one * (one @ b); b /= np.linalg.norm(b)
        out.append(abs(float(a @ b)))
    return np.array(out)


def permute_within_sites(M, mask, rng):
    """位点内置换残基标签：摧毁 site x residue 耦合，保留矩阵尺寸、
    稀疏模式的行边缘和每行的数值集合。"""
    M2 = np.full_like(M, np.nan)
    mk2 = np.zeros_like(mask)
    nc = M.shape[1]
    for i in range(M.shape[0]):
        obs = np.where(mask[i])[0]
        if len(obs) == 0:
            continue
        tgt = rng.choice(nc, size=len(obs), replace=False)
        M2[i, tgt] = M[i, obs]
        mk2[i, tgt] = True
    return M2, mk2


# ----------------------------------------------------------------------
# 4. bootstrap（按位点重抽）
# ----------------------------------------------------------------------
def bootstrap_subspaces(M, mask, n=2000, rank=2, seed=0, method="emsvd"):
    """返回 n 个 bootstrap 子空间 [(B, cols), ...]，失败的记 None。"""
    rng = np.random.default_rng(seed)
    nr = M.shape[0]
    out = []
    for _ in range(n):
        idx = rng.integers(0, nr, nr)
        B, cols, _ = extract_subspace(M[idx], mask[idx], rank=rank,
                                      method=method, seed=int(rng.integers(1e9)))
        out.append(None if B is None else (B, cols))
    return out


def ci(x, lo=2.5, hi=97.5):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)])
    if len(x) < 10:
        return (float("nan"), float("nan"))
    return (float(np.percentile(x, lo)), float(np.percentile(x, hi)))


# ----------------------------------------------------------------------
# 5. 数据加载
# ----------------------------------------------------------------------
CENSOR_WINDOW = 0.25   # 原生单位（log10 Ka 或 kcal/mol）；见下方自检


def censor_fraction(raw):
    """检测下限堆积的比例。

    **第一版是错的，而且静默。** 原来只数「恰好等于最小值」的个数
    （沿用 ligand_specific_rank.py 的约定）。那个约定在 Starr RBD 数据上
    有效，因为非结合变体被**赋成**完全相同的下限值；但 PDCoV 的
    bind_hAPN / bind_pAPN 的堆积被噪声摊开，没有重复值 ——
    恰好等于最小值的只有 0.04%，所以旧检测器报 0.0% 截尾。
    而已建立事实是 hAPN 给秩 0、pAPN 不可判（docs/12 §2），即重度截尾。

    改为：min + 0.25（约一个测量 SD）窗口内的比例，并与精确并列取大。
    已知事实标定（见 analysis/subspace_censor_check.py）：
      gAPN 1.5%（秩 2）、hAPN 47%、pAPN 48%（秩 0 / 不可判）、
      干净的 sars2_variants 0.3%、Lehner 0.1-0.3%、SKEMPI 0.4%。
    """
    a = np.asarray(raw, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return 0.0
    lo = a.min()
    return float(max((a <= lo + 1e-6).mean(),
                     (a <= lo + CENSOR_WINDOW).mean()))


def _mat_from_cells(cells):
    sites = sorted({s for s, _ in cells})
    si = {s: i for i, s in enumerate(sites)}
    M = np.full((len(sites), 20), np.nan)
    for (s, a), v in cells.items():
        if a in AA20:
            M[si[s], AA20.index(a)] = v
    return M, np.isfinite(M), sites


def _starr_matrix(path, col, site_key="position", target=None):
    """绝对 bind 列 -> 矩阵。截尾按最小值处的堆积检测（与
    ligand_specific_rank.py 同一约定），剔除并报告比例。"""
    vals = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if target is not None and r.get("target") != target:
                continue
            v = r.get(col, "")
            if v in ("", "NA", "nan"):
                continue
            vals.append((r[site_key], r["mutant"], float(v)))
    if len(vals) < 200:
        return None
    arr = np.array([v for _, _, v in vals])
    lo = arr.min()
    cfrac = censor_fraction(arr)
    keep = {(s, a): v for s, a, v in vals if v > lo + 1e-6}
    M, mask, sites = _mat_from_cells(keep)
    return {"M": M, "mask": mask, "sites": sites,
            "censor_frac": cfrac, "n_raw": len(vals),
            # raw = **未剔除截尾格子**的原生数值全集。密度型检测器
            # （analysis/censor_density.py）必须看到下限堆积本身，而
            # M[mask] 已经把恰好等于最小值的格子丢掉了（见上面的 keep）。
            "raw": arr,
            "n_obs": int(mask.sum())}


def load_all_dms():
    """返回 [{name, dataset, backbone, ligand, M, mask, censor_frac, ...}]"""
    jobs = []
    jobs.append(("animalACE2", "starr_animalACE2_final_variant_scores.csv",
                 "position",
                 ["bind_deer", "bind_hamster", "bind_bat", "bind_cat",
                  "bind_human_monomer"], True))
    jobs.append(("sarbecovirus",
                 "starr_SARSr_RBD_final_variant_scores_lib40_41.csv", "site",
                 ["huACE2", "Ra787ACE2", "Ra9479ACE2", "RlanACE2",
                  "RalcACE2"], True))
    jobs.append(("PDCoV", "starr_PDCoV_RBD_final_variant_scores.csv",
                 "position", ["bind_gAPN", "bind_hAPN", "bind_pAPN"], True))
    # 这五个文件是**同一批库的累积发布**：Alpha/Beta/Delta/Eta 等骨架在多个
    # 文件里逐字重复（n_obs 完全相同）。统一打到 sars2_variants 一个数据集下，
    # 按骨架去重，否则会把同一个矩阵当成多个"独立体系"反复计入。
    for tag, fn in (("sars2_variants",
                     "starr2022_variants_final_variant_scores.csv"),
                    ("sars2_variants", "starr_omicron_final_variant_scores.csv"),
                    ("sars2_variants",
                     "taylor2023_xbb_bq_final_variant_scores.csv"),
                    ("sars2_variants", "starr_EG5BA286_final_variant_scores.csv"),
                    ("sars2_variants", "starr_KP3LP8_final_variant_scores.csv")):
        jobs.append((tag, fn, "position", ["bind"], True))

    out = []
    seen = set()
    for tag, fn, skey, cols, bysplit in jobs:
        path = os.path.join(DMS, fn)
        if not os.path.exists(path):
            continue
        tgts = sorted({r["target"] for r in csv.DictReader(open(path))}) \
            if bysplit else [None]
        for tgt in tgts:
            for col in cols:
                lig = col.replace("bind_", "").replace("bind", "ACE2") or "lig"
                key = (tag, tgt, lig)
                if key in seen:
                    continue
                d = _starr_matrix(path, col, skey, tgt)
                if d is None:
                    continue
                seen.add(key)
                d.update({"dataset": tag, "backbone": tgt or tag,
                          "ligand": lig,
                          "name": f"{tag}/{(tgt or '')[:16]}/{lig}"})
                out.append(d)

    # SARS-CoV-2 2020（相对 WT 的 delta，无绝对下限堆积）
    cells = {}
    with open(os.path.join(DMS, "starr2020_rbd_single_mut_effects.csv")) as f:
        for r in csv.DictReader(f):
            v = r.get("bind_avg", "")
            if v in ("", "NA"):
                continue
            cells[(r.get("site_SARS2") or r.get("site_RBD"),
                   r["mutant"])] = float(v)
    arr = np.array(list(cells.values()))
    M, mask, sites = _mat_from_cells(cells)
    out.append({"M": M, "mask": mask, "sites": sites,
                "censor_frac": censor_fraction(arr), "n_raw": len(arr),
                "raw": arr,
                "n_obs": int(mask.sum()), "dataset": "starr2020",
                "backbone": "SARS-CoV-2-WH1", "ligand": "huACE2",
                "name": "starr2020/WH1/huACE2"})
    return out


def load_independent():
    """与冠状病毒 RBD 完全无关的体系 —— 这些才给出真正独立的比较。

    Lehner ddPCA: binding_coefficient 列是结合 ΔΔG（kcal/mol），
      三个完全不同的蛋白 x 完全不同的配体（CRIPT 肽 / GAB2 肽 / IgG-Fc）。
      加载逻辑复用 within_system_rank.load_lehner 的解析约定（第三列 = 结合）。
    Tite-seq 4-4-20 scFv x 荧光素: 小分子配体、抗体骨架、与上面全部无关。
      从 CDR1H_AA/CDR3H_AA 10-mer 里挑**单点**突变体，值取 -log10(KD)。
    """
    import re
    out = []
    three_ok = set(AA20)
    for nm, fn, lig in (("PSD95-PDZ3", "PSD95-PDZ3_ddG_weights.txt", "CRIPT"),
                        ("GRB2-SH3", "GRB2-SH3_ddG_weights.txt", "GAB2"),
                        ("GB1", "GB1_ddG_weights.txt", "IgG-Fc")):
        p = os.path.join(DMS, "lehner_ddPCA", fn)
        if not os.path.exists(p):
            continue
        cells = {}
        with open(p) as f:
            f.readline()
            for line in f:
                q = line.split()
                if len(q) < 3 or q[2] in ("NA", "", "nan"):
                    continue
                m = re.match(r"^([A-Z])(\d+)([A-Z])$", q[0])
                if not m or m.group(3) not in three_ok:
                    continue
                cells[(m.group(2), m.group(3))] = float(q[2])
        if len(cells) < 150:
            continue
        M, mask, sites = _mat_from_cells(cells)
        arr = M[mask]
        out.append({"M": M, "mask": mask, "sites": sites,
                    "censor_frac": censor_fraction(arr),
                    "n_raw": int(mask.sum()), "n_obs": int(mask.sum()),
                    "dataset": "lehner_ddPCA", "backbone": nm, "ligand": lig,
                    "name": f"lehner/{nm}/{lig}"})

    # Tite-seq 4-4-20 scFv x 荧光素
    WT1, WT3 = "TFSDYWMNWV", "GSYYGMDYWG"
    acc = {}
    nrep = 0
    for rep in (1, 2, 3):
        p = os.path.join(DMS, f"titeseq2016_replicate_{rep}.csv")
        if not os.path.exists(p):
            continue
        nrep += 1
        for r in csv.DictReader(open(p)):
            a1, a3 = r.get("CDR1H_AA", ""), r.get("CDR3H_AA", "")
            kd = r.get("fit_KD", "")
            if len(a1) != 10 or len(a3) != 10 or kd in ("", "NA", "nan"):
                continue
            try:
                kdv = float(kd)
            except ValueError:
                continue
            if not (1e-11 < kdv < 1e-5):      # 两端都是拟合边界（截尾）
                continue
            diff = [(i, c) for i, (c, w) in enumerate(zip(a1, WT1)) if c != w] \
                + [(10 + i, c) for i, (c, w) in enumerate(zip(a3, WT3))
                   if c != w]
            if len(diff) != 1:
                continue
            pos, aa = diff[0]
            if aa not in three_ok:
                continue
            acc.setdefault((f"CDR{'1' if pos < 10 else '3'}H_{pos % 10 + 1}",
                            aa), []).append(-math.log10(kdv))
    if nrep and len(acc) >= 80:
        cells = {k: float(np.mean(v)) for k, v in acc.items()}
        M, mask, sites = _mat_from_cells(cells)
        out.append({"M": M, "mask": mask, "sites": sites,
                    "censor_frac": censor_fraction(M[mask]),
                    "n_raw": int(mask.sum()),
                    "n_obs": int(mask.sum()), "dataset": "titeseq2016",
                    "backbone": "4-4-20 scFv", "ligand": "fluorescein",
                    "name": "titeseq/4-4-20/fluorescein",
                    "note": f"{nrep} 个重复取均值；拟合边界 KD 已剔除"})
    return out


def load_skempi(min_sub=4):
    """cache/skempi_thr4.csv -> 每个 system 一个矩阵（完全不同的蛋白）。"""
    rows = list(csv.DictReader(open(os.path.join(ROOT, "cache",
                                                 "skempi_thr4.csv"))))
    bysys = {}
    for r in rows:
        try:
            v = float(r["ddG_kcal"])
        except (ValueError, KeyError):
            continue
        key = r["system"]
        site = f"{r['pdb']}_{r['chain']}{r['position']}"
        bysys.setdefault(key, {})[(site, r["mut_res"])] = v
    # OMTKY3 额外按**蛋白酶**拆开：同一个抑制剂骨架、4 个完全不同的酶
    # （1CHO 胰凝乳蛋白酶 / 1PPF 人白细胞弹性蛋白酶 / 1R0R 枯草杆菌蛋白酶
    #  Carlsberg / 3SGB Streptomyces griseus 蛋白酶 B），各 10 位点 x 20 残基
    # 完整测全（Laskowski 加和性数据集）。这是一个**非病毒**体系上的
    # 「同骨架 x 多配体」对照，与 PDCoV/animalACE2 的设计同构。
    PROT = {"1CHO": "chymotrypsin", "1PPF": "HLE-elastase",
            "1R0R": "subtilisin-Carlsberg", "3SGB": "SGPB"}
    omt = {}
    for r in rows:
        if r["system"] != "Turkey ovomucoid third domain":
            continue
        try:
            v = float(r["ddG_kcal"])
        except (ValueError, KeyError):
            continue
        omt.setdefault(r["pdb"], {})[
            (f"{r['chain']}{r['position']}", r["mut_res"])] = v

    out = []
    for pdb, cells in sorted(omt.items()):
        M, mask, sites = _mat_from_cells(cells)
        if mask.sum() < 120 or len(sites) < 8:
            continue
        out.append({"M": M, "mask": mask, "sites": sites,
                    "censor_frac": censor_fraction(M[mask]),
                    "n_raw": int(mask.sum()), "n_obs": int(mask.sum()),
                    "dataset": "SKEMPI_OMTKY3", "backbone": "OMTKY3",
                    "ligand": PROT.get(pdb, pdb),
                    "name": f"OMTKY3/{PROT.get(pdb, pdb)}/{pdb}"})

    for sysname, cells in sorted(bysys.items()):
        M, mask, sites = _mat_from_cells(cells)
        if mask.sum() < 40 or len(sites) < 6:
            continue
        arr = M[mask]
        out.append({"M": M, "mask": mask, "sites": sites,
                    "censor_frac": censor_fraction(arr),
                    "n_raw": int(mask.sum()), "n_obs": int(mask.sum()),
                    "dataset": ("SKEMPI_OMTKY3"
                                if sysname == "Turkey ovomucoid third domain"
                                else "SKEMPI"),
                    # 合并版包含上面 4 个单酶矩阵的全部数据，所以 backbone
                    # 必须与它们相同，否则配对会被误判成「独立」而重复计入。
                    "backbone": ("OMTKY3"
                                 if sysname == "Turkey ovomucoid third domain"
                                 else sysname),
                    "ligand": ("4-proteases-pooled"
                               if sysname == "Turkey ovomucoid third domain"
                               else sysname),
                    "name": ("SKEMPI/OMTKY3/pooled-4proteases"
                             if sysname == "Turkey ovomucoid third domain"
                             else f"SKEMPI/{sysname}/{sysname}")})
    return out


# ----------------------------------------------------------------------
# 6. 残基性质标度（含强制自检）
# ----------------------------------------------------------------------
def load_scales(verbose=True):
    """返回 {scale_name: np.array(20)}（AA20 顺序，缺失为 nan）。
    aaindex 解析走 scale_redundancy.parse_aaindex + sanity_check
    —— 这个文件烧过本项目两次（残基顺序交错、缺失标记读成 0）。
    """
    from scale_redundancy import parse_aaindex, sanity_check
    want = {"MCMT640101": "molar_refractivity",
            "WOLR810101": "hydration_dG_Wolfenden",
            "FAUJ830101": "pi_octanol_FauchereP",
            "FAUJ880113": "pKa_RCOOH",
            "COHE430101": "partial_specific_volume",
            "CHAM810101": "steric_upsilon",
            "FAUJ880109": "n_hbond_donor",
            "FAUJ880110": "n_hbond_acceptor",
            "FAUJ880111": "pos_charge",
            "FAUJ880112": "neg_charge",
            "KYTJ820101": "kd_hydropathy",
            "ZIMJ680104": "isoelectric_point",
            "RADA880106": "asa_RadzickaWolfenden",   # 原误标为 Rose；RADA880106 是 Radzicka-Wolfenden 1988，Rose 的是 ROSG8501xx（docs/36）
            "CHOP780201": "helix_prop_Chou",
            "CHOP780202": "sheet_prop_Chou"}
    path = os.path.join(ROOT, "cache", "aaindex1.txt")
    data = parse_aaindex(path, set(want))
    if verbose:
        print("[自检] aaindex 解析（已知事实核验）")
    if not sanity_check(data):
        raise RuntimeError("aaindex sanity_check 未通过 —— 后续数字不可用")
    # 独立二次核验（不依赖上面那个函数的实现）
    mr = data["MCMT640101"]
    assert min(mr, key=lambda k: mr[k]) == "G", "折射度最小不是 Gly"
    assert max(mr, key=lambda k: mr[k]) == "W", "折射度最大不是 Trp"

    scales = {}
    for code, nm in want.items():
        if code not in data:
            continue
        scales[nm] = np.array([data[code].get(a, np.nan) for a in AA20],
                              float)

    three = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
             "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
             "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
             "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

    def from3(dd):
        v = np.full(20, np.nan)
        for k, x in dd.items():
            a = three.get(k.upper())
            if a and x is not None and np.isfinite(float(x)):
                v[AA20.index(a)] = float(x)
        return v

    # 本地测得的描述子
    pol = {}
    with open(os.path.join(ROOT, "cache", "sidechain_alpha.tsv")) as f:
        for line in f:
            if line.startswith("#") or line.startswith("residue"):
                continue
            p = line.split("\t")
            if len(p) >= 3:
                pol[p[0]] = float(p[2])
    scales["local_polarizability"] = from3(pol)

    dip = {}
    with open(os.path.join(ROOT, "cache", "sidechain_dipole.tsv")) as f:
        for line in f:
            if line.startswith("#") or line.startswith("residue"):
                continue
            p = line.split("\t")
            if len(p) >= 3 and p[2].strip() not in ("NA", ""):
                dip[p[0]] = float(p[2])
    scales["local_dipole"] = from3(dip)

    arm = json.load(open(os.path.join(ROOT, "cache", "sidechain_arm.json")))
    scales["local_arm"] = from3(arm.get("arm", arm))
    reach = json.load(open(os.path.join(ROOT, "cache",
                                        "sidechain_reach.json")))
    scales["local_hbond_reach"] = from3(reach)

    # 本项目已验证过的两个身份描述子
    scales["planar_sp2"] = np.array(
        [1.0 if a in "FWYHDENQR" else 0.0 for a in AA20])
    scales["beta_branch"] = np.array(
        [1.0 if a in "IVT" else 0.0 for a in AA20])
    if verbose:
        print(f"   载入 {len(scales)} 条标度；"
              f"折射度 Gly 最小/Trp 最大 二次核验通过")
    return scales


# ----------------------------------------------------------------------
# 7. 去重与「主代表」选择
# ----------------------------------------------------------------------
def _content_key(d):
    v = np.round(d["M"][d["mask"]], 6)
    return (d["M"].shape, int(d["mask"].sum()),
            hash(v.tobytes()), float(v.sum()))


def dedup_by_content(mats):
    """按矩阵内容去重。

    为什么需要：Starr 的多个发布文件里同一个库被逐字重复。
    实测 animalACE2/Omicron_BA2/human_monomer 与
    sars2_variants/Omicron_BA2/ACE2 是**同一个矩阵**，配对给 cos = (1.000,
    1.000)。若不去重，这种自配对会被当成「跨数据集复现」。
    """
    seen, out, drops = {}, [], []
    for d in mats:
        k = _content_key(d)
        if k in seen:
            drops.append((d["name"], seen[k]))
            continue
        seen[k] = d["name"]
        out.append(d)
    return out, drops


def pick_primary(mats):
    """每个 backbone 只留一个代表（观测最多者）用于共识与「独立配对」统计。
    其余保留，但只参与「同骨架 x 不同配体」那一问。"""
    best = {}
    for d in mats:
        k = (d["dataset"], d["backbone"])
        if k not in best or d["n_obs"] > best[k]["n_obs"]:
            best[k] = d
    prim = {id(v) for v in best.values()}
    for d in mats:
        d["primary"] = id(d) in prim
    return mats
