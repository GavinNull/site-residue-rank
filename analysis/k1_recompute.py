#!/usr/bin/env python3
"""docs/44 Step 3：在 K1（去偏定秩）下重算每一个头条量，并与现行程序（同一协议重算）并列。

预先登记：`cache/docs44_prereg.md`，sha256 锁在 `cache/docs44_prereg_lock.json`，入口断言一致（CLAUDE.md §4）。
先决（任一不过即中止，CLAUDE.md §10）：
  P0 预先登记锁；P1 rank_core.selfcheck + rank_k1.selfcheck + independence_stats.selfcheck；
  P2 Gen-3 截尾检测器已知事实自检；
  P3 重建核对：34 单元的 n/位点/cens_count 与 cache/censor_gen3_final.json 逐单元相等；Gen-3 g3 重算与缓存相等；
     现行程序臂的 3 种子逐种子 argmin 与缓存的 rank_noleak_seeds 逐单元相等；
  P4 老臂复现 docs/41：S1 面板的位点/联合区间与「5 上 / 9 跨 / 1 下」逐位复现 cache/independence_audit.json，否则中止，K1 臂不得使用。

产物：cache/k1_recompute.json、cache/k1_units34.json（Figure 1 的数据）、cache/k1_cells.pkl（逐格误差，可删，--reuse 才用）。
用法：python3 analysis/k1_recompute.py [--jobs 8] [--reuse] [--skip-recovery]
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import collections
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
DMS = os.path.join(ROOT, "cache", "dms")

import rank_core as RC                       # noqa: E402
import rank_k1 as K1                         # noqa: E402
import independence_stats as IS              # noqa: E402
import independence_audit as IA              # noqa: E402
import censor_density as CD                  # noqa: E402
import censor_recheck as CR                  # noqa: E402
import subspace_lib as L                     # noqa: E402

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
CVK = {k: v for k, v in K1.CV_HEADLINE.items() if k != "K"}
SEEDS = K1.SEEDS_HEADLINE
MAXR = K1.MAXR_HEADLINE
NB = 2000
NB_CORR = 4000
PAIRS = ((0, 1), (1, 2), (2, 3), (0, 2), (0, 3))
ARMS = ("old", "k1")
PROC = {"old": "noleak", "k1": "k1"}
CELL_FN = {"old": K1.cv_cell_errors_noleak, "k1": K1.cv_cell_errors_k1}
OUT_JSON = os.path.join(ROOT, "cache", "k1_recompute.json")
OUT_U34 = os.path.join(ROOT, "cache", "k1_units34.json")
CELLS_PKL = os.path.join(ROOT, "cache", "k1_cells.pkl")
LOG = []


def P(*x):
    s = " ".join(str(v) for v in x)
    print(s, flush=True)
    LOG.append(s)


# ============================================================ P0
def verify_lock():
    lk = json.load(open(os.path.join(ROOT, "cache", "docs44_prereg_lock.json")))
    h = hashlib.sha256(open(os.path.join(ROOT, lk["file"]), "rb").read()).hexdigest()
    assert h == lk["sha256"], f"docs/44 预先登记与锁不一致 —— 中止\n  锁定 {lk['sha256']}\n  现在 {h}"
    return {"prereg_sha256": h, "locked_at": lk["locked_at"]}


# ============================================================ 加载
def build_units34():
    """逐字照抄 censor_gen3_final.build_units 的筛选与建矩阵（不算秩、不算 g3 区间）。"""
    units = []
    for tag, fn, skey, cols in CR.JOBS:
        rows = list(csv.DictReader(open(os.path.join(DMS, fn))))
        bgs = sorted({r["target"] for r in rows})
        for bg in bgs:
            for col in cols:
                vals = []
                for r in rows:
                    if r.get("target") != bg:
                        continue
                    v = r.get(col, "")
                    if v in ("", "NA"):
                        continue
                    vals.append((r[skey], r["mutant"], float(v)))
                if not vals:
                    continue
                arr = np.array([v for _, _, v in vals])
                lo = arr.min()
                cens = int((arr <= lo + 1e-6).sum())
                keep = [(s, a, v) for s, a, v in vals if v > lo + 1e-6]
                if len(keep) < 500:
                    continue
                sites = sorted({s for s, _, _ in keep})
                si = {s: i for i, s in enumerate(sites)}
                M = np.full((len(sites), 20), np.nan)
                for s, a, v in keep:
                    if a in AA20:
                        M[si[s], AA20.index(a)] = v
                mask = np.isfinite(M)
                if mask.sum() < 500 or len(sites) < 30:
                    continue
                n = int(mask.sum())
                name = (f"animalACE2/{bg}/{col.replace('bind_', '')}" if tag == "animalACE2"
                        else f"sarbecovirus/{bg}/{col}")
                units.append({"dataset": tag, "bg": bg, "lig": col, "name": name,
                              "M": np.where(mask, M, 0.0), "mask": mask, "arr": arr,
                              "sites": len(sites), "site_labels": sites, "n": n,
                              "cens_count": cens, "n_raw": int(len(vals)),
                              "g1": cens / (cens + n)})
    return units


def load_scope():
    """范围内矩阵（B/C/D），与 independence_audit.load_all 同一来源，但不加载 CR9114。"""
    mats = {}
    for d in L.load_all_dms():
        if d["dataset"] in ("animalACE2", "PDCoV"):
            mats[d["name"]] = d
    for d in L.load_skempi():
        if d["dataset"] == "SKEMPI_OMTKY3" and d["ligand"] == "4-proteases-pooled":
            mats[d["name"]] = d
    for d in L.load_independent():
        if d["dataset"] in ("titeseq2016", "lehner_ddPCA"):
            mats[d["name"]] = d
    return mats


SYSTEM_OF = {"animalACE2": "S1", "PDCoV": "S2", "SKEMPI_OMTKY3": "S3", "titeseq2016": "S4"}
DDPCA = {"lehner/PSD95-PDZ3/CRIPT": "S6", "lehner/GRB2-SH3/GAB2": "S7", "lehner/GB1/IgG-Fc": "S8"}
S2_NAMES = ["PDCoV/PDCoV/gAPN", "PDCoV/PDCoV/hAPN", "PDCoV/PDCoV/pAPN"]
S3_NAME = "SKEMPI/OMTKY3/pooled-4proteases"
S4_NAME = "titeseq/4-4-20/fluorescein"
S1_BB = ["Eta", "Omicron_BA2", "Wuhan-Hu-1"]
S1_LG = ["deer", "hamster", "bat", "cat", "human_monomer"]
S1_ORDER = [f"animalACE2/{b}/{l}" for b in S1_BB for l in S1_LG]


# ============================================================ 并行任务
MATS = {}                # name -> (M, mask)


def cells_job(key):
    name, arm = key
    M, mask = MATS[name]
    per_seed = {}
    for sd in SEEDS:
        cells, E = CELL_FN[arm](M, mask, max_rank=MAXR, seed=sd, **CVK)
        per_seed[sd] = E
    return key, cells, per_seed


def recovery_job(name):
    """K1 的恢复性诊断：independence_audit.recovery 同法（_plant、种子 1000+17t+k、CV 种子 3），检出换成 K1，max_rank=3。"""
    M, mask = MATS[name]
    out = {}
    for k in (2, 1, 0):
        bests, g12 = [], []
        sig = None
        for t in range(8):
            Y, sig = IA._plant(M, mask, k, 1000 + 17 * t + k)
            e = K1.cv_rank_k1(Y, mask, max_rank=MAXR, seed=3, **CVK)
            br, trim = RC.best_rank(e)
            bests.append(int(br))
            g12.append((trim[1] - trim[2]) / trim[1])
        out[f"plant{k}"] = {"best": bests, "sigma": sig,
                            "p_detect2": float(np.mean([b == 2 for b in bests])),
                            "p_detect0": float(np.mean([b == 0 for b in bests])),
                            "p_ge2": float(np.mean([b >= 2 for b in bests])),
                            "p_r2_beats_r1": float(np.mean([g > 0 for g in g12]))}
    out["powered"] = bool(out["plant2"]["p_detect2"] >= 0.75 and out["plant0"]["p_detect0"] >= 0.75)
    return name, out


# ============================================================ 统计
def curve_stats(E_seed):
    """E_seed: {seed: {rank: err}} -> 每种子 trim95 曲线、均值曲线、三种口径的检出秩。"""
    curves = [{r: K1.trim95(E_seed[sd][r]) for r in range(MAXR + 1)} for sd in SEEDS]
    mc = {r: float(np.mean([c[r] for c in curves])) for r in range(MAXR + 1)}
    bests = [K1._argmin(c) for c in curves]
    vals, cnt = np.unique(bests, return_counts=True)
    Ebar = {r: np.mean([E_seed[sd][r] for sd in SEEDS], axis=0) for r in range(MAXR + 1)}
    cellc = {r: K1.trim95(Ebar[r]) for r in Ebar}
    return {"curves": curves, "mean_curve": mc, "rank_headline": K1._argmin(mc),
            "seed_bests": bests, "rank_mode": int(vals[np.argmax(cnt)]), "stab": float(cnt.max() / len(bests)),
            "cell_curve": cellc, "rank_cell": K1._argmin(cellc), "Ebar": Ebar}


def gain(t, a, b):
    return (t[a] - t[b]) / t[a]


def panel_arm(units15, names15, label):
    """面板 15 个子集：位点联合重抽 + 骨架/配体/位点同时重抽 + 每子集自己的区间。"""
    pts = {}
    for (a, b) in PAIRS:
        pts[(a, b)] = [gain({r: K1.trim95(u["E"][r]) for r in range(MAXR + 1)}, a, b) for u in units15]
    pts_star = [max(pts[(0, 1)][i], pts[(0, 2)][i], pts[(0, 3)][i]) for i in range(len(units15))]
    sb = IA.site_boot_gains(units15, NB, seed=41, pairs=PAIRS)
    star_b = np.maximum.reduce([sb[(0, 1)], sb[(0, 2)], sb[(0, 3)]])
    res = {"point_median": {}, "site_ci": {}, "combined_ci": {}, "combined_frac_le0": {}, "pos_matrices": {}}
    keymap = {"g01": (0, 1), "g12": (1, 2), "g23": (2, 3), "g02": (0, 2), "g03": (0, 3)}
    for k, pr in keymap.items():
        res["point_median"][k] = float(np.median(pts[pr]))
        res["site_ci"][k] = IS.pct_ci(np.median(sb[pr], axis=1))
        res["pos_matrices"][k] = int(sum(p > 0 for p in pts[pr]))
    res["point_median"]["g0star"] = float(np.median(pts_star))
    res["site_ci"]["g0star"] = IS.pct_ci(np.median(star_b, axis=1))
    # 同时重抽骨架、配体、位点：顺序 g12,g23,g02,g01,g03,g0star（docs/41 的前三个顺序不变 => 老臂可逐位复现）
    rngc = np.random.default_rng(99)
    for k in ("g12", "g23", "g02", "g01", "g03", "g0star"):
        gb = star_b if k == "g0star" else sb[keymap[k]]
        med = np.empty(NB)
        for t in range(NB):
            ri = rngc.integers(0, 3, 3)
            ci = rngc.integers(0, 5, 5)
            med[t] = np.median(gb[t].reshape(3, 5)[np.ix_(ri, ci)])
        res["combined_ci"][k] = IS.pct_ci(med)
        res["combined_frac_le0"][k] = float((med <= 0).mean())
    # 每个子集自己的区间（单矩阵位点重抽，docs/41 T1 同法）
    per = {}
    for u, nm in zip(units15, names15):
        s = 1500 + zlib.crc32(nm.encode()) % 1000
        sbx = IA.site_boot_gains([u], NB, seed=s, pairs=PAIRS)
        row = {}
        for k, pr in keymap.items():
            lo, hi = IS.pct_ci(sbx[pr][:, 0])
            row[k] = [gain({r: K1.trim95(u["E"][r]) for r in range(MAXR + 1)}, *pr), lo, hi]
        per[nm] = row
    res["per_subset"] = per
    cnt = {}
    for k in keymap:
        above = sum(per[n][k][1] > 0 for n in names15)
        below = sum(per[n][k][2] < 0 for n in names15)
        cnt[k] = {"above0": int(above), "span0": int(len(names15) - above - below), "below0": int(below)}
    res["subset_interval_counts"] = cnt
    res["label"] = label
    return res


def pooled(units, names, seed):
    """体系层：矩阵间中位数在联合位点重抽下的区间（docs/41 `pooled` 同法）。"""
    sbx = IA.site_boot_gains(units, NB, seed=seed, pairs=PAIRS)
    keymap = {"g01": (0, 1), "g12": (1, 2), "g23": (2, 3), "g02": (0, 2), "g03": (0, 3)}
    o = {}
    for k, pr in keymap.items():
        pts = [gain({r: K1.trim95(u["E"][r]) for r in range(MAXR + 1)}, *pr) for u in units]
        o[k] = {"point": float(np.median(pts)), "ci": IS.pct_ci(np.median(sbx[pr], axis=1))}
    return o


def sys_status(o):
    c12, c01, c23 = o["g12"]["ci"], o["g01"]["ci"], o["g23"]["ci"]
    st12 = "SUPPORTS_2OVER1" if c12[0] > 0 else ("SUPPORTS_LE1" if c12[1] < 0 else "INDETERMINATE_1v2")
    return {"g12_status": st12, "interaction": bool(c01[0] > 0), "rank3_worse": bool(c23[1] < 0),
            "interior_min_supported": bool(c12[0] > 0 and c23[1] < 0)}


def strat_stats(units, key, rng_seed=20261017):
    r = np.array([u[key] for u in units], float)
    g = np.array([u["g3"] for u in units], float)
    n = len(units)
    out = {"n_units": n, "n_by_rank": {}, "median_by_rank": {}, "median_ci": {}}
    for k in (0, 1, 2, 3):
        s = g[r == k]
        out["n_by_rank"][k] = int(len(s))
        out["median_by_rank"][k] = float(np.median(s)) if len(s) else None
    rng = np.random.default_rng(rng_seed)
    boots = {k: [] for k in (0, 1, 2, 3)}
    d01 = []
    for _ in range(NB_CORR):
        i = rng.integers(0, n, n)
        rr, gg = r[i], g[i]
        for k in (0, 1, 2, 3):
            s = gg[rr == k]
            if len(s):
                boots[k].append(float(np.median(s)))
        a, b = gg[rr == 0], gg[rr >= 1]
        if len(a) and len(b):
            d01.append(float(np.median(a) - np.median(b)))
    for k in (0, 1, 2, 3):
        out["median_ci"][k] = ([float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))]
                               if len(boots[k]) > 100 else None)
    a, b = g[r == 0], g[r >= 1]
    out["rank0_vs_ge1"] = {"n0": int(len(a)), "n_ge1": int(len(b)),
                           "median0": float(np.median(a)) if len(a) else None,
                           "median_ge1": float(np.median(b)) if len(b) else None,
                           "diff": float(np.median(a) - np.median(b)) if len(a) and len(b) else None,
                           "diff_ci": ([float(np.percentile(d01, 2.5)), float(np.percentile(d01, 97.5))]
                                       if len(d01) > 100 else None)}
    out["spearman"] = CR.boot_corr(g, r, NB_CORR, np.random.default_rng(rng_seed + 1), "spearman")
    out["pearson"] = CR.boot_corr(g, r, NB_CORR, np.random.default_rng(rng_seed), "pearson")
    # 单调：n>=3 的组，按检出秩从高到低，g3 中位不减
    grp = [(k, out["median_by_rank"][k]) for k in (3, 2, 1, 0) if out["n_by_rank"][k] >= 3]
    meds = [m for _, m in grp]
    out["monotone_groups"] = [k for k, _ in grp]
    out["monotone_nondecreasing"] = bool(all(meds[i] <= meds[i + 1] + 1e-12 for i in range(len(meds) - 1)))
    lt10 = [u for u in units if u["g3"] < 0.10]
    out["lt10"] = {"n": len(lt10), "rank2": int(sum(u[key] == 2 for u in lt10)),
                   "rank_ge1": int(sum(u[key] >= 1 for u in lt10))}
    strata = [(0.0, 0.02), (0.02, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 1.01)]
    xt = {}
    for lo, hi in strata:
        sel = [u for u in units if lo <= u["g3"] < hi]
        xt[f"{lo:.2f}-{min(hi, 1):.2f}"] = {"n": len(sel),
                                           "ranks": dict(collections.Counter(int(u[key]) for u in sel))}
    out["crosstab"] = xt
    return out


def fig1_stats(units, arm_curves, seed=20261018):
    low = [i for i, u in enumerate(units) if u["g1"] < 0.10]
    high = [i for i, u in enumerate(units) if u["g1"] >= 0.10]
    out = {"n_low": len(low), "n_high": len(high)}
    for arm, curves in arm_curves.items():
        norm = np.array([[c[r] / c[0] for r in range(4)] for c in curves])
        o = {}
        for lab, idx in (("low", low), ("high", high)):
            V = norm[idx]
            m = np.median(V, axis=0)
            rng = np.random.default_rng(seed)
            d12, d32, d10 = [], [], []
            for _ in range(NB):
                mb = np.median(V[rng.integers(0, len(V), len(V))], axis=0)
                d12.append(mb[1] - mb[2])
                d32.append(mb[3] - mb[2])
                d10.append(mb[1] - mb[0])
            o[lab] = {"median_curve": [float(x) for x in m], "argmin": int(np.argmin(m)),
                      "m1_minus_m2": float(m[1] - m[2]), "m1_minus_m2_ci": IS.pct_ci(d12),
                      "m3_minus_m2": float(m[3] - m[2]), "m3_minus_m2_ci": IS.pct_ci(d32),
                      "m1_minus_m0": float(m[1] - m[0]), "m1_minus_m0_ci": IS.pct_ci(d10),
                      "unit_argmin_counts": dict(collections.Counter(int(np.argmin(v)) for v in V))}
        out[arm] = o
    return out


# ============================================================ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--skip-recovery", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    R = {"search_volume": {}}
    P("=" * 100)
    P("docs/44 K1 重算")
    P("=" * 100)
    R["prereg"] = verify_lock()
    P(f"P0 docs/44 预先登记锁一致 sha256 {R['prereg']['prereg_sha256'][:16]}…  OK")
    RC.selfcheck(verbose=False)
    P("P1a rank_core.selfcheck 通过  OK")
    K1.selfcheck(verbose=False)
    P("P1b rank_k1.selfcheck 通过（命中/不误伤/回归/泄露/原型一致）  OK")
    IS.selfcheck(verbose=False)
    P("P1c independence_stats.selfcheck 通过  OK")
    P(IA.censor_selfcheck())

    # ---- 加载与重建核对
    units = build_units34()
    cache34 = json.load(open(os.path.join(ROOT, "cache", "censor_gen3_final.json")))["units"]
    assert len(units) == 34 == len(cache34), (len(units), len(cache34))
    for u, c in zip(units, cache34):
        assert (u["dataset"], u["bg"], u["lig"]) == (c["dataset"], c["bg"], c["lig"]), "单元顺序不一致"
        assert u["n"] == c["n"] and u["sites"] == c["sites"] and u["cens_count"] == c["cens_count"], \
            f"重建不符 {u['name']}"
        q = float(CD.censor_fraction_density(u["arr"]))          # Gen-3，喂原始值（含堆积）
        assert abs(q - c["g3"]) < 1e-9, f"{u['name']} Gen-3 重算 {q} ≠ 缓存 {c['g3']}"
        u["g3"] = q
        u["cache_rank_noleak_seeds"] = c["rank_noleak_seeds"]
        u["cache_rank_noleak"] = c["rank_noleak"]
        u["cache_rank_orig"] = c.get("rank_orig")
    P("P3a 重建核对：34 单元 n/位点/cens_count 与缓存逐单元相等；Gen-3 g3 重算 == 缓存（<1e-9）  OK")

    scope = load_scope()
    d41 = {m["name"]: m for m in json.load(open(os.path.join(ROOT, "cache", "independence_audit.json")))["matrices"]}
    need = S1_ORDER + S2_NAMES + [S3_NAME, S4_NAME] + list(DDPCA)
    for nm in need:
        assert nm in scope, f"缺矩阵 {nm}"
        d = scope[nm]
        assert int(d["mask"].sum()) == d41[nm]["n_obs"], f"{nm} 观测数与 docs/41 不符"
        assert int(d["mask"].any(1).sum()) == d41[nm]["sites"], f"{nm} 位点数与 docs/41 不符"
    # S1 面板矩阵 == 34 单元里的同名单元（逐格）
    byname = {u["name"]: u for u in units}
    for nm in S1_ORDER:
        u, d = byname[nm], scope[nm]
        assert u["M"].shape == d["M"].shape and np.array_equal(u["mask"], d["mask"]) and \
            np.allclose(np.where(u["mask"], u["M"], 0), np.where(d["mask"], d["M"], 0), atol=1e-12), nm
    P("P3b 范围内矩阵与 docs/41 的观测数/位点数相等；S1 面板矩阵与 34 单元里的同名单元逐格相等  OK")

    # ---- 矩阵表
    for u in units:
        MATS[u["name"]] = (u["M"], u["mask"])
    for nm in need:
        if nm not in MATS:
            d = scope[nm]
            MATS[nm] = (np.where(d["mask"], d["M"], 0.0), d["mask"])
    sites_of = {nm: list(scope[nm]["sites"]) for nm in need}

    # ---- 逐格误差（两臂 × 3 种子 × 秩 0–3）
    keys = [(nm, arm) for nm in MATS for arm in ARMS]
    keys.sort(key=lambda k: -int(MATS[k[0]][1].sum()))
    res = {}
    sig = hashlib.sha256(repr([(k, float(MATS[k[0]][0].sum()), int(MATS[k[0]][1].sum())) for k in keys]
                              + [CVK, SEEDS, MAXR]).encode()).hexdigest()
    if a.reuse and os.path.exists(CELLS_PKL):
        blob = pickle.load(open(CELLS_PKL, "rb"))
        if blob["sig"] == sig:
            res = blob["res"]
            P("  复用 cache/k1_cells.pkl（输入签名一致）")
    if not res:
        t0 = time.time()
        with mp.get_context("fork").Pool(a.jobs) as pool:
            for k, cells, per_seed in pool.imap_unordered(cells_job, keys, chunksize=1):
                res[k] = (cells, per_seed)
        pickle.dump({"sig": sig, "res": res}, open(CELLS_PKL, "wb"))
        P(f"逐格误差 {len(keys)} 个 (矩阵×程序) 任务，{time.time()-t0:.0f}s")
    R["search_volume"]["matrices_ranked"] = len(MATS)
    R["search_volume"]["cv_runs_ranking"] = len(keys) * len(SEEDS) * (MAXR + 1)

    CS = {k: curve_stats(res[k][1]) for k in res}
    cellsof = {k: res[k][0] for k in res}

    # ---- P3c 老臂逐种子 argmin == 缓存
    for u in units:
        old = CS[(u["name"], "old")]
        assert old["seed_bests"] == u["cache_rank_noleak_seeds"], \
            (u["name"], old["seed_bests"], u["cache_rank_noleak_seeds"])
        assert old["rank_mode"] == u["cache_rank_noleak"]
    P("P3c 现行程序臂：34 单元的 3 种子逐种子 argmin 与 cache/censor_gen3_final.json 的 rank_noleak_seeds 逐单元相等  OK")

    # ---- P4 老臂复现 docs/41 的面板区间
    def unit_of(nm, arm):
        return {"sites": sites_of[nm], "rows": cellsof[(nm, arm)][:, 0], "E": CS[(nm, arm)]["Ebar"]}

    panel = {}
    for arm in ARMS:
        us = [unit_of(nm, arm) for nm in S1_ORDER]
        panel[arm] = panel_arm(us, S1_ORDER, arm)
    ref = json.load(open(os.path.join(ROOT, "cache", "independence_audit.json")))["panel"]
    po = panel["old"]
    chk = [("site g12", po["site_ci"]["g12"], ref["U4_site"]["median_g12_ci"]),
           ("site g23", po["site_ci"]["g23"], ref["U4_site"]["median_g23_ci"]),
           ("site g02", po["site_ci"]["g02"], ref["U4_site"]["median_g02_ci"]),
           ("combined g12", po["combined_ci"]["g12"], ref["U3U4_combined"]["g12"]["ci"]),
           ("combined g23", po["combined_ci"]["g23"], ref["U3U4_combined"]["g23"]["ci"]),
           ("combined g02", po["combined_ci"]["g02"], ref["U3U4_combined"]["g02"]["ci"])]
    for nm, x, y in chk:
        assert max(abs(p - q) for p, q in zip(x, y)) < 1e-9, f"老臂未复现 docs/41 {nm}: {x} vs {y}"
    cnt12 = po["subset_interval_counts"]["g12"]
    assert (cnt12["above0"], cnt12["span0"], cnt12["below0"]) == (5, 9, 1), cnt12
    d41_t1 = {m["name"]: m for m in json.load(open(os.path.join(ROOT, "cache", "independence_audit.json")))["matrices"]}
    for nm in S1_ORDER:
        c = d41_t1[nm]["ci"]["g12"]["ci"]
        assert max(abs(p - q) for p, q in zip(po["per_subset"][nm]["g12"][1:], c)) < 1e-9, nm
    P("P4 老臂复现 docs/41：S1 面板 g12/g23/g02 的位点区间与联合区间、逐子集区间、「5 上/9 跨/1 下」逐位复现  OK")
    R["reproduction_old_arm_docs41"] = {"checks": [c[0] for c in chk], "subset_g12_counts": cnt12}
    R["panel"] = panel

    # ---- 34 单元：检出秩
    rows34 = []
    for u in units:
        o, k = CS[(u["name"], "old")], CS[(u["name"], "k1")]
        row = {"name": u["name"], "dataset": u["dataset"], "bg": u["bg"], "lig": u["lig"], "n": u["n"],
               "sites": u["sites"], "cens_count": u["cens_count"], "g1": u["g1"], "g3": u["g3"],
               "rank_old": o["rank_headline"], "rank_k1": k["rank_headline"],
               "rank_old_mode": o["rank_mode"], "rank_k1_mode": k["rank_mode"],
               "rank_old_cell": o["rank_cell"], "rank_k1_cell": k["rank_cell"],
               "seed_bests_old": o["seed_bests"], "seed_bests_k1": k["seed_bests"],
               "stab_k1": k["stab"], "mean_curve_old": o["mean_curve"], "mean_curve_k1": k["mean_curve"],
               "rank_orig_ligand_rank": u["cache_rank_orig"]}
        rows34.append(row)
        u["rank_old"], u["rank_k1"] = row["rank_old"], row["rank_k1"]
        u["rank_old_mode"], u["rank_k1_mode"] = row["rank_old_mode"], row["rank_k1_mode"]
        u["rank_old_cell"], u["rank_k1_cell"] = row["rank_old_cell"], row["rank_k1_cell"]
    R["units34"] = rows34
    strat = {}
    for key in ("rank_old", "rank_k1", "rank_old_mode", "rank_k1_mode", "rank_old_cell", "rank_k1_cell"):
        strat[key] = strat_stats(units, key)
    R["strat"] = strat
    shuffle = collections.Counter((r["rank_old"], r["rank_k1"]) for r in rows34)
    R["reshuffle_old_to_k1"] = {f"{a_}->{b_}": int(c) for (a_, b_), c in sorted(shuffle.items())}
    # 老臂的头条值：与发表的 43.2/10.9/0.1 对照（用原秩 rank_orig 与 g1 复现）
    orig = [dict(g3=u["g1"], r=u["cache_rank_orig"]) for u in units]
    R["published_stratification_check"] = {
        str(k): float(np.median([o["g3"] for o in orig if o["r"] == k])) for k in (0, 1, 2)}

    # ---- Figure 1 数据
    lr = {(c["dataset"], c["bg"], c["lig"]): c for c in json.load(open(os.path.join(ROOT, "cache", "ligand_rank.json")))
          if "skip" not in c}
    as_drawn = [{int(k): v for k, v in lr[(u["dataset"], u["bg"], u["lig"])]["mean_curve"].items()} for u in units]
    arm_curves = {"k1": [CS[(u["name"], "k1")]["mean_curve"] for u in units],
                  "old_noleak": [CS[(u["name"], "old")]["mean_curve"] for u in units],
                  "old_as_drawn": as_drawn}
    R["fig1"] = fig1_stats(units, arm_curves)
    json.dump([{"dataset": u["dataset"], "bg": u["bg"], "lig": u["lig"], "name": u["name"], "n": u["n"],
                "cens_count": u["cens_count"], "g1": u["g1"], "g3": u["g3"],
                "mean_curve_k1": {str(k): v for k, v in CS[(u["name"], "k1")]["mean_curve"].items()},
                "mean_curve_old_noleak": {str(k): v for k, v in CS[(u["name"], "old")]["mean_curve"].items()},
                "rank_k1": u["rank_k1"], "rank_old": u["rank_old"],
                "provenance": "analysis/k1_recompute.py; rank_k1.cv_cell_errors_k1, seeds 1-3, K=5, restarts=2, iters=150, ridge=0.1, max_rank=3"}
               for u in units], open(OUT_U34, "w"), indent=1)

    # ---- 恢复性诊断（K1）
    rec_names = S1_ORDER + S2_NAMES + [S3_NAME, S4_NAME] + list(DDPCA)
    if a.skip_recovery:
        recov = {}
    else:
        t0 = time.time()
        recov = {}
        with mp.get_context("fork").Pool(a.jobs) as pool:
            for nm, o in pool.imap_unordered(recovery_job, sorted(rec_names, key=lambda n: -int(MATS[n][1].sum())),
                                             chunksize=1):
                recov[nm] = o
        P(f"K1 恢复性诊断 {len(rec_names)} 个矩阵 × 3 秩 × 8，{time.time()-t0:.0f}s")
        R["search_volume"]["recovery_cv_runs_k1"] = len(rec_names) * 3 * 8
    R["recovery_k1"] = recov
    R["recovery_old_docs41"] = {nm: {"powered": d41[nm]["powered"],
                                     "p_detect2": d41[nm]["rec"]["plant2"]["p_detect2"],
                                     "p_detect0": d41[nm]["rec"]["plant0"]["p_detect0"],
                                     "p_r2_beats_r1_plant1": d41[nm]["rec"]["plant1"]["p_r2_beats_r1"]}
                                for nm in rec_names}

    # ---- 体系表
    systems = {}
    sysdefs = {"S1": S1_ORDER, "S2": S2_NAMES, "S3": [S3_NAME], "S4": [S4_NAME]}
    for arm in ARMS:
        systems[arm] = {}
        for sid, names in sysdefs.items():
            if arm == "k1":
                pw = [n for n in names if recov.get(n, {}).get("powered")]
            else:
                pw = [n for n in names if d41[n]["powered"]]
            info = {"matrices": names, "powered": pw, "n_powered": len(pw)}
            if not pw:
                info["status"] = "UNINFORMATIVE"
            else:
                o = pooled([unit_of(n, arm) for n in pw], pw, 500 + int(sid[1:]))
                info.update(o)
                info.update(sys_status(o))
            systems[arm][sid] = info
    R["systems"] = systems
    s1o = systems["old"]["S1"]
    if s1o["n_powered"] == 15:      # 老臂 S1 体系行必须复现 docs/41 T6 的区间（同种子 501）
        r41 = json.load(open(os.path.join(ROOT, "cache", "independence_audit.json")))["systems"]["S1"]
        for k, kk in (("g12", "g12_ci"), ("g23", "g23_ci"), ("g02", "g02_ci")):
            assert max(abs(x - y) for x, y in zip(s1o[k]["ci"], r41[kk])) < 1e-9, (k, s1o[k]["ci"], r41[kk])
        P("P4b 老臂 S1 体系行区间（seed 501）复现 docs/41 T6  OK")
    # 补充 D：ddPCA 的头条检出秩（描述性）
    R["ddpca"] = {nm: {"rank_old": CS[(nm, "old")]["rank_headline"], "rank_k1": CS[(nm, "k1")]["rank_headline"],
                       "k1_curve": CS[(nm, "k1")]["mean_curve"], "old_curve": CS[(nm, "old")]["mean_curve"]}
                  for nm in DDPCA}

    # ---- 范围内矩阵的检出秩汇总（三口径）与量级核对
    inscope = S1_ORDER + [S2_NAMES[0], S3_NAME]
    summ = {}
    for nm in S1_ORDER + S2_NAMES + [S3_NAME, S4_NAME]:
        summ[nm] = {arm: {"rank": CS[(nm, arm)]["rank_headline"], "mode": CS[(nm, arm)]["rank_mode"],
                          "cell": CS[(nm, arm)]["rank_cell"], "stab": CS[(nm, arm)]["stab"],
                          "mean_curve": CS[(nm, arm)]["mean_curve"]} for arm in ARMS}
    R["ranks_scope"] = summ
    R["n_rank2_inscope17"] = {arm: int(sum(summ[nm][arm]["rank"] == 2 for nm in inscope)) for arm in ARMS}
    R["n_inscope"] = len(inscope)
    ratio = [CS[(u["name"], "k1")]["mean_curve"][0] / CS[(u["name"], "old")]["mean_curve"][0] for u in units]
    R["magnitude"] = {"rank0_trim95_ratio_k1_over_old": [float(min(ratio)), float(np.median(ratio)), float(max(ratio))],
                      "trim95_range_k1": [float(min(min(CS[(u['name'], 'k1')]['mean_curve'].values()) for u in units)),
                                          float(max(max(CS[(u['name'], 'k1')]['mean_curve'].values()) for u in units))]}
    P(f"量级核对：K1/现行 rank-0 trim95 之比 min/median/max = {min(ratio):.4f}/{np.median(ratio):.4f}/{max(ratio):.4f}")
    R["search_volume"]["bootstrap_resample_sets"] = {
        "panel_site_joint": 2, "panel_per_subset": 2 * 15, "panel_combined_keys": 2 * 6,
        "system_level": sum(1 for arm in ARMS for sid in systems[arm] if systems[arm][sid].get("n_powered")),
        "nboot_each": NB, "strat_and_corr_sets": 6 * 4, "nboot_corr": NB_CORR, "fig1_sets": 3 * 2, "fig1_nboot": NB}
    R["search_volume"]["candidates_tried"] = 1
    R["seconds_total"] = time.time() - t_start
    json.dump(R, open(OUT_JSON, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    P(f"\n完成，用时 {R['seconds_total']:.0f}s。{OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
