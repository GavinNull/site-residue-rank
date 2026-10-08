#!/usr/bin/env python3
"""秩的配体特异性：在两个数据集的 45 个「骨架 × 配体」组合上检验。

PD-CoV RBD 上的发现（同骨架 114 位点，三个 APN 配体）：
  gAPN  12/12 种子给秩 2，真内部极小（0.885 < 0.950, 0.939）
  hAPN  10/12 给秩 0，CV 误差随秩单调上升
  pAPN  不可判（8/12 给到上限，曲线平坦非单调）
且不是数据质量问题：hAPN 信噪比 3.57 > gAPN 的 1.56。

若「秩是配体特异的」为真，则在二维设计里应看到：
  秩随配体变化（列效应）> 秩随骨架变化（行效应）

数据：
  animalACE2   3 骨架 x 5 ACE2 正交体 x 201 位点
  sarbecovirus 6 骨架 x 5 ACE2 正交体 x ~200 位点（RpearACE2 是 MFI，已排除）
"""
import collections
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coupling_structure import double_center
from rank_test_power import als_guarded

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DMS = os.path.join(ROOT, "cache", "dms")
AA20 = list("ACDEFGHIKLMNPQRSTVWY")
MAXR = 3
SEEDS = (1, 2, 3)


def cv_curve(Rm, mask, seed, K=5, restarts=2, iters=150):
    rng = np.random.default_rng(seed)
    idx = np.argwhere(mask)
    rng.shuffle(idx)
    folds = np.array_split(idx, K)
    out = {}
    for rank in range(MAXR + 1):
        e = []
        for f in folds:
            trn = mask.copy()
            for i, j in f:
                trn[i, j] = False
            Y = np.where(trn, Rm, 0.0)
            best, bp = np.inf, None
            for _ in range(restarts):
                P = als_guarded(Y, trn, rank, rng, iters=iters, ridge=0.1)
                tr = np.mean((Rm[trn] - P[trn]) ** 2)
                if tr < best:
                    best, bp = tr, P
            e += [(Rm[i, j] - bp[i, j]) ** 2 for i, j in f]
        e = np.array(e)
        out[rank] = math.sqrt(np.quantile(e, 0.95))
    return out


def analyse(rows, bg, col, site_key):
    vals = []
    for r in rows:
        if r.get("target") != bg:
            continue
        v = r.get(col, "")
        if v in ("", "NA"):
            continue
        vals.append((r[site_key], r["mutant"], float(v)))
    if not vals:
        return None
    arr = np.array([v for _, _, v in vals])
    # 按最小值处的堆积检测截尾下限
    lo = arr.min()
    cens = int((arr <= lo + 1e-6).sum())
    keep = [(s, a, v) for s, a, v in vals if v > lo + 1e-6]
    if len(keep) < 500:
        return {"n": len(keep), "cens": cens, "skip": "观测不足"}
    sites = sorted({s for s, _, _ in keep})
    si = {s: i for i, s in enumerate(sites)}
    M = np.full((len(sites), len(AA20)), np.nan)
    for s, a, v in keep:
        if a in AA20:
            M[si[s], AA20.index(a)] = v
    mask = np.isfinite(M)
    if mask.sum() < 500 or len(sites) < 30:
        return {"n": int(mask.sum()), "cens": cens, "skip": "矩阵太小"}
    Rm = double_center(np.where(mask, M, 0.0), mask)
    curves = [cv_curve(Rm, mask, s) for s in SEEDS]
    bests = [min(c, key=c.get) for c in curves]
    mean = {r: float(np.mean([c[r] for c in curves])) for r in range(MAXR + 1)}
    cnt = collections.Counter(bests)
    mode, nmode = cnt.most_common(1)[0]
    return {"n": int(mask.sum()), "sites": len(sites), "cens": cens,
            "coup_frac": float((Rm[mask].std() / M[mask].std()) ** 2),
            "mean_curve": mean, "best_mode": int(mode),
            "stability": nmode / len(SEEDS),
            "gain": (mean[0] - mean[mode]) / mean[0] if mean[0] else 0.0}


def main():
    jobs = [
        ("animalACE2", "starr_animalACE2_final_variant_scores.csv", "position",
         ["bind_deer", "bind_hamster", "bind_bat", "bind_cat",
          "bind_human_monomer"]),
        ("sarbecovirus", "starr_SARSr_RBD_final_variant_scores_lib40_41.csv",
         "site", ["huACE2", "Ra787ACE2", "Ra9479ACE2", "RlanACE2",
                  "RalcACE2"]),
    ]
    res = []
    for tag, fn, skey, cols in jobs:
        path = os.path.join(DMS, fn)
        if not os.path.exists(path):
            print(f"{tag}: 文件缺失")
            continue
        rows = list(csv.DictReader(open(path)))
        bgs = sorted({r["target"] for r in rows})
        print("=" * 92)
        print(f"{tag}: {len(bgs)} 骨架 x {len(cols)} 配体")
        print("=" * 92)
        print(f"{'骨架':<22}{'配体':<20}{'位点':>5}{'观测':>6}{'截尾':>6}"
              f"{'耦合%':>7}{'最优秩':>6}{'稳定':>6}{'增益':>8}")
        print("-" * 92)
        for bg in bgs:
            for col in cols:
                r = analyse(rows, bg, col, skey)
                if r is None:
                    continue
                if "skip" in r:
                    print(f"{bg[:20]:<22}{col[:18]:<20}{'':>5}{r['n']:>6}"
                          f"{r['cens']:>6}   {r['skip']}")
                    continue
                print(f"{bg[:20]:<22}{col[:18]:<20}{r['sites']:>5}{r['n']:>6}"
                      f"{r['cens']:>6}{r['coup_frac']:>7.1%}"
                      f"{r['best_mode']:>6}{r['stability']:>6.0%}"
                      f"{r['gain']:>+8.1%}", flush=True)
                res.append({"dataset": tag, "bg": bg, "lig": col, **r})
    json.dump(res, open(os.path.join(ROOT, "cache", "ligand_rank.json"), "w"),
              indent=1, default=float)
    # 二维模式分析
    print("\n" + "=" * 92)
    print("二维模式：秩随配体变 vs 随骨架变")
    print("=" * 92)
    for tag in {r["dataset"] for r in res}:
        sub = [r for r in res if r["dataset"] == tag]
        bylig = collections.defaultdict(list)
        bybg = collections.defaultdict(list)
        for r in sub:
            bylig[r["lig"]].append(r["best_mode"])
            bybg[r["bg"]].append(r["best_mode"])
        vl = np.mean([np.var(v) for v in bylig.values() if len(v) > 1]) \
            if bylig else float("nan")
        vb = np.mean([np.var(v) for v in bybg.values() if len(v) > 1]) \
            if bybg else float("nan")
        print(f"\n[{tag}] {len(sub)} 个组合")
        print(f"   同一配体内、跨骨架的秩方差 = {vl:.3f}")
        print(f"   同一骨架内、跨配体的秩方差 = {vb:.3f}")
        print("   => 后者 >> 前者 则秩是配体特异的；前者 >> 后者 则是骨架特异的")
        for lig, v in sorted(bylig.items()):
            print(f"      配体 {lig:<20} 各骨架的秩 {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
