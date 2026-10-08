#!/usr/bin/env python3
"""单体系内部定秩：功效比此前高一个量级。

此前的秩判定用 9 个体系**混合**的 819 个观测，而跨体系混合有 ΔΔG 尺度
不一致的问题。现在有若干独立的大矩阵，每个单独就比之前全部数据多：

  SARS-CoV-2 RBD x ACE2    201 x 20 = 4020
  PD-CoV RBD x 3 种 APN     114 x 20 = 2280 (x3 个配体)
  PSD95-PDZ3 x CRIPT        84 x 19 = 1596
  GRB2-SH3 x GAB2           56 x 19 = 1064
  4-4-20 scFv x 荧光素       20 x 19 = 380

截尾处理：Starr 系列把非结合变体堆在检测下限（log10Ka = 5 或 6），
这些值必须剔除而不是当数值用，否则会系统性压低埋藏位点的效应。
剔除比例逐个报告。
"""
import collections
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coupling_structure import double_center
# 用**带欠定行列保护的** ALS。未加保护的 als_complete 会在稀疏行/列上
# 产生爆点（实测 GRB2 的留出集里 4 个点单点误差 >10，最大 104，
# 它们支配 RMSE 并给出 -800% 这种无意义的「增益」）。
# 这个修正我在 rank_test_power.py 里已经做过，但没复用到这里。
from rank_test_power import als_guarded

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DMS = os.path.join(ROOT, "cache", "dms")
AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def cv_rank(Rm, mask, max_rank=5, K=5, restarts=3, seed=3):
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
            Y = np.where(trn, Rm, 0.0)
            best, bestP = np.inf, None
            for _ in range(restarts):
                P = als_guarded(Y, trn, rank, rng, iters=250,
                                ridge=0.1)
                tr = np.mean((Rm[trn] - P[trn]) ** 2)
                if tr < best:
                    best, bestP = tr, P
            for i, j in f:
                allerr.append((Rm[i, j] - bestP[i, j]) ** 2)
        e = np.array(allerr)
        out[rank] = (math.sqrt(e.mean()),
                     math.sqrt(np.quantile(e, 0.95)),   # 裁掉最差 5%
                     float(np.sqrt(np.median(e))))
    return out


def analyse(name, cells, censor_note=""):
    """cells: {(site, aa): value}"""
    sites = sorted({s for s, _ in cells})
    M = np.full((len(sites), len(AA20)), np.nan)
    si = {s: i for i, s in enumerate(sites)}
    for (s, a), v in cells.items():
        if a in AA20:
            M[si[s], AA20.index(a)] = v
    mask = np.isfinite(M)
    if mask.sum() < 200 or len(sites) < 15:
        print(f"{name:<34} 数据不足（{mask.sum()} 观测, {len(sites)} 位点）")
        return None
    Rm = double_center(np.where(mask, M, 0.0), mask)
    tot = M[mask].std()
    coup = Rm[mask].std()
    errs = cv_rank(Rm, mask)
    # 主指标用**裁掉最差 5% 后**的 RMSE —— 单点爆值不该决定秩
    trim = {r: errs[r][1] for r in errs}
    best = min(trim, key=trim.get)
    g = {r: (trim[0] - trim[r]) / trim[0] for r in trim}
    nexpl = sum(1 for r in errs if errs[r][0] > 3 * errs[0][0])
    print(f"{name:<34} {len(sites):>4} {mask.sum():>6} "
          f"{(coup / tot) ** 2:>7.1%} "
          + "".join(f"{g[r]:>+8.1%}" for r in range(1, 5))
          + f"{best:>5} {nexpl:>4}")
    return {"name": name, "sites": len(sites), "n": int(mask.sum()),
            "coup_frac": (coup / tot) ** 2, "best_rank": best,
            "gains": g, "censor": censor_note,
            "rmse_full": {r: errs[r][0] for r in errs},
            "rmse_trim": trim}


def load_starr(path, col, lower_bound):
    cells = {}
    cens = 0
    with open(path) as f:
        for r in csv.DictReader(f):
            v = r.get(col, "")
            if v in ("", "NA"):
                continue
            x = float(v)
            if x <= lower_bound + 1e-9:
                cens += 1
                continue
            tgt = r.get("target", "")
            cells[(f"{tgt}_{r['position']}", r["mutant"])] = x
    return cells, cens


def load_lehner(path):
    cells = {}
    with open(path) as f:
        head = f.readline().split()
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            mid = p[0]
            v = p[2]
            if v in ("NA", "", "nan"):
                continue
            # id 形如 "A12G" 或 "12A" —— 实测格式
            import re
            m = re.match(r"^([A-Z])(\d+)([A-Z])$", mid)
            if not m:
                continue
            cells[(m.group(2), m.group(3))] = float(v)
    return cells


def main():
    print("=" * 96)
    print("单体系内部定秩（CV 增益 vs 秩0）")
    print("=" * 96)
    print("主指标 = 裁掉最差 5% 后的 CV RMSE（单点爆值不该决定秩）")
    print(f"{'体系':<34} {'位点':>4} {'观测':>6} {'耦合占比':>7} "
          f"{'秩1':>8}{'秩2':>8}{'秩3':>8}{'秩4':>8}{'最优':>5}{'爆':>4}")
    print("-" * 96)
    res = []

    # PD-CoV x 3 种 APN（bind_* 是绝对 log10Ka）
    for lig, lb in (("gAPN", 5.0), ("hAPN", 4.0), ("pAPN", 4.0)):
        c, cens = load_starr(os.path.join(
            DMS, "starr_PDCoV_RBD_final_variant_scores.csv"),
            f"bind_{lig}", lb)
        r = analyse(f"PD-CoV RBD x {lig}", c, f"剔除下限 {cens}")
        if r:
            res.append(r)

    # SARS-CoV-2 RBD x ACE2（delta 相对 WT）
    c = {}
    with open(os.path.join(DMS, "starr2020_rbd_single_mut_effects.csv")) as f:
        for r0 in csv.DictReader(f):
            v = r0.get("bind_avg", "")
            if v in ("", "NA"):
                continue
            c[(r0.get("site_SARS2") or r0.get("site_RBD"),
               r0["mutant"])] = float(v)
    r = analyse("SARS-CoV-2 RBD x huACE2", c)
    if r:
        res.append(r)

    # Lehner ddPCA（binding ΔΔG，kcal/mol）
    for nm, fn in (("PSD95-PDZ3 x CRIPT", "PSD95-PDZ3_ddG_weights.txt"),
                   ("GRB2-SH3 x GAB2", "GRB2-SH3_ddG_weights.txt"),
                   ("GB1 x IgG-Fc（反推值）", "GB1_ddG_weights.txt")):
        p = os.path.join(DMS, "lehner_ddPCA", fn)
        if os.path.exists(p):
            r = analyse(nm, load_lehner(p))
            if r:
                res.append(r)

    print("-" * 96)
    print("\n最优秩汇总:")
    cnt = collections.Counter(r["best_rank"] for r in res)
    for k in sorted(cnt):
        print(f"   秩 {k}: {cnt[k]} 个体系")
    print(f"\n耦合占总方差: 中位 "
          f"{np.median([r['coup_frac'] for r in res]):.1%}, "
          f"范围 {min(r['coup_frac'] for r in res):.1%}–"
          f"{max(r['coup_frac'] for r in res):.1%}")
    import json
    json.dump(res, open(os.path.join(ROOT, "cache", "within_rank.json"), "w"),
              indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
