#!/usr/bin/env python3
"""docs/43 Step 6 的区间（CLAUDE.md §5）：现行程序与 K1 在 18 个真实矩阵上的秩间增益，
位点联合重抽 2000 次（复用 independence_audit.site_boot_gains）。

先决：预先登记锁；GC 已通过 K1（cache/overshoot_gc.json）；
已知事实自检：cv_cell_errs 的 trim95（取单种子）必须等于 cv_curve 的值（逐位），
且现行程序（joint=False）的 15 个 animalACE2 秩 1→2 中位增益逐位复现 docs/41 的 +3.54%（种子平均逐格误差口径）。
产物：cache/overshoot_k1_boot.json
"""
import json
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overshoot_lib as OL                       # noqa: E402
import independence_audit as IA                  # noqa: E402

ROOT = OL.ROOT
NB = 2000


def trim(e):
    return math.sqrt(np.quantile(e, 0.95))


def main():
    OL.verify_lock()
    gc = json.load(open(os.path.join(ROOT, "cache", "overshoot_gc.json")))
    assert "K1" in gc["passed"], "K1 未通过 GC，不得用于真实矩阵"
    geoms = pickle.load(open(os.path.join(ROOT, "cache", "overshoot_geoms.pkl"), "rb"))
    # 自检：trim95 一致
    g = geoms["PDCoV/PDCoV/gAPN"]
    for joint in (False, True):
        cells, E = OL.cv_cell_errs(g.M, g.mask, [1, 2], 3, joint=joint)
        cc = OL.cv_curve(g.M, g.mask, [1, 2], seed=3, joint=joint, **{k: v for k, v in OL.CV.items() if k != "K"})
        for r in (1, 2):
            assert abs(trim(E[r]) - cc[r]) < 1e-12, (joint, r, trim(E[r]), cc[r])
    print("  自检：cv_cell_errs 的 trim95 == cv_curve（base 与 joint, 逐位）  OK", flush=True)

    names = OL.scope_names()
    res = {"base": {}, "K1": {}}
    units = {"base": {}, "K1": {}}
    for nm in names:
        g = geoms[nm]
        for lab, joint in (("base", False), ("K1", True)):
            cells, E = OL.cell_errs_mean(g.M, g.mask, joint)
            units[lab][nm] = {"sites": list(range(g.nr)), "rows": cells[:, 0], "E": E}
        print(f"  逐格误差 {nm}", flush=True)
    # 自检（复现 docs/41）：base 的 15 个 animalACE2 秩 1→2 中位 = +3.54%
    s1 = [n for n in names if n.startswith("animalACE2/")]
    g12 = [(trim(units["base"][n]["E"][1]) - trim(units["base"][n]["E"][2])) / trim(units["base"][n]["E"][1]) for n in s1]
    print(f"  base S1 秩1→2 中位（种子平均逐格误差口径）{100*np.median(g12):.2f}%（docs/41 §8 记 +3.54%）", flush=True)
    assert abs(100 * np.median(g12) - 3.54) < 0.05, "未复现 docs/41 的 +3.54%（逐格误差口径）"
    for lab in ("base", "K1"):
        # 位点标签：各矩阵的真实位点标签（跨矩阵对齐）
        us = []
        for nm in names:
            u = dict(units[lab][nm])
            u["sites"] = geoms[nm].sites_labels if hasattr(geoms[nm], "sites_labels") else u["sites"]
            us.append(u)
        res[lab]["names"] = names
    # 位点标签对齐只对 S1 的 15 个（共享位点集合）有意义；其余体系各自重抽
    labs = {nm: list(OL.pick(nm)["sites"]) for nm in names}
    out = {}
    for lab in ("base", "K1"):
        out[lab] = {"per_matrix": {}, "S1": {}}
        for nm in names:
            u = dict(units[lab][nm])
            u["sites"] = labs[nm]
            sb = IA.site_boot_gains([u], NB, seed=43, pairs=((0, 1), (1, 2), (2, 3), (0, 2)))
            row = {}
            for pair in ((0, 1), (1, 2), (2, 3), (0, 2)):
                a, b = pair
                pt = (trim(u["E"][a]) - trim(u["E"][b])) / trim(u["E"][a])
                lo, hi = np.percentile(sb[pair][:, 0], [2.5, 97.5])
                row[f"g{a}{b}"] = [pt, float(lo), float(hi)]
            tt = {r: trim(u["E"][r]) for r in range(4)}
            row["best_rank_cellavg"] = int(min(tt, key=tt.get))
            out[lab]["per_matrix"][nm] = row
        us = []
        for nm in s1:
            u = dict(units[lab][nm])
            u["sites"] = labs[nm]
            us.append(u)
        sb = IA.site_boot_gains(us, NB, seed=44, pairs=((0, 1), (1, 2), (2, 3), (0, 2)))
        for pair in ((0, 1), (1, 2), (2, 3), (0, 2)):
            a, b = pair
            pts = [(trim(u["E"][a]) - trim(u["E"][b])) / trim(u["E"][a]) for u in us]
            med = np.median(sb[pair], axis=1)
            out[lab]["S1"][f"g{a}{b}"] = [float(np.median(pts)), float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]
            out[lab]["S1"][f"g{a}{b}_pos_matrices"] = int(sum(p > 0 for p in pts))
    out["nboot"] = NB
    json.dump(out, open(os.path.join(ROOT, "cache", "overshoot_k1_boot.json"), "w"), indent=1)
    print("\n秩间增益（trim95，种子平均逐格误差；区间 = 位点重抽 2000 次）")
    print(f"{'矩阵':<38}{'现行 g12':>24}{'K1 g12':>24}{'现行 g23':>24}{'K1 g23':>24}")
    f = lambda x: f"{100*x[0]:+.1f}% [{100*x[1]:+.1f},{100*x[2]:+.1f}]"
    for nm in names:
        b, k = out["base"]["per_matrix"][nm], out["K1"]["per_matrix"][nm]
        print(f"{nm.replace('animalACE2/',''):<38}{f(b['g12']):>24}{f(k['g12']):>24}{f(b['g23']):>24}{f(k['g23']):>24}")
    print("按逐格误差口径的最优秩（现行 / K1）：", {nm.replace('animalACE2/',''): (out['base']['per_matrix'][nm]['best_rank_cellavg'], out['K1']['per_matrix'][nm]['best_rank_cellavg']) for nm in names})
    for lab in ("base", "K1"):
        s = out[lab]["S1"]
        print(f"S1(15) {lab}: g01 {f(s['g01'])}（正的矩阵 {s['g01_pos_matrices']}/15）  g12 中位 {f(s['g12'])}（正的矩阵 {s['g12_pos_matrices']}/15）  g23 {f(s['g23'])}  g02 {f(s['g02'])}")


if __name__ == "__main__":
    main()
