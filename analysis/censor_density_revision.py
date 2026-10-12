#!/usr/bin/env python3
"""用密度型检测器重测 docs/13 的全部矩阵，给出修订后的保留集（docs/21 第 1 阶段）。

先强制跑 analysis/censor_density_check.py 的全部判据；不通过就中止，
不看任何结论（CLAUDE.md §10）。

重点是 docs/19 §7 标记的 4 个「盲区」矩阵：
  animalACE2/Omicron_BA2/bat、animalACE2/Eta/bat、animalACE2/Omicron_BA2/cat、
  PDCoV/PDCoV/gAPN
它们在固定窗口 w = 0.25 下都 < 10%，但 w = 1.00 下跳到 27.8–86.9%。
这四个的判决方向**两边都有**（docs/19 §7）：
  Eta/bat 与 BA2/cat 是 animalACE2 里唯二不给秩 2 的组合 —— 剔除它们
    **加强** docs/12；
  BA2/bat 给的是秩 2 —— 剔除它**削弱** docs/12。
所以本脚本对每个矩阵都打印「这次重测把结论推向哪一边」。

运行：python3 analysis/censor_density_revision.py
产物：cache/censor_density_revision.json
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import censor_density_check as CHK                            # noqa: E402
from censor_density import (censor_fraction_density_ci,        # noqa: E402
                            raw_values)
from subspace_lib import (DMS, ROOT, dedup_by_content,         # noqa: E402
                          load_all_dms, load_independent, load_skempi)

CENSOR_MAX = 0.25

# docs/19 §7 的 4 个盲区矩阵 + 它们的检出秩（docs/12 §3 / docs/19 §4）
BLIND = {
    "animalACE2/Omicron_BA2/bat": ("秩 2", "剔除会**削弱** docs/12"),
    "animalACE2/Eta/bat": ("秩 1（animalACE2 唯二不给秩 2 之一）",
                           "剔除会**加强** docs/12"),
    "animalACE2/Omicron_BA2/cat": ("秩 1（animalACE2 唯二不给秩 2 之一）",
                                   "剔除会**加强** docs/12"),
    "PDCoV/PDCoV/gAPN": ("秩 2，12/12 种子", "剔除会**削弱** docs/12"),
}
# docs/19 §7 的固定窗口扫描值（w = 0.25 / 1.00）
BLIND_W = {"animalACE2/Omicron_BA2/bat": (0.012, 0.869),
           "animalACE2/Eta/bat": (0.012, 0.524),
           "animalACE2/Omicron_BA2/cat": (0.005, 0.343),
           "PDCoV/PDCoV/gAPN": (0.015, 0.278)}

SRC = {"animalACE2": "starr_animalACE2_final_variant_scores.csv",
       "sarbecovirus":
           "starr_SARSr_RBD_final_variant_scores_lib40_41.csv",
       "PDCoV": "starr_PDCoV_RBD_final_variant_scores.csv"}


def file_floor(tag):
    """同一个源文件里**所有**骨架 × 配体的全局最小值 = 该 assay 的硬下限。

    这是一条独立的、不依赖检测器的事实：如果一个矩阵自己的 min 明显
    高于它所在文件的全局 min，那它**根本没有触到检测下限**，
    「下限堆积」在它里面不存在。animalACE2 文件的全局 min 正是
    5.000（非结合变体被硬赋成这个值）。
    """
    path = os.path.join(DMS, SRC.get(tag, ""))
    if not os.path.exists(path):
        return None
    vals = []
    for r in csv.DictReader(open(path)):
        for k, v in r.items():
            if not (k == "bind" or k.startswith("bind_")
                    or k in ("huACE2", "Ra787ACE2", "Ra9479ACE2",
                             "RlanACE2", "RalcACE2")):
                continue
            if v in ("", "NA", "nan", None):
                continue
            try:
                vals.append(float(v))
            except ValueError:
                pass
    return float(min(vals)) if vals else None


def main():
    print("#" * 92)
    print("# 第 0 步：检测器自检（不通过即中止）")
    print("#" * 92)
    if CHK.main() != 0:
        print("\n** 检测器自检未通过 —— 中止，不看任何结论 **")
        return 1

    mats = load_independent() + load_skempi() + load_all_dms()
    n0 = len(mats)
    mats, drops = dedup_by_content(mats)
    print("\n" + "#" * 92)
    print(f"# 第 1 步：重测全部矩阵  原始 {n0} 个 -> 内容去重后 {len(mats)} 个"
          f"（丢 {len(drops)} 个）")
    print("#" * 92)

    floors = {t: file_floor(t) for t in SRC}
    rows = []
    for d in mats:
        r = censor_fraction_density_ci(raw_values(d), seed=7)
        a = raw_values(d)
        fl = floors.get(d["dataset"])
        rows.append({
            "name": d["name"], "dataset": d["dataset"],
            "backbone": d["backbone"], "ligand": d["ligand"],
            "sites": len(d["sites"]), "n_obs": int(d["n_obs"]),
            "n_raw": int(d["n_raw"]),
            "old": float(d["censor_frac"]), "new": float(r["q"]),
            "lo": float(r["lo"]), "hi": float(r["hi"]),
            "sub_med": float(r["sub_med"]),
            "gate": r["info"]["gate"], "why": r["info"]["why"],
            "sigma_hat": r["info"]["sigma_hat"],
            "c_rel": r["info"]["c_rel"],
            "min": float(np.min(a)),
            "above_floor": (None if fl is None else float(np.min(a) - fl)),
            "file_floor": fl,
        })
    rows.sort(key=lambda z: -z["new"])

    print(f"\n{'矩阵':<44}{'旧':>7}{'新':>7}{'10-90%子抽样':>16}"
          f"{'位点':>5}{'离硬下限':>9}  判决")
    print("-" * 110)
    for z in rows:
        verdict = "剔除" if z["new"] >= CENSOR_MAX else "保留"
        af = ("—" if z["above_floor"] is None
              else f"{z['above_floor']:+.2f}")
        ci = f"[{z['lo']:.1%}, {z['hi']:.1%}]"
        print(f"{z['name'][:43]:<44}{z['old']:>7.1%}{z['new']:>7.1%}"
              f"{ci:>16}{z['sites']:>5}{af:>9}  {verdict}")

    new_keep = [z for z in rows if z["new"] < CENSOR_MAX]
    old_keep = [z for z in rows if z["old"] < CENSOR_MAX]
    print(f"\n旧口径保留 {len(old_keep)} 个（docs/13 §4 说 48）；"
          f"新口径保留 {len(new_keep)} 个")

    # 变动的矩阵
    flips = [z for z in rows
             if (z["old"] < CENSOR_MAX) != (z["new"] < CENSOR_MAX)]
    print("\n" + "=" * 92)
    print("分类发生变动的矩阵（及这次重测把结论推向哪一边）")
    print("=" * 92)
    if not flips:
        print("   无。新旧口径给出**同一个保留集**。")
    for z in flips:
        direction = ("旧剔除 -> 新保留（判别集变大）"
                     if z["old"] >= CENSOR_MAX else
                     "旧保留 -> 新剔除（判别集变小）")
        print(f"   {z['name']:<44} {z['old']:.1%} -> {z['new']:.1%}  "
              f"{direction}")

    # ---------------- 盲区四个矩阵的判决 ----------------
    print("\n" + "=" * 92)
    print("docs/19 §7 的 4 个盲区矩阵：判决 + 方向")
    print("=" * 92)
    byn = {z["name"]: z for z in rows}
    blind_out = []
    for nm, (rank, push) in BLIND.items():
        z = byn.get(nm)
        if z is None:
            print(f"   {nm}: 去重时被丢弃，不在判别集里")
            continue
        w25, w100 = BLIND_W[nm]
        keep = z["new"] < CENSOR_MAX
        print(f"\n   {nm}")
        print(f"     固定窗口 w=0.25: {w25:.1%}   w=1.00: {w100:.1%}"
              f"（docs/19 §7 的标记依据）")
        print(f"     密度型检测器: **{z['new']:.1%}**  "
              f"子抽样 10–90% [{z['lo']:.1%}, {z['hi']:.1%}]  "
              f"闸: {z['gate']}")
        if z["why"]:
            print(f"     理由: {z['why'][:80]}")
        if z["above_floor"] is not None:
            print(f"     机制核对: 本矩阵 min = {z['min']:.3f}，"
                  f"源文件全局硬下限 = {z['file_floor']:.3f}，"
                  f"**高出 {z['above_floor']:+.3f}**")
        print(f"     检出秩: {rank}；若剔除：{push}")
        print(f"     => 判决：**{'保留' if keep else '剔除'}**"
              f"（{'不是截尾' if keep else '是截尾'}）")
        blind_out.append({"name": nm, "w025": w25, "w100": w100,
                          "new": z["new"], "lo": z["lo"], "hi": z["hi"],
                          "keep": keep, "rank": rank, "push": push,
                          "above_floor": z["above_floor"]})

    out = {"censor_max": CENSOR_MAX, "n_raw_mats": n0,
           "n_dedup": len(mats), "n_keep_old": len(old_keep),
           "n_keep_new": len(new_keep),
           "retained": [z["name"] for z in new_keep],
           "excluded": [z["name"] for z in rows
                        if z["new"] >= CENSOR_MAX],
           "flips": [z["name"] for z in flips],
           "blind": blind_out, "rows": rows}
    p = os.path.join(ROOT, "cache", "censor_density_revision.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
