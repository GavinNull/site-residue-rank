#!/usr/bin/env python3
"""Part 1 的附表：不在 docs/21 §4.1 那 9 个成员里、但**后果分析必须用到**的
矩阵的同一套秩 2 下限 / 上限标定。

为什么必须单列：
- `SKEMPI/OMTKY3/pooled-4proteases`（40 位点 / 760 观测）**不在**那 9 个里，
  但它出现在 docs/21 §5.5 的 25 个跨家族配对里，而且它是「4 个单酶矩阵
  若被剔除之后，非病毒那一侧还剩什么」的唯一答案。
  它的几何比单酶矩阵大 4 倍，所以它的下限必然更低 —— 这个数决定了
  Part 3 的后果是「非病毒家族归零」还是「非病毒家族只剩一个矩阵」。
- `lehner/GB1`（人类已决定剔除）与 `lehner/PSD95-PDZ3` / `GRB2-SH3`
  （docs/13 §4.3 已排除）放在同一张表里，是为了给门槛一个**量级对照**：
  一个被排除的矩阵的下限长什么样。
- `titeseq/4-4-20`：docs/23 已判「闸无判别力」，这里给它的**秩 2** 版本。

参数与 `gate_rank2_calibration.py` 逐字相同，否则两张表不可并列。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_calib_lib as G  # noqa: E402
from gate_rank2_calibration import (GAP_MIN, N_BOOT, N_BOOT_OBS,  # noqa: E402
                                    N_REAL, STAB_FLAT, cached_stab,
                                    harness_gate)
from subspace_lib import ci  # noqa: E402

AUX = ["SKEMPI/OMTKY3/pooled-4proteases", "lehner/GB1/IgG-Fc",
       "lehner/PSD95-PDZ3/CRIPT", "lehner/GRB2-SH3/GAB2",
       "titeseq/4-4-20/fluorescein"]


def main():
    harness_gate()
    mats = G.load_evidence()
    pub = cached_stab()
    print("\n" + "=" * 112)
    print(f"Part 1 附表：秩 2 `stab cos2` 的下限 / 上限"
          f"（{N_REAL} × {N_BOOT}；实测 {N_BOOT_OBS}）")
    print("=" * 112)
    print(f"{'体系':<40}{'位点':>5}{'观测':>6}{'秩2解释':>8}"
          f"{'实测stab':>9}{'下限中位':>9}{'p90':>7}{'p95':>7}{'p99':>7}"
          f"{'上限中位':>9}{'gap':>8}")
    print("-" * 112)
    rows = []
    for nm in AUX:
        if nm not in mats:
            print(f"{nm}  <<< 不在加载集里，跳过")
            continue
        d = mats[nm]
        shape, mask = d["M"].shape, d["mask"]
        e2, _ = G.rankr_expl(d["M"], mask, rank=2)
        obs = G.stab_boot(d["M"], mask, rank=2, nboot=N_BOOT_OBS, seed=7)
        om = float(np.median(obs)) if obs else float("nan")
        flo = G.floor_dist(shape, mask, rank=2, n_real=N_REAL, nboot=N_BOOT)
        fp = G.pcts(flo)
        sg, sgtab = G.match_sigma(shape, mask, 2, e2)
        hi = G.ceiling_dist(shape, mask, rank=2, sigma=sg, n_real=N_REAL,
                            nboot=N_BOOT)
        hm = float(np.median(hi))
        gap = hm - fp[50]
        print(f"{nm[:39]:<40}{shape[0]:>5}{int(mask.sum()):>6}"
              f"{e2:>8.1%}{om:>9.3f}{fp[50]:>9.3f}{fp[90]:>7.3f}"
              f"{fp[95]:>7.3f}{fp[99]:>7.3f}{hm:>9.3f}{gap:>+8.3f}")
        readable = np.isfinite(gap) and gap >= GAP_MIN
        rows.append({"name": nm, "shape": list(shape),
                     "n_obs": int(mask.sum()), "expl_rank2": e2,
                     "stab_obs": om, "stab_pub": pub.get(nm),
                     "floor_med": fp[50], "floor_p90": fp[90],
                     "floor_p95": fp[95], "floor_p99": fp[99],
                     "floor_ci": list(ci(flo)), "ceiling_med": hm,
                     "ceiling_ci": list(ci(hi)), "sigma_matched": sg,
                     "sigma_table": sgtab, "gap": gap,
                     "readable": bool(readable),
                     "pass_flat": bool(om >= STAB_FLAT),
                     "pass_geo": bool(om > fp[95])})
    print("\n判读：")
    for r in rows:
        if not r["readable"]:
            v = f"**闸无判别力（gap = {r['gap']:+.3f}）—— 实测不可读**"
        elif r["pass_geo"]:
            v = "过几何相关门槛（下限 p95）"
        else:
            v = "**不过几何相关门槛 —— 与纯噪声不可区分**"
        r["verdict"] = v
        print(f"   {r['name'][:42]:<44}实测 {r['stab_obs']:.3f} vs "
              f"下限 p95 {r['floor_p95']:.3f}   {v}")
    p = os.path.join(G.ROOT, "cache", "gate_rank2_aux.json")
    json.dump({"rows": rows, "n_real": N_REAL, "n_boot": N_BOOT},
              open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print("\n产物:", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
