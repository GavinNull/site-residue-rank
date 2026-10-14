#!/usr/bin/env python3
"""λ2 条件检验的判决对 k_eff 的敏感性（docs/34 §7.2）。

为什么要单独测这一项
--------------------
`analysis/gate_consequences.py:154` 写的是 `keff = max(2, len(fam))`，
场景 A 有两个家族，所以 **λ2 的条件零分布是在 k_eff = 2 上建的**；
而观测 λ2 = 0.627 是 **8 个矩阵**（先家族内平均、再家族等权）的量。
二分法只把零分布的 **λ1** 标定到观测值（残差 0.0035），
**λ2 的分布形状不被这一步约束**，而它对 k_eff 极其敏感。

本脚本把**同一个观测 λ2 = 0.627**，对**同样标定到观测 λ1 = 0.839** 的条件
零分布，在 k_eff = 2 / 3 / 5 / 8 上各读一次 p。四行的零分布 λ1 中位都等于
0.839（标定残差一起报），所以这是**同口径**比较，差别只来自 k_eff。

这不决定「哪个 k_eff 才对」——那是检验构造的技术路线问题，
按 CLAUDE.md §9 属于人类决策点。本脚本只把敏感性摆出来。

数值层复用 `subspace_cond_null`（cond_null / _draw / ONE），未重写。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subspace_cond_null as CN  # noqa: E402
from subspace_lib import ROOT, extract_subspace  # noqa: E402
import gate_calib_lib as G  # noqa: E402
import lambda2_power as LP  # noqa: E402

NSIM = 4000
OBS_L1, OBS_L2 = 0.8391, 0.6268     # docs/24 §6.2，由 lambda2_power.stage1 复现


def main():
    LP.require_harness()
    mats = G.load_evidence()
    by = {}
    for nm in LP.ORDER:
        B, c, _ = extract_subspace(mats[nm]["M"], mats[nm]["mask"])
        assert B is not None and len(c) == 20, nm
        by[nm] = B
    w, V = LP._fam_avg([[by[n] for n in LP.COV], [by[n] for n in LP.OMT3]])
    u = V[:, 0]
    l1, l2 = float(w[0]), float(w[1])
    assert abs(l1 - OBS_L1) < 0.004 and abs(l2 - OBS_L2) < 0.004, (l1, l2)
    print(f"观测（8 个矩阵 / 2 个家族）λ1 = {l1:.4f}、λ2 = {l2:.4f}"
          f"  —— 与 docs/24 §6.2 一致")
    print(f"\n同一个观测 λ2，对**同样标定到观测 λ1** 的条件零分布，"
          f"在不同 k_eff 上读 p（每格 {NSIM} 次）：")
    print(f"{'k_eff':>6}{'抖动角°':>9}{'零分布λ1中位':>14}{'标定残差':>10}"
          f"{'零分布λ2中位':>14}{'λ2 p95':>9}{'观测λ2的p':>11}  判定")
    rows = []
    for k in (2, 3, 5, 8):
        cn, al = CN.cond_null(k, u, np.random.default_rng(4), nsim=NSIM,
                              target_l1=l1)
        m1 = float(np.median(cn[:, 0]))
        p = float(np.mean(cn[:, 1] >= l2))
        rec = {"k_eff": k, "alpha_deg": float(np.degrees(al)),
               "null_l1_med": m1, "cal_resid": abs(m1 - l1),
               "null_l2_med": float(np.median(cn[:, 1])),
               "null_l2_p95": float(np.percentile(cn[:, 1], 95)),
               "p_lambda2": p}
        rows.append(rec)
        print(f"{k:>6}{np.degrees(al):>9.1f}{m1:>14.3f}"
              f"{abs(m1-l1):>10.4f}{rec['null_l2_med']:>14.3f}"
              f"{rec['null_l2_p95']:>9.3f}{p:>11.4f}  "
              f"{'不超出' if p > 0.05 else '**超出**'}")
    print("\n四行的标定残差都 < 0.004，所以这是同口径比较，"
          "差别**只来自 k_eff**。")
    print("在用的那一行是 k_eff = 2（gate_consequences.py:154）。")
    p_ = os.path.join(ROOT, "cache", "lambda2_keff_sens.json")
    json.dump({"obs_lam1": l1, "obs_lam2": l2, "nsim": NSIM, "rows": rows},
              open(p_, "w"), ensure_ascii=False, indent=1, default=float)
    print("产物:", p_)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
