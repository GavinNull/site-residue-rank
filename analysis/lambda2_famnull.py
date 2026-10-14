#!/usr/bin/env python3
"""把 λ2 的零分布建在**统计量自己的家族结构**上（docs/34 §7.3 的诊断）。

问题（docs/34 §2.1 / §7.2）
--------------------------
在用的那个条件零分布抽的是 **k_eff = 2 个纯秩 2 投影算子**的平均，
而观测 λ2 的两个「单元」是 **5 个和 3 个**子空间的**家族平均** ——
家族平均不是秩 2 投影算子（迹仍为 2，但特征值被摊开）。
两边的结构不一样，而二分法只标定 **λ1**，λ2 的分布形状不被约束。

本脚本给出同一个问题的**结构匹配**版本：零分布的每一次抽样都走
「5 个冠状病毒式矩阵 + 3 个 OMTKY3 式矩阵、家族内第二轴按实测离散度、
**跨家族第二轴独立**（ρ2 = 0，即 H0）」这条**与观测同构**的路，
再用二分法把 `a_c`（第一轴跨家族共享度）调到使零分布的 **λ1 中位 = 观测
λ1 = 0.839**，然后在同一个 λ1 水平上读 λ2。

**这不是在改主检验。** 采用哪一个零分布属于检验构造的技术路线，
按 CLAUDE.md §9 是人类决策点。本脚本只把「结构匹配的零分布会给什么」
作为诊断报出来，并把三个 `b_w(omt)` 设定都跑一遍（该参数弱可识别）。

数值层复用 `lambda2_power`（synth_set / _fam_avg / fit_geometry）与
`subspace_lib`，未重写。**未 import `als_complete`。**
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_calib_lib as G  # noqa: E402
import lambda2_power as LP  # noqa: E402
from subspace_lib import ROOT, extract_subspace  # noqa: E402

NDRAW = 400
OBS_L1, OBS_L2 = 0.8391, 0.6268


def lam_draws(geo, par, rng, n):
    out = []
    for _ in range(n):
        fb, _ = LP.synth_set(geo, par, rng)
        if fb is not None:
            out.append(LP._fam_avg(fb)[0][:3])
    return np.array(out)


def calib_ac(geo, par, target_l1, rng, nrep=30, iters=12):
    """二分法把 a_c 调到使**结构匹配零分布**的 λ1 中位 = 观测 λ1。

    a_c 越小 -> 两个家族的第一轴越不一致 -> λ1 越低。单调。
    """
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        par["a_c"] = mid
        L = lam_draws(geo, par, rng, nrep)
        m = float(np.median(L[:, 0])) if len(L) else 0.0
        if m < target_l1:
            lo = mid
        else:
            hi = mid
    par["a_c"] = 0.5 * (lo + hi)
    return par["a_c"]


def main():
    LP.require_harness()
    mats = G.load_evidence()
    by, by_mat = {}, {}
    for nm in LP.ORDER:
        B, c, _ = extract_subspace(mats[nm]["M"], mats[nm]["mask"])
        assert B is not None and len(c) == 20, nm
        by[nm] = B
        by_mat[nm] = mats[nm]
    w, _ = LP._fam_avg([[by[n] for n in LP.COV], [by[n] for n in LP.OMT3]])
    l1, l2 = float(w[0]), float(w[1])
    assert abs(l1 - OBS_L1) < 0.004 and abs(l2 - OBS_L2) < 0.004
    print(f"观测 λ1 = {l1:.4f}、λ2 = {l2:.4f}（与 docs/24 §6.2 一致）")
    geo = LP.fit_geometry(by_mat)

    # 第一轴家族内共享度沿用 lambda2_power 的标定值（同一目标、同一方法）
    AW = {"cov": 0.984, "omt": 0.946}
    BW_COV = 0.941
    print(f"\n结构匹配零分布：5 + 3，ρ2 = 0（H0），a_w = {AW}、"
          f"b_w(cov) = {BW_COV}，a_c 二分法标定到观测 λ1")
    print(f"{'b_w(omt)':>10}{'a_c标定':>9}{'零分布λ1中位':>14}{'标定残差':>10}"
          f"{'零分布λ2中位':>14}{'λ2 p95':>9}{'观测λ2的p':>11}  判定")
    rows = []
    for bw_omt, lab in ((0.033, "H 低端"), (0.380, "H2 高端"),
                        (1.000, "F 上界")):
        par = {"a_c": 1.0, "a_w": dict(AW), "b_c": 0.0,
               "b_w": {"cov": BW_COV, "omt": bw_omt}}
        rng = np.random.default_rng(2026 + int(bw_omt * 1000))
        ac = calib_ac(geo, par, l1, rng)
        L = lam_draws(geo, par, rng, NDRAW)
        m1 = float(np.median(L[:, 0]))
        p = float(np.mean(L[:, 1] >= l2))
        rec = {"b_w_omt": bw_omt, "label": lab, "a_c": ac,
               "null_l1_med": m1, "cal_resid": abs(m1 - l1),
               "null_l2_med": float(np.median(L[:, 1])),
               "null_l2_p95": float(np.percentile(L[:, 1], 95)),
               "null_l2_q": [float(np.percentile(L[:, 1], 2.5)),
                             float(np.percentile(L[:, 1], 97.5))],
               "p_lambda2": p, "n": int(len(L))}
        rows.append(rec)
        print(f"{bw_omt:>10.3f}{ac:>9.3f}{m1:>14.3f}{abs(m1-l1):>10.4f}"
              f"{rec['null_l2_med']:>14.3f}{rec['null_l2_p95']:>9.3f}"
              f"{p:>11.4f}  "
              f"{'不超出' if p > 0.05 else '**超出**'}  [{lab}]")
    bad = [r for r in rows if r["cal_resid"] >= 0.02]
    if bad:
        print(f"\n**{len(bad)} 行标定残差 >= 0.02，那些行不可用**")
    print("\n口径声明：这是**诊断**，不是把主检验换掉。"
          "采用哪个零分布是 CLAUDE.md §9 的人类决策点。")
    p_ = os.path.join(ROOT, "cache", "lambda2_famnull.json")
    json.dump({"obs_lam1": l1, "obs_lam2": l2, "ndraw": NDRAW,
               "a_w": AW, "b_w_cov": BW_COV, "rows": rows},
              open(p_, "w"), ensure_ascii=False, indent=1, default=float)
    print("产物:", p_)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
