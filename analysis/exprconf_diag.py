#!/usr/bin/env python3
"""docs/47 修订 A1 的诊断：表达量交互层残差的重复可靠度，与秩 2 拟合结构占其方差的比例。"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import exprconf_lib as E
pairs = E.load_all(); Ed = E._GEOMS["Ed"]
out = {}
for src in ("WH1_v1", "WH1_v2", "Eta", "BA2", "PDCoV"):
    e = Ed["expr"][src]
    K = e["k1"] & e["k2"]
    R1, R2 = E.dcK(e["M1"], K), E.dcK(e["M2"], K)
    rho = float(np.corrcoef(R1[K], R2[K])[0, 1])
    Re = E.dcK(e["M"], e["mask"]); Pe = e["geom"].fit[2]["P"]
    out[src] = {"rep_corr_dc": rho, "reliability_avg": 2 * rho / (1 + rho), "rank2_fit_share_of_var": float(Pe[e["mask"]].var() / Re[e["mask"]].var())}
    print(src, out[src])
json.dump(out, open(os.path.join(E.OUT, "diag.json"), "w"), indent=1)
