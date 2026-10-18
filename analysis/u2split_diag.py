#!/usr/bin/env python3
"""docs/45 事后诊断（§4.3 与 §7）。不进判决。
D1：秩 1 世界 σ_meas 端的 cos θ₂ 虚高来自什么（截断 / 主效应 / 缺失），WH1/hamster 与 BA2/deer。
D2：每个矩阵的残基主效应方差有多少落在全数据二维残基子空间内（EM 抽取）。
输出 cache/u2split/diag.json。"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import u2split_lib as U, u2split_run as RUN


def d1(g, nz, n, variant, reps=20):
    G = g[n]
    pt = nz[n]["path_k1"][0]
    out = {}
    for ext in U.EXTRACTORS:
        vals = []
        for rep in range(reps):
            rng = np.random.default_rng(1000 + rep)
            I = G._scale_to(G.interaction(1, "real", rng), pt["s"])
            eps = rng.normal(0, pt["sigma"], (G.nr, G.nc))
            r, c = G.r[:, None], G.c[None, :]
            mask = G.mask
            if variant == "base":
                Y = np.clip(r + c + I + eps, G.lo, G.hi)
            elif variant == "noclip":
                Y = r + c + I + eps
            elif variant == "nomain":
                Y = I + eps
            elif variant == "full":
                Y = np.clip(r + c + I + eps, G.lo, G.hi); mask = np.ones_like(G.mask)
            elif variant == "noclip_nomain_full":
                Y = I + eps; mask = np.ones_like(G.mask)
            Y = np.where(mask, Y, 0.0)
            c_ = U.many_splits(Y, mask, ext, 6, U.seed_of("diag", n, variant, rep, ext))
            vals.append(float(np.nanmedian(c_[:, 1])))
        out[ext] = float(np.median(vals))
    return out


if __name__ == "__main__":
    U.verify_lock()
    g, nz = RUN.load_geoms()
    res = {"D1": {}, "D2": {}}
    for n in ("animalACE2/Wuhan-Hu-1/hamster", "animalACE2/Omicron_BA2/deer"):
        res["D1"][n] = {v: d1(g, nz, n, v) for v in ("base", "noclip", "nomain", "full", "noclip_nomain_full")}
        print(n, {k: {a: round(b, 2) for a, b in v.items()} for k, v in res["D1"][n].items()}, flush=True)
    for n in RUN.SCOPE:
        G = g[n]
        cc = G.c - G.c.mean()
        B, cols = U.extract_em(G.M, G.mask)
        Q, _ = np.linalg.qr(B - B.mean(0))
        v = cc[cols] - cc[cols].mean()
        res["D2"][n] = {"span": float(np.sum((Q.T @ v) ** 2) / np.sum(v ** 2)),
                        "axis1": float((Q[:, 0] @ v) ** 2 / np.sum(v ** 2)),
                        "axis2": float((Q[:, 1] @ v) ** 2 / np.sum(v ** 2)), "random_2d": 2.0 / (len(cols) - 1)}
        print(n.split('/',1)[1][:24].ljust(24), {a: round(b, 2) for a, b in res["D2"][n].items()}, flush=True)
    json.dump(res, open(os.path.join(RUN.OUT, "diag.json"), "w"), indent=1)
