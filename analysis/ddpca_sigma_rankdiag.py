#!/usr/bin/env python3
"""docs/28 §4.4：为什么反解锚点 C 在 ddPCA 上落到了体外上界 B **之外**。

`ddpca_sigma.backsolve_C` 照 docs/27 的做法反解 σ：扫 σ，取使**合成秩 1**
矩阵的位点对半可复现度 == 实测可复现度的那个 σ。
在 ddPCA 的六个矩阵上这个反解一致地给出 σ_C > σ_B（体外总 RMSE），
于是预先写定的判据 **A ≤ C ≤ B 不通过**。

A ≤ C ≤ B 不通过有两种完全不同的读法，必须分开（D-014 的同型要求）：
  (i) σ 真的比体外 RMSE 还大 —— 但体外 RMSE 本身已经是**上界**
      （它含文献误差与条件差异），所以这一读法自相矛盾；
  (ii) **反解的生成模型错了**：它假设真结构是秩 1，而一个秩更高 / 更弥散的
      真结构在同样的 σ 下，其「第一轴」本来就没有秩 1 那么稳定。

本脚本直接测 (ii)：在 σ = σ_B（与 σ_B'）下，把真结构的秩从 1 换到 2、3，
看哪个秩的合成可复现度落在实测附近。这把「判据不过」从一句否定变成
一个可检的陈述。

数值层复用 `generic_folding_axis.synth_like`（rank 参数本来就有）、
`gate_calib_lib.split_half`。改动为零。
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddpca_load as DL                                        # noqa: E402
import ddpca_sigma as S                                        # noqa: E402
import gate_calib_lib as G                                     # noqa: E402
import rank_core as RC                                         # noqa: E402
from generic_folding_axis import synth_like                    # noqa: E402
from subspace_lib import ROOT, haar_1d_null                    # noqa: E402

OUTJSON = os.path.join(ROOT, "cache", "ddpca_sigma_rankdiag.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrep", type=int, default=12)
    ap.add_argument("--nsplit", type=int, default=120)
    a = ap.parse_args()
    pd = S.predecl()
    SB = pd["anchors"]["B_upper"]["values"]
    sig = json.load(open(os.path.join(ROOT, "cache", "ddpca_sigma.json"))) \
        if os.path.exists(os.path.join(ROOT, "cache", "ddpca_sigma.json")) \
        else None
    bp = {}
    if sig:
        for r in sig["rows"]:
            bp[r["name"]] = r["sigma"]["Bp"] if "sigma" in r else \
                r["conventions"]["docs15"]["sigma"]["Bp"]

    print("=" * 96)
    print("docs/28 §4.4：反解锚点 C 越过上界 B 的归因 —— 是秩假设，不是 σ")
    print("=" * 96)
    rows = []
    for dom, ph in S.MATS:
        D = DL.load_domain(dom)
        DL.verify(dom, D, verbose=False)
        M, mask, cen = DL.matrix(D, ph)
        name = f"{dom}/{ph}"
        r, c = RC.center_fit(M, mask)
        R = M - r[:, None] - c[None, :]
        coup = float(R[mask].std())
        mr = float((r - r.mean()).std())
        mc = float((c - c.mean()).std())
        obs = G.split_half(M, mask, rank=1, nsplit=a.nsplit)
        om = float(np.median(obs))
        d = M.shape[1]
        p95 = float(np.percentile(haar_1d_null(d, n=20000, seed=3), 95))
        rec = {"name": name, "coup_sd_obs": coup, "obs_repro": om,
               "haar1d_p95": p95, "cells": {}}
        print(f"\n[{name}] coup_sd_obs {coup:.3f}，实测位点对半可复现 "
              f"{om:.3f}（Haar p95 {p95:.3f}）")
        for tag, sg in (("B", SB[name]), ("Bp", bp.get(name))):
            if sg is None:
                continue
            ct = math.sqrt(max(coup ** 2 - sg ** 2, 0.0))
            if ct <= 1e-9:
                print(f"   σ_{tag} = {sg:.3f}: coup_sd_true = 0，"
                      f"没有可植入的结构 —— 这一格无法做秩归因")
                rec["cells"][tag] = {"sigma": sg, "coup_sd_true": 0.0,
                                     "why": "coup_sd_true = 0"}
                continue
            byrank = {}
            for rk in (1, 2, 3):
                vals = []
                for t in range(a.nrep):
                    rng = np.random.default_rng(555 + 131 * t + rk * 7
                                                + int(sg * 1e4))
                    X = synth_like(mask, S._rand_dir(rng, d), ct, sg,
                                   mr, mc, rng, rank=rk)
                    v = G.split_half(np.where(mask, X, np.nan), mask,
                                     rank=1, nsplit=a.nsplit, seed=7 + t)
                    if v:
                        vals.append(float(np.median(v)))
                byrank[rk] = {"med": float(np.median(vals)),
                              "q10": float(np.percentile(vals, 10)),
                              "q90": float(np.percentile(vals, 90))}
            best = min(byrank, key=lambda k: abs(byrank[k]["med"] - om))
            inside = [k for k in byrank
                      if byrank[k]["q10"] <= om <= byrank[k]["q90"]]
            rec["cells"][tag] = {"sigma": sg, "coup_sd_true": ct,
                                 "by_rank": byrank, "best_rank": best,
                                 "ranks_covering_obs": inside}
            print(f"   σ_{tag} = {sg:.3f}（coup_sd_true {ct:.3f} = "
                  f"{ct / sg:.2f}σ）：合成可复现度 "
                  + "，".join(f"秩{k} {byrank[k]['med']:.3f} "
                              f"[{byrank[k]['q10']:.3f}, "
                              f"{byrank[k]['q90']:.3f}]"
                              for k in (1, 2, 3))
                  + f" -> 最接近实测的是 **秩 {best}**；"
                  + (f"实测落在秩 {inside} 的 10–90% 区间内"
                     if inside else "实测不落在任何秩的 10–90% 区间内"))
        rows.append(rec)
    json.dump({"rows": rows}, open(OUTJSON, "w"), ensure_ascii=False,
              indent=1, default=float)
    print("\n产物:", OUTJSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
