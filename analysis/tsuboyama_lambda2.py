#!/usr/bin/env python3
"""docs/27 §6.2：λ2 的条件检验在 k = 90 上**定不住大小**，所以必须降 k 重做。

发生了什么（照 §10 留痕）
----------------------------------------------------------------
`tsuboyama_crossdomain.py` 第一版在 λ2 的负对照上**断言失败并中止**：
合成「只共享一维」的构造给 λ2 = 0.117，而条件零分布检验给 p = **0.018**
—— 把一个只共享一维的构造判成「第二维也共享」。
加上 k 扫描之后发现：同一个构造换一个种子（101 而不是 77）在 k = 90 上给
p = **0.110**。两个种子一个误报一个不误报，**所以这不是「检验坏了」，
是「检验在 k = 90 上的大小定不住」**。

机制：k 越大，平均投影算子越集中，「只共享一维」的 λ2 越小
（k = 6 → 0.297、k = 90 → 0.111），而条件零分布的 λ2 上尾也同步下移，
两者贴到一起，p 值就由单次实现的抖动决定。
小 k 上余量很大（p = 0.48–0.95），大 k 上余量消失。

本脚本做两件事：
  (1) **把不稳定量化**：在 k = 90 与若干更小的 k 上，各用 **24 个独立种子**
      重跑负对照，报 p 的分布与「误报率」。误报率就是该 k 上检验的**实际大小**。
  (2) **在实际大小可控的 k 上重做观测 λ2**：从 90 个簇里不放回抽 k 个，
      重复多次，报 λ2 与条件 p 的分布、标定残差。

判据（在看结果之前写定）：取**最大的**那个 k，使负对照的误报率 ≤ 0.10
（即检验的实际大小不超过名义 0.05 的两倍）。只在那个 k 上读 λ2 的 p。

运行：python3 analysis/tsuboyama_lambda2.py
产物：cache/tsuboyama_lambda2.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subspace_cond_null as CN                                 # noqa: E402
from subspace_lib import ROOT                                   # noqa: E402

KS = [9, 15, 25, 40, 60, 90]
N_SEED = 24
NSIM = 1500
N_SUB = 40


def one_dim_only(k, uf, rng):
    """合成：k 个子空间，第一方向**精确**是 uf，第二方向随机（只共享一维）。"""
    Ps = []
    u1 = uf / np.linalg.norm(uf)
    for _ in range(k):
        a2 = CN.rand_in_complement(u1, rng)
        a2 -= u1 * (u1 @ a2)
        a2 /= np.linalg.norm(a2)
        Q, _ = np.linalg.qr(np.stack([u1, a2], 1))
        Ps.append(Q @ Q.T)
    return np.linalg.eigvalsh(np.mean(Ps, axis=0))[::-1]


def main():
    CD = json.load(open(os.path.join(ROOT, "cache",
                                     "tsuboyama_crossdomain.json")))
    uf = np.array(CD["u_fold"], float)
    obs_l1, obs_l2 = CD["eigs"][0], CD["eigs"][1]
    print("=" * 102)
    print("docs/27 §6.2：λ2 条件检验的**实际大小**，以及在可控 k 上重做")
    print("=" * 102)
    print(f"   观测（k = {CD['n_clusters']} 个簇，簇内平均 + 簇等权）："
          f"λ1 = {obs_l1:.3f}、λ2 = **{obs_l2:.3f}**")

    # ---- (1) 负对照的实际大小 ---------------------------------------
    print("\n[1] 负对照（只共享一维，真值 = 「第二维不共享」）"
          f"在每个 k 上用 {N_SEED} 个独立种子重跑")
    print(f"   {'k':>4}{'合成 λ2 中位':>13}{'p 中位':>9}{'p 的 10–90%':>18}"
          f"{'误报率 p<0.05':>14}{'判据 ≤0.10':>11}")
    size = []
    for k in KS:
        ps, l2s = [], []
        for s in range(N_SEED):
            w = one_dim_only(k, uf, np.random.default_rng(1000 + 37 * s))
            cn, al = CN.cond_null(k, uf, np.random.default_rng(5000 + s),
                                  nsim=NSIM, target_l1=float(w[0]))
            ps.append(float(np.mean(cn[:, 1] >= w[1])))
            l2s.append(float(w[1]))
        ps = np.array(ps)
        fp = float(np.mean(ps < 0.05))
        size.append({"k": k, "lam2_median": float(np.median(l2s)),
                     "p_median": float(np.median(ps)),
                     "p_decile": [float(np.percentile(ps, 10)),
                                  float(np.percentile(ps, 90))],
                     "false_pos_rate": fp, "ok": bool(fp <= 0.10),
                     "n_seed": N_SEED})
        print(f"   {k:>4}{np.median(l2s):>13.3f}{np.median(ps):>9.3f}"
              f"{'[%.3f, %.3f]' % (np.percentile(ps, 10), np.percentile(ps, 90)):>18}"
              f"{fp:>14.2f}{'OK' if fp <= 0.10 else '**超**':>11}")
    good = [z["k"] for z in size if z["ok"]]
    kuse = max(good) if good else None
    print(f"   => 实际大小 ≤ 0.10 的 k：{good}；**取最大的 k = {kuse}** 读 λ2")

    # ---- (2) 在 kuse 上重做观测 λ2 ----------------------------------
    info = json.load(open(os.path.join(ROOT, "cache",
                                       "tsuboyama_axis.json")))
    import csv

    import tsuboyama_load as TL
    from subspace_lib import extract_subspace
    _, mats = TL.load_cached()
    cl = {}
    for x in csv.DictReader(open(os.path.join(
            ROOT, "cache", "dms", "tsuboyama2023", "Single_DMS_list.csv"))):
        cl[x["name"]] = x.get("WT_cluster", "")
    names = [n for n in info["repro_pass"]]
    P, groups = {}, defaultdict(list)
    for n in names:
        B, cols, _ = extract_subspace(mats[n]["M"], mats[n]["mask"], rank=2)
        if B is None or len(cols) != 20:
            continue
        P[n] = CN.proj(B)[0]
        groups[cl.get(n, "?")].append(n)
    allc = sorted(groups)
    print(f"\n[2] 在 k = {kuse} 上重做观测 λ2："
          f"从 {len(allc)} 个簇里不放回抽 {kuse} 个，重复 {N_SUB} 次")
    rows = []
    rngs = np.random.default_rng(13)
    for t in range(N_SUB):
        pick = rngs.choice(len(allc), kuse, replace=False)
        Pb = np.mean([np.mean([P[n] for n in groups[allc[i]]], axis=0)
                      for i in pick], axis=0)
        ww, VV = np.linalg.eigh(Pb)
        ww = ww[::-1]; uu = VV[:, ::-1][:, 0]
        cn, al = CN.cond_null(kuse, uu, np.random.default_rng(900 + t),
                              nsim=NSIM, target_l1=float(ww[0]))
        rows.append({"lam1": float(ww[0]), "lam2": float(ww[1]),
                     "lam3": float(ww[2]),
                     "p": float(np.mean(cn[:, 1] >= ww[1])),
                     "cal_resid": abs(float(np.median(cn[:, 0]))
                                      - float(ww[0])),
                     "alpha_deg": float(np.degrees(al)),
                     "abscos_u": abs(float(uu @ (uf / np.linalg.norm(uf))))})
    l1 = np.array([z["lam1"] for z in rows])
    l2 = np.array([z["lam2"] for z in rows])
    pp = np.array([z["p"] for z in rows])
    cr = np.array([z["cal_resid"] for z in rows])
    ac = np.array([z["abscos_u"] for z in rows])
    print(f"   λ1 中位 {np.median(l1):.3f}、λ2 中位 **{np.median(l2):.3f}**"
          f"（10–90% [{np.percentile(l2, 10):.3f}, "
          f"{np.percentile(l2, 90):.3f}]）")
    print(f"   λ2 的条件 p：中位 **{np.median(pp):.4f}**、"
          f"10–90% [{np.percentile(pp, 10):.4f}, "
          f"{np.percentile(pp, 90):.4f}]；p < 0.05 的比例 "
          f"**{np.mean(pp < 0.05):.0%}**")
    print(f"   标定残差中位 {np.median(cr):.4f}（docs/21 的四组 ≤ 0.001）")
    print(f"   子抽样得到的 u 与全集 u_fold 的 |cos|：最小 {ac.min():.4f}、"
          f"中位 {np.median(ac):.4f}")
    sep = np.median(l2) / max(1e-9, [z["lam2_median"] for z in size
                                     if z["k"] == kuse][0])
    print(f"\n   **判读**：在实际大小可控的 k = {kuse} 上，"
          f"观测 λ2 = {np.median(l2):.3f} 而「只共享一维」的合成 λ2 = "
          f"{[z['lam2_median'] for z in size if z['k'] == kuse][0]:.3f}"
          f"（相差 {sep:.1f} 倍），条件 p 在 {np.mean(pp < 0.05):.0%} 的"
          f"子抽样上 < 0.05。")

    out = {"observed_k90": {"lam1": obs_l1, "lam2": obs_l2,
                            "n_clusters": CD["n_clusters"]},
           "size_scan": size, "k_used": kuse,
           "subsample": rows,
           "sub_lam1_median": float(np.median(l1)),
           "sub_lam2_median": float(np.median(l2)),
           "sub_p_median": float(np.median(pp)),
           "sub_p_decile": [float(np.percentile(pp, 10)),
                            float(np.percentile(pp, 90))],
           "sub_frac_p_lt_05": float(np.mean(pp < 0.05)),
           "sub_cal_resid_median": float(np.median(cr)),
           "sub_abscos_u_min": float(ac.min())}
    p = os.path.join(ROOT, "cache", "tsuboyama_lambda2.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
