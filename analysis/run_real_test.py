#!/usr/bin/env python3
"""在真实 ΔΔG 数据上跑 MP-1 的主检验。

数据: single_site_ddG_series.csv（72 系列，60 个完整 19 残基扫描）
预处理（每一条都有独立依据，不是为了好看）：
  - 剔除 censored 行（Tite-Seq 的 1e-5 上限）
  - 同取代重复测量取均值（SKEMPI 同一取代有多篇来源）
  - PRO 已由描述子表排除（D-022：侧链与骨架成环）
  - ARG 已由描述子表排除（无可用实验偶极矩）
  - **GLY 保留**：实测 |z| 中位数 0.99 vs 其余残基最大 2.05，不是离群点
"""
import collections
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from site_lfer_test import (load_descriptors, evaluate_site, nested_test,  # noqa
                            design, loo_rmse, FEATURE_SETS)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AA1to3 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
          "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
          "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
          "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}


def load_series(path, min_n=12):
    rows = list(csv.DictReader(open(path)))
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    meta = {}
    for r in rows:
        if r.get("censored") or not r["ddG_kcal"]:
            continue
        k = (r["system"], r["pdb"], r["position"], r.get("pmid", ""))
        m = r["mut_res"].strip().upper()
        m = AA1to3.get(m, m) if len(m) == 1 else m
        g[k][m].append(float(r["ddG_kcal"]))
        meta[k] = {"ligand": r.get("ligand", ""), "metric": r.get("metric", ""),
                   "wt": r.get("wt_res", "")}
    out = {}
    for k, d in g.items():
        rows_k = [(m, float(np.mean(v))) for m, v in d.items()]
        if len(rows_k) >= min_n:
            out[k] = (rows_k, meta[k])
    return out


def main():
    desc = load_descriptors()
    series = load_series(os.path.join(ROOT, "single_site_ddG_series.csv"))
    print("=" * 86)
    print(f"MP-1 主检验（嵌套 C vs A）在真实数据上  —  {len(series)} 个系列")
    print("=" * 86)
    print("A = 疏水性 π + 偏比容；C = A + 静电通道 (Δq, Δμ, ΔΘ)")
    print("统计量 D = LOO_RMSE(A) − LOO_RMSE(C)，正值 = 加静电有增量")
    print("零分布 = 位点内标签置换。实测假阳性率 0.03–0.07（名义 0.05）\n")

    print(f"{'系列':<40} {'n':>3} {'跨度':>6} {'LOO_A':>7} {'LOO_C':>7} "
          f"{'D':>7} {'p':>6} {'显著':>5}")
    print("-" * 86)
    res = []
    for k in sorted(series):
        rows_k, m = series[k]
        nt = nested_test(rows_k, desc, n_perm=2000)
        if nt is None:
            continue
        span = max(v for _, v in rows_k) - min(v for _, v in rows_k)
        lbl = f"{k[0][:20]}/{k[1][:10]}/{k[2]}"
        print(f"{lbl:<40} {nt['n']:>3} {span:>6.1f} {nt['loo_A']:>7.3f} "
              f"{nt['loo_C']:>7.3f} {nt['D']:>+7.3f} {nt['p']:>6.3f} "
              f"{'是' if nt['significant'] else '':>5}")
        res.append((k, nt, span, m))

    nsig = sum(1 for _, nt, _, _ in res if nt["significant"])
    n = len(res)
    print("\n" + "=" * 86)
    print(f"显著的系列: {nsig}/{n} = {nsig / n:.1%}")
    print(f"若静电无增量，期望显著率 ≈ 5%（即 {0.05 * n:.1f} 个）")
    # 二项检验
    from math import comb
    pval = sum(comb(n, i) * 0.05 ** i * 0.95 ** (n - i)
               for i in range(nsig, n + 1))
    print(f"二项检验 P(>= {nsig} | p0=0.05) = {pval:.3e}")

    # 按配体类型分层
    print("\n按配体类型分层:")
    for tag, sel in (("小分子（荧光素）",
                      lambda m: "luores" in m["ligand"].lower()
                      or "fluor" in m["ligand"].lower()),
                     ("蛋白配体", lambda m: not ("fluor" in m["ligand"].lower()))):
        sub = [(nt, s) for _, nt, s, m in res if sel(m)]
        if not sub:
            continue
        w = sum(1 for nt, _ in sub if nt["significant"])
        print(f"  {tag:<16} {w}/{len(sub)}  中位 D = "
              f"{np.median([nt['D'] for nt, _ in sub]):+.3f}")

    # 按动态范围分层（跨度小的系列信噪比低）
    print("\n按 ΔΔG 跨度分层:")
    for lo, hi in ((0, 3), (3, 6), (6, 20)):
        sub = [(nt, s) for _, nt, s, _ in res if lo <= s < hi]
        if not sub:
            continue
        w = sum(1 for nt, _ in sub if nt["significant"])
        print(f"  跨度 {lo}-{hi} kcal/mol: {w}/{len(sub)}  中位 D = "
              f"{np.median([nt['D'] for nt, _ in sub]):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
