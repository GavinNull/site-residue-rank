#!/usr/bin/env python3
"""提取耦合的**径向轮廓**，而不是固定一个两壳比值。

动机：ρ = free(1.5-2.5)/free(4.5-6.0) 同时配疏水性(p<1e-5)与平面性(p=2e-4)，
而那两个壳是我随手选的。正确做法是让数据给出轮廓：

    ΔΔG_耦合 ≈ Σ_k c_k^X · X(残基) · f_S(壳 k)

对残基描述子 X ∈ {疏水 π, 平面 P}，各壳系数 c_k^X 的轮廓就是该项的
径向结构。两个轮廓的差异本身就是新结构，且比单一比值信息多。

统计：整个模型（5 系数）对位点内置换零分布检验；轮廓形状为描述性。
"""
import csv
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coupling_structure import double_center, burial_at_site
from pairwise_shape_matched import add_true_volume
from run_real_test import load_series, ROOT
from site_lfer_test import load_descriptors

SHELLS = [(1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 6.0), (6.0, 8.0)]
PLANAR = {"PHE", "TYR", "TRP", "HIS", "ASP", "GLU", "ASN", "GLN"}


def load_all():
    desc = add_true_volume(load_descriptors())
    res_list = sorted(desc)
    series = load_series(os.path.join(ROOT, "single_site_ddG_series.csv"))
    chain_of = {}
    for r in csv.DictReader(open(os.path.join(
            ROOT, "single_site_ddG_series.csv"))):
        chain_of[(r["system"], r["pdb"], r["position"],
                  r.get("pmid", ""))] = r["chain"]
    sites = [k for k in sorted(series) if chain_of.get(k)]
    M = np.array([[dict(series[k][0]).get(r, np.nan) for r in res_list]
                  for k in sites])
    mask = np.isfinite(M)
    Rm = double_center(np.where(mask, M, 0.0), mask)
    SH = np.load(os.path.join(ROOT, "cache", "shell_free.npy"))
    bur = np.array([burial_at_site(k[1], chain_of[k], k[2]) or np.nan
                    for k in sites])
    return desc, res_list, sites, M, mask, Rm, SH, bur


def fit_profile(Rm, mask, SH, xres, nperm=4000, seed=17):
    """对单个残基描述子 x，拟合各壳系数 c_k，并给整体置换 p。"""
    good = np.where(np.isfinite(SH).all(axis=1))[0]
    mk = mask[good]
    Rg = Rm[good]
    cols = []
    for k in range(len(SHELLS)):
        C = np.outer(SH[good, k], xres)
        cols.append(double_center(np.where(mk, C, 0.0), mk)[mk])
    X = np.column_stack([np.ones(mk.sum())] + cols)
    y = Rg[mk]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(nperm):
        xp = rng.permutation(xres)
        cs = []
        for k in range(len(SHELLS)):
            C = np.outer(SH[good, k], xp)
            cs.append(double_center(np.where(mk, C, 0.0), mk)[mk])
        Xp = np.column_stack([np.ones(mk.sum())] + cs)
        bb, *_ = np.linalg.lstsq(Xp, y, rcond=None)
        ee = y - Xp @ bb
        null.append(1 - (ee @ ee) / ((y - y.mean()) @ (y - y.mean())))
    p = float(np.mean([v >= r2 for v in null]))
    return b[1:], r2, p, mk.sum()


def main():
    desc, res_list, sites, M, mask, Rm, SH, bur = load_all()
    pi = np.array([desc[r]["pi"] for r in res_list])
    P = np.array([1.0 if r in PLANAR else 0.0 for r in res_list])
    print("=" * 74)
    print("耦合的径向轮廓：各壳自由体积比例 × 残基描述子")
    print("=" * 74)
    print(f"{'残基描述子':<12} " + "".join(f"{f'{lo}-{hi}':>11}" for lo, hi in SHELLS)
          + f"{'R²':>8} {'p':>8}")
    print("-" * 74)
    out = {}
    for name, x in (("疏水 π", pi), ("平面 P", P)):
        c, r2, p, n = fit_profile(Rm, mask, SH, x)
        out[name] = c
        print(f"{name:<12} " + "".join(f"{v:>11.2f}" for v in c)
              + f"{r2:>8.2%} {p:>8.4f}")
    print(f"\n（n = {n} 观测；p 为整个 5 系数模型对残基标签置换的零分布）")
    print("\n轮廓解读：系数符号在哪个壳翻转，就是该项的几何定位。")
    for name, c in out.items():
        s = "".join("+" if v > 0 else "−" for v in c)
        print(f"   {name}: 符号序列 {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
