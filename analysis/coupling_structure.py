#!/usr/bin/env python3
"""不分层，直接对耦合建模。

### 为什么换做法

此前用分层去逼近「两者都极性 + 形状匹配」那一档，把 55 个位点切到 8 个，
功效 < 0.09。但那一档的科学问题可以在**全数据**上问：

    ΔΔG(位点, 残基) = a(残基) + b(位点) + 耦合(位点, 残基)

去溶剂化项是 a(残基)·burial(位点)，一个**秩 1** 耦合。
真正的问题是：去掉主效应之后残余耦合有多少，什么能解释它。

这是嵌套检验，用全部位点，功效远高于分层。
且它正是 docs/05 M1 的双线性框架 —— 我当初建了它却没用上。

### 步骤

1. 建 (位点 × 残基) 的 ΔΔG 矩阵（不完整）
2. 双向中心化 -> 纯耦合残差。**耦合占总方差的比例就是任何耦合模型的天花板**
3. 候选耦合矩阵：
   a. 疏水性(残基) × burial(位点)        —— 外积，秩 1
   b. 氢键可达性(位点, 残基)              —— 真二维
   c. 电荷(残基) × 局部场(位点)           —— 外积，秩 1
4. 每个对 ΔΔG 耦合做相关，零分布用**位点内残基标签置换**（保持边缘）
"""
import collections
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_orientation import parse_pdb, sub, norm, unit, dot, ROOT
from hbond_reach import hb_reach, REACH
from pairwise_shape_matched import add_true_volume
from site_lfer_test import load_descriptors
from run_real_test import load_series


def burial_at_site(pdb_key, chain, resi, radius=8.0):
    """位点埋藏度：Cβ 周围 radius 内的重原子数（含伙伴链与配体）。

    这是最少假设的埋藏代理：数邻居，不依赖溶剂可及面积算法。
    """
    import re
    m = re.match(r"^([0-9A-Za-z]{4})", pdb_key)
    pid = m.group(1).upper()
    path = os.path.join(ROOT, "cache", "pdb", f"{pid}.pdb")
    if not os.path.exists(path):
        return None
    atoms = parse_pdb(path)
    cb = None
    for a in atoms:
        if (a["chain"] == chain and a["resi"] == str(resi) and not a["het"]
                and a["name"] == "CB"):
            cb = a["xyz"]
    if cb is None:
        return None
    n = 0
    for a in atoms:
        if a["res"] == "HOH":
            continue
        if a["chain"] == chain and a["resi"] == str(resi):
            continue
        if norm(sub(a["xyz"], cb)) <= radius:
            n += 1
    return float(n)


def local_field_mag(pdb_key, chain, resi):
    from field_orientation import analyse
    g = analyse(pdb_key, chain, resi)
    return g["E"] if g else None


def double_center(M, mask, iters=60):
    """缺失值下的迭代双向中心化。返回残差（仅观测位置有意义）。"""
    r = np.zeros(M.shape[0]); c = np.zeros(M.shape[1])
    for _ in range(iters):
        R = M - r[:, None] - c[None, :]
        for i in range(M.shape[0]):
            if mask[i].any():
                r[i] += R[i][mask[i]].mean()
        R = M - r[:, None] - c[None, :]
        for j in range(M.shape[1]):
            if mask[:, j].any():
                c[j] += R[:, j][mask[:, j]].mean()
    return M - r[:, None] - c[None, :]


def masked_corr(A, B, mask):
    a = A[mask]; b = B[mask]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    desc = add_true_volume(load_descriptors())
    series = load_series(os.path.join(ROOT, "single_site_ddG_series.csv"))
    chain_of = {}
    for r in csv.DictReader(open(os.path.join(
            ROOT, "single_site_ddG_series.csv"))):
        chain_of[(r["system"], r["pdb"], r["position"],
                  r.get("pmid", ""))] = r["chain"]

    res_list = sorted(desc)
    rows, sites = [], []
    meta = {}
    for k in sorted(series):
        ch = chain_of.get(k)
        if not ch:
            continue
        d = dict(series[k][0])
        bu = burial_at_site(k[1], ch, k[2])
        fe = local_field_mag(k[1], ch, k[2])
        if bu is None or fe is None:
            continue
        hb = {}
        ok = True
        for rr in res_list:
            s = hb_reach(k[1], ch, k[2], rr)
            if s is None:
                ok = False
                break
            hb[rr] = s
        if not ok:
            continue
        sites.append(k)
        meta[k] = {"burial": bu, "field": fe, "hb": hb}
        rows.append([d.get(rr, np.nan) for rr in res_list])

    M = np.array(rows)
    mask = np.isfinite(M)
    print("=" * 78)
    print("耦合结构分析（不分层，用全部位点）")
    print("=" * 78)
    print(f"ΔΔG 矩阵: {M.shape[0]} 位点 x {M.shape[1]} 残基，"
          f"填充 {mask.mean():.0%}（{mask.sum()} 个观测）")

    M0 = np.where(mask, M, 0.0)
    Rm = double_center(M0, mask)
    tot = M[mask].std()
    coup = Rm[mask].std()
    print(f"\n[天花板] ΔΔG 总标准差 {tot:.3f}，去主效应后残差 {coup:.3f}")
    print(f"         => 耦合项占总方差 {(coup / tot) ** 2:.1%}")
    print(f"         主效应（残基 + 位点）占 {1 - (coup / tot) ** 2:.1%}")
    print("         **任何耦合模型最多只能解释这 {:.0%}**".format(
        (coup / tot) ** 2))

    bur = np.array([meta[k]["burial"] for k in sites])
    fld = np.array([meta[k]["field"] for k in sites])
    print(f"\n位点埋藏度（Cβ 8 Å 内重原子数）: "
          f"{bur.min():.0f}..{bur.max():.0f}，中位 {np.median(bur):.0f}")

    cands = {}
    pi = np.array([desc[r]["pi"] for r in res_list])
    q = np.array([desc[r]["q"] for r in res_list])
    mu = np.array([desc[r]["mu"] for r in res_list])
    cands["疏水 x 埋藏 (秩1)"] = np.outer(bur, pi)
    cands["电荷 x 场强 (秩1)"] = np.outer(fld, q)
    cands["偶极 x 场强 (秩1)"] = np.outer(fld, mu)
    H = np.array([[meta[k]["hb"][r] for r in res_list] for k in sites])
    cands["氢键可达性 (真二维)"] = np.log1p(H)

    print("\n[候选耦合矩阵 vs ΔΔG 耦合残差]")
    print("每个候选也做双向中心化（否则它的主效应会和 ΔΔG 主效应混淆）")
    print(f"\n{'候选':<24} {'r':>8} {'r²':>7} {'置换 p':>9}")
    print("-" * 54)
    rng = np.random.default_rng(20261007)
    for name, C in cands.items():
        Cc = double_center(np.where(mask, C, 0.0), mask)
        r = masked_corr(Rm, Cc, mask)
        if not math.isfinite(r):
            continue
        # 零分布：位点内置换残基标签（保持位点边缘）
        null = []
        for _ in range(2000):
            Cp = C.copy()
            for i in range(Cp.shape[0]):
                idx = np.where(mask[i])[0]
                if len(idx) > 1:
                    Cp[i, idx] = Cp[i, rng.permutation(idx)]
            Cpc = double_center(np.where(mask, Cp, 0.0), mask)
            v = masked_corr(Rm, Cpc, mask)
            if math.isfinite(v):
                null.append(abs(v))
        p = float(np.mean([x >= abs(r) for x in null])) if null else float("nan")
        star = " *" if p < 0.05 else ""
        print(f"{name:<24} {r:>+8.3f} {r ** 2:>7.1%} {p:>9.4f}{star}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
