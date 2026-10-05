#!/usr/bin/env python3
"""回到原始规格：形状匹配的位点内残基对的 pairwise accuracy。

### 为什么重做

前两轮偏离了 `docs/00-problem.md` 的规格，两处：
  (a) **从未施加「形状相近」**。系列里混了 Gly->Trp 这类剧烈改形状的取代，
      它们由体积/疏水支配，会掩盖掉电子效应那一档。
  (b) 用了回归，而原始指定的指标是 **pairwise accuracy**。

### 本检验

对每个位点，取所有残基对 (i,j) 满足
  - |体积_i − 体积_j| < thr_vol      （形状匹配 = 原始问题的条件）
  - |ΔΔG_i − ΔΔG_j| > thr_ddg       （同源数据 sigma~0.4，成对噪声 ~0.57）

**零拟合预测规则**（每条都是单一方向的断言，没有自由参数）：
  dipole     偶极大者结合更紧      <- 用的是数据已确立的 c2 < 0 (31/38, P=5.8e-5)
  charge     带负电者结合更紧      <- 多数配体/界面带正电互补，方向待检
  aromatic   芳香者结合更紧
  hydrophob  疏水性大者结合更紧    <- 常规竞争者
  volume     体积大者结合更紧      <- **内部对照**：已按体积匹配，应接近 50%

### 统计

位点内的对不独立，故主检验是**位点层面的符号检验**：
算每个位点的准确率，再检验这些准确率是否系统性高于 0.5。
同时报汇总准确率（给量级）与自助区间。
"""
import itertools
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from site_lfer_test import load_descriptors
from run_real_test import load_series, ROOT
from scale_redundancy import parse_aaindex, AA20

AA3 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
       "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
       "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
       "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}


def add_true_volume(desc):
    """加入**真正的分子体积**。

    早先用 COHE430101（偏比容, cm3/g）做形状匹配 —— 那不是体积。
    Trp 与 Gly 的偏比容分别 0.70 / 0.64，很接近，但一个巨大一个极小。
    按它匹配等于没匹配，而内部对照（volume 规则应接近 0.5）正是这样
    把它抓出来的：对照给出 0.585-0.679，p 低到 0.0002。

    改用 FAUJ880103（Fauchere 1988 归一化范德华体积，Gly=0，侧链增量形式）。
    已验：最小 = Gly，最大 = Trp。
    """
    aai = parse_aaindex(os.path.join(ROOT, "cache", "aaindex1.txt"),
                        {"FAUJ880103"})
    v = aai["FAUJ880103"]
    for aa in AA20:
        r3 = AA3[aa]
        if r3 in desc and math.isfinite(v[aa]):
            desc[r3]["vdwvol"] = v[aa]
    return {k: d for k, d in desc.items() if "vdwvol" in d}

RULES = {
    "dipole  偶极大者更紧": ("mu", -1),
    "charge  负电者更紧": ("q", +1),
    "aromatic 芳香者更紧": ("theta", -1),
    "hydrophob 疏水者更紧": ("pi", -1),
    "alpha   极化率大者更紧": ("alpha", -1),
    "vdw体积 大者更紧 [对照]": ("vdwvol", -1),
    "偏比容  大者更紧": ("vol", -1),
}


def pairs_for_site(rows, desc, thr_vol, thr_ddg):
    rs = [(r, v) for r, v in rows if r in desc]
    out = []
    for (ra, va), (rb, vb) in itertools.combinations(rs, 2):
        if abs(desc[ra]["vdwvol"] - desc[rb]["vdwvol"]) >= thr_vol:
            continue
        if abs(va - vb) <= thr_ddg:
            continue
        out.append((ra, va, rb, vb))
    return out


def accuracy(pairs, desc, feat, sgn):
    """sgn=-1: 描述子大者 ΔΔG 更小（结合更紧）。返回 (命中, 可判对数)。"""
    hit = tot = 0
    for ra, va, rb, vb in pairs:
        d = desc[ra][feat] - desc[rb][feat]
        if abs(d) < 1e-9:
            continue                      # 描述子无差别，不可判
        pred_a_tighter = (d > 0) if sgn < 0 else (d < 0)
        a_tighter = va < vb
        hit += (pred_a_tighter == a_tighter)
        tot += 1
    return hit, tot


def main():
    desc = add_true_volume(load_descriptors())
    series = load_series(os.path.join(ROOT, "single_site_ddG_series.csv"))
    print(f"可用残基 {len(desc)} 个；vdW 体积范围 "
          f"{min(d['vdwvol'] for d in desc.values()):.2f}.."
          f"{max(d['vdwvol'] for d in desc.values()):.2f}")
    print("=" * 84)
    print("形状匹配的位点内 pairwise accuracy（零拟合）")
    print("=" * 84)

    for thr_vol, thr_ddg in ((0.4, 0.8), (0.8, 0.8), (1.5, 0.8),
                             (0.8, 1.5)):
        per_site = {name: [] for name in RULES}
        pooled = {name: [0, 0] for name in RULES}
        nsite = npair = 0
        for k in sorted(series):
            rows, _ = series[k]
            prs = pairs_for_site(rows, desc, thr_vol, thr_ddg)
            if len(prs) < 5:
                continue
            nsite += 1
            npair += len(prs)
            for name, (feat, sgn) in RULES.items():
                h, t = accuracy(prs, desc, feat, sgn)
                if t >= 5:
                    per_site[name].append(h / t)
                    pooled[name][0] += h
                    pooled[name][1] += t
        print(f"\n--- |Δ体积| < {thr_vol}, |ΔΔG 差| > {thr_ddg} kcal/mol  "
              f"({nsite} 位点, {npair} 对) ---")
        print(f"{'规则':<26} {'汇总准确率':>10} {'可判对':>7} "
              f"{'位点中位':>9} {'>0.5 位点':>10} {'符号检验 p':>11}")
        print("-" * 84)
        rows_out = []
        for name in RULES:
            h, t = pooled[name]
            if t == 0:
                continue
            ps = per_site[name]
            above = sum(1 for x in ps if x > 0.5)
            ties = sum(1 for x in ps if abs(x - 0.5) < 1e-9)
            n_eff = len(ps) - ties
            from math import comb
            p = (sum(comb(n_eff, i) * 0.5 ** n_eff
                     for i in range(above, n_eff + 1)) if n_eff else float("nan"))
            star = " *" if p < 0.05 else ""
            print(f"{name:<26} {h / t:>10.3f} {t:>7} {np.median(ps):>9.3f} "
                  f"{above:>5}/{n_eff:<4} {p:>11.4f}{star}")
            rows_out.append((name, h / t, p))
    print("\n* = 位点层面符号检验 p < 0.05（位点内的对不独立，故以位点为单位）")
    print("vdw体积 那一行是内部对照：已按它匹配，它应当接近 0.5。")
    print("若它仍显著偏离 0.5，说明匹配带宽不够紧，其余结论都不可信。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
