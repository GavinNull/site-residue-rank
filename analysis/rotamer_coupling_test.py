#!/usr/bin/env python3
"""Rotamer 描述子 vs ΔΔG 的未解释耦合。

协议与 analysis/coupling_structure.py 相同（652 个观测，功效足够）：
双向中心化取出纯耦合，零分布用位点内残基标签置换。

候选：
  S_rot = ln(n_allowed)   rotamer 近似下的构象熵
  f_hb  = n_hb/n_allowed  位阻允许的构象里能成氢键的比例
  n_hb  本身
两者都是**不可分解的二维量**。
"""
import csv
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rotamer_descriptors as RD
from coupling_structure import double_center, masked_corr
from run_real_test import load_series, ROOT
from site_lfer_test import load_descriptors
from pairwise_shape_matched import add_true_volume

AA1 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
       "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
       "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
       "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}


def main():
    desc = add_true_volume(load_descriptors())
    res_list = sorted(desc)
    series = load_series(os.path.join(ROOT, "single_site_ddG_series.csv"))
    chain_of = {}
    for r in csv.DictReader(open(os.path.join(
            ROOT, "single_site_ddG_series.csv"))):
        chain_of[(r["system"], r["pdb"], r["position"],
                  r.get("pmid", ""))] = r["chain"]

    cache_path = os.path.join(ROOT, "cache", "rotamer_matrix.json")
    if os.path.exists(cache_path):
        blob = json.load(open(cache_path))
        sites = [tuple(k) for k in blob["sites"]]
        D = {tuple(k): v for k, v in zip(blob["sites"], blob["data"])}
        print(f"从 cache 载入 {len(sites)} 位点")
    else:
        sites, D = [], {}
        for i, k in enumerate(sorted(series)):
            ch = chain_of.get(k)
            if not ch:
                continue
            pid = re.match(r"^([0-9A-Za-z]{4})", k[1]).group(1).upper()
            pos = (re.sub(r"^[A-Za-z]", "", k[2])
                   if re.match(r"^[A-Za-z]\d", k[2]) else k[2])
            d = RD.descriptors_for_site(pid, ch, pos, res_list)
            if d is None:
                continue
            sites.append(k)
            D[k] = d
            print(f"  [{len(sites)}] {pid}/{pos} 完成", flush=True)
        json.dump({"sites": [list(s) for s in sites],
                   "data": [D[s] for s in sites]}, open(cache_path, "w"))
        print(f"写入 cache/rotamer_matrix.json（{len(sites)} 位点）")

    M = np.array([[dict(series[k][0]).get(r, np.nan) for r in res_list]
                  for k in sites])
    mask = np.isfinite(M)
    print(f"\nΔΔG 矩阵 {M.shape}，填充 {mask.mean():.0%}（{mask.sum()} 观测）")
    Rm = double_center(np.where(mask, M, 0.0), mask)
    tot = M[mask].std(); coup = Rm[mask].std()
    print(f"耦合占总方差 {(coup / tot) ** 2:.1%}，耦合 SD {coup:.3f} kcal/mol")

    cands = {}
    for name, key, tf in (("S_rot = ln(n_allowed)", "n_allowed",
                           lambda x: math.log(x) if x > 0 else np.nan),
                          ("f_hb = n_hb/n_allowed", None, None),
                          ("n_hb (氢键构象数)", "n_hb",
                           lambda x: math.log1p(x))):
        A = np.full(M.shape, np.nan)
        for i, k in enumerate(sites):
            for j, r in enumerate(res_list):
                v = D[k].get(r)
                if v is None:
                    continue
                if key is None:
                    A[i, j] = v["f_hb"] if v["n_allowed"] else np.nan
                else:
                    A[i, j] = tf(v[key])
        cands[name] = A

    print("\n[候选 vs ΔΔG 耦合残差]")
    print(f"{'候选':<26} {'可用':>6} {'r':>8} {'r²':>7} {'置换 p':>9}")
    print("-" * 62)
    rng = np.random.default_rng(20261008)
    for name, A in cands.items():
        m2 = mask & np.isfinite(A)
        if m2.sum() < 100:
            print(f"{name:<26} {m2.sum():>6}  可用观测不足")
            continue
        Ac = double_center(np.where(m2, np.nan_to_num(A), 0.0), m2)
        r = masked_corr(Rm, Ac, m2)
        null = []
        for _ in range(2000):
            Ap = A.copy()
            for i in range(Ap.shape[0]):
                idx = np.where(m2[i])[0]
                if len(idx) > 1:
                    Ap[i, idx] = Ap[i, rng.permutation(idx)]
            Apc = double_center(np.where(m2, np.nan_to_num(Ap), 0.0), m2)
            v = masked_corr(Rm, Apc, m2)
            if math.isfinite(v):
                null.append(abs(v))
        p = float(np.mean([x >= abs(r) for x in null])) if null else float("nan")
        star = " *" if p < 0.05 else ""
        print(f"{name:<26} {m2.sum():>6} {r:>+8.3f} {r ** 2:>7.1%} "
              f"{p:>9.4f}{star}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
