#!/usr/bin/env python3
"""方向性检验：自由拟合的 c2 符号 vs 结构算出的场方向。

预测 sign(c2) = -sign(u_Cbeta . E_hat)。见 field_orientation.py 的推导。

只对 **c2 显著非零** 的系列检验符号 —— 对噪声系数测符号一致性
必然给 50%，那不是检验。
"""
import collections
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_real_test import load_series, ROOT
from site_lfer_test import load_descriptors, design, ols
from field_orientation import analyse

FEATS = ["pi", "vol", "q", "mu"]      # 不含 theta：它只是芳香指示变量


def fit_c2(rows, desc):
    X, y, _ = design(rows, FEATS, desc)
    n, p = X.shape
    if n < p + 3:
        return None
    b = ols(X, y)
    resid = y - X @ b
    s2 = float(resid @ resid) / (n - p)
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    j = FEATS.index("mu") + 1
    se = math.sqrt(max(cov[j, j], 0.0))
    return {"c2": float(b[j]), "se": se, "t": float(b[j] / se) if se else 0.0,
            "n": n}


def main():
    desc = load_descriptors()
    series = load_series(os.path.join(ROOT, "single_site_ddG_series.csv"))
    rows_out = []
    for k in sorted(series):
        sys_name, pdb_key, pos, pmid = k
        data, meta = series[k]
        fit = fit_c2(data, desc)
        if fit is None:
            continue
        # 突变链：从 CSV 的 chain 列（run_real_test 没带，这里重读）
        rows_out.append((k, fit))
    # 补 chain 信息
    import csv
    chain_of = {}
    for r in csv.DictReader(open(os.path.join(ROOT,
                                              "single_site_ddG_series.csv"))):
        chain_of[(r["system"], r["pdb"], r["position"], r.get("pmid", ""))] = \
            r["chain"]

    print("=" * 96)
    print("方向性检验：自由拟合的 c2 符号 vs 结构算出的场方向")
    print("=" * 96)
    print("预测 sign(c2) = -sign(u_Cbeta . E_hat)。场从 PDB 的可电离基团算，")
    print("c2 从 ΔΔG 拟。两边不相交。只检验 |t(c2)| > 2 的系列。\n")
    print(f"{'系列':<34} {'n':>3} {'c2':>8} {'t':>6} {'cos':>7} "
          f"{'预测':>5} {'实际':>5} {'一致':>5}")
    print("-" * 96)
    agree = tot = 0
    byenz = collections.defaultdict(dict)
    for k, fit in rows_out:
        sys_name, pdb_key, pos, pmid = k
        ch = chain_of.get(k)
        if not ch:
            continue
        g = analyse(pdb_key, ch, pos)
        if g is None:
            continue
        sig = abs(fit["t"]) > 2.0
        pred = g["pred_sign_c2"]
        act = 1 if fit["c2"] > 0 else -1
        ok = pred == act
        lbl = f"{sys_name[:14]}/{pdb_key[:9]}/{pos}"
        if sig:
            agree += ok; tot += 1
            print(f"{lbl:<34} {fit['n']:>3} {fit['c2']:>+8.3f} "
                  f"{fit['t']:>+6.2f} {g['cos']:>+7.3f} {pred:>+5} {act:>+5} "
                  f"{'是' if ok else '否':>5}")
        byenz[pos][pdb_key] = (g["cos"], fit["c2"], fit["t"], sig)

    print(f"\n符号一致: {agree}/{tot}")
    if tot:
        from math import comb
        pv = sum(comb(tot, i) * 0.5 ** tot for i in range(agree, tot + 1))
        print(f"二项检验 P(>= {agree} | p0=0.5) = {pv:.4f}")

    print("\n" + "=" * 96)
    print("OMTKY3 的 4 酶同位点对照 —— 核心判别")
    print("=" * 96)
    print("同一抑制剂骨架 => u 几乎相同；不同酶 => E 不同")
    print("|Δμ| 版本预测 c2 符号不随酶变；Δμ·E 版本预测它随 cos 的符号变\n")
    enz = {"1CHO_EFG_I": "胰凝乳蛋白酶", "3SGB_E_I": "SGPB",
           "1PPF_E_I": "弹性蛋白酶", "1R0R_E_I": "枯草Carlsberg"}
    print(f"{'位点':>6} " + "".join(f"{v:>16}" for v in enz.values()))
    print("-" * 72)
    flips = same = 0
    for pos in sorted(byenz, key=lambda x: (len(x), x)):
        row = byenz[pos]
        if not any(e in row for e in enz):
            continue
        cells = []
        signs = []
        for e in enz:
            if e in row:
                cos, c2, t, sig = row[e]
                mark = "*" if sig else " "
                cells.append(f"{cos:+.2f}/{c2:+.2f}{mark}")
                if sig:
                    signs.append((1 if cos > 0 else -1, 1 if c2 > 0 else -1))
            else:
                cells.append("—")
        print(f"{pos:>6} " + "".join(f"{c:>16}" for c in cells))
        if len(signs) >= 2:
            cs = {s[0] for s in signs}
            c2s = {s[1] for s in signs}
            if len(cs) > 1:
                flips += 1
                if len(c2s) > 1:
                    same += 1
    print("\n格式 cos/c2，* = |t|>2")
    print(f"\ncos 符号在酶间翻转的位点数: {flips}")
    print(f"其中 c2 符号也跟着翻转的: {same}")
    print("=> 若 Δμ·E 为真，这两个数应当接近；")
    print("   若 |Δμ| 版本为真（c2 不随酶变），第二个数应接近 0。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
