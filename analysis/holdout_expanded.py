#!/usr/bin/env python3
"""扩充后的留一体系交叉验证。

此前只有 3 个体系且 OMTKY3 占主导，样本外测试近乎无意义（+0.012）。
现在：SKEMPI 门槛降到 4 后 9 个体系有描述子（6 个非 OMTKY3），
加上 4-4-20/荧光素（唯一小分子体系）共 10 个。

关键是现在可以**留出 OMTKY3、在其余体系上训练** —— 这是之前做不到的。
"""
import collections
import csv
import itertools
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pairwise_shape_matched import add_true_volume
from run_real_test import load_series, ROOT
from site_lfer_test import load_descriptors

PLANAR = {"PHE", "TYR", "TRP", "HIS", "ASP", "GLU", "ASN", "GLN"}
AA1 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
       "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
       "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
       "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}


def build():
    desc = add_true_volume(load_descriptors())
    blob = json.load(open(os.path.join(ROOT, "cache",
                                       "expanded_site_desc.json")))
    SD = blob["sites"]
    rows = []
    # --- SKEMPI 扩充集 ---
    for r in csv.DictReader(open(os.path.join(ROOT, "cache",
                                              "skempi_thr4.csv"))):
        key = "|".join([r["system"], r["pdb"], r["chain"], r["position"]])
        if key not in SD:
            continue
        m = AA1.get(r["mut_res"].strip().upper(), r["mut_res"].strip().upper())
        if m not in desc:
            continue
        sh = SD[key]["shells"]
        rows.append({"sys": r["system"], "site": key, "res": m,
                     "y": float(r["ddG_kcal"]),
                     "bur": SD[key]["burial"], "df": sh[0] - sh[3]})
    # --- 4-4-20/荧光素（唯一小分子体系）---
    series = load_series(os.path.join(ROOT, "single_site_ddG_series.csv"))
    chain_of = {}
    for r in csv.DictReader(open(os.path.join(
            ROOT, "single_site_ddG_series.csv"))):
        chain_of[(r["system"], r["pdb"], r["position"],
                  r.get("pmid", ""))] = r["chain"]
    SH = np.load(os.path.join(ROOT, "cache", "shell_free.npy"))
    sites_old = [k for k in sorted(series) if chain_of.get(k)]
    from coupling_structure import burial_at_site
    for i, k in enumerate(sites_old):
        if "1FLR" not in k[1]:
            continue
        pos = (re.sub(r"^[A-Za-z]", "", k[2])
               if re.match(r"^[A-Za-z]\d", k[2]) else k[2])
        bu = burial_at_site(k[1], chain_of[k], pos)
        if bu is None or not np.isfinite(SH[i, 0]):
            continue
        key = f"4-4-20/荧光素|1FLR|{chain_of[k]}|{k[2]}"
        for rr, v in series[k][0]:
            if rr in desc:
                rows.append({"sys": "4-4-20/荧光素（小分子）", "site": key,
                             "res": rr, "y": v, "bur": bu,
                             "df": SH[i, 0] - SH[i, 3]})
    return desc, rows


def main():
    desc, rows = build()
    pi = {r: desc[r]["pi"] for r in desc}
    P = {r: (1.0 if r in PLANAR else 0.0) for r in desc}
    # 位点内中心化
    bysite = collections.defaultdict(list)
    for r in rows:
        bysite[r["site"]].append(r)
    feats = []
    for site, rs in bysite.items():
        if len(rs) < 4:
            continue
        m = np.mean([r["y"] for r in rs])
        F = np.array([[r["bur"] * pi[r["res"]], r["df"] * pi[r["res"]],
                       r["df"] * P[r["res"]]] for r in rs])
        F = F - F.mean(axis=0)
        for idx, r in enumerate(rs):
            feats.append({"sys": r["sys"], "site": site, "res": r["res"],
                          "y": r["y"] - m, "f": F[idx]})
    bysys = collections.defaultdict(list)
    for f in feats:
        bysys[f["sys"]].append(f)
    print("=" * 84)
    print("扩充后的留一体系交叉验证")
    print("=" * 84)
    print(f"体系 {len(bysys)} 个，位点 {len({f['site'] for f in feats})}，"
          f"观测 {len(feats)}")
    print(f"\n{'留出体系':<40} {'位点':>4} {'可判对':>7} {'仅疏水':>7} "
          f"{'三项':>7} {'增益':>8}")
    print("-" * 84)
    tot_h = tot_b = tot_n = 0
    for held in sorted(bysys, key=lambda s: -len(bysys[s])):
        tr = [f for s, v in bysys.items() if s != held for f in v]
        te = bysys[held]
        if len(tr) < 40:
            print(f"{held[:38]:<40} 训练数据不足，跳过")
            continue
        X = np.array([f["f"] for f in tr])
        y = np.array([f["y"] for f in tr])
        b = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y,
                            rcond=None)[0][1:]
        bs = collections.defaultdict(list)
        for f in te:
            bs[f["site"]].append(f)
        h = bh = n = 0
        for rs in bs.values():
            if len(rs) < 3:
                continue
            pred = np.array([f["f"] @ b for f in rs])
            yv = np.array([f["y"] for f in rs])
            for a, c in itertools.combinations(range(len(rs)), 2):
                if abs(yv[a] - yv[c]) <= 0.8:
                    continue
                n += 1
                h += (pred[a] < pred[c]) == (yv[a] < yv[c])
                bh += (pi[rs[a]["res"]] > pi[rs[c]["res"]]) == (yv[a] < yv[c])
        if n < 8:
            print(f"{held[:38]:<40} {len(bs):>4} {n:>7}  可判对不足")
            continue
        tot_h += h; tot_b += bh; tot_n += n
        print(f"{held[:38]:<40} {len(bs):>4} {n:>7} {bh / n:>7.3f} "
              f"{h / n:>7.3f} {(h - bh) / n:>+8.3f}")
    print("-" * 84)
    print(f"{'合计':<40} {'':>4} {tot_n:>7} {tot_b / tot_n:>7.3f} "
          f"{tot_h / tot_n:>7.3f} {(tot_h - tot_b) / tot_n:>+8.3f}")
    print(f"\n净增 {tot_h - tot_b} 对 / {tot_n} 对")
    print("λ 在其余体系上拟，预测没见过的体系。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
