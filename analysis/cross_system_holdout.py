#!/usr/bin/env python3
"""留一体系交叉验证：三项模型能否推广到没见过的蛋白体系？

此前整个耦合分析用了全部 55 个系列，且搜索路径自适应（先看载荷再选描述子）。
置换检验对最终描述子非循环，但「测什么」被数据引导了 —— Bonferroni 不解决
分叉路径问题，唯一解药是样本外。

### 检验设计

按**蛋白体系**分组（OMTKY3 / BPTI / 4-4-20 / 溶菌酶 / SSI / eglin / barnase …），
留出一个体系，在其余体系上拟 λ，预测留出体系各位点内的残基排序。

指标用**位点内 pairwise accuracy** —— 这正是原始问题要的量，
且位点主效应在成对比较中自动抵消（预测时不需要知道留出位点的均值）。
"""
import collections
import csv
import itertools
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coupling_structure import burial_at_site
from pairwise_shape_matched import add_true_volume
from run_real_test import load_series, ROOT
from site_lfer_test import load_descriptors

PLANAR = {"PHE", "TYR", "TRP", "HIS", "ASP", "GLU", "ASN", "GLN"}


def system_of(k):
    """按蛋白体系分组。pdb 码的前四位足以区分。"""
    pid = re.match(r"^([0-9A-Za-z]{4})", k[1]).group(1).upper()
    return {"1CHO": "OMTKY3", "3SGB": "OMTKY3", "1PPF": "OMTKY3",
            "1R0R": "OMTKY3", "2FTL": "BPTI", "1CBW": "BPTI",
            "1FLR": "4-4-20/荧光素", "3HFM": "溶菌酶/HyHEL10",
            "2SIC": "SSI", "1BRS": "barnase", "1CSE": "eglin",
            "1ACB": "eglin", "1AO7": "A6 TCR"}.get(pid, pid)


def main():
    desc = add_true_volume(load_descriptors())
    res_list = sorted(desc)
    series = load_series(os.path.join(ROOT, "single_site_ddG_series.csv"))
    chain_of = {}
    for r in csv.DictReader(open(os.path.join(
            ROOT, "single_site_ddG_series.csv"))):
        chain_of[(r["system"], r["pdb"], r["position"],
                  r.get("pmid", ""))] = r["chain"]
    sites = [k for k in sorted(series) if chain_of.get(k)]
    SH = np.load(os.path.join(ROOT, "cache", "shell_free.npy"))
    df = SH[:, 0] - SH[:, 3]              # 有界的 近端−远端 差
    def _pos(x):
        # 位点编号可能带链字母前缀（如 "H102"），PDB 里是 "102"
        return re.sub(r"^[A-Za-z]", "", x) if re.match(r"^[A-Za-z]\d", x) else x

    bur = np.array([burial_at_site(k[1], chain_of[k], _pos(k[2])) or np.nan
                    for k in sites])
    pi = np.array([desc[r]["pi"] for r in res_list])
    P = np.array([1.0 if r in PLANAR else 0.0 for r in res_list])

    # 每个 (位点, 残基) 的三个特征 + 位点内中心化后的 ΔΔG
    rows = []
    for i, k in enumerate(sites):
        d = dict(series[k][0])
        vals = [(j, d[r]) for j, r in enumerate(res_list) if r in d]
        if len(vals) < 6 or not np.isfinite(df[i]) or not np.isfinite(bur[i]):
            continue
        m = np.mean([v for _, v in vals])
        for j, v in vals:
            rows.append({"sys": system_of(k), "site": i, "res": j,
                         "y": v - m,                       # 位点内中心化
                         "f1": bur[i] * pi[j],
                         "f2": df[i] * pi[j],
                         "f3": df[i] * P[j]})
    bysys = collections.defaultdict(list)
    for r in rows:
        bysys[r["sys"]].append(r)
    print("=" * 76)
    print("留一体系交叉验证")
    print("=" * 76)
    print(f"{'体系':<18} {'位点':>5} {'观测':>5} {'可判对':>7} "
          f"{'基线':>7} {'三项':>7} {'增益':>8}")
    print("-" * 76)

    def fit(tr):
        X = np.array([[1.0, r["f1"], r["f2"], r["f3"]] for r in tr])
        y = np.array([r["y"] for r in tr])
        # 位点内中心化后各特征也须按位点中心化，否则位点效应漏进来
        bysite = collections.defaultdict(list)
        for idx, r in enumerate(tr):
            bysite[r["site"]].append(idx)
        for ids in bysite.values():
            X[ids, 1:] -= X[ids, 1:].mean(axis=0)
        return np.linalg.lstsq(X, y, rcond=None)[0]

    tot_h = tot_n = 0
    base_h = 0
    for held in sorted(bysys):
        tr = [r for s, v in bysys.items() if s != held for r in v]
        te = bysys[held]
        if len(tr) < 40 or len(te) < 6:
            continue
        b = fit(tr)
        # 在留出体系的每个位点内做成对预测
        bysite = collections.defaultdict(list)
        for r in te:
            bysite[r["site"]].append(r)
        h = n = bh = 0
        for ids in bysite.values():
            if len(ids) < 3:
                continue
            F = np.array([[r["f1"], r["f2"], r["f3"]] for r in ids])
            F = F - F.mean(axis=0)
            pred = F @ b[1:]
            yv = np.array([r["y"] for r in ids])
            for a, c in itertools.combinations(range(len(ids)), 2):
                if abs(yv[a] - yv[c]) <= 0.8:
                    continue
                n += 1
                if (pred[a] < pred[c]) == (yv[a] < yv[c]):
                    h += 1
                # 基线：只用疏水性（已知项的残基侧）
                if (pi[ids[a]["res"]] > pi[ids[c]["res"]]) == (yv[a] < yv[c]):
                    bh += 1
        if n < 8:
            continue
        tot_h += h; tot_n += n; base_h += bh
        print(f"{held:<18} {len(bysite):>5} {len(te):>5} {n:>7} "
              f"{bh / n:>7.3f} {h / n:>7.3f} {(h - bh) / n:>+8.3f}")
    print("-" * 76)
    print(f"{'合计':<18} {'':>5} {'':>5} {tot_n:>7} "
          f"{base_h / tot_n:>7.3f} {tot_h / tot_n:>7.3f} "
          f"{(tot_h - base_h) / tot_n:>+8.3f}")
    # 二项检验：三项模型 vs 基线（配对，按对为单位偏保守）
    from math import comb
    d = tot_h - base_h
    print(f"\n三项模型 pairwise accuracy = {tot_h / tot_n:.3f}")
    print(f"仅疏水基线               = {base_h / tot_n:.3f}")
    print(f"净增 {d} 对 / {tot_n} 对")
    print("\n注意：这是**样本外**数字 —— λ 在其他体系上拟，预测没见过的体系。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
