#!/usr/bin/env python3
"""在 10 个体系上重测耦合的秩，并把 WCN 加进来作竞争描述子。

动机（文献校准，docs/12）：
- 「位点效应 + 残基效应 + 乘性交互」的方差分解已有先例（Rosace-AA 2025，
  >500 domain，同样报告位点效应占 40-90%）。**不是新框架。**
- 唯一没有先例的是「site×residue 交互矩阵的秩是多少」
  （site×site 的秩有先例：Husain & Murugan MBE 2020，但那是另一个索引）。
- 所以秩判定是唯一可辩护的新意，而它此前只在 3 个体系上测过。

WCN（Shahmoradi & Wilke, Proteins 2016, 10.1002/prot.25034）用 1/r² 加权，
而 1/r² 恰好抵消壳层残基数的几何增长 => WCN 是「各距离等权」的径向积分，
**刻意把径向剖面压成标量**。我的 Δf 正是它丢弃的自由度。
把 WCN 加进来是为了回答「相对已有描述子的增量是多少」。
"""
import collections
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_orientation import parse_pdb, sub, norm, ROOT
from rotamer_build import CB_SPEC, nerf
from rank_power import als_complete
from coupling_structure import double_center, masked_corr
from pairwise_shape_matched import add_true_volume
from site_lfer_test import load_descriptors

AA1 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
       "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
       "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
       "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}


def wcn_at(pid, chain, resi):
    """WCN = Σ_j 1/r_ij²，j 遍历其他残基的 CA（Shahmoradi-Wilke 的定义）。"""
    path = os.path.join(ROOT, "cache", "pdb", f"{pid}.pdb")
    if not os.path.exists(path):
        return None
    atoms = parse_pdb(path)
    own = {a["name"]: a["xyz"] for a in atoms
           if a["chain"] == chain and a["resi"] == str(resi) and not a["het"]}
    if "CA" not in own:
        return None
    s = 0.0
    for a in atoms:
        if a["het"] or a["name"] != "CA":
            continue
        if a["chain"] == chain and a["resi"] == str(resi):
            continue
        r = norm(sub(a["xyz"], own["CA"]))
        if r > 1e-6:
            s += 1.0 / r ** 2
    return s


def main():
    desc = add_true_volume(load_descriptors())
    res_list = sorted(desc)
    SD = json.load(open(os.path.join(ROOT, "cache",
                                     "expanded_site_desc.json")))["sites"]
    ddg = collections.defaultdict(dict)
    meta = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "cache",
                                              "skempi_thr4.csv"))):
        key = "|".join([r["system"], r["pdb"], r["chain"], r["position"]])
        if key not in SD:
            continue
        m = AA1.get(r["mut_res"].strip().upper(), r["mut_res"].strip().upper())
        if m not in desc:
            continue
        ddg[key][m] = float(r["ddG_kcal"])
        meta[key] = (r["system"], r["pdb"], r["chain"], r["position"])
    sites = [k for k in sorted(ddg) if len(ddg[k]) >= 4]
    M = np.array([[ddg[k].get(r, np.nan) for r in res_list] for k in sites])
    mask = np.isfinite(M)
    nsys = len({meta[k][0] for k in sites})
    print("=" * 76)
    print(f"扩充后的秩判定：{len(sites)} 位点 x {len(res_list)} 残基, "
          f"{mask.sum()} 观测, {nsys} 个体系")
    print("=" * 76)
    Rm = double_center(np.where(mask, M, 0.0), mask)
    tot = M[mask].std(); coup = Rm[mask].std()
    print(f"ΔΔG 总 SD {tot:.3f}，耦合残差 SD {coup:.3f} "
          f"=> 耦合占总方差 {(coup / tot) ** 2:.1%}")

    def cv(rank, K=5, restarts=4, seed=3):
        rng = np.random.default_rng(seed)
        idx = np.argwhere(mask); rng.shuffle(idx)
        folds = np.array_split(idx, K); tot_e = cnt = 0
        for f in folds:
            trn = mask.copy()
            for i, j in f:
                trn[i, j] = False
            Y = np.where(trn, Rm, 0.0)
            best, bestP = np.inf, None
            for _ in range(restarts):
                P = als_complete(Y, trn, rank, rng, iters=300)
                tr = np.mean((Rm[trn] - P[trn]) ** 2)
                if tr < best:
                    best, bestP = tr, P
            tot_e += sum((Rm[i, j] - bestP[i, j]) ** 2 for i, j in f)
            cnt += len(f)
        return math.sqrt(tot_e / cnt)

    print(f"\n{'拟合秩':>6} {'5折 CV RMSE':>12} {'vs 秩0':>9}")
    print("-" * 32)
    base = cv(0); errs = {}
    for rk in range(0, 5):
        e = cv(rk); errs[rk] = e
        print(f"{rk:>6} {e:>12.4f} {(base - e) / base:>+8.1%}")
    print(f"\n=> CV 最优秩 = {min(errs, key=errs.get)}"
          f"   （3 体系样本上曾判为 2）")

    # WCN vs Δf 的正交性与各自的耦合解释力
    print("\n" + "=" * 76)
    print("竞争描述子：WCN（Shahmoradi-Wilke）vs Δf（本项目）")
    print("=" * 76)
    wcn = np.array([wcn_at(meta[k][1], meta[k][2], meta[k][3]) or np.nan
                    for k in sites])
    bur = np.array([SD[k]["burial"] for k in sites])
    sh = np.array([SD[k]["shells"] for k in sites])
    df = sh[:, 0] - sh[:, 3]
    ok = np.isfinite(wcn) & np.isfinite(df) & np.isfinite(bur)
    print(f"可用位点 {ok.sum()}/{len(sites)}")
    print(f"   r(WCN, 埋藏度) = {np.corrcoef(wcn[ok], bur[ok])[0, 1]:+.3f}")
    print(f"   r(WCN, Δf)     = {np.corrcoef(wcn[ok], df[ok])[0, 1]:+.3f}"
          f"   <= 低则 Δf 是 WCN 丢弃的自由度")
    print(f"   r(Δf, 埋藏度)  = {np.corrcoef(df[ok], bur[ok])[0, 1]:+.3f}")
    pi = np.array([desc[r]["pi"] for r in res_list])
    print(f"\n各位点量 × 疏水性 对耦合残差的相关:")
    for nm, v in (("埋藏度", bur), ("WCN", wcn), ("Δf", df)):
        C = np.outer(v, pi)
        m2 = mask & np.isfinite(C)
        Cc = double_center(np.where(m2, np.nan_to_num(C), 0.0), m2)
        print(f"   {nm:<8} r = {masked_corr(Rm, Cc, m2):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
