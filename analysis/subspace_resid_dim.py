#!/usr/bin/env python3
"""第二维到底是「一条普适轴」还是「一片受限区域」？

两个统计量给出看似矛盾的结果：
  - 成对主角度：独立配对的 cos th2 中位 0.33，区间 0.03-0.84（随机 p95 0.30）
  - 平均投影算子的 lambda_2：显著超出标定后的条件零分布（p = 0.0005）
这两件事可以同时为真，而且含义明确：各体系的第二方向**不相同**，
但也**不是各向同性**的 —— 它们被约束在 R^19 的一小块区域里。

本脚本直接量化那块区域的维数：
  1. 把每个体系的 2 维子空间按普适方向 u 去卷（deflate），得到一条残差方向 d_i；
  2. 看 {d_i} 的平均投影算子谱、participation ratio、以及
     「要覆盖 90% 需要几维」；
  3. 与 k 条**各向同性**随机一维方向（u 的正交补 18 维内）的零分布比。
如果有效维数 ~1 => 第二维也是一条普适轴；
如果 ~k（体系数）=> 完全体系特异；
介于之间 => 受限但非单一。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import (AA20, ROOT, dedup_by_content,  # noqa: E402
                          extract_subspace, load_all_dms, load_independent,
                          load_scales, load_skempi)

ONE = np.ones(20) / np.sqrt(20)
NSIM = 4000


def centered_orth(B):
    Q = B - np.outer(ONE, ONE @ B)
    Q, _ = np.linalg.qr(Q)
    return Q


def spectrum(dirs):
    P = np.mean([np.outer(d, d) for d in dirs], axis=0)
    w = np.linalg.eigvalsh(P)[::-1]
    w = np.clip(w, 0, None)
    pr = w.sum() ** 2 / (w ** 2).sum()
    n90 = int(np.searchsorted(np.cumsum(w) / w.sum(), 0.90) + 1)
    return w, float(pr), n90


def main():
    rng = np.random.default_rng(20261011)
    scales = load_scales(verbose=False)
    mats, _ = dedup_by_content(load_independent() + load_skempi()
                               + load_all_dms())
    by = {}
    for d in mats:
        B, c, _ = extract_subspace(d["M"], d["mask"])
        if B is not None and len(c) == 20 and d["censor_frac"] < 0.25:
            by[d["name"]] = B

    COV = ["starr2020/WH1/huACE2", "animalACE2/Wuhan-Hu-1/bat",
           "PDCoV/PDCoV/gAPN",
           "sarbecovirus/SARS-CoV-1_Urban/Ra9479ACE2",
           "animalACE2/Omicron_BA2/human_monomer"]
    OMT = ["OMTKY3/chymotrypsin/1CHO", "OMTKY3/HLE-elastase/1PPF",
           "OMTKY3/subtilisin-Carlsberg/1R0R", "OMTKY3/SGPB/3SGB"]
    GROUPS = [("可分辨全集（4 OMTKY3 单酶 + GB1 + 5 冠状病毒）",
               OMT + ["lehner/GB1/IgG-Fc"] + COV),
              ("冠状病毒家族（阳性对照：已知共享两维）", COV),
              ("OMTKY3 x 4 蛋白酶（同骨架、不同配体）", OMT)]

    print("=" * 88)
    print("第二维的有效维数：一条普适轴，还是一片受限区域？")
    print("=" * 88)
    out = []
    for label, names in GROUPS:
        Qs = [centered_orth(by[n]) for n in names]
        P = np.mean([Q @ Q.T for Q in Qs], axis=0)
        w, V = np.linalg.eigh(P)
        u = V[:, ::-1][:, 0]
        k = len(Qs)
        # 每个体系在子空间内与 u 最正交的那一维
        dirs = []
        for Q in Qs:
            c = Q.T @ u
            c /= np.linalg.norm(c)
            dirs.append(Q @ np.array([-c[1], c[0]]))
        ws, pr, n90 = spectrum(dirs)
        # 零分布：u 正交补（18 维）内各向同性的 k 条一维方向
        prn, n90n, w1n = [], [], []
        for _ in range(NSIM):
            ds = []
            for _ in range(k):
                a = rng.normal(size=20)
                a -= ONE * (ONE @ a)
                a -= u * (u @ a)
                ds.append(a / np.linalg.norm(a))
            wn, p, n = spectrum(ds)
            prn.append(p); n90n.append(n); w1n.append(wn[0])
        print(f"\n[{label}]  k = {k}")
        print(f"    残差方向谱（前 5）: " +
              " ".join(f"{x:.3f}" for x in ws[:5]))
        print(f"    participation ratio = {pr:.2f}"
              f"（完全相同 = 1，完全各向同性 ~ min(k,18) = {min(k, 18)}）")
        print(f"      零分布 PR 中位 {np.median(prn):.2f} "
              f"[{np.percentile(prn, 2.5):.2f}, "
              f"{np.percentile(prn, 97.5):.2f}]")
        print(f"    首特征值 = {ws[0]:.3f}；零分布中位 "
              f"{np.median(w1n):.3f} p95 {np.percentile(w1n, 95):.3f}  "
              f"{'超出' if ws[0] > np.percentile(w1n, 95) else '未超出'}")
        print(f"    覆盖 90% 需要 {n90} 维；零分布中位 "
              f"{int(np.median(n90n))} 维")
        # 残差方向的化学身份
        vr = np.linalg.eigh(np.mean([np.outer(d, d) for d in dirs],
                                    axis=0))[1][:, -1]
        rs = []
        for nm, s in scales.items():
            idx = [i for i in range(20) if np.isfinite(s[i])]
            if len(idx) < 10:
                continue
            a = vr[idx] - np.mean(vr[idx]); b = s[idx] - np.mean(s[idx])
            if np.std(b) == 0:
                continue
            rs.append((abs(float(np.corrcoef(a, b)[0, 1])),
                       float(np.corrcoef(a, b)[0, 1]), nm))
        print(f"    残差方向主轴的化学身份（前 4）: " + "; ".join(
            f"{nm} r={r:+.2f}" for _, r, nm in sorted(rs, reverse=True)[:4]))
        o = np.argsort(-vr)
        print(f"      载荷 最正->最负: " +
              " ".join(f"{AA20[i]}{vr[i]:+.2f}" for i in o))
        out.append({"label": label, "k": k, "resid_eigs": ws[:5].tolist(),
                    "pr": pr, "pr_null_med": float(np.median(prn)),
                    "pr_null_ci": [float(np.percentile(prn, 2.5)),
                                   float(np.percentile(prn, 97.5))],
                    "w1": float(ws[0]),
                    "w1_null_p95": float(np.percentile(w1n, 95)),
                    "n90": n90, "n90_null_med": float(np.median(n90n)),
                    "resid_axis": vr.tolist(),
                    "resid_scale_corr": [{"scale": nm, "r": r}
                                         for _, r, nm in
                                         sorted(rs, reverse=True)[:8]]})

    print("\n" + "=" * 88)
    print("读法")
    print("  PR 接近 1  => 第二维也是一条普适轴")
    print("  PR 接近 k  => 第二维完全体系特异（各向同性）")
    print("  介于之间   => 受限但非单一：存在一片低维的「第二轴家族」")
    p = os.path.join(ROOT, "cache", "subspace_resid_dim.json")
    json.dump({"nsim": NSIM, "aa_order": AA20, "groups": out},
              open(p, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
