#!/usr/bin/env python3
"""docs/28 Step 5(b)：配对的 folding–binding 对照还能不能做成。

docs/15 §7.1 把「同一批变体上两个表型都足够精确」列为**本项目最重要的
那一个检验**，并判定这批数据做不成（结合一侧 0/3 有功效）。
docs/27 §8 也明确说它**仍然没有做成**（Tsuboyama 只有折叠表型）。

Step 3 改变了一件事：ddPCA 的两个 assay 各自有**三次独立生物学重复**，
所以配对比较可以在**原始表型（fitness）层**上做，而 D-036 的前置条件
（每个方向自身可复现）可以用重复**直接测**，不靠位点对半。

本脚本因此分两层报，不混：
  (I)  **ΔΔG 层**（docs/15 §5 的那一问，配对 mask）—— 结论预期不变；
  (II) **fitness 层**（本文档新增）—— 同一批变体、两次**不同的选择实验**
       （abundance-PCA vs binding-PCA），每条轴自身可复现度由三重复给出。

(II) 必须带三条限定，否则会被过度解读：
  1. fitness 不是 ΔΔG：三态模型的连接函数是饱和的，纯可加的 additive trait
     过饱和连接会产生**表观**交互。所以同时报「交互第一轴 vs 自己的残基
     主效应」的 |cos| 作为判别指标，以及两个 assay 的**主效应**之间的 |cos|
     （若主效应几乎相同，交互层的对齐可能是被诱导的）。
  2. 两个 assay 共用同一个质粒库与同一个细胞系统，所以是
     **同源、跨 assay**，不是跨实验室。按 docs/25 §6.3 的要求分类报出。
  3. 置换零分布（位点内置换残基标签）与 Haar 零分布都在**本比较自己的 d**
     上生成。

数值层复用 `ddpca_sigma.rep_matrices` / `rank_core.center_fit` /
`subspace_lib.extract_subspace`（内部 `als_guarded`）/ `haar_1d_null` /
`permute_within_sites`。改动为零。
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddpca_load as DL                                        # noqa: E402
import ddpca_sigma as S                                        # noqa: E402
import gate_calib_lib as G                                     # noqa: E402
import rank_core as RC                                         # noqa: E402
from subspace_lib import (ROOT, extract_subspace,              # noqa: E402
                          haar_1d_null, permute_within_sites)

OUTJSON = os.path.join(ROOT, "cache", "ddpca_paired.json")
# 两个 assay 都有独立选择实验的域（GB1 没有 abundance assay，docs/15 §6.1）
DOMS = ["GRB2-SH3", "PSD95-PDZ3"]


def _ax(M, mk):
    B, cols, _ = extract_subspace(np.where(mk, M, np.nan), mk, rank=1)
    if B is None:
        return None, None
    return B[:, 0], cols


def _cos(w1, c1, w2, c2):
    if w1 is None or w2 is None:
        return None
    common = [x for x in c1 if x in set(c2)]
    if len(common) < 10:
        return None
    v1 = np.asarray(w1)[[list(c1).index(x) for x in common]]
    v2 = np.asarray(w2)[[list(c2).index(x) for x in common]]
    v1 = v1 - v1.mean()
    v2 = v2 - v2.mean()
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    return abs(float(v1 @ v2 / (n1 * n2))), len(common)


def rep_axis_repro(Ms, mk):
    """三次独立重复各自抽第一轴，返回两两 |cos| 的中位（D-036 口径）。"""
    ax = [_ax(X, mk) for X in Ms]
    vals = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        z = _cos(ax[i][0], ax[i][1], ax[j][0], ax[j][1])
        if z:
            vals.append(z[0])
    return (float(np.median(vals)) if vals else float("nan")), vals


def run(dom, nsplit=200, nperm=400, verbose=True):
    D = DL.load_domain(dom)
    DL.verify(dom, D, verbose=False)
    pos = sorted(D["wt"])
    out = {"domain": dom}

    # ---------------- (I) ΔΔG 层，配对 mask ----------------
    Mf, mkf, cf = DL.matrix(D, "folding")
    Mb, mkb, cb = DL.matrix(D, "binding")
    mp = mkf & mkb
    keep = mp.sum(axis=1) >= 4
    mp = mp & keep[:, None]
    design = cf["design"]
    af = _ax(Mf, mp)
    ab = _ax(Mb, mp)
    z = _cos(af[0], af[1], ab[0], ab[1])
    d = z[1] if z else 20
    p95 = float(np.percentile(haar_1d_null(d, n=20000, seed=3), 95))
    rf = G.split_half(np.where(mp, Mf, np.nan), mp, rank=1, nsplit=nsplit)
    rb = G.split_half(np.where(mp, Mb, np.nan), mp, rank=1, nsplit=nsplit)
    rfm = float(np.median(rf)) if rf else float("nan")
    rbm = float(np.median(rb)) if rb else float("nan")
    rngd = np.random.default_rng(23)
    pvd = []
    for _ in range(nperm):
        Bp, mkp = permute_within_sites(np.where(mp, Mb, np.nan), mp, rngd)
        apd = _ax(Bp, mkp)
        qd = _cos(af[0], af[1], apd[0], apd[1])
        if qd:
            pvd.append(qd[0])
    attd = (z[0] / math.sqrt(rfm * rbm)
            if z and rfm > 0 and rbm > 0 else None)
    out["ddg"] = {
        "n_paired": int(mp.sum()), "design": design,
        "censor_paired": 1.0 - mp.sum() / design,
        "abscos": z[0] if z else None, "d": d, "haar1d_p95": p95,
        "abscos_attenuation_corrected": attd,
        "perm_med": float(np.median(pvd)) if pvd else None,
        "perm_p95": float(np.percentile(pvd, 95)) if pvd else None,
        "perm_p": (float(np.mean(np.array(pvd) >= z[0]))
                   if pvd and z else None),
        "repro_folding": rfm, "repro_binding": rbm,
        "d036_folding": bool(rfm > p95), "d036_binding": bool(rbm > p95),
        "d036_both": bool(rfm > p95 and rbm > p95)}
    if verbose:
        print(f"[{dom}] (I) ΔΔG 层，配对 mask: {int(mp.sum())}/{design} "
              f"（截尾 {out['ddg']['censor_paired']:.1%}）")
        print(f"    |cos(折叠第一轴, 结合第一轴)| = "
              + (f"{z[0]:.3f}" if z else "n/a")
              + f"，d = {d}，Haar p95 = {p95:.3f}"
              + (f"；衰减校正后 {attd:.3f}" if attd else "")
              + (f"；位点内置换零分布中位 {out['ddg']['perm_med']:.3f}、"
                 f"p95 {out['ddg']['perm_p95']:.3f}，p = "
                 f"{out['ddg']['perm_p']:.4f}" if pvd else ""))
        print(f"    **D-036 前置**：折叠轴自身可复现 {rfm:.3f} "
              f"{'过' if rfm > p95 else '**不过**'}；"
              f"结合轴自身可复现 {rbm:.3f} "
              f"{'过' if rbm > p95 else '**不过**'}"
              f" -> 配对主张"
              f"{'**可以**提' if out['ddg']['d036_both'] else '**不允许**提'}")

    # ---------------- (II) fitness 层，同一批变体 ----------------
    Fs, Fm, _ = S.rep_matrices(f"{dom}/abundance", D["wt"], pos)
    Bs, Bm, _ = S.rep_matrices(f"{dom}/binding", D["wt"], pos)
    mk = (np.all([np.isfinite(X) for X in Fs], axis=0)
          & np.all([np.isfinite(X) for X in Bs], axis=0)
          & np.isfinite(Fm) & np.isfinite(Bm))
    keep = mk.sum(axis=1) >= 4
    mk = mk & keep[:, None]
    if mk.sum() < 200:
        out["fitness"] = {"ok": False,
                          "why": f"两 assay 都齐全的格子只有 {int(mk.sum())}"}
        return out
    aF = _ax(Fm, mk)
    aB = _ax(Bm, mk)
    zf = _cos(aF[0], aF[1], aB[0], aB[1])
    d2 = zf[1] if zf else 20
    p95b = float(np.percentile(haar_1d_null(d2, n=20000, seed=3), 95))
    repF, vF = rep_axis_repro(Fs, mk)
    repB, vB = rep_axis_repro(Bs, mk)
    # 主效应对照：两个 assay 的残基主效应有多像（诱导风险）
    rF, cF = RC.center_fit(np.where(mk, Fm, np.nan), mk)
    rB, cB = RC.center_fit(np.where(mk, Bm, np.nan), mk)
    allc = list(range(Fm.shape[1]))
    zmain = _cos(cF - cF.mean(), allc, cB - cB.mean(), allc)
    zlinkF = _cos(aF[0], aF[1], cF - cF.mean(), allc)
    zlinkB = _cos(aB[0], aB[1], cB - cB.mean(), allc)
    # 置换零分布：位点内置换残基标签（摧毁耦合、保留边缘与缺失模式）
    rng = np.random.default_rng(17)
    pv = []
    for _ in range(nperm):
        Bp, mkp = permute_within_sites(np.where(mk, Bm, np.nan), mk, rng)
        ap = _ax(Bp, mkp)
        q = _cos(aF[0], aF[1], ap[0], ap[1])
        if q:
            pv.append(q[0])
    att = (zf[0] / math.sqrt(repF * repB)
           if zf and repF > 0 and repB > 0 else None)
    out["fitness"] = {
        "ok": True, "n_cells": int(mk.sum()), "n_sites": int(keep.sum()),
        "abscos": zf[0] if zf else None, "d": d2, "haar1d_p95": p95b,
        "repro_folding_reps": repF, "repro_folding_pairs": vF,
        "repro_binding_reps": repB, "repro_binding_pairs": vB,
        "d036_both": bool(repF > p95b and repB > p95b),
        "abscos_attenuation_corrected": att,
        "perm_med": float(np.median(pv)) if pv else None,
        "perm_p95": float(np.percentile(pv, 95)) if pv else None,
        "perm_p": (float(np.mean(np.array(pv) >= zf[0]))
                   if pv and zf else None),
        "main_effect_abscos": zmain[0] if zmain else None,
        "link_folding": zlinkF[0] if zlinkF else None,
        "link_binding": zlinkB[0] if zlinkB else None,
        "pairing_class": "同源（同库同细胞系统）、跨 assay（两次独立选择实验）"}
    if verbose:
        f = out["fitness"]
        print(f"    (II) fitness 层，同一批变体: {f['n_cells']} 格 / "
              f"{f['n_sites']} 位点，d = {d2}")
        print(f"        **D-036 前置（由三次独立重复直接测，不靠位点对半）**："
              f"abundance 轴 {repF:.3f}、binding 轴 {repB:.3f} vs Haar p95 "
              f"{p95b:.3f} -> "
              f"{'**两条都过**' if f['d036_both'] else '**不过**'}")
        print(f"        |cos(abundance 交互第一轴, binding 交互第一轴)| = "
              f"**{f['abscos']:.3f}**"
              f"（衰减校正后 {att:.3f}）" if att else "")
        print(f"        位点内置换零分布: 中位 {f['perm_med']:.3f}、p95 "
              f"{f['perm_p95']:.3f}，p = {f['perm_p']:.4f}")
        print(f"        限定 1（连接非线性）: 交互轴 vs 自己的残基主效应 "
              f"|cos| = abundance {f['link_folding']:.3f}、binding "
              f"{f['link_binding']:.3f}")
        print(f"        限定 2（诱导风险）: 两个 assay 的**残基主效应**之间 "
              f"|cos| = {f['main_effect_abscos']:.3f}")
        print(f"        限定 3（配对类别）: {f['pairing_class']}")
    return out


def main():
    print("=" * 96)
    print("docs/28 Step 5(b)：配对的 folding–binding 对照（docs/15 §7.1）")
    print("=" * 96)
    rows = [run(d) for d in DOMS]
    print("\n[GB1] 不做：GB1 **没有独立的 abundance assay**（docs/15 §6.1，"
          "由 zip 索引断言），folding 与 binding 系数来自**同一组** "
          "binding-PCA 观测，两者之间的任何一致性都不是两个表型的"
          "生物学一致性。")
    json.dump({"rows": rows}, open(OUTJSON, "w"), ensure_ascii=False,
              indent=1, default=float)
    print("\n产物:", OUTJSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
