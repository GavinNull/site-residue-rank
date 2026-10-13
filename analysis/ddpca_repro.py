#!/usr/bin/env python3
"""docs/28 Step 3：**直接测量**，不是论证。

docs/27 §4.1 / D-042 之所以站得住，不是因为它对 σ 讲了一个道理，而是因为
它拿**两条独立蛋白酶读出**在同一批矩阵上测到 r = 0.904、|cos| = 0.983
—— σ 判据说那里什么都没有的地方，两个独立测量一致。

ddPCA 一侧的同型把手（本模块找到并验证）：
  zip 内 `Data/fitness/<domain>/*/*_fitness_replicates.RData` 的
  `fitness{1,2,3}_uncorr` = **三次独立的选择实验**（e1/e2/e3 两个时间点计数）。
  这与 bootstrap 重拟合**完全不是一回事**（D-028 已禁止后者当噪声模型）。

本模块测四件事（全部在**双向中心化之后**的交互项上）：
  (1) 三对重复之间交互残差的相关 r —— 与 docs/27 的 0.904 同口径；
  (2) 各重复**独立**抽出的第一残基轴之间的 |cos| —— 与 docs/27 的 0.983 同口径，
      并与该比较自己的 d 上生成的 Haar 一维零分布 p95 比；
  (3) 由 r 直接给出的噪声占比 σ_centered/coup_sd_obs = sqrt(1 − r)
      —— 这是 anchor A 的**独立交叉核对**（A 走的是差值 SD）；
  (4) 重复 fitness 矩阵的第一轴 vs mochi ΔΔG 矩阵的第一轴
      —— 跨提取路径的同源配对。

不误伤对照（D-007 / §10.2）：把格子↔重复的配对在 mask 内打乱，
(1) 与 (2) 都必须塌到零分布水平。只测「能命中」的护栏最终会被绕过。
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddpca_load as DL                                        # noqa: E402
import ddpca_replicates as RP                                  # noqa: E402
import ddpca_sigma as S                                        # noqa: E402
import rank_core as RC                                         # noqa: E402
from generic_folding_axis import first_axis                    # noqa: E402
from subspace_lib import ROOT, haar_1d_null                    # noqa: E402

OUTJSON = os.path.join(ROOT, "cache", "ddpca_repro.json")


def _centered(X, mk):
    r, c = RC.center_fit(np.where(mk, X, np.nan), mk)
    return np.where(mk, X, np.nan) - r[:, None] - c[None, :]


def _axis(X, mk, d_expect=20):
    w, cols = first_axis(np.where(mk, X, np.nan), mk)
    if w is None:
        return None, None
    return w, cols


def _abscos(w1, c1, w2, c2):
    if w1 is None or w2 is None:
        return None
    common = [a for a in c1 if a in set(c2)]
    if len(common) < 10:
        return None
    i1 = [list(c1).index(a) for a in common]
    i2 = [list(c2).index(a) for a in common]
    v1, v2 = np.asarray(w1)[i1], np.asarray(w2)[i2]
    v1 = v1 - v1.mean()
    v2 = v2 - v2.mean()
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    return abs(float(v1 @ v2 / (n1 * n2))), len(common)


def run(dom, ph, nperm=200, verbose=True):
    D = DL.load_domain(dom)
    DL.verify(dom, D, verbose=False)
    M, mask, cen = DL.matrix(D, ph)
    pos = sorted(D["wt"])
    assay = S.REP_ASSAY[(dom, ph)]
    Ms, Mm, n = S.rep_matrices(assay, D["wt"], pos)
    mk = np.all([np.isfinite(X) for X in Ms], axis=0) & np.isfinite(Mm)
    # 位点保护：rank1 需要每行有足够观测
    keep = mk.sum(axis=1) >= 4
    mk = mk & keep[:, None]
    if mk.sum() < 200:
        return {"name": f"{dom}/{ph}", "ok": False,
                "why": f"三重复齐全的格子只有 {int(mk.sum())}"}
    Rs = [_centered(X, mk) for X in Ms]

    # (1) 交互残差相关
    pairs = ((0, 1), (0, 2), (1, 2))
    rr = [float(np.corrcoef(Rs[i][mk], Rs[j][mk])[0, 1]) for i, j in pairs]
    # (2) 各重复独立抽第一轴
    ax = [_axis(X, mk) for X in Ms]
    cc = []
    for i, j in pairs:
        z = _abscos(ax[i][0], ax[i][1], ax[j][0], ax[j][1])
        if z:
            cc.append(z[0])
    d = len(ax[0][1]) if ax[0][1] is not None else 20
    p95 = float(np.percentile(haar_1d_null(d, n=20000, seed=3), 95))
    # (3) 由 r 反推噪声占比
    rmed = float(np.median(rr))
    frac = math.sqrt(max(1.0 - rmed, 0.0))        # σ_ml 对 coup_sd_obs 之比
    frac_ml = frac / math.sqrt(3.0)
    # (4) 重复 fitness 矩阵的第一轴 vs mochi ΔΔG 矩阵的第一轴
    am = _axis(np.where(mk, Mm, np.nan), mk)
    ad = _axis(M, mask)
    cross = _abscos(am[0], am[1], ad[0], ad[1])
    # (5) **连接函数非线性的判别指标**（这条限定不加，(1) 会被过度解读）：
    #     一个纯可加的 additive trait 过一个饱和连接函数，在 fitness 上会
    #     产生**完全可复现**的表观 site×residue 交互，而 mochi 的三态模型
    #     正是去掉它的。所以必须测：fitness 的交互第一轴有多像
    #     ΔΔG 的**残基主效应**方向。像，就说明 (1) 的高 r 主要来自连接函数。
    rM, cM = RC.center_fit(M, mask)
    col_main = cM - cM.mean()
    cols_all = list(range(M.shape[1]))
    link = _abscos(am[0], am[1], col_main, cols_all)
    # 同一指标在 fitness 侧自己的主效应上（对照）
    rF, cF = RC.center_fit(np.where(mk, Mm, np.nan), mk)
    link_f = _abscos(am[0], am[1], cF - cF.mean(), cols_all)

    # 不误伤对照：mask 内打乱格子↔重复的配对
    rng = np.random.default_rng(31)
    idx = np.argwhere(mk)
    rr_n, cc_n = [], []
    for _ in range(nperm):
        Rp = [R.copy() for R in Rs]
        Mp = [X.copy() for X in Ms]
        for t in (1, 2):
            pm = rng.permutation(len(idx))
            for (i0, j0), q in zip(idx, pm):
                Rp[t][i0, j0] = Rs[t][idx[q][0], idx[q][1]]
                Mp[t][i0, j0] = Ms[t][idx[q][0], idx[q][1]]
        rr_n.append(float(np.corrcoef(Rp[0][mk], Rp[1][mk])[0, 1]))
        if len(rr_n) <= 24:          # 轴的置换较贵，少做几次够判
            a0, a1 = _axis(Mp[0], mk), _axis(Mp[1], mk)
            z = _abscos(a0[0], a0[1], a1[0], a1[1])
            if z:
                cc_n.append(z[0])

    out = {"name": f"{dom}/{ph}", "ok": True, "assay": assay,
           "n_cells": int(mk.sum()), "n_sites": int(keep.sum()), "d": d,
           "resid_corr": rr, "resid_corr_med": rmed,
           "axis_abscos": cc,
           "axis_abscos_med": float(np.median(cc)) if cc else None,
           "haar1d_p95": p95,
           "frac_noise_single": frac, "frac_noise_ml": frac_ml,
           "cross_axis_fitness_vs_ddg": cross[0] if cross else None,
           "cross_axis_d": cross[1] if cross else None,
           "link_fitness_int_vs_ddg_colmain": link[0] if link else None,
           "link_fitness_int_vs_fit_colmain": link_f[0] if link_f else None,
           "perm_resid_corr_med": float(np.median(rr_n)),
           "perm_resid_corr_p95": float(np.percentile(rr_n, 95)),
           "perm_axis_abscos_med": (float(np.median(cc_n)) if cc_n else None),
           "perm_axis_abscos_p95": (float(np.percentile(cc_n, 95))
                                    if cc_n else None),
           "ddg_censor": cen}
    # §10.2 不误伤断言
    assert out["perm_resid_corr_p95"] < 0.10, (
        f"{dom}/{ph}: 打乱配对后交互残差相关的 p95 还有 "
        f"{out['perm_resid_corr_p95']:.3f} —— 不误伤对照失败")
    if verbose:
        print(f"[{dom}/{ph}]  重复来源 {assay}，{int(mk.sum())} 个三重复齐全"
              f"的格子 / {int(keep.sum())} 个位点，d = {d}")
        print(f"   (1) 中心化交互残差的三对重复相关 r = "
              + ", ".join(f"{x:.3f}" for x in rr)
              + f"（中位 **{rmed:.3f}**）；打乱配对后中位 "
                f"{out['perm_resid_corr_med']:+.3f}、p95 "
                f"{out['perm_resid_corr_p95']:.3f}")
        print(f"   (2) 三对重复**独立**第一轴 |cos| = "
              + ", ".join(f"{x:.3f}" for x in cc)
              + (f"（中位 **{np.median(cc):.3f}**）" if cc else "")
              + f"；Haar 一维零分布 p95 = {p95:.3f}"
              + (f"；打乱后中位 {out['perm_axis_abscos_med']:.3f}"
                 if cc_n else ""))
        print(f"   (3) 由 r 直接反推：单重复噪声 / coup_sd_obs = "
              f"{frac:.3f}，三重复合并 = **{frac_ml:.3f}**")
        if cross:
            print(f"   (4) 重复 fitness 第一轴 vs mochi ΔΔG 第一轴 |cos| = "
                  f"**{cross[0]:.3f}**（{cross[1]} 维公共）")
        print(f"   (5) 连接非线性判别: fitness 交互第一轴 vs ΔΔG 残基主效应 "
              f"|cos| = "
              + (f"**{link[0]:.3f}**" if link else "n/a")
              + "；vs fitness 自己的残基主效应 |cos| = "
              + (f"**{link_f[0]:.3f}**" if link_f else "n/a")
              + "（高 = (1) 的高 r 主要来自饱和连接函数，不是真交互）")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=200)
    a = ap.parse_args()
    print("=" * 96)
    print("docs/28 Step 3：ddPCA 的独立生物学重复 —— 直接测量")
    print("=" * 96)
    RP.selfcheck(verbose=True)
    rows = []
    for dom, ph in S.MATS:
        rows.append(run(dom, ph, nperm=a.nperm))
        print()
    json.dump({"rows": rows}, open(OUTJSON, "w"), ensure_ascii=False,
              indent=1, default=float)
    print("产物:", OUTJSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
