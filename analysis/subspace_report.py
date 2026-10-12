#!/usr/bin/env python3
"""读 cache/subspace_universality.json，出三张判决表。

1. 共识方向的残基载荷 + **用 docs/12 已建立的成分2 特征做已知事实核对**
   （已建立：成分2 与平面 sp2 指示量 r = -0.768；最正 Ile/Val/Thr/Cys，
     最负 Glu/Trp/Asp）。若共识 v2 复现不出这个特征，说明抽取或对齐有问题。
2. 衰减分析：独立配对的 cos2 与「可达水平」是否同步。
   这是区分「子空间体系特异」与「我的数据分辨不出第二维」的关键。
3. 分辨力达标的独立配对 vs 三道零分布。
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import AA20, ROOT, load_scales  # noqa: E402

CEIL_MIN = 0.45
STAB_MIN = 0.60   # 「第二维可分辨」的门槛，见文中说明


def main():
    J = json.load(open(os.path.join(ROOT, "cache",
                                    "subspace_universality.json")))
    scales = load_scales(verbose=False)
    V = np.array(J["consensus_ind_V2"])
    print("=" * 88)
    print("判决表 1：独立体系共识 2 维子空间的残基载荷")
    print("=" * 88)
    print(f"  共识特征值 {['%.3f' % x for x in J['consensus_ind_eigs']]}")
    print(f"  随机零分布 p95 {['%.3f' % x for x in J['consensus_null_eig_p95']]}")
    print(f"  bootstrap 2.5/97.5% {np.array(J['consensus_boot_ci']).round(3).tolist()}")
    print(f"\n  {'残基':<5}{'v1':>9}{'v2':>9}")
    o1 = np.argsort(-V[:, 0]); o2 = np.argsort(-V[:, 1])
    for i in range(20):
        print(f"  {AA20[i]:<5}{V[i,0]:>9.3f}{V[i,1]:>9.3f}")
    print(f"\n  v1 最正 -> 最负: {''.join(AA20[i] for i in o1)}")
    print(f"  v2 最正 -> 最负: {''.join(AA20[i] for i in o2)}")

    print("\n  已知事实核对（docs/12 建立的成分身份；不是自洽性检查）")
    hyd = scales["kd_hydropathy"]
    sp2 = scales["planar_sp2"]
    vol = scales["partial_specific_volume"]
    mr = scales["molar_refractivity"]
    for nm, s, direction in (("成分1 ~ 疏水（Kyte-Doolittle）", hyd, "v1"),
                             ("成分1 ~ 体积（偏比容）", vol, "v1"),
                             ("成分2 ~ 平面 sp2 指示量", sp2, "v2"),
                             ("成分2 ~ 摩尔折射度", mr, "v2")):
        k = 0 if direction == "v1" else 1
        r = float(np.corrcoef(V[:, k], s)[0, 1])
        print(f"    r({direction}, {nm}) = {r:+.3f}")
    pos4 = set("IVTC"); neg3 = set("EWD")
    top4 = set(AA20[i] for i in o2[:4]); bot3 = set(AA20[i] for i in o2[-3:])
    top4b = set(AA20[i] for i in o2[-4:]); bot3b = set(AA20[i] for i in o2[:3])
    hit = max(len(pos4 & top4) + len(neg3 & bot3),
              len(pos4 & top4b) + len(neg3 & bot3b))
    print(f"    docs/12 的成分2 签名（最正 IVTC / 最负 EWD，符号任意）: "
          f"命中 {hit}/7")
    print(f"      共识 v2 最正 4 个 = {''.join(sorted(top4))}，"
          f"最负 3 个 = {''.join(sorted(bot3))}")

    # ---------------- 2. 衰减分析 ----------------
    print("\n" + "=" * 88)
    print("判决表 2：衰减分析 —— cos2 低是「体系特异」还是「分辨不出第二维」")
    print("=" * 88)
    fr = J["focus_pairs"]
    ind = [r for r in fr if r["cls"] == "independent" and r.get("prim", True)]
    c2 = np.array([r["cos2"] for r in ind])
    ce = np.array([r["ceiling_cos2"] for r in ind])
    rho, p = spearmanr(ce, c2)
    print(f"  独立配对 {len(ind)} 对")
    print(f"  Spearman(可达水平 ceiling_cos2, 实测 cos2) = {rho:+.3f}  p = {p:.2g}")
    print("  读法：rho 高 => 低 cos2 主要由分辨力不足造成（共享但测不出）；")
    print("        rho ~ 0 且 cos2 普遍低 => 第二维真的体系特异。")
    print(f"\n  {'配对':<44}{'ceiling2':>9}{'cos2':>8}{'cos1':>8}"
          f"{'投疏水后':>9}{'N1_1d p95':>10}")
    for r in sorted(ind, key=lambda x: -x["ceiling_cos2"]):
        nm = f"{r['a'].split('/')[1][:14]} x {r['b'].split('/')[1][:14]}"
        print(f"  {nm:<44}{r['ceiling_cos2']:>9.3f}{r['cos2']:>8.3f}"
              f"{r['cos1']:>8.3f}"
              f"{(r['defl'] if r['defl'] is not None else np.nan):>9.3f}"
              f"{r['null_1d_p95']:>10.3f}")

    # ---------------- 3. 分辨力达标子集 ----------------
    # ---- 2b. 用 bootstrap 稳定性定「可分辨」（对半分对 10 位点矩阵算不出来）----
    print("\n" + "=" * 88)
    print(f"判决表 2b：按 bootstrap 稳定性筛「第二维可分辨」的**全部**独立配对")
    print("=" * 88)
    print("  可分辨判据：双方 stab_cos2（bootstrap 子空间 vs 全数据子空间的")
    print(f"  对齐中位数）都 >= {STAB_MIN}。OMTKY3 的 4 个单酶矩阵只有 10 位点，")
    print("  对半分后过不了列保护，所以必须用这个度量。")
    ind_all = [r for r in fr if r["cls"] == "independent"]
    def sok(r):
        return (r.get("stab_cos2_a") or 0) >= STAB_MIN and \
               (r.get("stab_cos2_b") or 0) >= STAB_MIN
    good2 = [r for r in ind_all if sok(r)]
    print(f"\n  可分辨的独立配对 {len(good2)} / {len(ind_all)}")
    print(f"  {'配对':<44}{'stab2 a':>8}{'stab2 b':>8}{'cos1':>8}{'cos2':>8}"
          f"{'投疏水后':>9}{'N1 cos2 p95':>12}{'1d p95':>8}")
    for r in sorted(good2, key=lambda x: -x["cos2"]):
        nm = f"{r['a'].split('/')[1][:14]} x {r['b'].split('/')[1][:14]}"
        print(f"  {nm:<44}{r['stab_cos2_a']:>8.3f}{r['stab_cos2_b']:>8.3f}"
              f"{r['cos1']:>8.3f}{r['cos2']:>8.3f}"
              f"{(r['defl'] if r['defl'] is not None else np.nan):>9.3f}"
              f"{r['null_cos2_p95']:>12.3f}{r['null_1d_p95']:>8.3f}")
    if good2:
        v1 = np.array([r["cos1"] for r in good2])
        v2 = np.array([r["cos2"] for r in good2])
        vd = np.array([r["defl"] for r in good2 if r["defl"] is not None])
        p95_1 = good2[0]["null_cos1_p95"]; p95_2 = good2[0]["null_cos2_p95"]
        p95_d = good2[0]["null_1d_p95"]
        print(f"\n  cos1: 中位 {np.median(v1):.3f}  区间 {v1.min():.3f}-"
              f"{v1.max():.3f}  超 N1 p95({p95_1:.3f}) 的比例 "
              f"{np.mean(v1 > p95_1):.0%}")
        print(f"  cos2: 中位 {np.median(v2):.3f}  区间 {v2.min():.3f}-"
              f"{v2.max():.3f}  超 N1 p95({p95_2:.3f}) 的比例 "
              f"{np.mean(v2 > p95_2):.0%}")
        print(f"  投疏水后一维残差: 中位 {np.median(vd):.3f}  区间 "
              f"{vd.min():.3f}-{vd.max():.3f}  超 N1 p95({p95_d:.3f}) 的比例 "
              f"{np.mean(vd > p95_d):.0%}")
        rho2, pp2 = spearmanr([min(r["stab_cos2_a"], r["stab_cos2_b"])
                               for r in ind_all if r.get("stab_cos2_a")],
                              [r["cos2"] for r in ind_all
                               if r.get("stab_cos2_a")])
        print(f"\n  Spearman(min(stab_cos2), cos2) 在全部 {len(ind_all)} 个"
              f"独立配对上 = {rho2:+.3f}  p = {pp2:.2g}")
        print("  => 若接近 0，低 cos2 不是分辨力问题。")

    print("\n" + "=" * 88)
    print(f"判决表 3：第二维可分辨（对半分 ceiling_cos2 >= {CEIL_MIN}，保守口径）"
          f"的独立配对")
    print("=" * 88)
    good = [r for r in ind if r["ceiling_cos2"] >= CEIL_MIN]
    bad = [r for r in ind if r["ceiling_cos2"] < CEIL_MIN]
    print(f"  可分辨 {len(good)} 对 / 不可分辨 {len(bad)} 对")
    if good:
        print(f"\n  {'配对':<42}{'cos1 [boot CI]':>22}{'cos2 [boot CI]':>22}"
              f"{'N1p95':>7}{'N2 97.5%':>9}{'投疏水后 [CI]':>20}{'1d N1p95':>9}")
        for r in sorted(good, key=lambda x: -x["cos2"]):
            nm = f"{r['a'].split('/')[1][:13]} x {r['b'].split('/')[1][:13]}"
            print(f"  {nm:<42}"
                  f"{r['cos1']:>8.3f} [{r['cos1_ci'][0]:.2f},{r['cos1_ci'][1]:.2f}]"
                  f"{r['cos2']:>8.3f} [{r['cos2_ci'][0]:.2f},{r['cos2_ci'][1]:.2f}]"
                  f"{r['null_cos2_p95']:>7.3f}{r['perm_cos2_ci'][1]:>9.3f}"
                  f"{(r['defl'] if r['defl'] is not None else np.nan):>8.3f}"
                  f" [{r['defl_ci'][0]:.2f},{r['defl_ci'][1]:.2f}]"
                  f"{r['null_1d_p95']:>9.3f}")
        for key, lbl in (("cos1", "cos1"), ("cos2", "cos2"),
                         ("defl", "投掉疏水后的一维残差")):
            v = np.array([r[key] for r in good if r[key] is not None])
            nn = np.array([r["null_cos2_p95"] if key == "cos2"
                           else (r["null_cos1_p95"] if key == "cos1"
                                 else r["null_1d_p95"]) for r in good])
            print(f"\n  {lbl}: 中位 {np.median(v):.3f}  范围 "
                  f"{v.min():.3f}-{v.max():.3f}  "
                  f"超 N1 p95 的比例 {np.mean(v > nn):.0%}")
        lows = [r["cos2_ci"][0] for r in good]
        print(f"\n  cos2 的 bootstrap 下界范围 {min(lows):.3f}-{max(lows):.3f}；"
              f"N1 p95 = {good[0]['null_cos2_p95']:.3f}")
        print("  注意：按位点有放回重抽会把不同位点数**降到约 63%**，对"
              "「子空间对齐」这种有下界的统计量系统性向下偏。")
        bm = [r["boot_med_cos2"] for r in good if r["boot_med_cos2"]]
        pm = [r["cos2"] for r in good if r["boot_med_cos2"]]
        print(f"  量化这个偏：bootstrap 中位 / 点估计 = "
              f"{np.median(np.array(bm) / np.array(pm)):.2f}（<1 即向下偏）")

    # ---------------- 4. 类别汇总 ----------------
    print("\n" + "=" * 88)
    print("判决表 4：全部配对的类别汇总（点估计）")
    print("=" * 88)
    s = J["all_pair_summary"]
    print(f"  {'类别':<18}{'n 对':>6}{'cos1 中位':>11}{'cos2 中位':>11}"
          f"{'sumcos2 中位':>13}{'超 N1 p95':>11}")
    for k in ("within_dataset", "across_cov", "same_backbone",
              "independent", "independent(主代表)"):
        if k not in s:
            continue
        v = s[k]
        print(f"  {k:<18}{v['n']:>6}{v['cos1_med']:>11.3f}"
              f"{v['cos2_med']:>11.3f}{v['ssc_med']:>13.3f}"
              f"{v['frac_above_null_p95']:>10.0%}")
    print("\n" + "=" * 88)
    print("判决表 5：OMTKY3 同骨架 x 4 种蛋白酶（非病毒体系上的配体对照）")
    print("=" * 88)
    sb = [r for r in fr if r["cls"] == "same_backbone"]
    print(f"  {'配对':<44}{'cos1':>8}{'cos2':>8}{'ceil2':>8}"
          f"{'投疏水后':>9}{'N2 97.5%':>10}")
    for r in sorted(sb, key=lambda x: -x["cos2"]):
        nm = f"{r['a'].split('/')[1][:14]} x {r['b'].split('/')[1][:14]}"
        print(f"  {nm:<44}{r['cos1']:>8.3f}{r['cos2']:>8.3f}"
              f"{r['ceiling_cos2']:>8.3f}"
              f"{(r['defl'] if r['defl'] is not None else np.nan):>9.3f}"
              f"{r['perm_cos2_ci'][1]:>10.3f}")
    if sb:
        print(f"\n  cos1 中位 {np.median([r['cos1'] for r in sb]):.3f}，"
              f"cos2 中位 {np.median([r['cos2'] for r in sb]):.3f}")
        print("  对照：冠状病毒的同数据集跨配体 cos2 中位见判决表 4 的 "
              "within_dataset 行。")

    print(f"\n  N1（随机）d=20: cos1 中位 0.404 p95 0.634；"
          f"cos2 中位 0.104 p95 0.304；sumcos2 中位 0.185 p95 0.455")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
