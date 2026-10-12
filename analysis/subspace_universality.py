#!/usr/bin/env python3
"""二维残基载荷子空间是否跨独立体系普适？

问题：docs/12-rank-two.md 定下 site x residue 交互是秩 2。秩2 双线性里
V（残基载荷）只在可逆 2x2 变换下确定，所以**载荷向量不可辨识，
二维子空间 span(V) 可辨识**。本脚本只比较子空间（主角度）。

三道零分布（缺任何一道，数字都没有意义）：
  N1 随机子空间（Haar，同环境维数、同共同残基数）
  N2 位点内残基标签置换（保尺寸、保稀疏模式、保行边缘）
  N3 **成分1 混淆**：两边成分1 都是去溶剂化，必然共享一个方向。
     所以分开报 cos th1 与 cos th2，并报把疏水方向投掉之后的一维残差夹角。

先跑 analysis/subspace_selftest.py（CLAUDE.md §10），不通过不要看这里的数。

用法: python3 analysis/subspace_universality.py [--boot 2000] [--quick]
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import (AA20, ROOT, align, align_deflated,  # noqa: E402
                          bootstrap_subspaces, ci, extract_subspace,
                          haar_1d_null, haar_pair_null, load_all_dms,
                          dedup_by_content, load_independent, load_scales,
                          load_skempi, permute_within_sites, pick_primary)

CENSOR_MAX = 0.25          # >=25% 截尾 -> 8/9 给秩 0（docs/19 重算，原记 7/7）
CENSOR_CLEAN = 0.10
N_PERM = 200

# 「完全独立」= 不同的蛋白 + 不同的配体，与冠状病毒 RBD 无关
IND_DATASETS = {"lehner_ddPCA", "titeseq2016", "SKEMPI",
                "SKEMPI_OMTKY3"}
COV_REPS = [
    "animalACE2/Omicron_BA2/human_monomer",
    "animalACE2/Wuhan-Hu-1/bat",
    "sarbecovirus/SARS-CoV-1_Urban/Ra9479ACE2",
    "PDCoV/PDCoV/gAPN",
    "sars2_variants/Omicron_BA2/ACE2",
    "starr2020/WH1/huACE2",
]


def _ceil(a, b, relkey, stabkey):
    """两边「可达水平」的几何均值。对半分算不出来时退回 bootstrap 稳定性。
    注意这是**半数据量**（或 bootstrap 有放回）下的水平，是全数据可达对齐的
    偏低估计，不是严格上界 —— 文档里按这个口径解释。"""
    vs = []
    for d in (a, b):
        v = d.get(relkey)
        if v is None or not np.isfinite(v):
            v = d.get(stabkey)
        vs.append(0.0 if v is None or not np.isfinite(v) else max(0.0, v))
    return float(np.sqrt(vs[0] * vs[1]))


def pair_class(a, b):
    da, db = a["dataset"], b["dataset"]
    if da in IND_DATASETS or db in IND_DATASETS:
        if a["backbone"] != b["backbone"]:
            return "independent"
        return "same_backbone"
    if da == db:
        return "within_dataset"
    return "across_cov"


def main():
    boot = 2000
    quick = "--quick" in sys.argv
    if "--boot" in sys.argv:
        boot = int(sys.argv[sys.argv.index("--boot") + 1])
    if quick:
        boot = 200

    print("=" * 92)
    print("二维残基载荷子空间的跨体系普适性")
    print("=" * 92)

    scales = load_scales()

    # ---------------- 1. 清点与截尾筛选 ----------------
    mats = load_independent() + load_skempi() + load_all_dms()
    n0 = len(mats)
    mats, dups = dedup_by_content(mats)
    mats = pick_primary(mats)
    print(f"\n[1] 清点：{n0} 个 site x residue 矩阵 -> 按矩阵内容去重后 "
          f"{len(mats)} 个")
    for a, b in dups:
        print(f"      [内容重复，已丢弃] {a}  ==  {b}")
    kept, drop_cens, drop_fail = [], [], []
    for d in mats:
        if d["censor_frac"] >= CENSOR_MAX:
            drop_cens.append(d)
            continue
        B, cols, err = extract_subspace(d["M"], d["mask"])
        if B is None:
            drop_fail.append((d, err))
            continue
        d["B"], d["cols"] = B, cols
        kept.append(d)
    print(f"    截尾 >= {CENSOR_MAX:.0%} 剔除: {len(drop_cens)} 个"
          f"（rho(截尾, 检出秩) = -0.77 [-0.89,-0.54]，>=25% 时 8/9 给秩 0；docs/19）")
    print(f"    抽取失败剔除: {len(drop_fail)} 个")
    print(f"    保留: {len(kept)} 个")
    for d in drop_cens:
        print(f"      [截尾 {d['censor_frac']:.0%}] {d['name']}")
    for d, e in drop_fail:
        print(f"      [失败 {e}] {d['name']}")

    nind = [d for d in kept if d["dataset"] in IND_DATASETS]
    print(f"\n    其中**完全独立体系**（非冠状病毒 RBD）{len(nind)} 个:")
    for d in nind:
        print(f"      {d['name']:<34} {len(d['sites']):>4} 位点 "
              f"{d['n_obs']:>5} 观测 截尾 {d['censor_frac']:.1%}")

    # ---------------- 2. 零分布 ----------------
    print(f"\n[2] 零分布（{boot} 次重抽 / 4000 次随机抽样）")
    nulls = {}
    for d20 in sorted({r["n_res"] for r in
                       [{"n_res": len(np.intersect1d(a["cols"], b["cols"]))}
                        for a, b in itertools.combinations(kept, 2)]}):
        if d20 >= 8:
            nulls[d20] = haar_pair_null(d20, 2, n=4000, seed=d20)
    n1d = {d: haar_1d_null(d, n=4000, seed=100 + d) for d in nulls}
    for d20 in sorted(nulls):
        nn = nulls[d20]
        print(f"    N1 d={d20:>2}: cos1 中位 {np.median(nn['cos1']):.3f} "
              f"p95 {np.percentile(nn['cos1'], 95):.3f} | "
              f"cos2 中位 {np.median(nn['cos2']):.3f} "
              f"p95 {np.percentile(nn['cos2'], 95):.3f} | "
              f"sumcos2 p95 {np.percentile(nn['ssc'], 95):.3f}")

    # ---------------- 3. 全部配对的点估计 ----------------
    print(f"\n[3] 全部 {len(kept) * (len(kept) - 1) // 2} 个配对的点估计"
          f"（按配对类别汇总）")
    rows = []
    for a, b in itertools.combinations(kept, 2):
        r = align(a["B"], a["cols"], b["B"], b["cols"])
        if r is None:
            continue
        cls = pair_class(a, b)
        nn = nulls.get(r["n_res"])
        rows.append({"a": a["name"], "b": b["name"], "cls": cls,
                     "prim": bool(a["primary"] and b["primary"]),
                     "cos1": r["cos"][0], "cos2": r["cos"][1],
                     "ssc": r["ssc"], "n_res": r["n_res"],
                     "null_ssc_p95": float(np.percentile(nn["ssc"], 95))
                     if nn is not None else None,
                     "null_cos2_p95": float(np.percentile(nn["cos2"], 95))
                     if nn is not None else None})
    print(f"    {'类别':<16}{'n 对':>6}{'cos1 中位':>11}{'cos2 中位':>11}"
          f"{'sumcos2 中位':>13}{'超 N1 p95 的比例':>18}")
    print("-" * 78)
    summary = {}
    for cls in ("within_dataset", "across_cov", "same_backbone",
                "independent", "independent(主代表)"):
        if cls == "independent(主代表)":
            sub = [r for r in rows if r["cls"] == "independent" and r["prim"]]
        else:
            sub = [r for r in rows if r["cls"] == cls]
        if not sub:
            continue
        above = np.mean([r["ssc"] > r["null_ssc_p95"] for r in sub
                         if r["null_ssc_p95"]])
        print(f"    {cls:<16}{len(sub):>6}"
              f"{np.median([r['cos1'] for r in sub]):>11.3f}"
              f"{np.median([r['cos2'] for r in sub]):>11.3f}"
              f"{np.median([r['ssc'] for r in sub]):>13.3f}"
              f"{above:>17.0%}")
        summary[cls] = {"n": len(sub),
                        "cos1_med": float(np.median([r['cos1'] for r in sub])),
                        "cos2_med": float(np.median([r['cos2'] for r in sub])),
                        "ssc_med": float(np.median([r['ssc'] for r in sub])),
                        "frac_above_null_p95": float(above)}

    # ---------------- 4. 重点对：bootstrap + 置换零分布 ----------------
    focus = nind + [d for d in kept if d["name"] in COV_REPS]
    print(f"\n[4] 重点矩阵 {len(focus)} 个：bootstrap({boot}, 按位点重抽) "
          f"+ 置换零分布({N_PERM})")
    for d in focus:
        print(f"    抽取 {d['name']} ...", flush=True)
        d["boot"] = bootstrap_subspaces(d["M"], d["mask"], n=boot,
                                        seed=abs(hash(d["name"])) % 10 ** 6)
        rng = np.random.default_rng(7)
        perms = []
        for _ in range(N_PERM):
            Mp, kp = permute_within_sites(d["M"], d["mask"], rng)
            B, c, _ = extract_subspace(Mp, kp)
            perms.append(None if B is None else (B, c))
        d["perm"] = perms
        nb = sum(1 for x in d["boot"] if x)
        print(f"       bootstrap 成功 {nb}/{boot}，置换成功 "
              f"{sum(1 for x in perms if x)}/{N_PERM}", flush=True)

    # -------- 4b. 可达上限诊断（CLAUDE.md §10.3 强制） --------
    # 「测不出共享子空间」可能是现象没有，也可能是我的数据量不够。
    # 必须先量化后者：把同一体系的位点随机对半分，两半各自抽子空间再对齐。
    # 这是同一个真子空间 + 同样噪声水平下**可达的最好对齐**。
    print("\n[4b] 可达上限诊断：同体系位点对半分的自对齐（§10.3）")
    print("     若某体系的自对齐就已经接近随机零分布，则它给出的"
          "「不对齐」什么也不说明。")
    print(f"     {'体系':<34}{'位点':>5}{'自对齐 cos1 中位 [2.5,97.5]':>30}"
          f"{'自对齐 cos2 中位 [2.5,97.5]':>30}")
    nsplit = 100 if quick else 400
    for d in focus:
        rng = np.random.default_rng(42)
        nr = d["M"].shape[0]
        c1, c2 = [], []
        for _ in range(nsplit):
            p = rng.permutation(nr)
            h1, h2 = p[: nr // 2], p[nr // 2:]
            B1, k1, _ = extract_subspace(d["M"][h1], d["mask"][h1])
            B2, k2, _ = extract_subspace(d["M"][h2], d["mask"][h2])
            if B1 is None or B2 is None:
                continue
            r = align(B1, k1, B2, k2)
            if r:
                c1.append(r["cos"][0]); c2.append(r["cos"][1])
        d["rel_cos1"] = float(np.median(c1)) if c1 else float("nan")
        d["rel_cos2"] = float(np.median(c2)) if c2 else float("nan")
        d["rel_cos1_ci"] = ci(c1); d["rel_cos2_ci"] = ci(c2)
        print(f"     {d['name'][:33]:<34}{nr:>5}"
              f"{d['rel_cos1']:>14.3f} [{d['rel_cos1_ci'][0]:.2f},"
              f"{d['rel_cos1_ci'][1]:.2f}]"
              f"{d['rel_cos2']:>14.3f} [{d['rel_cos2_ci'][0]:.2f},"
              f"{d['rel_cos2_ci'][1]:.2f}]", flush=True)
    # 小矩阵（OMTKY3 各 10 位点）对半分后每列只剩 5 个观测，过不了列保护，
    # 自对齐会是 NaN。对所有矩阵再给一个能算的稳定性度量：
    # bootstrap 子空间与全数据子空间的对齐中位数。
    print("\n     稳定性（bootstrap 子空间 vs 全数据子空间，对半分算不出来时用这个）")
    print(f"     {'体系':<36}{'cos1 中位':>11}{'cos2 中位':>11}")
    for d in focus:
        s1, s2 = [], []
        for x in d["boot"]:
            if not x:
                continue
            r = align(d["B"], d["cols"], x[0], x[1])
            if r:
                s1.append(r["cos"][0]); s2.append(r["cos"][1])
        d["stab_cos1"] = float(np.median(s1)) if s1 else float("nan")
        d["stab_cos2"] = float(np.median(s2)) if s2 else float("nan")
        print(f"     {d['name'][:35]:<36}{d['stab_cos1']:>11.3f}"
              f"{d['stab_cos2']:>11.3f}")
    print(f"     （对照：随机零分布 d=20 的 cos1 p95 = "
          f"{np.percentile(nulls[20]['cos1'], 95):.3f}，"
          f"cos2 p95 = {np.percentile(nulls[20]['cos2'], 95):.3f}）")

    hyd = scales["kd_hydropathy"]
    vol = scales["partial_specific_volume"]
    hcols = np.array([i for i in range(20) if np.isfinite(hyd[i])])

    print(f"\n[5] 重点配对的对齐统计 + 三道零分布"
          f"（bootstrap 2.5-97.5% 区间；按 CLAUDE.md §5 比区间重叠）")
    hdr = (f"{'配对':<40}{'d':>3}{'cos1':>7}{'上限':>7}"
           f"{'cos2':>7}{'上限':>7}{'N1p95':>7}{'N2 95%':>7}"
           f"{'投疏水后':>9}{'N1_1d':>7}"
           f"{'cos2 boot CI':>16}")
    focus_rows = []
    for a, b in itertools.combinations(focus, 2):
        base = align(a["B"], a["cols"], b["B"], b["cols"])
        if base is None:
            continue
        cls = pair_class(a, b)
        bc1, bc2, bssc = [], [], []
        for x, y in zip(a["boot"], b["boot"]):
            if not x or not y:
                continue
            r = align(x[0], x[1], y[0], y[1])
            if r:
                bc1.append(r["cos"][0]); bc2.append(r["cos"][1])
                bssc.append(r["ssc"])
        pc1, pc2 = [], []
        for x, y in zip(a["perm"], b["perm"]):
            if not x or not y:
                continue
            r = align(x[0], x[1], y[0], y[1])
            if r:
                pc1.append(r["cos"][0]); pc2.append(r["cos"][1])
        dfl = align_deflated(a["B"], a["cols"], b["B"], b["cols"],
                             hyd[hcols], hcols)
        bd = []
        for x, y in zip(a["boot"], b["boot"]):
            if not x or not y:
                continue
            r = align_deflated(x[0], x[1], y[0], y[1], hyd[hcols], hcols)
            if r:
                bd.append(r["cos_resid"])
        nn = nulls.get(base["n_res"])
        rec = {"a": a["name"], "b": b["name"], "cls": cls,
               "prim": bool(a["primary"] and b["primary"]),
               "n_res": base["n_res"],
               "cos1": base["cos"][0], "cos1_ci": ci(bc1),
               "cos2": base["cos"][1], "cos2_ci": ci(bc2),
               "ssc": base["ssc"], "ssc_ci": ci(bssc),
               "perm_cos1_ci": ci(pc1), "perm_cos2_ci": ci(pc2),
               "perm_cos2_med": float(np.median(pc2)) if pc2 else None,
               "defl": dfl["cos_resid"] if dfl else None,
               "defl_ci": ci(bd),
               "h_in_a": dfl["h_in_a"] if dfl else None,
               "h_in_b": dfl["h_in_b"] if dfl else None,
               "null_cos1_p95": float(np.percentile(nn["cos1"], 95)),
               "null_cos2_p95": float(np.percentile(nn["cos2"], 95)),
               "null_ssc_p95": float(np.percentile(nn["ssc"], 95)),
               "null_1d_p95": float(np.percentile(n1d[base["n_res"]], 95)),
               # bootstrap 中位 vs 点估计：量化 bootstrap 的向下偏（见文档）
               "boot_med_cos1": float(np.median(bc1)) if bc1 else None,
               "boot_med_cos2": float(np.median(bc2)) if bc2 else None,
               "rel_cos1_a": a.get("rel_cos1"), "rel_cos1_b": b.get("rel_cos1"),
               "rel_cos2_a": a.get("rel_cos2"), "rel_cos2_b": b.get("rel_cos2"),
               "stab_cos1_a": a.get("stab_cos1"), "stab_cos1_b": b.get("stab_cos1"),
               "stab_cos2_a": a.get("stab_cos2"), "stab_cos2_b": b.get("stab_cos2"),
               # 可达上限：两边自对齐的几何均值（点估计能达到的最好水平）
               "ceiling_cos1": _ceil(a, b, "rel_cos1", "stab_cos1"),
               "ceiling_cos2": _ceil(a, b, "rel_cos2", "stab_cos2")}
        focus_rows.append(rec)

    for cls in ("independent", "across_cov", "within_dataset",
                "same_backbone"):
        sub = [r for r in focus_rows if r["cls"] == cls]
        if not sub:
            continue
        print(f"\n--- {cls}  ({len(sub)} 对) ---")
        print(hdr)
        for r in sorted(sub, key=lambda x: -x["cos2"]):
            nm = (f"{r['a'].split('/')[1][:13]} x "
                  f"{r['b'].split('/')[1][:13]}")
            print(f"{nm:<40}{r['n_res']:>3}"
                  f"{r['cos1']:>7.3f}{r['ceiling_cos1']:>7.3f}"
                  f"{r['cos2']:>7.3f}{r['ceiling_cos2']:>7.3f}"
                  f"{r['null_cos2_p95']:>7.3f}"
                  f"{r['perm_cos2_ci'][1]:>7.3f}"
                  f"{(r['defl'] if r['defl'] is not None else float('nan')):>9.3f}"
                  f"{r['null_1d_p95']:>7.3f}"
                  f"   [{r['cos2_ci'][0]:.2f},{r['cos2_ci'][1]:.2f}]")

    # ---------------- 6. 共识子空间 ----------------
    print("\n[6] 共识子空间：平均投影算子的特征分解")
    print("    平均投影算子 Pbar = mean_i Q_i Q_i^T 的特征值 = 「该方向被多少")
    print("    比例的体系共享」。随机情形每个方向期望 2/(d-1) = %.3f。"
          % (2 / 19))

    def consensus(group, label):
        Ps, ok = [], []
        for d in group:
            if len(d["cols"]) != 20:
                continue
            Q = d["B"]
            one = np.ones((20, 1)) / np.sqrt(20)
            Q = Q - one @ (one.T @ Q)
            Q, _ = np.linalg.qr(Q)
            Ps.append(Q @ Q.T)
            ok.append(d["name"])
        if len(Ps) < 2:
            return None
        Pb = np.mean(Ps, axis=0)
        w, V = np.linalg.eigh(Pb)
        w = w[::-1]; V = V[:, ::-1]
        print(f"\n    [{label}] {len(Ps)} 个体系")
        print(f"      特征值前 5: " +
              " ".join(f"{x:.3f}" for x in w[:5]))
        return {"w": w, "V": V, "names": ok, "n": len(Ps)}

    nind_prim = [d for d in nind if d["primary"]]
    print(f"\n    （共识只用每个 backbone 的主代表：{len(nind_prim)} 个独立体系 "
          f"{[d['name'] for d in nind_prim]}）")
    cons_ind = consensus(nind_prim, "完全独立体系（每骨架一个代表）")
    cons_cov = consensus([d for d in kept if d["name"] in COV_REPS],
                         "冠状病毒代表")
    cons_all = consensus(focus, "全部重点体系")

    # 共识特征值的随机零分布
    rng = np.random.default_rng(11)
    nsim = 2000
    null_w = []
    for _ in range(nsim):
        Ps = []
        for _ in range(cons_ind["n"]):
            one = np.ones((20, 1)) / np.sqrt(20)
            A = rng.normal(size=(20, 2)); A -= one @ (one.T @ A)
            Q, _ = np.linalg.qr(A)
            Ps.append(Q @ Q.T)
        w = np.linalg.eigvalsh(np.mean(Ps, axis=0))[::-1]
        null_w.append(w[:3])
    null_w = np.array(null_w)
    print(f"\n    随机零分布（{cons_ind['n']} 个随机 2 维面，{nsim} 次）:")
    for k in range(3):
        print(f"      lambda_{k+1} 中位 {np.median(null_w[:, k]):.3f}  "
              f"p95 {np.percentile(null_w[:, k], 95):.3f}")

    # bootstrap 共识特征值
    bw = []
    for t in range(min(boot, 1000)):
        Ps = []
        for d in nind_prim:
            x = d["boot"][t] if t < len(d["boot"]) else None
            if not x or len(x[1]) != 20:
                continue
            one = np.ones((20, 1)) / np.sqrt(20)
            Q = x[0] - one @ (one.T @ x[0])
            Q, _ = np.linalg.qr(Q)
            Ps.append(Q @ Q.T)
        if len(Ps) >= 2:
            bw.append(np.linalg.eigvalsh(np.mean(Ps, axis=0))[::-1][:3])
    bw = np.array(bw)
    print(f"\n    完全独立体系共识特征值 bootstrap（{len(bw)} 次有效）:")
    for k in range(3):
        lo, hi = np.percentile(bw[:, k], [2.5, 97.5])
        print(f"      lambda_{k+1} = {cons_ind['w'][k]:.3f} "
              f"[{lo:.3f}, {hi:.3f}]   vs 零分布 p95 "
              f"{np.percentile(null_w[:, k], 95):.3f}  "
              f"{'区间不重叠 -> 真共享' if lo > np.percentile(null_w[:, k], 95) else '区间与零分布重叠'}")

    # ---------------- 7. 化学身份 ----------------
    print("\n[7] 共识方向的化学身份：|cos(标度, 共识方向)|")
    print("    对每条标度，取其有定义的残基 -> 在该子集内去均值 -> 与共识方向求夹角。")
    out_chem = {}
    for label, cons in (("独立体系共识", cons_ind),
                        ("全部重点体系共识", cons_all)):
        if cons is None:
            continue
        print(f"\n    [{label}]")
        print(f"      {'标度':<28}{'n':>4}{'|cos v1|':>10}{'|cos v2|':>10}"
              f"{'在 2 维内的投影占比':>20}")
        res = []
        for nm, s in sorted(scales.items()):
            idx = np.array([i for i in range(20) if np.isfinite(s[i])])
            if len(idx) < 10:
                continue
            v = s[idx] - s[idx].mean()
            if np.linalg.norm(v) < 1e-12:
                continue
            v = v / np.linalg.norm(v)
            # 把共识方向限制到同一残基子集并再中心化
            Vs = cons["V"][:, :2][idx, :]
            Vs = Vs - Vs.mean(0, keepdims=True)
            Q, _ = np.linalg.qr(Vs)
            c1 = abs(float(Q[:, 0] @ v)); c2 = abs(float(Q[:, 1] @ v))
            frac = float(np.linalg.norm(Q.T @ v) ** 2)
            res.append((nm, len(idx), c1, c2, frac))
        for nm, n, c1, c2, frac in sorted(res, key=lambda t: -t[4]):
            print(f"      {nm:<28}{n:>4}{c1:>10.3f}{c2:>10.3f}{frac:>20.3f}")
        out_chem[label] = res

    # 已知标度能张成共识子空间多少？
    print("\n    已知标度对共识 2 维子空间的覆盖（贪心选 2 条）:")
    for label, cons in (("独立体系共识", cons_ind),):
        full = [nm for nm, s in scales.items()
                if np.isfinite(s).all()]
        Vs = cons["V"][:, :2]
        one = np.ones((20, 1)) / np.sqrt(20)
        Vs = Vs - one @ (one.T @ Vs)
        Qc, _ = np.linalg.qr(Vs)
        best = None
        for p, q in itertools.combinations(full, 2):
            A = np.stack([scales[p], scales[q]], 1).astype(float)
            A = A - A.mean(0, keepdims=True)
            Qa, _ = np.linalg.qr(A)
            s = np.linalg.svd(Qa.T @ Qc, compute_uv=False)
            ssc = float((s ** 2).sum())
            if best is None or ssc > best[0]:
                best = (ssc, p, q, s.tolist())
        print(f"      最佳两条标度 = {best[1]} + {best[2]}")
        print(f"      sum cos^2 = {best[0]:.3f} / 2.0 "
              f"（主角度 cos = {best[3][0]:.3f}, {best[3][1]:.3f}）")
        print(f"      => 共识子空间里 {best[0] / 2:.0%} 被已知标度张成，"
              f"剩 {1 - best[0] / 2:.0%} 不在任何两条已知标度的平面内")
        out_chem["best_two_scales"] = {"ssc": best[0], "a": best[1],
                                       "b": best[2], "cos": best[3]}

    # ---------------- 8. 量级校验 ----------------
    print("\n[8] 量级校验（CLAUDE.md §8）")
    print(f"    随机两个 2 维面 E[sum cos^2] = 4/19 = {4/19:.4f}；"
          f"实测零分布中位 {np.median(nulls[20]['ssc']):.4f}")
    ii = [r for r in focus_rows if r["cls"] == "independent"]
    if ii:
        print(f"    独立配对 sum cos^2 中位 {np.median([r['ssc'] for r in ii]):.3f}"
              f"（上限 2.0）；若超过 2.0 或低于 0 则实现有错")
        assert all(-1e-9 <= r["ssc"] <= 2 + 1e-9 for r in ii)
    print("    cos 必须落在 [0,1]：", all(
        0 <= r["cos2"] <= r["cos1"] <= 1 + 1e-9 for r in focus_rows))

    outp = os.path.join(ROOT, "cache", "subspace_universality.json")
    json.dump({"censor_max": CENSOR_MAX, "n_boot": boot, "n_perm": N_PERM,
               "kept": [d["name"] for d in kept],
               "dropped_censor": [{"name": d["name"],
                                   "censor_frac": d["censor_frac"]}
                                  for d in drop_cens],
               "dropped_fail": [{"name": d["name"], "err": e}
                                for d, e in drop_fail],
               "independent_systems": [d["name"] for d in nind],
               "all_pair_summary": summary,
               "all_pairs": rows, "focus_pairs": focus_rows,
               "consensus_ind_eigs": cons_ind["w"][:5].tolist(),
               "consensus_ind_V2": cons_ind["V"][:, :2].tolist(),
               "consensus_null_eig_p95":
                   np.percentile(null_w, 95, axis=0).tolist(),
               "consensus_boot_ci":
                   np.percentile(bw, [2.5, 97.5], axis=0).tolist(),
               "aa_order": AA20,
               "chem": {k: v for k, v in out_chem.items()}},
              open(outp, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(outp, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
