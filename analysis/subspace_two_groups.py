#!/usr/bin/env python3
"""docs/21 第 2 阶段：在**修订后的保留集**上、**去掉 GB1** 之后重算 docs/13。

两项输入都是人类已决定的（CLAUDE.md §9，本脚本不再讨论）：
  1. 保留集由密度型检测器定（`cache/censor_density_revision.json`）；
  2. GB1 binding 从跨家族证据里剔除（docs/15 §7.2：它没有独立的
     abundance assay，coup_sd_true = 0，定秩给秩 0）。
     于是跨家族证据从**三组**变成**两组**：OMTKY3 蛋白酶组 + 冠状病毒家族。

数值层全部复用，没有重写：
  subspace_lib（extract_subspace / align / align_deflated / haar_*/ load_scales）
  subspace_cond_null（proj / eigs / cond_null / uncond_null / _draw）
  subspace_resid_dim（centered_orth / spectrum）
  rank_test_power.als_guarded（经 subspace_lib 导入，**不是**未保护的 als_complete）

**零分布必须在新的 k 上重生成**：一个 k 的 PR 不能和另一个 k 建的零分布比。
本脚本对每一组都现场生成零分布，k 直接取该组的大小。

运行：python3 analysis/subspace_two_groups.py
产物：cache/subspace_two_groups.json
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subspace_cond_null as CN                                # noqa: E402
import subspace_resid_dim as RD                                # noqa: E402
from subspace_lib import (AA20, ROOT, align, align_deflated,    # noqa: E402
                          dedup_by_content, extract_subspace,
                          haar_1d_null, haar_pair_null,
                          load_all_dms, load_independent, load_scales,
                          load_skempi, pick_primary)

CENSOR_MAX = 0.25
STAB_MIN = 0.60
NSIM = 4000

COV_REPS = ["starr2020/WH1/huACE2", "animalACE2/Wuhan-Hu-1/bat",
            "PDCoV/PDCoV/gAPN",
            "sarbecovirus/SARS-CoV-1_Urban/Ra9479ACE2",
            "animalACE2/Omicron_BA2/human_monomer"]
OMT = ["OMTKY3/chymotrypsin/1CHO", "OMTKY3/HLE-elastase/1PPF",
       "OMTKY3/subtilisin-Carlsberg/1R0R", "OMTKY3/SGPB/3SGB"]
IND_DATASETS = {"lehner_ddPCA", "titeseq2016", "SKEMPI", "SKEMPI_OMTKY3"}
DROPPED = "lehner/GB1/IgG-Fc"      # 人类决定：从跨家族证据里剔除


def load_new_censor():
    p = os.path.join(ROOT, "cache", "censor_density_revision.json")
    if not os.path.exists(p):
        raise RuntimeError("先跑 analysis/censor_density_revision.py")
    J = json.load(open(p))
    return {z["name"]: z["new"] for z in J["rows"]}, J


def main():
    newc, REV = load_new_censor()
    scales = load_scales(verbose=True)       # 内含 aaindex 强制自检
    # 已知事实的**第三次**独立核验（§10.1：用已知事实，不用自洽性）
    mr = scales["molar_refractivity"]
    gi, wi = AA20.index("G"), AA20.index("W")
    assert int(np.nanargmin(mr)) == gi, "摩尔折射度最小不是 Gly"
    assert int(np.nanargmax(mr)) == wi, "摩尔折射度最大不是 Trp"
    print(f"[自检] 摩尔折射度 Gly 最小({mr[gi]:.1f}) / "
          f"Trp 最大({mr[wi]:.1f}) —— 独立核验通过")

    mats, _ = dedup_by_content(load_independent() + load_skempi()
                               + load_all_dms())
    miss = [d["name"] for d in mats if d["name"] not in newc]
    if miss:
        raise RuntimeError(f"缓存里缺这些矩阵的新截尾值: {miss[:3]}")
    for d in mats:
        d["censor_frac_old"] = d["censor_frac"]
        d["censor_frac"] = newc[d["name"]]      # 新口径作为唯一真相
    mats = pick_primary(mats)
    keep = [d for d in mats if d["censor_frac"] < CENSOR_MAX]
    print(f"\n保留集：{len(keep)} 个矩阵（旧口径 48，见 docs/21 §3）")

    U = json.load(open(os.path.join(ROOT, "cache",
                                    "subspace_universality.json")))
    stab = {}
    for r in U["focus_pairs"]:
        for side in ("a", "b"):
            if r.get(f"stab_cos2_{side}") is not None:
                stab[r[side]] = r[f"stab_cos2_{side}"]

    by, cols_by, info = {}, {}, {}
    for d in keep:
        B, c, _ = extract_subspace(d["M"], d["mask"])
        if B is None:
            continue
        cols_by[d["name"]] = c
        info[d["name"]] = d
        if len(c) == 20:
            by[d["name"]] = B

    # ---------------- 1. 两组证据的定义 ----------------
    groups = {
        "冠状病毒家族（阳性对照：已知共享两维）": [n for n in COV_REPS
                                                 if n in by],
        "OMTKY3 × 4 蛋白酶（同骨架、不同配体）": [n for n in OMT if n in by],
        "跨家族（两组各 1 代表：冠状病毒 / OMTKY3）":
            [n for n in ("starr2020/WH1/huACE2",
                         "SKEMPI/OMTKY3/pooled-4proteases") if n in by],
        "可分辨全集（4 OMTKY3 单酶 + 5 冠状病毒，**已去 GB1**）":
            [n for n in OMT + COV_REPS if n in by],
    }
    print(f"\n已按人类决定剔除 {DROPPED}"
          f"（在保留集里？{'是' if DROPPED in by else '否'}）")
    for k, v in groups.items():
        print(f"   [{k}] k = {len(v)}")
        for n in v:
            print(f"       {n}  截尾 {info[n]['censor_frac']:.1%}"
                  f"（旧 {info[n]['censor_frac_old']:.1%}）"
                  f"  stab cos2 = "
                  f"{stab.get(n, float('nan')):.3f}")

    # ---------------- 2. 共识 λ + 标定后的条件零分布 ----------------
    rng = np.random.default_rng(20261012)
    print("\n" + "=" * 92)
    print(f"共识层面：λ1 / λ2 与标定后的条件零分布（每格 {CN.NSIM} 次模拟）")
    print("=" * 92)
    cons = []
    for label, names in groups.items():
        if len(names) < 2:
            print(f"\n[{label}] k = {len(names)} < 2，跳过")
            continue
        r = CN.report(label, names, by, rng)
        cons.append(r)
    print("\n量级校验：标定后条件零分布的 λ1 中位必须等于观测 λ1")
    cal_ok = True
    for r in cons:
        dd = abs(r["cond_med"][0] - r["eigs"][0])
        cal_ok &= dd < 0.02
        print(f"   [{r['label'][:44]:<46}] 零分布 λ1 中位 "
              f"{r['cond_med'][0]:.3f} vs 观测 {r['eigs'][0]:.3f}  "
              f"差 {dd:.3f}  {'OK' if dd < 0.02 else '<< 标定失败'}")

    print("\n负对照（§10.2 不误伤）：合成「只共享一维」的子空间，"
          "同一检验必须**不**超出")
    rng2 = np.random.default_rng(99)
    u0 = np.array(cons[-1]["u"])
    negc = []
    for k in sorted({len(v) for v in groups.values() if len(v) >= 2}):
        tgt = float([r for r in cons
                     if len(r["names"]) == k][0]["eigs"][0])
        al = CN.cond_null(k, u0, rng2, nsim=10, target_l1=tgt)[1]
        obs = float(np.median([CN._draw(k, u0, al, rng2)[1]
                               for _ in range(400)]))
        cn2, _ = CN.cond_null(k, u0, rng2, nsim=1500, target_l1=tgt)
        pp = float(np.mean(cn2[:, 1] >= obs))
        negc.append({"k": k, "target_l1": tgt, "lam2": obs, "p": pp})
        print(f"   k = {k}、λ1 标到 {tgt:.3f}: 合成 λ2 = {obs:.3f}，"
              f"条件零分布 p = {pp:.3f}  "
              f"{'OK（未误报）' if pp > 0.05 else '<< 误报'}")

    # ---------------- 3. 残差方向的参与率（零分布在新 k 上重生成） ----
    print("\n" + "=" * 92)
    print("第二方向的有效维数 PR —— **零分布在每组自己的 k 上重新生成**")
    print("=" * 92)
    prs = []
    for label, names in groups.items():
        if len(names) < 3:
            print(f"\n[{label}] k = {len(names)} < 3，PR 无意义，跳过")
            continue
        Qs = [RD.centered_orth(by[n]) for n in names]
        P = np.mean([Q @ Q.T for Q in Qs], axis=0)
        w, V = np.linalg.eigh(P)
        u = V[:, ::-1][:, 0]
        k = len(Qs)
        dirs = []
        for Q in Qs:
            c = Q.T @ u
            c /= np.linalg.norm(c)
            dirs.append(Q @ np.array([-c[1], c[0]]))
        ws, pr, n90 = RD.spectrum(dirs)
        prn, n90n, w1n = [], [], []
        for _ in range(NSIM):
            ds = []
            for _ in range(k):
                a = rng.normal(size=20)
                a -= RD.ONE * (RD.ONE @ a)
                a -= u * (u @ a)
                ds.append(a / np.linalg.norm(a))
            wn, p, n = RD.spectrum(ds)
            prn.append(p); n90n.append(n); w1n.append(wn[0])
        lo, hi = np.percentile(prn, [2.5, 97.5])
        p_one = float(np.mean(np.array(prn) <= pr))
        print(f"\n[{label}]  k = {k}")
        print(f"    残差谱前 3: " + " ".join(f"{x:.3f}" for x in ws[:3]))
        print(f"    PR = {pr:.2f}（完全相同 = 1，各向同性 ~ "
              f"min(k,18) = {min(k, 18)}）")
        print(f"      **在 k = {k} 上重生成的**零分布 PR 中位 "
              f"{np.median(prn):.2f} [{lo:.2f}, {hi:.2f}]")
        print(f"      单尾 p（零分布 PR <= 观测 PR 的比例） = {p_one:.4f}")
        print(f"    覆盖 90% 需要 {n90} 维；零分布中位 "
              f"{int(np.median(n90n))} 维")
        vr = np.linalg.eigh(np.mean([np.outer(d, d) for d in dirs],
                                    axis=0))[1][:, -1]
        rs = []
        for nm, s in scales.items():
            idx = [i for i in range(20) if np.isfinite(s[i])]
            if len(idx) < 10 or np.std(s[idx]) == 0:
                continue
            rs.append((abs(float(np.corrcoef(vr[idx], s[idx])[0, 1])),
                       float(np.corrcoef(vr[idx], s[idx])[0, 1]), nm))
        print("    残差主轴化学身份（前 4）: " + "; ".join(
            f"{nm} r={r:+.2f}" for _, r, nm in sorted(rs, reverse=True)[:4]))
        prs.append({"label": label, "k": k, "pr": pr,
                    "resid_eigs": ws[:3].tolist(),
                    "pr_null_med": float(np.median(prn)),
                    "pr_null_ci": [float(lo), float(hi)],
                    "p_one_tail": p_one, "n90": n90,
                    "n90_null_med": float(np.median(n90n)),
                    "resid_axis": vr.tolist(),
                    "resid_scale_corr": [{"scale": nm, "r": r}
                                         for _, r, nm in
                                         sorted(rs, reverse=True)[:6]]})

    # ---------------- 4. 成对统计（新保留集） ----------------
    print("\n" + "=" * 92)
    print("成对统计（新保留集；每骨架一主代表的「完全独立」类别已去 GB1）")
    print("=" * 92)
    N1 = haar_pair_null(20, 2, n=NSIM, seed=1)
    N1d = haar_1d_null(20, n=NSIM, seed=2)
    p95 = {"cos1": float(np.percentile(N1["cos1"], 95)),
           "cos2": float(np.percentile(N1["cos2"], 95)),
           "ssc": float(np.percentile(N1["ssc"], 95)),
           "d1": float(np.percentile(N1d, 95))}
    print(f"N1（d = 20）p95: cos1 {p95['cos1']:.3f}  cos2 {p95['cos2']:.3f}"
          f"  Σcos² {p95['ssc']:.3f}；一维 p95 {p95['d1']:.3f}")

    kd = scales["kd_hydropathy"]
    cols_h = np.arange(20)
    cats = {"同数据集内（冠状病毒）": [], "跨冠状病毒数据集": [],
            "OMTKY3 同骨架 × 不同蛋白酶": [], "完全独立（每骨架一主代表）": []}
    pairs = []
    nms = sorted(by)
    for a, b in itertools.combinations(nms, 2):
        da, db = info[a], info[b]
        if da["backbone"] == db["backbone"]:
            if da["backbone"] != "OMTKY3":
                continue
            cat = "OMTKY3 同骨架 × 不同蛋白酶"
        elif (da["dataset"] in IND_DATASETS or db["dataset"]
              in IND_DATASETS):
            if not (da["primary"] and db["primary"]):
                continue
            cat = "完全独立（每骨架一主代表）"
        elif da["dataset"] == db["dataset"]:
            cat = "同数据集内（冠状病毒）"
        else:
            cat = "跨冠状病毒数据集"
        r = align(by[a], cols_by[a], by[b], cols_by[b])
        if r is None:
            continue
        rd = align_deflated(by[a], cols_by[a], by[b], cols_by[b],
                            kd, cols_h)
        rec = {"a": a, "b": b, "cat": cat, "cos1": r["cos"][0],
               "cos2": r["cos"][1], "ssc": r["ssc"],
               "resid": (rd["cos_resid"] if rd else None)}
        pairs.append(rec)
        cats[cat].append(rec)

    print(f"\n{'配对类别':<30}{'n':>5}{'cos θ1 中位':>12}{'cos θ2 中位':>12}"
          f"{'Σcos² 中位':>12}{'超 N1 p95':>10}")
    print("-" * 92)
    pair_out = {}
    for cat, v in cats.items():
        if not v:
            continue
        c1 = np.median([z["cos1"] for z in v])
        c2 = np.median([z["cos2"] for z in v])
        sc = np.median([z["ssc"] for z in v])
        ex = np.mean([z["ssc"] > p95["ssc"] for z in v])
        print(f"{cat:<30}{len(v):>5}{c1:>12.3f}{c2:>12.3f}{sc:>12.3f}"
              f"{ex:>10.0%}")
        pair_out[cat] = {"n": len(v), "cos1_med": float(c1),
                         "cos2_med": float(c2), "ssc_med": float(sc),
                         "frac_over_p95": float(ex)}

    res = [z for z in cats["完全独立（每骨架一主代表）"]
           if min(stab.get(z["a"], 0), stab.get(z["b"], 0)) >= STAB_MIN]
    print(f"\n第二维两边都可分辨（stab cos2 >= {STAB_MIN}）的独立配对："
          f"{len(res)} 个")
    if res:
        for key, lab in (("cos1", "cos θ1"), ("cos2", "cos θ2"),
                         ("resid", "投掉疏水后的一维残差")):
            v = [z[key] for z in res if z[key] is not None]
            thr = p95["d1"] if key == "resid" else p95[key]
            print(f"   {lab:<22} 中位 {np.median(v):.3f}  "
                  f"范围 {min(v):.3f}–{max(v):.3f}  "
                  f"超 p95({thr:.3f}) 的比例 "
                  f"{np.mean(np.array(v) > thr):.0%}")
            pair_out.setdefault("resolvable_independent", {})[key] = {
                "n": len(v), "median": float(np.median(v)),
                "min": float(min(v)), "max": float(max(v)),
                "p95": float(thr),
                "frac_over": float(np.mean(np.array(v) > thr))}

    # OMTKY3 内部（同一骨架，不是独立跨家族证据 —— 必须明说）
    om = cats["OMTKY3 同骨架 × 不同蛋白酶"]
    if om:
        print(f"\nOMTKY3 内部 {len(om)} 个配对（**同一个骨架**，"
              f"不计为独立跨家族证据）：")
        print(f"   cos θ1 {min(z['cos1'] for z in om):.3f}–"
              f"{max(z['cos1'] for z in om):.3f}"
              f"（中位 {np.median([z['cos1'] for z in om]):.3f}）；"
              f"cos θ2 {min(z['cos2'] for z in om):.3f}–"
              f"{max(z['cos2'] for z in om):.3f}"
              f"（中位 {np.median([z['cos2'] for z in om]):.3f}）")

    # 两组之间的那**一个**跨家族配对
    print("\n两组证据之间的跨家族配对（冠状病毒代表 × OMTKY3 代表）：")
    xf = []
    for cv in COV_REPS:
        for ot in OMT + ["SKEMPI/OMTKY3/pooled-4proteases"]:
            if cv not in by or ot not in by:
                continue
            r = align(by[cv], cols_by[cv], by[ot], cols_by[ot])
            rd = align_deflated(by[cv], cols_by[cv], by[ot], cols_by[ot],
                                kd, cols_h)
            if r is None:
                continue
            xf.append({"cov": cv, "omt": ot, "cos1": r["cos"][0],
                       "cos2": r["cos"][1],
                       "resid": (rd["cos_resid"] if rd else None)})
            print(f"   {cv[:34]:<36} × {ot[:30]:<32} "
                  f"cos θ1 = {r['cos'][0]:.3f}  cos θ2 = {r['cos'][1]:.3f}")
    c1s = [z["cos1"] for z in xf]
    c2s = [z["cos2"] for z in xf]
    print(f"   => cos θ1 中位 {np.median(c1s):.3f}"
          f"（{min(c1s):.3f}–{max(c1s):.3f}），超 N1 p95 "
          f"{np.mean(np.array(c1s) > p95['cos1']):.0%}")
    print(f"      cos θ2 中位 {np.median(c2s):.3f}"
          f"（{min(c2s):.3f}–{max(c2s):.3f}），超 N1 p95 "
          f"{np.mean(np.array(c2s) > p95['cos2']):.0%}")

    # ---------------- 5. 第一轴的化学身份（两组证据） ----------------
    print("\n" + "=" * 92)
    print("第一轴的化学身份：**两组**家族均衡（冠状病毒 / OMTKY3），已去 GB1")
    print("=" * 92)
    FAM2 = [n for n in ("starr2020/WH1/huACE2",
                        "SKEMPI/OMTKY3/pooled-4proteases") if n in by]
    Pb = np.mean([CN.proj(by[n])[0] for n in FAM2], axis=0)
    w, V = np.linalg.eigh(Pb)
    w = w[::-1]; V = V[:, ::-1]
    u = V[:, 0]
    print(f"   λ 前 3 = " + " ".join(f"{x:.3f}" for x in w[:3]))
    un = CN.uncond_null(len(FAM2), np.random.default_rng(5), nsim=NSIM)
    up = np.percentile(un, 95, axis=0)
    print(f"   无条件随机零分布 p95 = " + " ".join(f"{x:.3f}" for x in up))
    for i in range(3):
        print(f"   λ{i+1} = {w[i]:.3f} vs p95 {up[i]:.3f}  "
              f"{'超出' if w[i] > up[i] else '未超出'}")
    order = np.argsort(-u)
    print("   普适方向 u 载荷（最正 -> 最负）:")
    print("     " + "  ".join(f"{AA20[i]}{u[i]:+.2f}" for i in order))

    print("\n   换冠状病毒代表时方向是否稳定（OMTKY3 代表固定）:")
    alt = []
    for rep in COV_REPS:
        if rep not in by:
            continue
        w2, V2 = np.linalg.eigh(np.mean(
            [CN.proj(by[n])[0] for n in [rep, FAM2[-1]]], axis=0))
        u2 = V2[:, ::-1][:, 0]
        alt.append({"rep": rep, "lam1": float(w2[::-1][0]),
                    "abscos": abs(float(u0 @ u2)) if False
                    else abs(float(u @ u2))})
        print(f"     {rep[:44]:<46} λ1 = {w2[::-1][0]:.3f}  "
              f"|cos(u, u')| = {abs(float(u @ u2)):.3f}")

    print("\n   与已知标度的相关:")
    rs = []
    for nm, s in scales.items():
        idx = np.array([i for i in range(20) if np.isfinite(s[i])])
        if len(idx) < 10 or s[idx].std() == 0:
            continue
        r = float(np.corrcoef(u[idx], s[idx])[0, 1])
        rs.append((abs(r), r, nm, len(idx)))
    for ar, r, nm, n in sorted(rs, reverse=True):
        print(f"     {nm:<30}{n:>4}  r = {r:+.3f}   r² = {r * r:.3f}")
    full = [nm for nm, s in scales.items() if np.isfinite(s).all()]
    best1 = max((abs(float(np.corrcoef(u, scales[p])[0, 1])), p)
                for p in full)
    bb = None
    for p, q in itertools.combinations(full, 2):
        A = np.stack([scales[p], scales[q], np.ones(20)], 1)
        coef, *_ = np.linalg.lstsq(A, u, rcond=None)
        r2 = 1 - ((u - A @ coef) ** 2).sum() / ((u - u.mean()) ** 2).sum()
        if bb is None or r2 > bb[0]:
            bb = (r2, p, q)
    print(f"   最佳单条: {best1[1]}  R² = {best1[0] ** 2:.3f}")
    print(f"   最佳两条: {bb[1]} + {bb[2]}  R² = {bb[0]:.3f}")
    kdr = float(np.corrcoef(u, scales["kd_hydropathy"])[0, 1])
    print(f"   对照 Kyte-Doolittle 疏水性: r = {kdr:+.3f}  "
          f"r² = {kdr * kdr:.3f}")

    # ---------------- 5b. 家族平均的两组共识（k = 2 退化的修补） ----
    # k = 2 时 P̄ 只由**一个**配对决定，λ 没有共识含义，而且 λ1 与 λ2
    # 挨得很近（0.971 / 0.907），首特征向量因此不可辨识 —— 上面换冠状病毒
    # 代表时 |cos(u,u')| 掉到 0.53–0.55 就是这个问题的直接显示。
    # 正确的两组构造：**先在每个家族内部取平均投影算子，再把两个家族
    # 等权平均**。这样家族权重仍是 1:1，但 u 由 9 个矩阵共同决定。
    print("\n" + "=" * 92)
    print("5b. 家族平均的两组共识 P̄ = ½(P̄_冠状病毒 + P̄_OMTKY3)")
    print("=" * 92)
    fam = {"冠状病毒": [n for n in COV_REPS if n in by],
           "OMTKY3": [n for n in OMT if n in by]}
    Pf = {k: np.mean([CN.proj(by[n])[0] for n in v], axis=0)
          for k, v in fam.items()}
    Pb2 = 0.5 * (Pf["冠状病毒"] + Pf["OMTKY3"])
    w2, V2 = np.linalg.eigh(Pb2)
    w2 = w2[::-1]; V2 = V2[:, ::-1]
    u2 = V2[:, 0]
    print(f"   每家族 k: " + "，".join(f"{k} = {len(v)}"
                                       for k, v in fam.items()))
    print(f"   λ 前 3 = " + " ".join(f"{x:.3f}" for x in w2[:3]))
    # 零分布：两组各自 k 个随机 2 维面，同样先家族内平均再等权
    nl = []
    rngf = np.random.default_rng(railguard := 20261012)
    for _ in range(NSIM):
        Ps = []
        for k, v in fam.items():
            Qs = []
            for _ in range(len(v)):
                A = rngf.normal(size=(20, 2))
                A -= CN.ONE @ (CN.ONE.T @ A)
                Q, _ = np.linalg.qr(A)
                Qs.append(Q @ Q.T)
            Ps.append(np.mean(Qs, axis=0))
        nl.append(np.linalg.eigvalsh(0.5 * (Ps[0] + Ps[1]))[::-1][:3])
    nl = np.array(nl)
    npc = np.percentile(nl, 95, axis=0)
    print(f"   **同结构**无条件零分布 p95 = "
          f"{np.round(npc, 3).tolist()}")
    for i in range(3):
        print(f"   λ{i+1} = {w2[i]:.3f} vs p95 {npc[i]:.3f}  "
              f"{'超出' if w2[i] > npc[i] else '未超出'}")
    cnd, alf = CN.cond_null(2, u2, np.random.default_rng(4),
                            nsim=NSIM, target_l1=float(w2[0]))
    p2f = float(np.mean(cnd[:, 1] >= w2[1]))
    print(f"   λ2 的条件零分布 p（标定到观测 λ1，抖动 "
          f"{np.degrees(alf):.1f}°）= {p2f:.4f}")
    o2 = np.argsort(-u2)
    print("   u 载荷（最正 -> 最负）:")
    print("     " + "  ".join(f"{AA20[i]}{u2[i]:+.2f}" for i in o2))
    print("\n   留一稳定性（每次从一个家族里去掉一个矩阵，重算 u）:")
    loo = []
    for fk, v in fam.items():
        for n in v:
            v2 = [x for x in v if x != n]
            if len(v2) < 2:
                continue
            Pq = {k: (np.mean([CN.proj(by[x])[0] for x in (v2 if k == fk
                                                           else vv)],
                              axis=0))
                  for k, vv in fam.items()}
            ww, VV = np.linalg.eigh(0.5 * (Pq["冠状病毒"] + Pq["OMTKY3"]))
            uu = VV[:, ::-1][:, 0]
            ac = abs(float(u2 @ uu))
            loo.append({"dropped": n, "family": fk, "abscos": ac,
                        "lam1": float(ww[::-1][0])})
            print(f"     去掉 {n[:42]:<44} λ1 = {ww[::-1][0]:.3f}  "
                  f"|cos(u, u')| = {ac:.3f}")
    print(f"   => 留一 |cos| 最小 {min(z['abscos'] for z in loo):.3f}，"
          f"中位 {np.median([z['abscos'] for z in loo]):.3f}")
    rs2 = []
    for nm, sc_ in scales.items():
        idx = np.array([i for i in range(20) if np.isfinite(sc_[i])])
        if len(idx) < 10 or sc_[idx].std() == 0:
            continue
        r = float(np.corrcoef(u2[idx], sc_[idx])[0, 1])
        rs2.append((abs(r), r, nm, len(idx)))
    print("\n   家族平均 u 与已知标度的相关（前 8 + Kyte-Doolittle）:")
    for ar, r, nm, n in sorted(rs2, reverse=True)[:8]:
        print(f"     {nm:<30}{n:>4}  r = {r:+.3f}   r² = {r * r:.3f}")
    kdr2 = float(np.corrcoef(u2, scales["kd_hydropathy"])[0, 1])
    print(f"     {'kd_hydropathy (对照)':<30}{20:>4}  r = {kdr2:+.3f}"
          f"   r² = {kdr2 * kdr2:.3f}")
    best1b = max((abs(float(np.corrcoef(u2, scales[p])[0, 1])), p)
                 for p in full)
    print(f"   最佳单条: {best1b[1]}  R² = {best1b[0] ** 2:.3f}")

    out = {"censor_max": CENSOR_MAX, "n_retained": len(keep),
           "n_retained_old": REV["n_keep_old"],
           "dropped": DROPPED, "aa_order": AA20,
           "groups": {k: v for k, v in groups.items()},
           "consensus": cons, "cal_ok": bool(cal_ok),
           "neg_controls": negc, "pr": prs,
           "pair_null_p95": p95, "pairs_by_cat": pair_out,
           "cross_family_pairs": xf,
           "axis1": {"u": u.tolist(), "eigs": w[:3].tolist(),
                     "uncond_p95": up.tolist(),
                     "family_reps": FAM2,
                     "alt_cov": alt,
                     "scale_corr": [{"scale": nm, "r": r, "n": n}
                                    for _, r, nm, n in
                                    sorted(rs, reverse=True)],
                     "best1": {"scale": best1[1], "r2": best1[0] ** 2},
                     "best2": {"a": bb[1], "b": bb[2], "r2": bb[0]},
                     "kd_r": kdr},
           "n_pairs": len(pairs),
           "axis1_family_avg": {
               "u": u2.tolist(), "eigs": w2[:3].tolist(),
               "null_p95": npc.tolist(), "p_lambda2": p2f,
               "alpha_deg": float(np.degrees(alf)),
               "families": {k: v for k, v in fam.items()},
               "loo": loo,
               "scale_corr": [{"scale": nm, "r": r, "n": n}
                              for _, r, nm, n in sorted(rs2, reverse=True)],
               "kd_r": kdr2,
               "best1": {"scale": best1b[1], "r2": best1b[0] ** 2}}}
    p = os.path.join(ROOT, "cache", "subspace_two_groups.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
