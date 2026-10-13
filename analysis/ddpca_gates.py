#!/usr/bin/env python3
"""docs/28 Step 4/5：闸判 + 「如果 GB1 binding 回来」的三组统计。

Step 4 三道闸，口径全部不新造
----------------------------
(a) **截尾闸**。Gen-3 (`censor_density.censor_fraction_density`) **原样**
    喂 **raw**（docs/22：喂 masked 会把一个矩阵从 83.1% 塌到 28.7%，
    足以静默地翻转一个结论），并按 **D-043** 同时报判别指标
    `(ĉ − min)/s_up`：**≲ 1.2 为右偏假阳性区，真堆积时 ≈ 0**。
    同时报与 ddPCA `censor_frac` 同口径的缺失率（docs/15 §3），
    因为那才是功效后果已知的那个量
    （GRB2-SH3/folding 3.0% → P = 1.00；GB1 13.1% → 0.88；PSD95 24.4% → **0.00**）。
(b) **可分辨闸**（docs/24）。硬前置：`cache/gate_harness_check.json` 的
    `ok` 必须为 true。门槛用**逐几何**秩 2 下限 **p95**（D-046 人类裁定），
    **p90 一并报**。功效判据用 docs/24 修正后的那条：
    `上限中位 − 下限 p95 ≥ 0.10`，否则判「**不可读**」而不是「是噪声」。
(c) **D-036 方向自身可复现**。任何对齐主张之前，每个方向都要先过
    位点对半 \\|cos\\| > 该比较自己的 d 上的 Haar 一维 p95。

Step 5：三组假设统计
-------------------
把 `lehner/GB1/IgG-Fc` 作为**第三个家族**加进 docs/24 场景 A 的构造，
**零分布一律在新的 k 上重生成**，条件零分布**二分法标定到观测 λ1**
（报标定残差 + 负对照）。成对统计**同源 / 跨源分开报**（docs/25 §6.3 的
假对齐签名是「同源在碰巧水平而跨源高」—— 次序相反）。

**本脚本不做纳入决定**（CLAUDE.md §9.1）：它只算出「如果纳入会是什么」。
数值层全部复用 `gate_calib_lib` / `gate_consequences` / `subspace_cond_null`
/ `subspace_resid_dim` / `subspace_lib` / `censor_density`，改动为零。
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import censor_density as CD                                    # noqa: E402
import ddpca_load as DL                                        # noqa: E402
import ddpca_sigma as S                                        # noqa: E402
import gate_calib_lib as G                                     # noqa: E402
import gate_consequences as GC                                 # noqa: E402
import subspace_cond_null as CN                                # noqa: E402
import subspace_resid_dim as RD                                # noqa: E402
from subspace_lib import (ROOT, align, extract_subspace,        # noqa: E402
                          haar_1d_null, haar_pair_null)

OUTJSON = os.path.join(ROOT, "cache", "ddpca_gates.json")
GB1B = "lehner/GB1/IgG-Fc"      # subspace_lib 里的 GB1 binding 矩阵
NSIM = 4000


def raw_values(D, ph):
    """**raw** = 未经 SE / |ΔΔG| 排除的全部该表型系数（docs/22 的口径）。"""
    return np.array([v for v in D["vals"][ph].values() if np.isfinite(v)])


def gate_censor(name, raw, censor_frac, seed=0):
    ci = CD.censor_fraction_density_ci(raw, seed=seed)
    q, info = ci["q"], ci["info"]
    ratio = None
    if info.get("c_rel") is not None and info.get("s_up"):
        ratio = info["c_rel"] / info["s_up"]
    # **逐子抽样审计（D-043）**：点估计报「无堆积」而某些子抽样却报
    # 70–90%，这件事只有把区间报出来才看得见（docs/21：区间不是装饰）。
    # 对每个 q > 2% 的子抽样都算出 D-043 的判别指标与 σ̂/s_up。
    rng = np.random.default_rng(seed)
    a = np.asarray(raw, float)
    a = a[np.isfinite(a)]
    k = max(200, int(CD.SUB_FRAC * len(a)))
    fired = []
    for t in range(CD.N_SUB):
        if len(a) <= k:
            break
        idx = rng.choice(len(a), size=k, replace=False)
        qs, inf = CD.censor_fraction_density(a[idx], detail=True)
        if qs > 0.02:
            fired.append({"sub": t, "q": float(qs),
                          "sigma_over_s": inf.get("sigma_over_s"),
                          "disc": (inf["c_rel"] / inf["s_up"]
                                   if inf.get("c_rel") and inf.get("s_up")
                                   else None),
                          "ev_over_zstar": inf.get("ev_over_zstar")})
    skew = float(((a - a.mean()) ** 3).mean() / a.std() ** 3)
    return {"name": name, "n": info["n"], "q_gen3": float(q),
            "q_ci": [ci["lo"], ci["hi"]], "q_sub_med": ci["sub_med"],
            "gate": info.get("gate"), "why": info.get("why"),
            "q_atom": info.get("q_atom"),
            "sigma_over_s": info.get("sigma_over_s"),
            "c_rel": info.get("c_rel"), "s_up": info.get("s_up"),
            "disc_c_minus_min_over_s_up": ratio,
            "d043_false_positive_zone": (ratio is not None and ratio <= 1.2),
            "skew": skew, "n_sub_fired": len(fired), "sub_fired": fired,
            "censor_frac_ddpca": float(censor_frac)}


def gate_resolvability(name, M, mask, n_real, nboot, verbose=True):
    expl, err = G.rankr_expl(M, mask, rank=2)
    if expl is None:
        return {"name": name, "ok": False, "why": err}
    sb = G.stab_boot(M, mask, rank=2, nboot=nboot)
    stab_obs = float(np.median(sb)) if sb else float("nan")
    fl = G.floor_dist(M.shape, mask, rank=2, n_real=n_real, nboot=nboot)
    sg, tab = G.match_sigma(M.shape, mask, 2, expl)
    ce = G.ceiling_dist(M.shape, mask, rank=2, sigma=sg, n_real=n_real,
                        nboot=nboot)
    hs = G.hi_snr_check(M.shape, mask, rank=2)
    fp = G.pcts(fl)
    cp = G.pcts(ce)
    gap = cp[50] - fp[95]
    out = {"name": name, "ok": True, "expl_r2": float(expl),
           "stab_obs": stab_obs, "floor_p90": fp[90], "floor_p95": fp[95],
           "floor_med": fp[50], "ceiling_med": cp[50],
           "sigma_matched": float(sg) if sg is not None else None,
           "match_table": tab,
           "hi_snr_med": float(np.median(hs)) if hs else None,
           "gap_ceiling_minus_floor_p95": float(gap),
           "readable": bool(gap >= 0.10),
           "pass_p95": bool(stab_obs > fp[95]),
           "pass_p90": bool(stab_obs > fp[90]),
           "quantile_sensitive": bool(stab_obs > fp[90]
                                      and not stab_obs > fp[95])}
    if verbose:
        print(f"   [可分辨闸] {name}: 实测 stab cos2 {stab_obs:.3f}；"
              f"下限 p95 {fp[95]:.3f} / p90 {fp[90]:.3f}；"
              f"上限中位 {cp[50]:.3f}（匹配 σ = {sg}）；"
              f"判别力 gap = {gap:+.3f} "
              f"{'（≥0.10，可读）' if gap >= 0.10 else '（<0.10，**不可读**）'}")
        print(f"      高 SNR 命中自检（σ=0.05 下已知秩 2 必须被还原）: "
              f"{out['hi_snr_med']:.3f}")
        print(f"      **p95 判读（D-046）**: "
              f"{'过' if out['pass_p95'] else '**不过**'}；"
              f"p90 判读: {'过' if out['pass_p90'] else '**不过**'}；"
              f"分位数敏感: {out['quantile_sensitive']}")
    return out


def gate_d036(name, M, mask, nsplit, seed=42, verbose=True):
    v = G.split_half(M, mask, rank=1, nsplit=nsplit, seed=seed)
    B, cols, _ = extract_subspace(M, mask, rank=1)
    d = len(cols) if cols is not None else 20
    p95 = float(np.percentile(haar_1d_null(d, n=20000, seed=3), 95))
    med = float(np.median(v)) if v else float("nan")
    out = {"name": name, "split_half_med": med, "n": len(v), "d": d,
           "haar1d_p95": p95, "pass": bool(med > p95)}
    if verbose:
        print(f"   [D-036] {name}: 位点对半第一轴 |cos| 中位 {med:.3f}"
              f"（n={len(v)}，d={d}）vs Haar p95 {p95:.3f} -> "
              f"{'过' if out['pass'] else '**不过**'}")
    return out


# ----------------------------------------------------------------------
def three_group(verbose=True):
    """把 GB1 binding 作为第三个家族加进场景 A，算出「会是什么」。"""
    cal = json.load(open(os.path.join(ROOT, "cache",
                                      "gate_rank2_calibration.json")))
    aux = json.load(open(os.path.join(ROOT, "cache",
                                      "gate_rank2_aux.json")))
    allrows = {r["name"]: r for r in cal["rows"] + aux["rows"]}
    mats = G.load_evidence()
    by, cols_by = {}, {}
    for nm, d in mats.items():
        B, c, _ = extract_subspace(d["M"], d["mask"])
        if B is not None and len(c) == 20:
            by[nm] = B
            cols_by[nm] = c
    keep = [nm for nm in G.EVIDENCE9
            if allrows[nm]["readable"] and allrows[nm]["pass_geo"]]
    cov = [n for n in G.COV_REPS if n in keep]
    omt = [n for n in G.OMT if n in keep]
    assert GB1B in by, f"{GB1B} 不在证据矩阵里"

    scen = {
        "A2. 现状（D-046 标定后，两组）": {"冠状病毒": cov, "OMTKY3": omt},
        "A3. 假设：GB1 binding 作为第三组": {"冠状病毒": cov, "OMTKY3": omt,
                                            "GB1": [GB1B]},
    }
    rng = np.random.default_rng(20261013)
    out = []
    for label, fam in scen.items():
        fam = {k: [n for n in v if n in by] for k, v in fam.items()}
        sizes = [len(v) for v in fam.values()]
        ktot = sum(sizes)
        w, V = GC.fam_avg_eigs(fam, by)
        u = V[:, 0]
        nl = GC.fam_avg_null(sizes, rng, nsim=NSIM)
        p95 = np.percentile(nl, 95, axis=0)
        keff = max(2, len(fam))
        cnd, alf = CN.cond_null(keff, u, np.random.default_rng(4),
                                nsim=NSIM, target_l1=float(w[0]))
        resid = abs(float(np.median(cnd[:, 0])) - float(w[0]))
        p2 = float(np.mean(cnd[:, 1] >= w[1]))
        rng2 = np.random.default_rng(99)
        al = CN.cond_null(keff, u, rng2, nsim=10,
                          target_l1=float(w[0]))[1]
        synth = float(np.median([CN._draw(keff, u, al, rng2)[1]
                                 for _ in range(400)]))
        cn2, _ = CN.cond_null(keff, u, rng2, nsim=1500,
                              target_l1=float(w[0]))
        pneg = float(np.mean(cn2[:, 1] >= synth))
        loo = []
        for fk, v in fam.items():
            for n in v:
                v2 = [x for x in v if x != n]
                f2 = dict(fam)
                if v2:
                    f2[fk] = v2
                else:
                    f2.pop(fk)       # 整组消失（GB1 只有一个矩阵）
                if sum(len(x) for x in f2.values()) < 2:
                    continue
                ww, VV = GC.fam_avg_eigs(f2, by)
                loo.append({"dropped": n,
                            "abscos": abs(float(u @ VV[:, 0])),
                            "lam1": float(ww[0])})
        # **代表互换诊断 —— 逐字照 docs/21 §5.1 的构造**：
        # 「两组**各取一个代表**」，然后换冠状病毒代表看首特征向量动多少。
        # docs/13 三组时是 0.936–0.985，docs/21 两组时掉到 0.533/0.547，
        # 那就是 k = 2 不可辨识的签名。参照方向 u_ref 取**该构造下第一个
        # 代表**给出的 u，和 docs/21 同一口径（不是和全集的 u 比）。
        # **穷举所有「每组一个代表」的组合**，而不是只换一侧 ——
        # 换哪一个本来就是一个选择，穷举掉它。
        # 报全部组合两两之间的 |cos(u, u')| 的最小值与中位数。
        import itertools as _it
        reps = [v for v in fam.values() if v]
        combos = [dict(zip(fam.keys(), c)) for c in _it.product(*reps)]
        us = []
        for cb in combos:
            ww, VV = GC.fam_avg_eigs({k2: [v2] for k2, v2 in cb.items()}, by)
            us.append((cb, VV[:, 0], float(ww[0]), float(ww[1])))
        swaps = []
        for (c1, u1, l11, l12), (c2, u2, _, _) in _it.combinations(us, 2):
            swaps.append({"reps_a": list(c1.values()),
                          "reps_b": list(c2.values()),
                          "abscos": abs(float(u1 @ u2))})
        for cb, uu, l1, l2 in us:
            swaps.append({"reps_a": list(cb.values()), "reps_b": "全集 u",
                          "abscos": abs(float(u @ uu)),
                          "lam1": l1, "lam2": l2})
        # PR（零分布在新的 k 上重生成）
        Qs = [RD.centered_orth(by[n]) for v in fam.values() for n in v]
        P = np.mean([Q @ Q.T for Q in Qs], axis=0)
        wq, Vq = np.linalg.eigh(P)
        uq = Vq[:, ::-1][:, 0]
        dirs = []
        for Q in Qs:
            c = Q.T @ uq
            c /= np.linalg.norm(c)
            dirs.append(Q @ np.array([-c[1], c[0]]))
        ws, pr, n90 = RD.spectrum(dirs)
        prn = []
        for _ in range(NSIM):
            ds = []
            for _ in range(len(Qs)):
                a = rng.normal(size=20)
                a -= RD.ONE * (RD.ONE @ a)
                a -= uq * (uq @ a)
                ds.append(a / np.linalg.norm(a))
            prn.append(RD.spectrum(ds)[1])
        lo, hi = np.percentile(prn, [2.5, 97.5])
        rec = {"label": label, "sizes": sizes, "k_total": ktot,
               "families": fam, "eigs": w[:3].tolist(),
               "null_p95": p95.tolist(), "p_lambda2": p2,
               "alpha_deg": float(np.degrees(alf)), "cal_resid": resid,
               "neg_control": {"lam2": synth, "p": pneg},
               "loo_min": min(z["abscos"] for z in loo) if loo else None,
               "loo_med": (float(np.median([z["abscos"] for z in loo]))
                           if loo else None),
               "loo": loo, "rep_swap": swaps,
               "n_rep_combos": len(combos),
               "rep_swap_min": min(z["abscos"] for z in swaps),
               "rep_swap_med": float(np.median([z["abscos"]
                                                for z in swaps])),
               "rep_swap_lam1": [z[2] for z in us],
               "rep_swap_lam2": [z[3] for z in us],
               "pr": float(pr), "pr_k": len(Qs),
               "pr_null_med": float(np.median(prn)),
               "pr_null_ci": [float(lo), float(hi)],
               "pr_p_one": float(np.mean(np.array(prn) <= pr)),
               "u": u.tolist()}
        out.append(rec)
        if verbose:
            print(f"\n[{label}]  家族 {sizes}（共 {ktot} 个矩阵）")
            print(f"   λ 前 3 = " + " ".join(f"{x:.3f}" for x in w[:3])
                  + "；**在该 k 上重生成的**同结构零分布 p95 = "
                  + " ".join(f"{x:.3f}" for x in p95))
            print(f"   λ2 的条件零分布（标定到观测 λ1，抖动 "
                  f"{np.degrees(alf):.1f}°，k_eff = {keff}）: p = {p2:.4f}"
                  f"；标定残差 {resid:.4f}"
                  f" {'OK' if resid < 0.02 else '<< 标定失败'}")
            print(f"   负对照（合成只共享一维）λ2 = {synth:.3f}, p = "
                  f"{pneg:.3f} {'OK（未误报）' if pneg > 0.05 else '<< 误报'}")
            print(f"   留一 |cos(u,u')| 最小 {rec['loo_min']:.3f} / 中位 "
                  f"{rec['loo_med']:.3f}")
            print(f"   **代表互换诊断**（docs/21 §5.1 口径：每组各取一个代表；"
                  f"本处**穷举全部 {len(combos)} 种组合**）"
                  f"|cos(u, u')| 最小 {rec['rep_swap_min']:.3f} / 中位 "
                  f"{rec['rep_swap_med']:.3f}；这些单代表构造的 λ1 范围 "
                  f"{min(rec['rep_swap_lam1']):.3f}–"
                  f"{max(rec['rep_swap_lam1']):.3f}、λ2 "
                  f"{min(rec['rep_swap_lam2']):.3f}–"
                  f"{max(rec['rep_swap_lam2']):.3f}"
                  f"（docs/21 §5.1 两组退化时报 0.533/0.547，"
                  f"docs/13 三组时 0.936–0.985）")
            print(f"   PR = {pr:.2f}（k = {len(Qs)}）vs 在该 k 上重生成的"
                  f"零分布 {np.median(prn):.2f} [{lo:.2f}, {hi:.2f}]，"
                  f"单尾 p = {rec['pr_p_one']:.4f}")
    # 成对统计：同源 / 跨源分开（docs/25 §6.3）
    N1 = haar_pair_null(20, 2, n=NSIM, seed=1)
    q1 = float(np.percentile(N1["cos1"], 95))
    q2 = float(np.percentile(N1["cos2"], 95))
    pw = {}
    groups = {"冠状病毒": cov, "OMTKY3": omt, "GB1": [GB1B]}
    names = list(groups)
    for i in range(len(names)):
        for j in range(i, len(names)):
            A, B = groups[names[i]], groups[names[j]]
            rec = []
            for a in A:
                for b in B:
                    if a == b:
                        continue
                    r = align(by[a], cols_by[a], by[b], cols_by[b])
                    if r:
                        rec.append((r["cos"][0], r["cos"][1]))
            if not rec:
                continue
            c1 = np.array([z[0] for z in rec])
            c2 = np.array([z[1] for z in rec])
            lab = ("同源（同一家族内）" if i == j else "跨源（跨家族）") \
                + f" {names[i]} × {names[j]}"
            pw[lab] = {"n": len(rec), "cos1_med": float(np.median(c1)),
                       "cos1_min": float(c1.min()),
                       "cos1_max": float(c1.max()),
                       "cos1_frac": float(np.mean(c1 > q1)),
                       "cos2_med": float(np.median(c2)),
                       "cos2_frac": float(np.mean(c2 > q2))}
            if verbose:
                print(f"   {lab}: n={len(rec)}，cos θ1 中位 "
                      f"{np.median(c1):.3f}（{c1.min():.3f}–{c1.max():.3f}），"
                      f"超 p95({q1:.3f}) {np.mean(c1 > q1):.0%}；"
                      f"cos θ2 中位 {np.median(c2):.3f}，超 p95({q2:.3f}) "
                      f"{np.mean(c2 > q2):.0%}")
    return {"scenarios": out, "pairwise": pw,
            "null_p95": {"cos1": q1, "cos2": q2}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=24)
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--nsplit", type=int, default=400)
    ap.add_argument("--skip-resolv", action="store_true")
    a = ap.parse_args()

    hz = json.load(open(os.path.join(ROOT, "cache",
                                     "gate_harness_check.json")))
    assert hz.get("ok") is True, \
        "cache/gate_harness_check.json 的 ok 不为 true —— docs/24 的硬前置"
    print("[硬前置] cache/gate_harness_check.json ok = true")

    out = {"censor": [], "resolv": [], "d036": []}
    print("\n" + "=" * 96)
    print("Step 4(a) 截尾闸：Gen-3 喂 **raw**，并报 D-043 的判别指标")
    print("=" * 96)
    for dom, ph in S.MATS:
        D = DL.load_domain(dom)
        DL.verify(dom, D, verbose=False)
        M, mask, cen = DL.matrix(D, ph)
        raw = raw_values(D, ph)
        z = gate_censor(f"{dom}/{ph}", raw, cen["censor_frac"])
        out["censor"].append(z)
        print(f"   {z['name']:<22} n={z['n']:<5} Gen-3 q = "
              f"{z['q_gen3']:.2%}"
              + f" [{z['q_ci'][0]:.2%}, {z['q_ci'][1]:.2%}]"
              + f"，闸「{z['gate']}」"
              + (f"，σ̂/s_up {z['sigma_over_s']:.3f}"
                 if z["sigma_over_s"] else "")
              + (f"，**(ĉ−min)/s_up = {z['disc_c_minus_min_over_s_up']:.2f}**"
                 f"{'（≲1.2，**D-043 假阳性区**）' if z['d043_false_positive_zone'] else '（>1.2）'}"
                 if z["disc_c_minus_min_over_s_up"] is not None else "")
              + f"；偏度 {z['skew']:+.2f}"
              + f"；ddPCA 同口径缺失率 {z['censor_frac_ddpca']:.1%}")
        if z["n_sub_fired"]:
            print(f"      **D-043 实例**：{z['n_sub_fired']}/{CD.N_SUB} 个"
                  f"子抽样报 q > 2%（" + ", ".join(
                      f"{s['q']:.0%}（σ̂/s_up {s['sigma_over_s']:.3f}，"
                      f"(ĉ−min)/s_up {s['disc']:.2f}）" for s in
                      z["sub_fired"]) + "），而点估计报「无堆积」、"
                  f"真实缺失率只有 {z['censor_frac_ddpca']:.1%}")

        if not a.skip_resolv:
            out["d036"].append(gate_d036(f"{dom}/{ph}", M, mask, a.nsplit))
            out["resolv"].append(gate_resolvability(
                f"{dom}/{ph}", M, mask, a.n_real, a.nboot))

    print("\n" + "=" * 96)
    print("Step 5：如果 GB1 binding 回来，三组会是什么（**不做纳入决定**）")
    print("=" * 96)
    out["three_group"] = three_group()

    json.dump(out, open(OUTJSON, "w"), ensure_ascii=False, indent=1,
              default=float)
    print("\n产物:", OUTJSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
