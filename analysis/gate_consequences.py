#!/usr/bin/env python3
"""Part 3：标定后的闸把谁剔出去之后，headline 还剩什么。

输入（不是本脚本自己决定的）：
  cache/gate_rank2_calibration.json  —— 9 个成员的秩 2 下限 / 上限 / 判读
  cache/gate_rank2_aux.json          —— pooled OMTKY3 等附表矩阵的同一标定

本脚本把「过几何相关门槛」的成员当成新的证据集，然后**原样重跑** docs/21 §4–5
的那套构造，数值层全部复用、没有重写：
  `subspace_cond_null`（proj / eigs / cond_null / uncond_null / _draw）
  `subspace_resid_dim`（centered_orth / spectrum）
  `subspace_lib`（extract_subspace / align / align_deflated / haar_*/ load_scales）

两条纪律必须写进代码而不是写进文档：
 1. **零分布一律在新的 k 上重生成**（docs/21 §5.4：一个 k 的 PR 或 λ
    不能和另一个 k 建的零分布比）。本脚本对每一组现场生成。
 2. **条件零分布必须用二分法标定到观测 λ1**（docs/13 §5、docs/21 §5.3）。
    让每个子空间精确含 u 会使 λ1 = 1.000，而 tr(P̄) ≡ 2，于是 λ2 被人为
    压低 —— 那是一个**结构性偏向阳性**的零假设。标定残差必须一起报。
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subspace_cond_null as CN  # noqa: E402
import subspace_resid_dim as RD  # noqa: E402
import gate_calib_lib as G  # noqa: E402
from subspace_lib import (AA20, ROOT, align, align_deflated,  # noqa: E402
                          extract_subspace, haar_1d_null, haar_pair_null,
                          load_scales)

NSIM = 4000
POOLED = "SKEMPI/OMTKY3/pooled-4proteases"


def fam_avg_eigs(fam, by):
    """P̄ = 等权平均（先家族内平均）。返回 (λ, u)。"""
    Ps = [np.mean([CN.proj(by[n])[0] for n in v], axis=0)
          for v in fam.values() if v]
    Pb = np.mean(Ps, axis=0)
    w, V = np.linalg.eigh(Pb)
    return w[::-1], V[:, ::-1]


def fam_avg_null(sizes, rng, nsim=NSIM):
    """**同结构**零分布：每个家族 k_i 个 Haar 随机 2 维面，先家族内平均
    再家族等权。必须在新的 k 上重生成。"""
    out = []
    for _ in range(nsim):
        Ps = []
        for k in sizes:
            Qs = []
            for _ in range(k):
                A = rng.normal(size=(20, 2))
                A -= CN.ONE @ (CN.ONE.T @ A)
                Q, _ = np.linalg.qr(A)
                Qs.append(Q @ Q.T)
            Ps.append(np.mean(Qs, axis=0))
        out.append(np.linalg.eigvalsh(np.mean(Ps, axis=0))[::-1][:3])
    return np.array(out)


def main():
    cal = json.load(open(os.path.join(ROOT, "cache",
                                      "gate_rank2_calibration.json")))
    aux = json.load(open(os.path.join(ROOT, "cache",
                                      "gate_rank2_aux.json")))
    allrows = {r["name"]: r for r in cal["rows"] + aux["rows"]}
    scales = load_scales(verbose=False)
    mats = G.load_evidence()

    by, cols_by = {}, {}
    for nm, d in mats.items():
        B, c, _ = extract_subspace(d["M"], d["mask"])
        if B is not None and len(c) == 20:
            by[nm] = B
            cols_by[nm] = c

    print("=" * 100)
    print("Part 3：标定后的闸对 headline 的后果")
    print("=" * 100)
    print(f"{'成员':<42}{'实测stab':>9}{'下限p95':>9}{'判读':>8}")
    print("-" * 100)
    keep, drop = [], []
    for nm in G.EVIDENCE9:
        r = allrows[nm]
        good = r["readable"] and r["pass_geo"]
        (keep if good else drop).append(nm)
        print(f"{nm[:41]:<42}{r['stab_obs']:>9.3f}{r['floor_p95']:>9.3f}"
              f"{'保留' if good else '**剔除**':>8}")
    rp = allrows.get(POOLED)
    pooled_ok = bool(rp and rp["readable"] and rp["pass_geo"])
    if rp:
        print(f"\n[附] {POOLED}  实测 {rp['stab_obs']:.3f} vs 下限 p95 "
              f"{rp['floor_p95']:.3f} -> {'保留' if pooled_ok else '**剔除**'}")

    cov = [n for n in G.COV_REPS if n in keep]
    omt = [n for n in G.OMT if n in keep]
    print(f"\n标定后的证据集：冠状病毒 {len(cov)} 个、OMTKY3 单酶 {len(omt)} 个"
          f"（共 {len(keep)} 个；原 9 个）")
    print(f"剔除 {len(drop)} 个：" + ("、".join(drop) if drop else "无"))

    out = {"keep": keep, "drop": drop, "pooled_ok": pooled_ok,
           "cov": cov, "omt": omt}

    # ------------------------------------------------------------------
    # 场景定义。非病毒一侧有三种可能，必须都摆出来而不是只报有利的那个。
    # ------------------------------------------------------------------
    scen = {}
    if omt:
        scen["A. 标定后原构造（冠状病毒 k=%d / OMTKY3 单酶 k=%d）"
             % (len(cov), len(omt))] = {"冠状病毒": cov, "OMTKY3": omt}
    if pooled_ok:
        scen["B. 非病毒一侧退回 pooled 单矩阵（冠状病毒 k=%d / OMTKY3 k=1）"
             % len(cov)] = {"冠状病毒": cov, "OMTKY3": [POOLED]}
    scen["C. 只剩冠状病毒一组（非病毒一侧全部剔除）"] = {"冠状病毒": cov}
    # 对照：docs/21 §5.2 的原构造，用来比「变了多少」
    scen["D. 对照：docs/21 §5.2 原构造（冠状病毒 5 / OMTKY3 单酶 4）"] = {
        "冠状病毒": G.COV_REPS, "OMTKY3": G.OMT}

    rng = np.random.default_rng(20261012)
    cons = []
    for label, fam in scen.items():
        fam = {k: [n for n in v if n in by] for k, v in fam.items()}
        fam = {k: v for k, v in fam.items() if v}
        sizes = [len(v) for v in fam.values()]
        ktot = sum(sizes)
        print("\n" + "=" * 100)
        print(f"[{label}]")
        print(f"   家族 / k: " + "，".join(f"{k} = {len(v)}"
                                          for k, v in fam.items())
              + f"（共 {ktot} 个矩阵）")
        if ktot < 2:
            print("   k < 2，P̄ 无共识含义，跳过")
            continue
        if len(fam) < 2:
            print("   **只有一个家族 —— 这不是跨家族检验**，")
            print("   下面的 λ 只能读成『该家族内部共享』，不能读成普适。")
        w, V = fam_avg_eigs(fam, by)
        u = V[:, 0]
        nl = fam_avg_null(sizes, rng)
        p95 = np.percentile(nl, 95, axis=0)
        print(f"   观测 λ 前 3 = " + " ".join(f"{x:.3f}" for x in w[:3]))
        print(f"   **在该 k 上重生成的**同结构零分布 p95 = "
              + " ".join(f"{x:.3f}" for x in p95))
        for i in range(3):
            print(f"      λ{i+1} = {w[i]:.3f} vs p95 {p95[i]:.3f}  "
                  f"{'超出' if w[i] > p95[i] else '**未超出**'}")
        # λ2 的标定条件零分布（二分法标到观测 λ1）
        keff = max(2, len(fam))
        cnd, alf = CN.cond_null(keff, u, np.random.default_rng(4),
                                nsim=NSIM, target_l1=float(w[0]))
        resid = abs(float(np.median(cnd[:, 0])) - float(w[0]))
        p2 = float(np.mean(cnd[:, 1] >= w[1]))
        print(f"   λ2 的条件零分布（标定到观测 λ1，抖动 "
              f"{np.degrees(alf):.1f}°，k_eff = {keff}）：p = {p2:.4f}")
        print(f"      标定残差 |零分布 λ1 中位 − 观测 λ1| = {resid:.4f}  "
              f"{'OK' if resid < 0.02 else '<< 标定失败，该格不可用'}")
        # 负对照（§10.2 不误伤）：合成「只共享一维」必须不超出
        rng2 = np.random.default_rng(99)
        al = CN.cond_null(keff, u, rng2, nsim=10,
                          target_l1=float(w[0]))[1]
        synth = float(np.median([CN._draw(keff, u, al, rng2)[1]
                                 for _ in range(400)]))
        cn2, _ = CN.cond_null(keff, u, rng2, nsim=1500,
                              target_l1=float(w[0]))
        pneg = float(np.mean(cn2[:, 1] >= synth))
        print(f"   负对照：合成「只共享一维」的 λ2 = {synth:.3f}，"
              f"条件零分布 p = {pneg:.3f}  "
              f"{'OK（未误报）' if pneg > 0.05 else '<< 误报'}")
        # 留一稳定性
        loo = []
        for fk, v in fam.items():
            for n in v:
                v2 = [x for x in v if x != n]
                if not v2:
                    continue
                f2 = dict(fam)
                f2[fk] = v2
                ww, VV = fam_avg_eigs(f2, by)
                loo.append({"dropped": n, "abscos": abs(float(u @ VV[:, 0])),
                            "lam1": float(ww[0])})
        if loo:
            print(f"   留一稳定性 |cos(u, u')|：最小 "
                  f"{min(z['abscos'] for z in loo):.3f}，中位 "
                  f"{np.median([z['abscos'] for z in loo]):.3f}"
                  f"；λ1 范围 {min(z['lam1'] for z in loo):.3f}–"
                  f"{max(z['lam1'] for z in loo):.3f}")
        # PR（零分布在新的 k 上重生成）
        pr_rec = None
        if ktot >= 3:
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
            pone = float(np.mean(np.array(prn) <= pr))
            print(f"   残差方向 PR = {pr:.2f}（k = {len(Qs)}）；"
                  f"**在 k = {len(Qs)} 上重生成的**零分布中位 "
                  f"{np.median(prn):.2f} [{lo:.2f}, {hi:.2f}]，单尾 p = "
                  f"{pone:.4f}")
            pr_rec = {"k": len(Qs), "pr": float(pr),
                      "resid_eigs": ws[:3].tolist(),
                      "null_med": float(np.median(prn)),
                      "null_ci": [float(lo), float(hi)], "p_one": pone,
                      "n90": int(n90)}
        else:
            print(f"   k = {ktot} < 3，PR 无意义（docs/21 §5.4 同一口径）")
        # 第一轴化学身份
        rs = []
        for nm_, s in scales.items():
            idx = np.array([i for i in range(20) if np.isfinite(s[i])])
            if len(idx) < 10 or s[idx].std() == 0:
                continue
            r = float(np.corrcoef(u[idx], s[idx])[0, 1])
            rs.append((abs(r), r, nm_))
        print("   第一轴化学身份（前 4）: " + "; ".join(
            f"{nm_} r={r:+.2f} (r²={r*r:.3f})"
            for _, r, nm_ in sorted(rs, reverse=True)[:4]))
        cons.append({"label": label,
                     "families": {k: v for k, v in fam.items()},
                     "sizes": sizes, "k_total": ktot,
                     "eigs": w[:3].tolist(), "null_p95": p95.tolist(),
                     "u": u.tolist(), "p_lambda2": p2,
                     "alpha_deg": float(np.degrees(alf)),
                     "cal_resid": resid, "neg_control": {"lam2": synth,
                                                         "p": pneg},
                     "loo": loo, "pr": pr_rec,
                     "scale_corr": [{"scale": nm_, "r": r}
                                    for _, r, nm_ in
                                    sorted(rs, reverse=True)[:8]]})
    out["consensus"] = cons

    # ------------------------------------------------------------------
    # 跨家族成对统计（标定后）
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("跨家族成对统计（标定后的成员；零分布 d = 20）")
    print("=" * 100)
    N1 = haar_pair_null(20, 2, n=NSIM, seed=1)
    N1d = haar_1d_null(20, n=NSIM, seed=2)
    q = {"cos1": float(np.percentile(N1["cos1"], 95)),
         "cos2": float(np.percentile(N1["cos2"], 95)),
         "d1": float(np.percentile(N1d, 95))}
    print(f"N1 p95: cos1 {q['cos1']:.3f}  cos2 {q['cos2']:.3f}；"
          f"一维 p95 {q['d1']:.3f}")
    kd = scales["kd_hydropathy"]
    ch = np.arange(20)
    sets = {}
    nonvir_cal = omt + ([POOLED] if pooled_ok else [])
    sets["标定后（冠状病毒保留 × 非病毒保留）"] = (cov, nonvir_cal)
    sets["对照：docs/21 §5.5 的 25 对（全 9 个 + pooled）"] = (
        G.COV_REPS, G.OMT + [POOLED])
    pw = {}
    for lab, (A, B) in sets.items():
        A = [n for n in A if n in by]
        B = [n for n in B if n in by]
        if not A or not B:
            print(f"\n[{lab}] 一侧为空 —— **没有跨家族配对可算**")
            pw[lab] = None
            continue
        rec = []
        for a in A:
            for b in B:
                r = align(by[a], cols_by[a], by[b], cols_by[b])
                rd = align_deflated(by[a], cols_by[a], by[b], cols_by[b],
                                    kd, ch)
                if r:
                    rec.append({"a": a, "b": b, "cos1": r["cos"][0],
                                "cos2": r["cos"][1],
                                "resid": rd["cos_resid"] if rd else None})
        c1 = np.array([z["cos1"] for z in rec])
        c2 = np.array([z["cos2"] for z in rec])
        print(f"\n[{lab}]  n = {len(rec)} 对")
        print(f"   cos θ1 中位 {np.median(c1):.3f}"
              f"（{c1.min():.3f}–{c1.max():.3f}），超 p95 "
              f"{np.mean(c1 > q['cos1']):.0%}")
        print(f"   cos θ2 中位 {np.median(c2):.3f}"
              f"（{c2.min():.3f}–{c2.max():.3f}），超 p95 "
              f"{np.mean(c2 > q['cos2']):.0%}")
        pw[lab] = {"n": len(rec), "cos1_med": float(np.median(c1)),
                   "cos1_min": float(c1.min()), "cos1_max": float(c1.max()),
                   "cos1_frac": float(np.mean(c1 > q["cos1"])),
                   "cos2_med": float(np.median(c2)),
                   "cos2_min": float(c2.min()), "cos2_max": float(c2.max()),
                   "cos2_frac": float(np.mean(c2 > q["cos2"])),
                   "pairs": rec}
    out["pairwise"] = pw
    out["null_p95"] = q

    p = os.path.join(ROOT, "cache", "gate_consequences.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print("\n产物:", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
