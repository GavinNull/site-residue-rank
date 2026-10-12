#!/usr/bin/env python3
"""docs/26 第 3 层：Part 1 的分层与区间。

三件 generic_axis_test.py 没做的事：

1. **按残基 bootstrap 的区间**（CLAUDE.md §5：点估计一律不单独汇报）。
   n = 20 时「相关 r」的抽样单位只能是**氨基酸**，所以 bootstrap 必须
   对 20 个残基有放回重抽。这会给出很宽的区间 —— 这正是
   「20 个点上分不分得开」这一问的正确答案形式。
2. **按家族分层**：两组证据（冠状病毒 / OMTKY3）各自的轴，与去稳定度
   的关系是否相同。这是必须做的，因为 D3（Starr 表达量）与冠状病毒
   那一半证据**同源**（同一批文库），它与 u 的任何一致都可能是技术性的。
3. **D3 逐矩阵的覆盖**，同源（冠状病毒）与跨源（OMTKY3）分列。

运行：python3 analysis/generic_axis_strat.py
产物：cache/generic_axis_strat.json
"""
import json
import math
import os
import sys

import numpy as np
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subspace_cond_null as CN                                  # noqa: E402
from subspace_lib import (AA20, ROOT, dedup_by_content,          # noqa: E402
                          extract_subspace, load_all_dms,
                          load_independent, load_scales,
                          load_skempi, pick_primary)

NBOOT = 2000
NSIM = 4000
ONE = np.ones(20) / math.sqrt(20)
DISRUPT_KEYS = ["D1_folding_ddG_ddPCA", "D2_blosum62_nonexchangeability",
                "D3_starr_expression_loss", "D4_backbone_disruption_PG",
                "D5_disrupt_PC1"]


def cen(v):
    v = np.asarray(v, float) - np.mean(v)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def boot_r(a, b, nboot=NBOOT, seed=3):
    """对 20 个残基有放回重抽的 r 的区间。"""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = []
    for _ in range(nboot):
        i = rng.integers(0, 20, 20)
        if np.std(a[i]) == 0 or np.std(b[i]) == 0:
            continue
        out.append(float(np.corrcoef(a[i], b[i])[0, 1]))
    out = np.array(out)
    return (float(np.corrcoef(a, b)[0, 1]),
            float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def boot_partial(y, x, zs, nboot=NBOOT, seed=4):
    rng = np.random.default_rng(seed)
    y, x = np.asarray(y, float), np.asarray(x, float)
    Zs = [np.asarray(c, float) for c in zs]

    def pr(ix):
        Z = np.column_stack([np.ones(len(ix))] + [c[ix] for c in Zs])
        ry = y[ix] - Z @ np.linalg.lstsq(Z, y[ix], rcond=None)[0]
        rx = x[ix] - Z @ np.linalg.lstsq(Z, x[ix], rcond=None)[0]
        if np.std(ry) == 0 or np.std(rx) == 0:
            return None
        return float(np.corrcoef(ry, rx)[0, 1])
    pt = pr(np.arange(20))
    out = [v for v in (pr(rng.integers(0, 20, 20)) for _ in range(nboot))
           if v is not None]
    out = np.array(out)
    return pt, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    scales = load_scales(verbose=False)
    mr = scales["molar_refractivity"]
    assert AA20[int(np.nanargmin(mr))] == "G" and float(mr[AA20.index("G")]) == 0.0
    assert AA20[int(np.nanargmax(mr))] == "W" and abs(float(mr[AA20.index("W")]) - 42.53) < 1e-6
    print("[自检] 折射度 Gly 0.00 / Trp 42.53 —— 独立核验通过")

    TG = json.load(open(os.path.join(ROOT, "cache",
                                     "subspace_two_groups.json")))
    DS = json.load(open(os.path.join(ROOT, "cache",
                                     "generic_disrupt_scales.json")))
    dis = {k: np.array(v, float) for k, v in DS["scales"].items()}
    u = np.array(TG["axis1_family_avg"]["u"], float)
    pol = scales["local_polarizability"]
    sheet = scales["sheet_prop_Chou"]

    # ------------------------------------------------------------
    # 1. 按残基 bootstrap 的区间
    # ------------------------------------------------------------
    print("\n" + "=" * 92)
    print(f"1. 按**残基**有放回重抽（{NBOOT} 次）的区间 —— §5 要求的那个区间")
    print("=" * 92)
    print("   抽样单位是 20 个氨基酸。区间宽度本身就是「20 个点够不够」的答案。")
    rows = []
    print(f"\n   {'量':<52}{'点估计':>9}{'95% 区间':>22}")
    print("   " + "-" * 83)
    for nm, a, b in [("r(u, 极化率)", u, pol),
                     ("r(u, sheet propensity)", u, sheet),
                     ("r(u, Kyte-Doolittle)", u, scales["kd_hydropathy"])] + \
            [(f"r(u, {k})", u, dis[k]) for k in DISRUPT_KEYS]:
        pt, lo, hi = boot_r(a, b)
        rows.append({"q": nm, "pt": pt, "lo": lo, "hi": hi,
                     "excl0": bool(lo > 0 or hi < 0)})
        print(f"   {nm:<52}{pt:>+9.3f}  [{lo:>+7.3f}, {hi:>+7.3f}]"
              f"  {'不含 0' if (lo > 0 or hi < 0) else '含 0'}")
    print()
    for k in DISRUPT_KEYS:
        pt, lo, hi = boot_partial(u, dis[k], [pol])
        rows.append({"q": f"r(u, {k} | 极化率)", "pt": pt, "lo": lo,
                     "hi": hi, "excl0": bool(lo > 0 or hi < 0)})
        print(f"   {'r(u, ' + k + ' | 极化率)':<52}{pt:>+9.3f}"
              f"  [{lo:>+7.3f}, {hi:>+7.3f}]"
              f"  {'不含 0' if (lo > 0 or hi < 0) else '含 0'}")
        pt, lo, hi = boot_partial(u, pol, [dis[k]], seed=5)
        rows.append({"q": f"r(u, 极化率 | {k})", "pt": pt, "lo": lo,
                     "hi": hi, "excl0": bool(lo > 0 or hi < 0)})
        print(f"   {'r(u, 极化率 | ' + k + ')':<52}{pt:>+9.3f}"
              f"  [{lo:>+7.3f}, {hi:>+7.3f}]"
              f"  {'不含 0' if (lo > 0 or hi < 0) else '含 0'}")
    print()
    for k in DISRUPT_KEYS:
        pt, lo, hi = boot_r(dis[k], pol, seed=6)
        rows.append({"q": f"共线性 r({k}, 极化率)", "pt": pt, "lo": lo,
                     "hi": hi, "excl0": bool(lo > 0 or hi < 0)})
        print(f"   {'共线性 r(' + k + ', 极化率)':<52}{pt:>+9.3f}"
              f"  [{lo:>+7.3f}, {hi:>+7.3f}]"
              f"  {'不含 0' if (lo > 0 or hi < 0) else '含 0'}")

    # ------------------------------------------------------------
    # 2. 按家族分层
    # ------------------------------------------------------------
    print("\n" + "=" * 92)
    print("2. 按家族分层：冠状病毒自己的轴 / OMTKY3 自己的轴")
    print("=" * 92)
    newc = {z["name"]: z["new"] for z in json.load(
        open(os.path.join(ROOT, "cache",
                          "censor_density_revision.json")))["rows"]}
    mats, _ = dedup_by_content(load_independent() + load_skempi()
                               + load_all_dms())
    for d in mats:
        d["censor_frac"] = newc[d["name"]]
    mats = pick_primary(mats)
    by = {}
    for d in mats:
        if d["censor_frac"] >= 0.25:
            continue
        B, c, _ = extract_subspace(d["M"], d["mask"])
        if B is not None and len(c) == 20:
            by[d["name"]] = B
    fam = {k: [n for n in v if n in by]
           for k, v in TG["axis1_family_avg"]["families"].items()}
    axes, S2s = {}, {}
    for g, names in fam.items():
        P = np.mean([CN.proj(by[n])[0] for n in names], axis=0)
        w, V = np.linalg.eigh(P)
        V = V[:, ::-1]; w = w[::-1]
        axes[g] = V[:, 0]
        S2s[g] = V[:, :2]
        print(f"   [{g}] k = {len(names)}  λ = {np.round(w[:3], 3).tolist()}")
    print(f"   |cos(u_冠状病毒, u_OMTKY3)| = "
          f"{abs(float(axes['冠状病毒'] @ axes['OMTKY3'])):.3f}；"
          f"|cos(u, u_冠状病毒)| = {abs(float(u @ axes['冠状病毒'])):.3f}；"
          f"|cos(u, u_OMTKY3)| = {abs(float(u @ axes['OMTKY3'])):.3f}")

    print(f"\n   {'标度':<36}{'r vs u_冠状病毒':>18}{'r vs u_OMTKY3':>18}")
    strat = {}
    for k in DISRUPT_KEYS + ["local_polarizability", "sheet_prop_Chou"]:
        v = dis[k] if k in dis else scales[k]
        ra = float(np.corrcoef(axes["冠状病毒"], v)[0, 1])
        rb = float(np.corrcoef(axes["OMTKY3"], v)[0, 1])
        strat[k] = {"r_cov": ra, "r_omt": rb}
        print(f"   {k:<36}{ra:>+18.3f}{rb:>+18.3f}")

    # 零分布：1 维方向 vs 2 维子空间（d = 20，中心化 19 维）
    rngn = np.random.default_rng(21)
    P1 = np.eye(20) - np.outer(ONE, ONE)
    n1v2 = []
    for _ in range(NSIM):
        a = P1 @ rngn.normal(size=20); a /= np.linalg.norm(a)
        A = P1 @ rngn.normal(size=(20, 2))
        Q, _ = np.linalg.qr(A)
        n1v2.append(float(np.linalg.norm(Q.T @ a)))
    p95_1v2 = float(np.percentile(n1v2, 95))
    print(f"\n   1 维 vs 2 维零分布 p95（d = 20）= {p95_1v2:.3f}")

    print(f"\n3. 逐矩阵覆盖 ‖P_S v̂‖，**同源（冠状病毒 = Starr 文库）"
          f"与跨源（OMTKY3）分列**")
    print(f"   {'矩阵':<42}" + "".join(f"{k[:11]:>13}" for k in
                                       ["D1_folding", "D2_blosum",
                                        "D3_expr", "D5_PC1", "极化率"]))
    permat = []
    keys5 = ["D1_folding_ddG_ddPCA", "D2_blosum62_nonexchangeability",
             "D3_starr_expression_loss", "D5_disrupt_PC1"]
    for g, names in fam.items():
        for n in names:
            A = by[n] - np.outer(ONE, ONE @ by[n])
            Q, _ = np.linalg.qr(A)
            vals = [float(np.linalg.norm(Q.T @ cen(dis[k]))) for k in keys5]
            vp = float(np.linalg.norm(Q.T @ cen(pol)))
            permat.append({"name": n, "family": g,
                           **{k: v for k, v in zip(keys5, vals)},
                           "pol": vp})
            print(f"   {n[:40]:<42}"
                  + "".join(f"{v:>13.3f}" for v in vals) + f"{vp:>13.3f}")
    for g in fam:
        v = [z for z in permat if z["family"] == g]
        print(f"   [{g}] 中位: "
              + "; ".join(f"{k.split('_')[0]} "
                          f"{np.median([z[k] for z in v]):.3f}"
                          for k in keys5)
              + f"; 极化率 {np.median([z['pol'] for z in v]):.3f}")

    print(f"\n   判读要点：D3（Starr 表达量）与冠状病毒那一半证据**同源**"
          f"（同一批酵母展示文库）。")
    print(f"   若 D3 的覆盖在冠状病毒矩阵上高、在 OMTKY3 上低，"
          f"那是同实验的技术方向，不是跨家族的一般去稳定度。")
    cov3 = [z["D3_starr_expression_loss"] for z in permat
            if z["family"] == "冠状病毒"]
    omt3 = [z["D3_starr_expression_loss"] for z in permat
            if z["family"] == "OMTKY3"]
    print(f"   实测：冠状病毒中位 {np.median(cov3):.3f}（同源）vs "
          f"OMTKY3 中位 {np.median(omt3):.3f}（跨源）")

    out = {"boot_rows": rows, "nboot": NBOOT,
           "family_axes": {k: v.tolist() for k, v in axes.items()},
           "cos_cov_omt": abs(float(axes["冠状病毒"] @ axes["OMTKY3"])),
           "cos_u_cov": abs(float(u @ axes["冠状病毒"])),
           "cos_u_omt": abs(float(u @ axes["OMTKY3"])),
           "strat_corr": strat, "p95_1d_vs_2d": p95_1v2,
           "per_matrix": permat,
           "d3_cov_med": float(np.median(cov3)),
           "d3_omt_med": float(np.median(omt3))}
    p = os.path.join(ROOT, "cache", "generic_axis_strat.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
