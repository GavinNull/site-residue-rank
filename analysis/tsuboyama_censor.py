#!/usr/bin/env python3
"""docs/27 Step 2：截尾 / 动态范围闸。**在任何统计之前。**

口径选择（本步最重要的判断，先写理由，再给数）
================================================================
本仓库此前见过三种弱端机制，Tsuboyama 是**第四种**：

| 数据 | 机制 | 正确口径 |
|---|---|---|
| Starr ACE2 系列 | **硬赋值**：不结合的变体被赋成 5.000 / 6.000（一个原子） | Gen-3 `q` |
| ddPCA（docs/15） | 超出范围**记成 NA** | 缺失率 |
| CR9114 Tite-seq（docs/25） | `nls` 有界但**不裁剪**，返回内点外插值 → 平滑斜坡 | `f_out`（窗外比例，需反推） |
| **Tsuboyama（本文档）** | 拟合给出 ΔG，但若落在作者验证过的窗口 **[−1, 5]** 之外，数值被**换成字面标记** `<-1` / `>5`，ΔΔG 直接扣留 | **`f_out`（直接计数，不需估计）** |

与前三种都不同的两点：
  * **不是堆积。** `tsuboyama_load` 的 T10 已断言数值型 `dG_ML` 严格落在
    开区间 (−1, 5) 内，边界质量 0.00026% / 0.00000% —— 越界值被**扣留**
    而不是被压到边界。Gen-3 的假设 (A1)（「压平 + 叠噪声」）不成立。
  * **比 docs/25 的 `f_out` 更强。** CR9114 的 `f_out` 要靠反推「实际滴定窗」；
    这里窗外身份是作者**显式标注**的，`f_out` 是直接计数。

**并且本文档在这里抓到一个 Gen-3 的新失效区（第七次工具事故，§10）：**
不是「报 ~0 且无信息量」（docs/21 §2.5 已登记的那种），而是
**在右偏分布上系统性假阳性**。合成对照（`skewnorm`，**完全没有截尾**）：

    α = 0 → q = 0.1%（闸「无堆积」，正确）
    α = 2 → q = 6.2%、α = 4 → 26.4%、α = 6 → 24.2%、α = 8 → 24.6%
            （三道存在性闸全部**通过**，两个估计量残差 1e-4 —— 它们
             一致地在拟合**同一个错的东西**：分布自己的众数）

机制：ΔΔG 在「正 = 去稳定」约定下是右偏的（单取代能稳定蛋白的幅度有上限，
多数取代去稳定），众数离最小值只有约 1 个 `s_up`。Gen-3 的截断拟合把
**众数附近的主体质量**读成「下限堆积」，`σ̂/s_up` 落在 0.19–0.33，
恰好在 γ = 0.35 之下，于是存在性闸放行。
判别指标：`(ĉ − min) / s_up ≲ 1.2` 时假阳性，真堆积时该比值 ≈ 0。
**这是检测器自身的错，不是数据的错**（§10 要求分开记，否则会去改正确的那个）。

于是：**`f_out` 是本数据集唯一适用的口径，Gen-3 的 `q` 在这里既不是判据，
也不能当成「没有堆积」的证据。** 两个方向都报（见 §Gen-3 一节），
并在 ddG_ML 取向（真正的去稳定截尾端落在下尾）上做 §10.3 还原性诊断。

闸判（判据照 docs/25 §3.5，不新造）
================================================================
用跨三代检测器都不动的那条单调事实（D-029）：
秩 0 / 1 / 2 对应截尾中位 **43.2% / 10.9% / 0.1%**。
比的是**区间与这些层的位置关系**，不是均值大小（CLAUDE.md §5）。
要求 `f_out` 的 95% bootstrap 区间**整体低于秩 1 层 10.9%**；
`miss_frac`（全部不可得格子 / 完整设计）同判据 —— 它才是与 ddPCA
`censor_frac`（GRB2-SH3/folding 3.0% → 功效 1.00；GB1 13.1% → 0.88；
PSD95-PDZ3 24.4% → **0.00**）**同口径**的量。两个都必须过。

运行：python3 analysis/tsuboyama_censor.py
产物：cache/tsuboyama_censor.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import skewnorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsuboyama_load as TL                                     # noqa: E402
from censor_density import (censor_fraction_density,            # noqa: E402
                            censor_fraction_density_ci,
                            upper_scale)
from censor_gen3_atom_power import implant                      # noqa: E402
from subspace_lib import ROOT, censor_fraction                  # noqa: E402

STRATA = {"rank0": 0.432, "rank1": 0.109, "rank2": 0.001}
NBOOT, NSUB = 2000, 12


def boot_frac(k, n, nboot=NBOOT, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.binomial(n, k / n, nboot) / n
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def gen3_fp_control():
    """§10.2 不误伤对照：**完全没有截尾**的右偏分布上 Gen-3 报什么。

    这一项是本文档新增的，因为它**不通过** —— 必须留痕而不是绕过。
    """
    rows = []
    for alpha in (0, 1, 2, 4, 6, 8):
        v = skewnorm.rvs(alpha, size=1000, scale=1.0, random_state=7)
        q, info = censor_fraction_density(v, detail=True)
        s = np.sort(v)
        rows.append({"alpha": alpha, "truth": 0.0, "q_G3": float(q),
                     "gate": info["gate"],
                     "sigma_over_s": info["sigma_over_s"],
                     "c_rel_over_s_up": (info["c_rel"] / info["s_up"]
                                         if info["c_rel"] is not None
                                         else None),
                     "ok": bool(q < 0.02)})
    return rows


def main():
    info, mats = TL.load_cached()
    print("=" * 102)
    print("docs/27 Step 2：截尾 / 动态范围闸（口径 = f_out）")
    print("=" * 102)
    T10 = info["selfcheck"]["T10_window"]
    print(f"\n[机制] 数值型 dG_ML ∈ ({T10['lo']}, {T10['hi']})，"
          f"实测 {T10['min']:.4f}…{T10['max']:.4f}（n = {T10['n']}）；"
          f"边界质量 {T10['mass_at_lo']:.5%} / {T10['mass_at_hi']:.5%}")
    print("        => 没有堆积；越界值被**扣留**。两端都会被截："
          "`<-1` 截去稳定端、`>5` 截稳定端。")

    # ------------------------------------------------------------------
    # 0. Gen-3 的不误伤对照（本文档新增，**不通过**，必须先说）
    # ------------------------------------------------------------------
    print("\n" + "=" * 102)
    print("[G-FP] Gen-3 不误伤对照（§10.2）：合成**完全无截尾**的右偏分布")
    print("=" * 102)
    fp = gen3_fp_control()
    print(f"   {'α':>4}{'真值':>7}{'q_G3':>9}{'σ̂/s_up':>10}"
          f"{'(ĉ−min)/s_up':>14}{'存在性闸':>10}{'判据 q<2%':>11}")
    for r in fp:
        print(f"   {r['alpha']:>4}{0:>7.0%}{r['q_G3']:>9.2%}"
              f"{('—' if r['sigma_over_s'] is None else '%.3f' % r['sigma_over_s']):>10}"
              f"{('—' if r['c_rel_over_s_up'] is None else '%.2f' % r['c_rel_over_s_up']):>14}"
              f"{str(r['gate']):>10}{'OK' if r['ok'] else '**未过**':>11}")
    nfail = sum(1 for r in fp if not r["ok"])
    print(f"   => {nfail}/{len(fp)} 格**未过**。右偏（α ≥ 2）时 Gen-3 把"
          f"分布自己的众数读成下限堆积，q 到 6%–26%，而真值是 0。")
    print("   => 这是**检测器自身的错**（不是数据的错），本仓库第七次工具事故，")
    print("      且部分静默：三道存在性闸放行、两个估计量残差 ~1e-4。")
    print("      判别指标：(ĉ−min)/s_up ≲ 1.2 为假阳性区；真堆积时该比值 ≈ 0。")

    # ------------------------------------------------------------------
    # 1. 逐矩阵的量（Gen-3 喂 raw，两个取向都报，带子抽样区间）
    # ------------------------------------------------------------------
    rows = {}
    for i, (w, m) in enumerate(sorted(mats.items())):
        raw = m["raw"]                      # 仓库取向：+ = 去稳定
        g3p = censor_fraction_density_ci(raw, n_sub=NSUB, seed=i)
        g3n = censor_fraction_density_ci(-raw, n_sub=NSUB, seed=i)
        dp = g3p["info"]; dn = g3n["info"]
        nout = m["n_hi"] + m["n_lo"]
        fo = nout / m["n_var"]
        fo_ci = boot_frac(nout, m["n_var"], seed=1000 + i)
        nmiss = m["design"] - m["n_obs"]
        mf = nmiss / m["design"]
        mf_ci = boot_frac(nmiss, m["design"], seed=2000 + i)
        rows[w] = {
            "n_raw": int(len(raw)),
            "q_G3_dest": g3p["q"], "q_G3_dest_sub": [g3p["lo"], g3p["hi"]],
            "q_G3_dest_gate": dp["gate"],
            "q_G3_dest_crel_over_sup": (dp["c_rel"] / dp["s_up"]
                                        if dp["c_rel"] is not None else None),
            "q_G3_ddG": g3n["q"], "q_G3_ddG_sub": [g3n["lo"], g3n["hi"]],
            "q_G3_ddG_gate": dn["gate"],
            "q_G3_ddG_crel_over_sup": (dn["c_rel"] / dn["s_up"]
                                       if dn["c_rel"] is not None else None),
            "q_atom": float((raw <= raw.min() + 1e-9).mean()),
            "q_G2": float(censor_fraction(raw)),
            "s_up_dest": dp["s_up"], "s_up_ddG": dn["s_up"],
            "f_out": fo, "f_out_ci": list(fo_ci),
            "f_out_hi": m["n_hi"] / m["n_var"],
            "f_out_lo": m["n_lo"] / m["n_var"],
            "miss_frac": mf, "miss_frac_ci": list(mf_ci),
            "sites": m["sites"], "n_obs": m["n_obs"],
            "parent": m.get("parent"), "natural": m.get("parent") is not None,
        }
    qp = np.array([r["q_G3_dest"] for r in rows.values()])
    qn = np.array([r["q_G3_ddG"] for r in rows.values()])
    print("\n" + "=" * 102)
    print("[Gen-3 原样跑在 raw 上，两个取向] 数都报，但都不作判据")
    print("=" * 102)
    print(f"   仓库取向（+ = 去稳定；真正的去稳定截尾端在**上**尾，"
          f"Gen-3 只看下尾）：q 中位 {np.median(qp):.3%}、"
          f"最大 {qp.max():.2%}、>2% 的 {int((qp > 0.02).sum())}/{len(qp)} 个")
    print(f"   ddG_ML 取向（− = 去稳定；去稳定截尾端在**下**尾，"
          f"这是唯一可能有效的取向）：q 中位 {np.median(qn):.3%}、"
          f"最大 {qn.max():.2%}、>2% 的 {int((qn > 0.02).sum())}/{len(qn)} 个")
    hiq = [w for w, r in rows.items() if r["q_G3_dest"] > 0.02]
    if hiq:
        cr = np.array([rows[w]["q_G3_dest_crel_over_sup"] for w in hiq
                       if rows[w]["q_G3_dest_crel_over_sup"] is not None])
        print(f"   仓库取向下 q > 2% 的 {len(hiq)} 个矩阵，其 "
              f"(ĉ−min)/s_up 中位 **{np.median(cr):.2f}** "
              f"—— 落在 [G-FP] 标出的假阳性区（≲1.2），"
              f"而不是真堆积区（≈0）")
    print(f"   q_atom 中位 {np.median([r['q_atom'] for r in rows.values()]):.5%}"
          f"  => 数值列里连原子层都没有（与 T10 一致）")

    # ------------------------------------------------------------------
    # 2. §10.3 还原性诊断（在 ddG_ML 取向上，σ 按**底材自己的 s_up** 缩放）
    # ------------------------------------------------------------------
    print("\n" + "=" * 102)
    print("§10.3 还原性诊断：在 ddG_ML 取向、这些矩阵自己的几何上，"
          "若真有堆积，Gen-3 看得见吗")
    print("（σ 按底材自己的 s_up 缩放 —— docs/25 §3.4(b) 自捉的错不重犯）")
    print("=" * 102)
    clean = [w for w, r in rows.items()
             if r["q_G3_ddG"] < 0.01 and r["n_raw"] >= 1000]
    base_name = max(clean, key=lambda w: rows[w]["n_raw"])
    sub = -mats[base_name]["raw"]          # ddG_ML 取向
    s_up = upper_scale(np.sort(sub))
    print(f"   底材 = {base_name}（ddG_ML 取向） n = {len(sub)}  "
          f"s_up = {s_up:.4f}  Gen-1 "
          f"{float((sub <= sub.min() + 1e-9).mean()):.3%}  "
          f"Gen-2 {censor_fraction(sub):.3%}  "
          f"Gen-3 {censor_fraction_density(sub):.3%}（底材须先干净）")
    assert censor_fraction_density(sub) < 0.03, "底材自己被判截尾，诊断无效"
    REL, REPS = [0.05, 0.10, 0.20, 0.30, 0.49], 8
    cases = ([("P1", qa, 0.0) for qa in (0.02, 0.05, 0.10, 0.20, 0.40)]
             + [("P2", 0.0, qs) for qs in (0.03, 0.05, 0.10, 0.20, 0.40)]
             + [("P3", 0.002, 0.03), ("P3", 0.002, 0.10),
                ("P3", 0.01, 0.20)])
    diag = []
    print(f"\n   {'构型':<5}{'q_atom':>8}{'q_smear':>9}{'σ/s_up':>8}"
          f"{'真值':>7}{'Gen-3 均值':>12}{'Gen-3 跨度':>20}{'判据':>8}")
    for tag, qa, qs in cases:
        for rel in ([None] if qs == 0 else REL):
            sg = 0.0 if rel is None else rel * s_up
            got = np.array([censor_fraction_density(
                implant(sub, qa, qs, sg, np.random.default_rng(900 + 7 * k)))
                for k in range(REPS)])
            truth = qa + qs
            ok = (abs(got.mean() - truth) <= 0.02 if tag == "P1"
                  else (got.mean() >= truth / 2 if tag == "P2"
                        else got.mean() >= qa + 0.5 * qs))
            diag.append({"cfg": tag, "q_atom": qa, "q_smear": qs,
                         "sigma_over_s": rel, "truth": truth,
                         "mean": float(got.mean()),
                         "span": [float(got.min()), float(got.max())],
                         "ok": bool(ok)})
            print(f"   {tag:<5}{qa:>8.1%}{qs:>9.1%}"
                  f"{('—' if rel is None else '%.2f' % rel):>8}"
                  f"{truth:>7.1%}{got.mean():>12.1%}"
                  f"{'[%.1f%%, %.1f%%]' % (100 * got.min(), 100 * got.max()):>20}"
                  f"{'OK' if ok else '未过':>8}")
    byc = {}
    for d in diag:
        byc.setdefault(d["cfg"], []).append(d["ok"])
    print("   合计: " + "、".join(f"{k} {sum(v)}/{len(v)}"
                                  for k, v in byc.items()))
    print("\n   **判读**：在 q_smear ≥ 3%、σ/s_up ≤ 0.30 的格子上 Gen-3 命中，")
    print("   所以 ddG_ML 取向下报的 ~0.1% 是一个**有信息量的非检出 ——"
          "数值列里确实没有堆积**。")
    print("   但这**不等于没有截尾**：截尾在这里表现为**扣留**，"
          "所以判据必须是 f_out。")

    # ------------------------------------------------------------------
    # 3. 闸判
    # ------------------------------------------------------------------
    print("\n" + "=" * 102)
    print(f"闸判（docs/25 §3.5：f_out 与 miss_frac 的 95% 区间都须整体低于"
          f"秩 1 层 {STRATA['rank1']:.1%}）")
    print("=" * 102)
    for r in rows.values():
        r["pass_fout"] = bool(r["f_out_ci"][1] < STRATA["rank1"])
        r["pass_miss"] = bool(r["miss_frac_ci"][1] < STRATA["rank1"])
        r["pass_step2"] = bool(r["pass_fout"] and r["pass_miss"])
    npass = sum(r["pass_step2"] for r in rows.values())
    nat_pass = sorted(w for w, r in rows.items()
                      if r["pass_step2"] and r["natural"])
    print(f"   候选 {len(rows)} 个 → **通过 {npass} 个**"
          f"（天然 {len(nat_pass)}、设计 {npass - len(nat_pass)}）")
    print(f"   f_out 不过的 "
          f"{sum(1 for r in rows.values() if not r['pass_fout'])} 个；"
          f"f_out 过而 miss_frac 不过的 "
          f"{sum(1 for r in rows.values() if r['pass_fout'] and not r['pass_miss'])} 个")
    prim = info["primary_natural"]
    prim_pass = {p: w for p, w in prim.items()
                 if w in rows and rows[w]["pass_step2"]}
    print(f"   逐母体 PDB 一个背景（Step 1 的确定性规则）后："
          f"天然独立结构域 {len(prim)} → **过闸 {len(prim_pass)} 个**")
    fp_ = np.array([rows[w]["f_out"] for w in prim_pass.values()])
    mf_ = np.array([rows[w]["miss_frac"] for w in prim_pass.values()])
    print(f"\n   过闸的天然独立集：f_out 中位 {np.median(fp_):.2%}、"
          f"最大 {fp_.max():.2%}；miss_frac 中位 {np.median(mf_):.2%}、"
          f"最大 {mf_.max():.2%}")
    print(f"   对照（同口径，docs/15 §2.2）：ddPCA censor_frac = "
          f"GRB2-SH3/folding 3.0%（功效 1.00）、GB1 13.1%（0.88）、"
          f"PSD95-PDZ3 24.4%（**0.00**）")

    show = sorted(prim_pass.values(), key=lambda w: -rows[w]["n_obs"])[:12]
    print("\n   前 12 个（按观测数）：")
    print(f"   {'矩阵':<22}{'位点':>5}{'观测':>7}{'f_out':>8}"
          f"{'f_out 95%CI':>17}{'miss':>7}{'q_G3(ddG)':>11}"
          f"{'子抽样':>15}")
    for w in show:
        r = rows[w]
        print(f"   {w:<22}{r['sites']:>5}{r['n_obs']:>7}{r['f_out']:>8.2%}"
              f"{'[%.1f%%, %.1f%%]' % (100 * r['f_out_ci'][0], 100 * r['f_out_ci'][1]):>17}"
              f"{r['miss_frac']:>7.1%}{r['q_G3_ddG']:>11.3%}"
              f"{'[%.2f%%, %.2f%%]' % (100 * r['q_G3_ddG_sub'][0], 100 * r['q_G3_ddG_sub'][1]):>15}")

    out = {"caliber": "f_out",
           "caliber_reason": "fourth limit mechanism: out-of-window dG "
                             "replaced by literal markers '<-1'/'>5' and "
                             "ddG withheld; no pile, no ramp",
           "strata": STRATA, "gen3_false_positive_control": fp,
           "gen3_fp_fail": nfail,
           "base_substrate": base_name, "base_s_up": float(s_up),
           "recovery_diag": diag, "rows": rows,
           "n_candidates": len(rows), "n_pass": npass,
           "natural_pass": nat_pass, "primary_pass": prim_pass}
    p = os.path.join(ROOT, "cache", "tsuboyama_censor.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
