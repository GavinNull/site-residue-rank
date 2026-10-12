#!/usr/bin/env python3
"""Gen-3 在「硬赋原子 + 叠加被摊开的堆积」这种混合构型下的**还原性诊断**。

为什么必须有这一项（CLAUDE.md §10.3）
--------------------------------------------------------------
`censor_gen3_final.py` 的实测结果是：在 docs/19 的 34 个单元上，
密度型检测器的混合模型分支**一个都没有命中**（`gate = "无堆积"` 34/34），
于是 Gen-3 退化成 `max(q_atom, 0) = q_atom`，**逐单元等于 Gen-1**。

docs/21 §9 的机制说法是「这些矩阵 σ = 0，原子比例就是真值」。
但 34 个单元里有几个的**锐度**（精确并列 / 窗口内比例）只有 0.48–0.54
（SARS-CoV-1_Urbani/RlanACE2 0.54、PRD-0038/RlanACE2 0.50、
 WH1/RalcACE2 0.48）—— 即下限附近有**一半的质量不是精确并列**。
对这些矩阵，「原子 = 全部堆积」不是可以假设的，必须检验。

于是必须分清两件在数值上长得一样的事：
  (A) 原子**就是**全部堆积，检测器正确地说「没有额外堆积」；
  (B) 原子之上**确实**叠着一层被摊开的堆积，而检测器看不见它 ——
      那就是又一次静默欠检，而「Gen-3 = Gen-1」这个吻合是
      **没有信息量的**，不能当成对 Gen-1 口径的确认。
CLAUDE.md §10.3：任何「测不出结构」的结论，必须先过一次理想条件下的
还原性诊断。本脚本就是那一次。

设计（既测能命中，也测不误伤 —— §10.2）
--------------------------------------------------------------
底材用**真实的干净矩阵**（animalACE2/Omicron_BA2/bind_cat，Gen-1 = Gen-2 =
Gen-3 = 0.0%），不用合成高斯，这样底材的形状就是真实 DMS 的形状。
在它上面植入已知构型，**植入的「错误」必须在被实际扫描的区间内真的错**
（§10.2，D-017 的教训）：

  [P1] 纯原子 q_a：全部压平到同一个值，σ = 0。真值 = q_a。
       （正控制：这是 34 个单元被认定的构型）
  [P2] 纯摊开堆积 q_s，σ > 0：压平后整层叠噪声。真值 = q_s。
       （正控制：docs/21 §2.6 盲区表已过，这里在真实底材上重测）
  [P3] **原子 + 摊开堆积的混合**：先取最低 q_a 压成原子，
       再取紧接着的 q_s 压到同一下限后叠 σ 噪声。真值 = q_a + q_s。
       这是 34 个单元里锐度 ≈ 0.5 的那几个的**嫌疑构型**。
       判据：`measured >= q_a + 0.5·q_s`（至少看见一半的摊开层）。
       若实测 ≈ q_a（只看见原子），则 (B) 成立。

输出一张表，并给出明确判定。本脚本**不改任何检测器参数**。
运行：python3 analysis/censor_gen3_atom_power.py [--reps 6]
产物：cache/censor_gen3_atom_power.json
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from censor_density import censor_fraction_density      # noqa: E402
from subspace_lib import censor_fraction                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DMS = os.path.join(ROOT, "cache", "dms")
BASE = ("starr_animalACE2_final_variant_scores.csv", "Omicron_BA2",
        "bind_cat")


def base_values():
    """真实干净底材：Gen-1/2/3 都 ~0 的那个单元（已知事实）。"""
    fn, bg, col = BASE
    out = []
    for r in csv.DictReader(open(os.path.join(DMS, fn))):
        if r.get("target") != bg:
            continue
        v = r.get(col, "")
        if v in ("", "NA"):
            continue
        out.append(float(v))
    return np.array(out)


def implant(v, q_a, q_s, sigma, rng):
    """植入 [P1]/[P2]/[P3] 的构型。返回观测数组。

    机制与 docs/19 §2.2 / censor_density.synth_censored 同构：
    压平到下限，再对**被压平的那一层**叠 σ 噪声。
    原子层不叠噪声（σ = 0 的硬赋），摊开层叠 σ。
    """
    v = np.sort(v.copy())
    n = len(v)
    na = int(round(q_a * n))
    ns = int(round(q_s * n))
    c = v[na + ns] if na + ns < n else v[-1]
    out = v.copy()
    if na + ns > 0:
        out[:na + ns] = c
    if ns > 0 and sigma > 0:
        # 摊开层 = 紧接原子之上的那 ns 个（它们被压到同一个 c，再叠噪声）
        out[na:na + ns] = c + rng.normal(0, sigma, ns)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6)
    a = ap.parse_args()

    v = base_values()
    print("=" * 100)
    print("Gen-3 还原性诊断：原子 + 摊开堆积的混合构型（CLAUDE.md §10.3）")
    print("=" * 100)
    g1 = float((v <= v.min() + 1e-6).mean())
    print(f"底材 = animalACE2/{BASE[1]}/{BASE[2]}  n = {len(v)}  "
          f"Gen-1 = {g1:.1%}  Gen-2 = {censor_fraction(v):.1%}  "
          f"Gen-3 = {censor_fraction_density(v):.1%}  （已知事实：无截尾）")
    if censor_fraction_density(v) > 0.03:
        print("** 底材自己就被判成截尾 —— 诊断无效，中止 **")
        return 1

    rows = []
    print(f"\n{'构型':<6}{'q_atom':>8}{'q_spread':>10}{'σ':>7}{'真值':>8}"
          f"{'Gen-3 均值':>12}{'Gen-3 跨度':>18}{'Gen-2':>8}{'判据':>26}")
    print("-" * 100)

    cases = []
    # [P1] 纯原子（正控制：34 个单元被认定的构型）
    for q in (0.10, 0.20, 0.40):
        cases.append(("P1", q, 0.0, 0.0, q))
    # [P2] 纯摊开（正控制：docs/21 盲区表的构型，换成真实底材）
    for q in (0.20, 0.40):
        for s in (0.07, 0.15):
            cases.append(("P2", 0.0, q, s, q))
    # [P3] 混合（34 个单元中锐度 ~0.5 的嫌疑构型）
    #      锐度 ~0.5 对应 q_atom ≈ q_spread
    for q_a, q_s in ((0.10, 0.10), (0.20, 0.20), (0.10, 0.30),
                     (0.20, 0.10)):
        for s in (0.07, 0.15):
            cases.append(("P3", q_a, q_s, s, q_a + q_s))

    for tag, q_a, q_s, s, truth in cases:
        vals, g2s = [], []
        for k in range(a.reps):
            rng = np.random.default_rng(1000 + k)
            w = implant(v, q_a, q_s, s, rng)
            vals.append(censor_fraction_density(w))
            g2s.append(censor_fraction(w))
        vals = np.array(vals)
        mu = float(vals.mean())
        if tag == "P1":
            good = abs(mu - truth) <= 0.02
            crit = "|测 − 真| <= 2pp"
        elif tag == "P2":
            good = abs(mu - truth) <= 0.15 and mu >= truth / 2
            crit = "|测 − 真| <= 15pp 且 >= 真/2"
        else:
            good = mu >= q_a + 0.5 * q_s
            crit = f"测 >= q_a + 0.5·q_s = {q_a + 0.5 * q_s:.0%}"
        rows.append({"case": tag, "q_atom": q_a, "q_spread": q_s,
                     "sigma": s, "truth": truth, "g3_mean": mu,
                     "g3_min": float(vals.min()), "g3_max": float(vals.max()),
                     "g2_mean": float(np.mean(g2s)), "pass": bool(good),
                     "criterion": crit})
        print(f"{tag:<6}{q_a:>8.0%}{q_s:>10.0%}{s:>7.2f}{truth:>8.0%}"
              f"{mu:>12.1%}"
              f"{'['+format(vals.min(),'.1%')+','+format(vals.max(),'.1%')+']':>18}"
              f"{np.mean(g2s):>8.1%}"
              f"  {crit:<22}{'OK' if good else '<< 未过'}")

    p1 = [r for r in rows if r["case"] == "P1"]
    p2 = [r for r in rows if r["case"] == "P2"]
    p3 = [r for r in rows if r["case"] == "P3"]
    print("\n" + "=" * 100)
    print(f"[P1] 纯原子（正控制）        : "
          f"{sum(r['pass'] for r in p1)}/{len(p1)} 过")
    print(f"[P2] 纯摊开（正控制）        : "
          f"{sum(r['pass'] for r in p2)}/{len(p2)} 过")
    print(f"[P3] 原子 + 摊开（关键诊断） : "
          f"{sum(r['pass'] for r in p3)}/{len(p3)} 过")
    # 关键判定：P3 下 Gen-3 是否只看见原子
    only_atom = [r for r in p3
                 if abs(r["g3_mean"] - r["q_atom"]) <= 0.02]
    print(f"\n[P3] 里 Gen-3 实测 ≈ q_atom（只看见原子层，误差 <= 2pp）的格数 "
          f"= {len(only_atom)}/{len(p3)}")
    verdict = ("B：原子之上的摊开层对 Gen-3 不可见 —— "
               "「Gen-3 = Gen-1」在锐度 < 1 的单元上是**退化**，不是确认"
               if len(only_atom) >= len(p3) / 2 else
               "A：Gen-3 能看见原子之上的摊开层 —— "
               "它在 34 个单元上报「无额外堆积」是一个有信息量的判定")
    print(f"\n判定：{verdict}")
    print("=" * 100)
    json.dump({"base": list(BASE), "base_n": int(len(v)), "reps": a.reps,
               "rows": rows, "n_only_atom": len(only_atom),
               "n_p3": len(p3), "verdict": verdict,
               "provenance": "analysis/censor_gen3_atom_power.py; "
                             "底材 cache/dms/" + BASE[0] +
                             "; 检测器 censor_density（参数原样）"},
              open(os.path.join(ROOT, "cache",
                                "censor_gen3_atom_power.json"), "w"),
              indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
