#!/usr/bin/env python3
"""Part 2：docs/13 §4.3 排除 PSD95-PDZ3 / GRB2-SH3 的那个数字，可读吗？

docs/13 §4.3 的排除理由是「对半分自对齐 cos1 = 0.431 / 0.424，
而随机零分布的 cos1 中位就是 0.404 —— 连第一维都不可辨识」。
docs/23 §5.1 证明**对半分口径在小几何上会无功效**（4-4-20 的同几何理想
上限只有 0.230，低于一维零分布 p95 = 0.446），并在 §8.2 第 2 条、§9.2 第 3 条
把「这两个数的可读性从未核实」登记为待办。人类已授权执行。

做法与 docs/23 §5.1 完全一致，只把统计量换成**这两个数字自己的那个**：
秩 2 对半分的 `cos θ1`（`subspace_universality.py` 里的 `rel_cos1`）。
沿用矩阵自己的 mask 几何，只把数值换成「已知秩 2 真结构 + 可调噪声」。

三种结果各有不同含义（任务定义）：
 (a) 上限远高于实测 -> 排除成立，理由也成立；
 (b) 上限贴近实测   -> 结论可能对，但**陈述的理由是错的**（即 4-4-20 情形）；
 (c) 上限低于零分布 -> 该统计量在此几何上根本没有意义。

不误伤对照：`starr2020/WH1/huACE2`（201 位点）与
`lehner/GB1/IgG-Fc`（55 位点，docs/13 §4.3 判「第二维可分辨」）
必须给出 (a) 那一类的图景，否则是我的诊断坏了而不是它们的数据坏了。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_calib_lib as G  # noqa: E402
from subspace_lib import ci  # noqa: E402

N_SPLIT = 400
TARGETS = ["lehner/PSD95-PDZ3/CRIPT", "lehner/GRB2-SH3/GAB2"]
CONTROLS = ["lehner/GB1/IgG-Fc", "starr2020/WH1/huACE2"]
# docs/13 §4.3 的已发表值（已知事实一侧）
PUB_SPLIT = {"lehner/PSD95-PDZ3/CRIPT": 0.431, "lehner/GRB2-SH3/GAB2": 0.424}


def main():
    mats = G.load_evidence()
    n2 = G.null2(20)
    null_med = float(np.median(n2["cos1"]))
    null_p95 = float(np.percentile(n2["cos1"], 95))
    print("=" * 108)
    print("Part 2：秩 2 对半分 cos θ1 的几何上限诊断（CLAUDE.md §10.3）")
    print("=" * 108)
    print(f"秩 2 随机零分布（d = 20）cos1 中位 {null_med:.3f}、"
          f"p95 {null_p95:.3f}  —— docs/13 §4.3 引的「0.404」就是这个中位")

    sigmas = [0.15, 0.3, 0.5, 0.8, 1.2, 2.0]
    out = {"null_cos1_med": null_med, "null_cos1_p95": null_p95,
           "n_split": N_SPLIT, "rows": []}
    for nm in TARGETS + CONTROLS:
        d = mats[nm]
        shape, mask = d["M"].shape, d["mask"]
        e2, _ = G.rankr_expl(d["M"], mask, rank=2)
        obs = G.split_half(d["M"], mask, rank=2, nsplit=N_SPLIT, which=0)
        om = float(np.median(obs)) if obs else float("nan")
        oc = ci(obs) if obs else (float("nan"),) * 2
        print("\n" + "-" * 108)
        tag = "【排除对象】" if nm in TARGETS else "【不误伤对照】"
        print(f"{tag} {nm}")
        print(f"  几何：{shape[0]} 位点 × {shape[1]} 残基，"
              f"{int(mask.sum())} 观测；秩 2 解释交互 = {e2:.1%}")
        rep = ""
        if nm in PUB_SPLIT:
            dd = abs(om - PUB_SPLIT[nm])
            rep = (f"（docs/13 §4.3 发表值 {PUB_SPLIT[nm]:.3f}，"
                   f"复现差 {dd:.3f} "
                   f"{'OK' if dd < 0.03 else '**复现失败，停**'}）")
        print(f"  实测对半分 cos1 中位 = {om:.3f} "
              f"[{oc[0]:.2f}, {oc[1]:.2f}]（{len(obs)}/{N_SPLIT} 次成功）{rep}")
        print(f"  同几何的理想上限（已知秩 2 真结构 + 可调噪声）：")
        print(f"     {'sigma':>7}{'秩2 解释交互':>13}"
              f"{'对半分 cos1 中位 [2.5,97.5]':>30}{'成功次数':>9}")
        tab = []
        for sg in sigmas:
            ee = []
            for t in range(6):
                rng = np.random.default_rng(4100 + t + int(sg * 1e4))
                X = G.synth_rank(shape, 2, sg, rng)
                x, _ = G.rankr_expl(X, mask, rank=2)
                if x is not None:
                    ee.append(x)
            vv = G.split_half_ceiling(shape, mask, rank=2, sigma=sg,
                                      n_real=12, nsplit=N_SPLIT, which=0,
                                      seed=4100)
            m = float(np.median(vv)) if vv else float("nan")
            c = ci(vv) if vv else (float("nan"),) * 2
            em = float(np.median(ee)) if ee else float("nan")
            tab.append({"sigma": float(sg), "expl": em, "split": m,
                        "split_ci": list(c), "n": len(vv)})
            print(f"     {sg:>7.2f}{em:>12.1%}{m:>16.3f} "
                  f"[{c[0]:.2f},{c[1]:.2f}]{len(vv):>9}")
        good = [r for r in tab if np.isfinite(r["expl"])]
        near = min(good, key=lambda r: abs(r["expl"] - e2)) if good else None
        verdict, outcome = "", None
        if near is not None:
            ceil_ = near["split"]
            print(f"  -> 实测解释率 {e2:.1%} 最接近 sigma = {near['sigma']}"
                  f"（{near['expl']:.1%}），该格的对半分 cos1 上限 = "
                  f"{ceil_:.3f}")
            print(f"     动态范围：上限 − 零分布 p95 = "
                  f"{ceil_ - null_p95:+.3f}；实测在 [零分布中位 "
                  f"{null_med:.3f}, 上限 {ceil_:.3f}] 中的位置 = "
                  f"{(om - null_med) / max(1e-9, ceil_ - null_med):.2f}")
            # 三分法只对**低实测值**（排除对象）有定义。实测本身已超零分布
            # p95 的矩阵不是排除对象：它「贴近上限」的含义是**已经达到上限**，
            # 与「测不出来」正好相反。混为一谈会把两个对照误判成 (b)。
            if ceil_ < null_p95:
                outcome = "c"
                verdict = (f"**(c) 上限 {ceil_:.3f} 低于零分布 p95 "
                           f"{null_p95:.3f} —— 该统计量在此几何上无意义**")
            elif om > null_p95:
                outcome = "-"
                verdict = (f"（不适用三分法）实测 {om:.3f} 本身超零分布 p95 "
                           f"{null_p95:.3f}，不是排除对象；它贴近上限 "
                           f"{ceil_:.3f} 只说明**已达该几何的上限**")
            elif ceil_ - om < 0.10:
                outcome = "b"
                verdict = (f"**(b) 上限 {ceil_:.3f} 贴近实测 {om:.3f}"
                           f"（差 {ceil_ - om:+.3f} < 0.10）—— 结论可能对，"
                           f"但陈述的理由错了**")
            else:
                outcome = "a"
                verdict = (f"(a) 上限 {ceil_:.3f} 远高于实测 {om:.3f}"
                           f"（差 {ceil_ - om:+.3f}），且上限超零分布 p95 "
                           f"—— 排除成立，理由也成立")
            print(f"  => {verdict}")
        out["rows"].append({"name": nm, "role": ("target" if nm in TARGETS
                                                 else "control"),
                            "shape": list(shape),
                            "n_obs": int(mask.sum()), "expl_rank2": e2,
                            "split_obs": om, "split_obs_ci": list(oc),
                            "n_split_obs": len(obs),
                            "pub": PUB_SPLIT.get(nm), "ceiling_table": tab,
                            "sigma_matched": (near["sigma"] if near else None),
                            "ceiling": (near["split"] if near else None),
                            "outcome": outcome, "verdict": verdict})

    print("\n" + "=" * 108)
    print("汇总")
    print("=" * 108)
    for r in out["rows"]:
        print(f"  {r['name'][:40]:<42}实测 {r['split_obs']:.3f}  上限 "
              f"{(r['ceiling'] if r['ceiling'] is not None else float('nan')):.3f}"
              f"  -> 结果 ({r['outcome']})")
    p = os.path.join(G.ROOT, "cache", "gate_splithalf_power.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print("\n产物:", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
