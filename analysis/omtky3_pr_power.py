#!/usr/bin/env python3
"""k = 4 的 PR 检验到底有多少功效？以及把 k 提到 8/10 能买到多少？

背景（docs/13 §6.5、state/open-questions.md 的 Q-005）：
OMTKY3 × 4 蛋白酶的残差方向参与率 PR = 2.71，k=4 随机零分布
中位 3.48、95% 区间 [2.90, 3.86]。docs/13 把它读成「勉强超出」，
并指出提高 k 是最便宜的提功效路径。

Q-005 的扩样本在数据上做不到（analysis/omtky3_panel.py 的普查：
完整 SKEMPI v2 里 OMTKY3 只有 4 个蛋白酶伙伴，thr4 过滤没有丢掉任何一个）。
既然 k 不能提，就必须回答另一个问题，否则「第二轴不普适」这个否定结论
无法与「我的方法在 k=4 下看不见」区分开（CLAUDE.md §10.3：
任何「测不出结构」的结论必须先过理想条件下的还原性诊断）。

本脚本做的就是那个还原性诊断：
  在**已知答案**的合成数据上（k 个子空间共享第一轴 u，第二方向围绕
  一条共同方向以角度散布 σ 分布），问 PR 检验在给定 k 下能把多大的
  「共享程度」与各向同性零分布分开。
  σ = 0  => 第二轴完全共享（PR 应 ≈ 1）
  σ = 90°=> 第二轴各向同性（PR 应落在零分布里）
输出：每个 k 的 功效曲线 power(σ)，以及 80% 功效对应的 σ 上限。

判据既测能命中（σ 小必须显著）也测不误伤（σ=90° 的拒绝率必须 ≈ α）。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import ROOT                       # noqa: E402
from subspace_resid_dim import spectrum             # noqa: E402

ONE = np.ones(20) / np.sqrt(20)
NNULL = 4000
NSIM = 600
KS = [4, 6, 8, 10, 14]
# 集中度 kappa：0 = 严格各向同性（阴性对照），越大越集中，None = 完全共享
KAPPAS = [0.0, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0, None]


def rand_perp(u, rng):
    a = rng.normal(size=20)
    a -= ONE * (ONE @ a)
    a -= u * (u @ a)
    return a / np.linalg.norm(a)


def draw_dirs(k, u, v0, kappa, rng):
    """k 条残差方向，全在 u 的正交补（18 维）内，围绕 v0 以集中度 kappa 分布。

    **第一版这个函数是错的，而且阴性对照当场抓到了**（CLAUDE.md §10.2）：
    原版写成 d = cos(a) v0 + sin(a) p，a ~ N(0, sigma)，并把 sigma = 90°
    当成「各向同性」。但对 sigma = pi/2，E[cos^2 a] = (1 + e^{-2 sigma^2})/2
    ≈ 0.504，而 18 维里真各向同性是 E[(d·v0)^2] = 1/18 ≈ 0.056 ——
    那个参数化**永远到不了各向同性**，所以 sigma = 90° 的拒绝率是 0.79
    而不是 0.05。植入的「错误」其实不是错误，正是 D-017 那一类。

    改为投影正态（projected normal）：d ∝ kappa * v0 + g，g 为补空间里的
    标准正态。kappa = 0 **严格等于**各向同性（阴性对照由构造保证正确），
    kappa -> inf 趋于完全共享。横轴不再报 kappa，而报由模拟**实测**的
    平均 |cos(d, v0)|，这是一个可直接解释的「共享程度」。
    """
    out = []
    for _ in range(k):
        if kappa is None:
            out.append(v0.copy())
            continue
        g = rand_perp(u, rng)             # 补空间内的各向同性方向
        d = kappa * v0 + g
        out.append(d / np.linalg.norm(d))
    return out


def null_pr(k, u, rng, n=NNULL):
    return np.array([spectrum([rand_perp(u, rng) for _ in range(k)])[1]
                     for _ in range(n)])


def main():
    rng = np.random.default_rng(424242)
    u = rand_perp(np.zeros(20) + 1e-9, rng)   # 任一中心化方向（旋转不变）
    u = rng.normal(size=20); u -= ONE * (ONE @ u); u /= np.linalg.norm(u)

    print("=" * 84)
    print("PR 检验的还原性诊断（CLAUDE.md §10.3）：k 与可检出的「第二轴共享程度」")
    print("=" * 84)
    obs_pr = 2.71          # docs/13 §6.5 / cache/omtky3_panel.json 的 OMTKY3 k=4
    res = {}
    for k in KS:
        nl = null_pr(k, u, rng)
        crit = float(np.percentile(nl, 5))     # 单尾 5%：PR 显著偏小
        row = []
        for kp in KAPPAS:
            prs, cosb = [], []
            for _ in range(NSIM):
                v0 = rand_perp(u, rng)
                ds = draw_dirs(k, u, v0, kp, rng)
                cosb += [abs(float(d @ v0)) for d in ds]
                prs.append(spectrum(ds)[1])
            prs = np.array(prs)
            row.append({"kappa": kp, "share_cos": float(np.mean(cosb)),
                        "power": float((prs < crit).mean()),
                        "pr_med": float(np.median(prs))})
        res[k] = {"null_med": float(np.median(nl)),
                  "null_ci": [float(np.percentile(nl, 2.5)),
                              float(np.percentile(nl, 97.5))],
                  "crit_p05": crit, "rows": row}
        print(f"\n[k = {k}]  零分布 PR 中位 {np.median(nl):.2f} "
              f"[{np.percentile(nl, 2.5):.2f}, {np.percentile(nl, 97.5):.2f}]"
              f"；单尾 5% 临界 PR < {crit:.2f}")
        print("     共享度 mean|cos| " + "".join(f"{r['share_cos']:>7.2f}"
                                                 for r in row))
        print("     PR 中位          " + "".join(f"{r['pr_med']:>7.2f}"
                                                 for r in row))
        print("     功效             " + "".join(f"{r['power']:>7.2f}"
                                                 for r in row))
        # 80% 功效对应的最低共享度（线性插值；共享度随 kappa 单增）
        xs = [r["share_cos"] for r in row]
        ps = [r["power"] for r in row]
        s80 = None
        for i in range(len(xs) - 1):
            if ps[i] < 0.8 <= ps[i + 1]:
                t = (0.8 - ps[i]) / (ps[i + 1] - ps[i])
                s80 = xs[i] + t * (xs[i + 1] - xs[i])
                break
        res[k]["share80"] = s80
        print(f"     达到 80% 功效所需的最低共享度 mean|cos| ≈ "
              + (f"{s80:.2f}" if s80 is not None else "（区间外）"))
        neg = row[0]["power"]
        print(f"     不误伤（阴性对照 kappa = 0，构造上严格各向同性）"
              f"拒绝率 {neg:.3f}（应 ≈ 0.05）  "
              f"{'OK' if neg < 0.09 else 'FAIL'}")
        if neg >= 0.09:
            raise RuntimeError("PR 功效模拟的阴性对照未通过 —— 数字不可用")
        if k == 4:
            hit = [r["share_cos"] for r in row if r["pr_med"] <= obs_pr]
            print(f"     观测 PR = {obs_pr} 对应的合成共享度 ≈ "
                  f"{min(hit):.2f}（即第二轴之间平均 |cos| 约这么大）"
                  if hit else "     观测 PR 低于全部合成水平")

    print("\n" + "=" * 84)
    print("读法")
    print("  「共享度」= 合成时实测的 k 条残差方向与共同方向的平均 |cos|。")
    print("  18 维里各向同性的基线共享度就有 mean|cos| ≈ 0.21（量级校准：")
    print("   随机方向远非正交），所以共享度必须读成「比 0.21 高多少」。")
    print("  kappa = 0 列是阴性对照：构造上严格各向同性，拒绝率必须 ≈ 0.05。")
    p = os.path.join(ROOT, "cache", "omtky3_pr_power.json")
    json.dump({"nnull": NNULL, "nsim": NSIM, "obs_pr_k4": obs_pr,
               "by_k": {str(k): v for k, v in res.items()}},
              open(p, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
