#!/usr/bin/env python3
"""M5 的单边功效分析：只**排除 r=1**需要多少测点？

动机（state/gates.md 的可行性冲突）：
C6 要求约 320 个测点才能「定秩」，而 D-019 把可分辨阈值提到 1.8 kcal/mol
（跨源）之后，同时满足两个条件的对可能远不足 320。

结构性理由：
  定秩（在 r=0..4 里选）需要数据量足以稳定估计**最大的那个模型**；
  而嵌套的单边比较 r=1 vs r>=2 只需分辨**增量**
  —— 秩 1->2 多出 (n_r+n_c) - 3 个参数，远少于秩 2 的总参数数。
所以单边检验的功效需求应当显著更低。本脚本量化这个「显著更低」。

检验构造：
  H0: 去主效应后交互矩阵秩为 1
  H1: 秩 >= 2
  统计量 Delta = cv_err(rank 1) - cv_err(rank 2)   （正值 => 秩 2 更优）
  阈值取 H0 下 Delta 的 95 分位（故 alpha = 0.05 被**校准**，不是假定）
  功效 = P(Delta > 阈值 | 真实秩 2)

噪声取值已按 evidence/assay-noise.md 更新：
  每个矩阵元是一个 ΔG 量，其单次测量误差为
    同源（单篇/单实验室）  ~0.35 kcal/mol
    跨文献混合            ~0.64 kcal/mol   (Hahn 2022 社区推荐值)
    跨文献生物物理 assay   ~0.78            (= Ross 2023 的成对 1.10 / sqrt2)
  早先的 rank_power.py 用 0.5，介于两者之间 —— 跨源情形比它模拟的更糟。

用法:
  python3 analysis/rank_test_power.py            # 默认网格
  python3 analysis/rank_test_power.py --quick
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_power import als_complete, decenter, n_params  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAIN_SCALE = 1.5
LAM1 = 1.5          # 主耦合项的元素级 RMS，kcal/mol

SIGMA = {
    "same_source": 0.35,
    "cross_mixed": 0.64,
    "cross_biophys": 0.78,
}


def make(nr, nc, rng, lam2=None):
    """主效应 + 秩1 耦合 (+ 可选秩2 耦合)。lam2=None 即 H0。"""
    a = rng.normal(0, MAIN_SCALE, nr)
    b = rng.normal(0, MAIN_SCALE, nc)
    M = a[:, None] + b[None, :]
    lams = [LAM1] + ([lam2] if lam2 else [])
    for lam in lams:
        phi = rng.normal(0, 1, nr); phi /= np.linalg.norm(phi)
        psi = rng.normal(0, 1, nc); psi /= np.linalg.norm(psi)
        M += lam * np.sqrt(nr * nc) * np.outer(phi, psi)
    return M


def cv_err(M, obs, rank, rng, K=4, restarts=2, iters=150):
    idx = np.argwhere(obs); rng.shuffle(idx)
    folds = np.array_split(idx, K)
    tot, cnt = 0.0, 0
    for f in folds:
        trn = obs.copy()
        for i, j in f:
            trn[i, j] = False
        if trn.sum() < max(M.shape):
            return None
        Y = decenter(M, trn)
        best = np.inf
        for _ in range(restarts):
            P = als_complete(np.where(trn, Y, 0.0), trn, rank, rng, iters=iters)
            e = np.mean([(Y[i, j] - P[i, j]) ** 2 for i, j in f])
            best = min(best, e)
        tot += best * len(f); cnt += len(f)
    return np.sqrt(tot / cnt) if cnt else None


def statistic(M, obs, rng):
    e1 = cv_err(M, obs, 1, rng)
    e2 = cv_err(M, obs, 2, rng)
    if e1 is None or e2 is None:
        return None
    return e1 - e2


def run_cell(nr, nc, fill, sigma, lam2, nsim, rng):
    null, alt = [], []
    for _ in range(nsim):
        o = rng.random((nr, nc)) < fill
        if o.sum() < 2 * max(nr, nc):
            continue
        M0 = make(nr, nc, rng) + rng.normal(0, sigma, (nr, nc))
        s = statistic(M0, o, rng)
        if s is not None:
            null.append(s)
    for _ in range(nsim):
        o = rng.random((nr, nc)) < fill
        if o.sum() < 2 * max(nr, nc):
            continue
        M1 = make(nr, nc, rng, lam2=lam2) + rng.normal(0, sigma, (nr, nc))
        s = statistic(M1, o, rng)
        if s is not None:
            alt.append(s)
    if len(null) < 10 or len(alt) < 10:
        return None
    thr = float(np.quantile(null, 0.95))
    power = float(np.mean([a > thr for a in alt]))
    return {"thr": thr, "power": power,
            "n_null": len(null), "n_alt": len(alt),
            "n_obs": int(fill * nr * nc),
            "dparams": (nr + nc) - 3,
            "params_r2": n_params(nr, nc, 2)}


def main():
    quick = "--quick" in sys.argv
    rng = np.random.default_rng(20261003)
    nsim = 30 if quick else 60

    print("=" * 78)
    print("M5 单边检验功效：只排除 r=1 需要多少测点")
    print("=" * 78)
    print(f"H0: 秩1   H1: 秩2   统计量 = cv_err(r1) - cv_err(r2)")
    print(f"阈值 = H0 下的 95 分位（alpha 被校准，不是假定）；每格 {nsim}+{nsim} 次模拟")
    print(f"lambda_1 = {LAM1}  main = {MAIN_SCALE}")

    shapes = [(8, 8), (10, 10), (12, 12), (14, 14)] if quick else \
             [(6, 6), (8, 8), (10, 10), (12, 12), (14, 14), (16, 16)]
    rows = []

    for label, sigma in (("cross_mixed", SIGMA["cross_mixed"]),
                         ("same_source", SIGMA["same_source"])):
        for lam2 in (0.8, 0.5):
            print(f"\n--- sigma = {sigma} ({label}),  lambda_2 = {lam2} "
                  f"(lam2/sigma = {lam2 / sigma:.2f}) ---")
            print(f"{'shape':>8} {'fill':>5} {'测点':>5} {'Δ参数':>6} "
                  f"{'秩2参数':>7} | {'功效':>6}")
            print("-" * 56)
            for (nr, nc) in shapes:
                for fill in (0.8,):
                    r = run_cell(nr, nc, fill, sigma, lam2, nsim, rng)
                    if r is None:
                        continue
                    flag = "  <= 功效 >= 0.8" if r["power"] >= 0.8 else ""
                    print(f"{nr:>3}x{nc:<4} {fill:>5.1f} {r['n_obs']:>5} "
                          f"{r['dparams']:>6} {r['params_r2']:>7} | "
                          f"{r['power']:>6.2f}{flag}")
                    rows.append({"sigma_label": label, "sigma": sigma,
                                 "lam2": lam2, "nr": nr, "nc": nc,
                                 "fill": fill, **r})

    print("\n" + "=" * 78)
    print("对照：定秩（rank_power.py，在 r=0..4 里选）需要的测点")
    print("  20x20 填充 0.8 = 320 测点 -> r=2 定对率 1.00")
    print("  12x12 填充 0.8 = 115 测点 -> r=2 定对率 0.32")
    print("单边检验只需分辨 Δ参数 = (nr+nc)-3 个增量，不需估计整个秩2模型，")
    print("故所需测点应显著更低。上表给出具体数字。")

    out = os.path.join(ROOT, "state", "m5_onesided_power.json")
    with open(out, "w") as f:
        json.dump({"lam1": LAM1, "main_scale": MAIN_SCALE,
                   "sigma_table": SIGMA, "nsim": nsim, "rows": rows}, f,
                  indent=2)
    print(f"\n写入 {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
