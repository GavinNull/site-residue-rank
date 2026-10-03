#!/usr/bin/env python3
"""M5 的功效分析：多大的矩阵才能把秩 2 和秩 3 分开？

docs/02-pair-criteria.md 的 C6 原本写「约 5 行 × 8 列、填充率 >= 40%」，
那是拍出来的。本脚本把它变成推导出来的数字。

合成模型（M1 的双因素形式）：
    M[i,j] = a[i] + b[j] + sum_k lambda_k * phi_k[i] * psi_k[j] + eps
    eps ~ N(0, sigma_exp)

流程与真实 M5 完全一致：去主效应 -> 固定秩拟合（ALS，处理缺失）
-> 留出集误差随拟合秩的曲线 -> 取极小点作为估计秩。

报告 P(估计秩 == 真实秩)。这是「能不能定秩」的直接答案，
而不是「拟合得好不好」。

用法:
  python3 analysis/rank_power.py              # 默认网格
  python3 analysis/rank_power.py --quick
"""
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- 现实效应量级（kcal/mol）。来源与理由见下方 PROVENANCE ----
SIGMA_EXP = 0.5          # 实验 ddG 误差，docs/05 M2.3
MAIN_SCALE = 1.5         # 行/列主效应的标准差
LAMBDAS = [1.5, 0.8, 0.4]   # 耦合项奇异值，递减
# lambda_3 = 0.4 < sigma_exp = 0.5 —— 这正是难的情形，不是人为放水

PROVENANCE = """
SIGMA_EXP = 0.5: docs/05 M2.3。待 evidence/assay-noise.md 用文献值替换。
MAIN_SCALE = 1.5: 单点突变 ddG 的典型分布宽度量级。待文献确认。
LAMBDAS: 形状（递减、第三项落在噪声以下）来自 Hammett 系谱的先验
  —— 秩 1 不够、秩 2 是主要修正、秩 3 处于可察觉边缘。
  绝对值是量级估计，不是测量值。
结论对 LAMBDAS 的绝对尺度敏感，故脚本同时扫 lambda_3 / sigma 比值。
"""


def make_matrix(nr, nc, r_true, rng, lam3_over_sigma=None):
    a = rng.normal(0, MAIN_SCALE, nr)
    b = rng.normal(0, MAIN_SCALE, nc)
    M = a[:, None] + b[None, :]
    lams = list(LAMBDAS[:r_true])
    if lam3_over_sigma is not None and r_true >= 3:
        lams[2] = lam3_over_sigma * SIGMA_EXP
    for k in range(r_true):
        phi = rng.normal(0, 1, nr)
        psi = rng.normal(0, 1, nc)
        phi /= np.linalg.norm(phi)
        psi /= np.linalg.norm(psi)
        # 归一化后乘 sqrt(nr*nc) 使 lambda 保持 kcal/mol 的逐元素量级
        M += lams[k] * np.sqrt(nr * nc) * np.outer(phi, psi)
    return M


def decenter(M, mask, iters=30):
    """只用观测元素迭代去行列主效应（缺失值下无闭式解）。"""
    X = np.where(mask, M, 0.0)
    r = np.zeros(M.shape[0])
    c = np.zeros(M.shape[1])
    for _ in range(iters):
        R = X - r[:, None] - c[None, :]
        for i in range(M.shape[0]):
            if mask[i].any():
                r[i] += R[i][mask[i]].mean()
        R = X - r[:, None] - c[None, :]
        for j in range(M.shape[1]):
            if mask[:, j].any():
                c[j] += R[:, j][mask[:, j]].mean()
    return M - r[:, None] - c[None, :]


def als_complete(Y, mask, rank, rng, iters=120, ridge=1e-2):
    """固定秩的交替最小二乘，处理缺失。"""
    nr, nc = Y.shape
    if rank == 0:
        return np.zeros_like(Y)
    U = rng.normal(0, 0.1, (nr, rank))
    V = rng.normal(0, 0.1, (nc, rank))
    I = np.eye(rank) * ridge
    for _ in range(iters):
        for i in range(nr):
            m = mask[i]
            if m.sum() < 1:
                continue
            Vm = V[m]
            U[i] = np.linalg.solve(Vm.T @ Vm + I, Vm.T @ Y[i, m])
        for j in range(nc):
            m = mask[:, j]
            if m.sum() < 1:
                continue
            Um = U[m]
            V[j] = np.linalg.solve(Um.T @ Um + I, Um.T @ Y[m, j])
    return U @ V.T


def select_rank_kfold(M, obs, rng, max_rank=4, K=5, restarts=3):
    """K 折 + 重启的定秩。单次留出严重欠功效（同 320 观测：0.82 vs 1.00），
    故这是方法要求不是细节。见 docs/02 C6「方法要求」。"""
    idx = np.argwhere(obs)
    rng.shuffle(idx)
    folds = np.array_split(idx, K)
    err = np.zeros(max_rank + 1)
    for f in folds:
        trn = obs.copy()
        for i, j in f:
            trn[i, j] = False
        if trn.sum() < max(M.shape):
            return None
        Y = decenter(M, trn)
        for rk in range(max_rank + 1):
            best = np.inf
            for _ in range(restarts):
                P = als_complete(np.where(trn, Y, 0.0), trn, rk, rng, iters=200)
                e = np.sqrt(np.mean([(Y[i, j] - P[i, j]) ** 2 for i, j in f]))
                best = min(best, e)
            err[rk] += best
    return int(np.argmin(err))


def n_params(nr, nc, r):
    """双因素 + 秩 r 耦合的自由参数数。"""
    return r * (nr + nc) - r * r + nr + nc - 1


def rho(nr, nc, fill, r):
    """观测/参数比。C6 判据：r=1 vs r>=2 需 rho >= 2.8。"""
    return fill * nr * nc / n_params(nr, nc, r)


def one_trial(nr, nc, fill, r_true, rng, max_rank=4, val_frac=0.25,
              lam3_over_sigma=None):
    M0 = make_matrix(nr, nc, r_true, rng, lam3_over_sigma)
    M = M0 + rng.normal(0, SIGMA_EXP, M0.shape)
    obs = rng.random((nr, nc)) < fill
    if obs.sum() < max(nr, nc) * 2:
        return None
    # 留出集
    idx = np.argwhere(obs)
    rng.shuffle(idx)
    nval = max(3, int(len(idx) * val_frac))
    val = idx[:nval]
    trn = np.zeros_like(obs)
    for i, j in idx[nval:]:
        trn[i, j] = True
    if trn.sum() < max(nr, nc):
        return None
    Y = decenter(M, trn)
    errs = []
    for rk in range(0, max_rank + 1):
        P = als_complete(np.where(trn, Y, 0.0), trn, rk, rng)
        e = np.sqrt(np.mean([(Y[i, j] - P[i, j]) ** 2 for i, j in val]))
        errs.append(e)
    return int(np.argmin(errs)), errs


def run(grid, nsim, rng, lam3_over_sigma=None):
    out = {}
    for nr, nc, fill, r_true in grid:
        hits, picks = 0, []
        n = 0
        for _ in range(nsim):
            res = one_trial(nr, nc, fill, r_true, rng,
                            lam3_over_sigma=lam3_over_sigma)
            if res is None:
                continue
            pick, _ = res
            picks.append(pick)
            hits += (pick == r_true)
            n += 1
        if n == 0:
            continue
        out[(nr, nc, fill, r_true)] = {
            "p_correct": hits / n,
            "mean_pick": float(np.mean(picks)),
            "n": n,
        }
    return out


def main():
    quick = "--quick" in sys.argv
    rng = np.random.default_rng(20261003)
    nsim = 40 if quick else 120

    print("=" * 70)
    print("M5 功效分析：多大的矩阵能定秩")
    print("=" * 70)
    print(f"sigma_exp={SIGMA_EXP}  main={MAIN_SCALE}  lambdas={LAMBDAS}")
    print(f"每格 {nsim} 次模拟；拟合秩 0..4，取留出误差极小点")

    shapes = [(5, 8), (8, 8), (8, 12), (12, 12), (12, 20), (20, 20)]
    fills = [0.4, 0.6, 0.8] if quick else [0.3, 0.4, 0.6, 0.8, 1.0]
    grid = [(nr, nc, f, r) for (nr, nc), f, r
            in itertools.product(shapes, fills, [2, 3])]

    res = run(grid, nsim, rng)

    print(f"\n{'shape':>9} {'fill':>5} | {'r=2 定对率':>11} {'r=3 定对率':>11}"
          f" | {'r=2 均值':>9} {'r=3 均值':>9}")
    print("-" * 70)
    rows = []
    for (nr, nc) in shapes:
        for f in fills:
            k2 = (nr, nc, f, 2)
            k3 = (nr, nc, f, 3)
            if k2 not in res or k3 not in res:
                continue
            p2, p3 = res[k2]["p_correct"], res[k3]["p_correct"]
            m2, m3 = res[k2]["mean_pick"], res[k3]["mean_pick"]
            nobs = int(nr * nc * f)
            flag = ""
            if p2 >= 0.8 and p3 >= 0.8:
                flag = "  <= 可定秩"
            elif p2 >= 0.8 > p3:
                flag = "  (只能确认 r>=2)"
            print(f"{nr:>4}x{nc:<4} {f:>5.1f} | {p2:>11.2f} {p3:>11.2f}"
                  f" | {m2:>9.2f} {m3:>9.2f}{flag}")
            rows.append({"nr": nr, "nc": nc, "fill": f, "n_obs": nobs,
                         "p_r2": p2, "p_r3": p3,
                         "mean_pick_r2": m2, "mean_pick_r3": m3})

    # 灵敏度：lambda_3 / sigma 比值
    print("\n" + "=" * 70)
    print("灵敏度：第三个耦合项相对噪声的强度（固定 12x12, fill=0.6）")
    print("=" * 70)
    sens = []
    for ratio in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        r = run([(12, 12, 0.6, 3)], nsim, rng, lam3_over_sigma=ratio)
        v = list(r.values())[0]
        print(f"  lambda_3/sigma = {ratio:<4} -> r=3 定对率 {v['p_correct']:.2f}"
              f"   估计秩均值 {v['mean_pick']:.2f}")
        sens.append({"ratio": ratio, "p_correct": v["p_correct"],
                     "mean_pick": v["mean_pick"]})

    out = os.path.join(ROOT, "state", "m5_power.json")
    with open(out, "w") as f:
        json.dump({"sigma_exp": SIGMA_EXP, "main_scale": MAIN_SCALE,
                   "lambdas": LAMBDAS, "nsim": nsim,
                   "grid": rows, "sensitivity_lambda3": sens,
                   "provenance": PROVENANCE.strip()}, f, indent=2)
    print(f"\n写入 {os.path.relpath(out, ROOT)}")
    print("\n注意：结论依赖 LAMBDAS 的量级假设，那是估计不是测量。")
    print("灵敏度那一栏是为此而做——它说明结论对 lambda_3/sigma 有多敏感。")


if __name__ == "__main__":
    main()
