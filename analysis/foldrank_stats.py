#!/usr/bin/env python3
"""docs/42 的统计层（簇级校准检验、簇自助、混合比例）+ **统计自检**。

统计量在被用来读真实数据之前，必须先在已知真值上自检（CLAUDE.md §10）：
  S1 从世界 w 生成「检出秩向量」，世界 w 不应被拒（误拒率 ≈ 名义 5%）；
  S2 从世界 2 生成，世界 1 在可分辨时应被拒（命中）；
  S3 簇自助区间对已知中位数的覆盖率。
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NSIM_CELL = 40          # 与 foldrank_power.NSIM 一致（预先登记 §3）
NMC = 20000


def smooth_rows(conf_prop, n=NSIM_CELL):
    """conf_prop: (4,4) 比例 [真实k, 检出j] -> Jeffreys 平滑 (n_jk+0.5)/(n+2)。"""
    C = np.round(np.asarray(conf_prop) * n)
    return (C + 0.5) / (n + 2.0)


def logl(P_by_cluster, w, det):
    """P_by_cluster: (n_cl, 4, 4)；det: (n_cl,) 检出秩。世界 w 下的对数似然。"""
    idx = np.arange(len(det))
    return float(np.log(P_by_cluster[idx, w, det]).sum())


def world_gof(P_by_cluster, det, nmc=NMC, seed=11):
    """每个纯世界 w∈{0..3}：单尾 p_w = P(logL_sim ≤ logL_obs)。"""
    rng = np.random.default_rng(seed)
    n = len(det)
    out = {}
    for w in range(4):
        pw = P_by_cluster[:, w, :]                       # (n, 4)
        pw = pw / pw.sum(axis=1, keepdims=True)
        cdf = np.cumsum(pw, axis=1)
        u = rng.random((nmc, n))
        sim = (u[:, :, None] > cdf[None, :, :]).sum(axis=2)   # (nmc, n)
        sim = np.minimum(sim, 3)
        ll_sim = np.log(pw[np.arange(n)[None, :], sim]).sum(axis=1)
        ll_obs = float(np.log(pw[np.arange(n), det]).sum())
        out[w] = {"logL": ll_obs, "p": float((ll_sim <= ll_obs).mean()),
                  "ll_sim_med": float(np.median(ll_sim))}
    return out


def lambda12(P_by_cluster, det, nmc=NMC, seed=12):
    """对数似然比 log(L2/L1) 及其在世界 1 / 世界 2 下的 MC 分布。"""
    n = len(det)
    idx = np.arange(n)
    l1 = P_by_cluster[:, 1, :]
    l2 = P_by_cluster[:, 2, :]
    l1 = l1 / l1.sum(1, keepdims=True)
    l2 = l2 / l2.sum(1, keepdims=True)
    obs = float(np.log(l2[idx, det]).sum() - np.log(l1[idx, det]).sum())
    rng = np.random.default_rng(seed)
    res = {"obs": obs}
    for nm, src in (("under_w1", l1), ("under_w2", l2)):
        cdf = np.cumsum(src, axis=1)
        u = rng.random((nmc, n))
        sim = np.minimum((u[:, :, None] > cdf[None, :, :]).sum(axis=2), 3)
        lam = (np.log(l2[idx[None, :], sim]).sum(1)
               - np.log(l1[idx[None, :], sim]).sum(1))
        res[nm] = {"median": float(np.median(lam)),
                   "q05": float(np.percentile(lam, 5)),
                   "q95": float(np.percentile(lam, 95)),
                   "p_ge_obs": float((lam >= obs).mean())}
    return res


def mixture_em(P_by_cluster, det, iters=500):
    """最大似然世界混合比例 π（secondary）。"""
    n = len(det)
    idx = np.arange(n)
    L = np.stack([P_by_cluster[idx, w, det] for w in range(4)], axis=1)
    pi = np.ones(4) / 4
    for _ in range(iters):
        r = L * pi[None, :]
        r /= r.sum(1, keepdims=True)
        pi = r.mean(0)
    return pi


def exact_sign(n2, n1):
    """二项精确双侧 p（H0 = 0.5）。"""
    n = n1 + n2
    if n == 0:
        return float("nan")
    k = min(n1, n2)
    from scipy.stats import binom
    return float(min(1.0, 2 * binom.cdf(k, n, 0.5)))


def boot_median(x, nboot=4000, seed=5):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = np.array([np.median(x[rng.integers(0, len(x), len(x))])
                  for _ in range(nboot)])
    return {"median": float(np.median(x)),
            "ci": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
            "n": int(len(x))}


def boot_spearman(x, y, nboot=4000, seed=6):
    from scipy.stats import spearmanr
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    r0 = float(spearmanr(x, y)[0])
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(nboot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
            continue
        rs.append(float(spearmanr(x[i], y[i])[0]))
    return {"rho": r0, "ci": [float(np.percentile(rs, 2.5)),
                              float(np.percentile(rs, 97.5))], "n": len(x)}


# ----------------------------------------------------------------------
def selftest(P_by_cluster, verbose=True):
    """统计自检：已知真值上的命中与不误伤。P_by_cluster 用真实纳入簇的混淆行。"""
    P = print if verbose else (lambda *a, **k: None)
    rng = np.random.default_rng(2026)
    n = len(P_by_cluster)
    log = {}
    # S1 不误伤：世界 w 生成 -> 世界 w 误拒率
    rej = {}
    for w in range(4):
        pw = P_by_cluster[:, w, :]
        pw = pw / pw.sum(1, keepdims=True)
        cdf = np.cumsum(pw, 1)
        r = 0
        T = 100
        for t in range(T):
            u = rng.random(n)
            det = np.minimum((u[:, None] > cdf).sum(1), 3)
            g = world_gof(P_by_cluster, det, nmc=2000, seed=t)
            r += g[w]["p"] < 0.05
        rej[w] = r / T
    log["S1_false_reject"] = rej
    for w, v in rej.items():
        assert v <= 0.12, f"[S1] 世界 {w} 的误拒率 {v:.2f} 远超名义 0.05"
    P("  [S1] 世界 w 生成 → 世界 w 误拒率 " +
      " ".join(f"w{w}:{v:.2f}" for w, v in rej.items()) + "  OK（≤0.12）")
    # S2 命中：世界 2 生成 -> 世界 1 被拒的比例（功效；只记录，是否可分辨取决于混淆）
    out = {}
    for src, tgt in ((2, 1), (1, 2), (2, 0), (2, 3)):
        pw = P_by_cluster[:, src, :]
        pw = pw / pw.sum(1, keepdims=True)
        cdf = np.cumsum(pw, 1)
        r = 0
        T = 100
        for t in range(T):
            u = rng.random(n)
            det = np.minimum((u[:, None] > cdf).sum(1), 3)
            g = world_gof(P_by_cluster, det, nmc=2000, seed=1000 + t)
            r += g[tgt]["p"] < 0.05
        out[f"true{src}_reject{tgt}"] = r / T
    log["S2_power"] = out
    P("  [S2] 统计功效（真实世界→被拒世界）: " +
      " ".join(f"{k}:{v:.2f}" for k, v in out.items()))
    # S3 自助区间覆盖：已知中位数的分布
    cover = 0
    T = 200
    pop = rng.normal(0.05, 0.1, 5000)
    true_med = float(np.median(pop))
    for t in range(T):
        smp = rng.choice(pop, n, replace=True)
        ci = boot_median(smp, nboot=500, seed=t)["ci"]
        cover += ci[0] <= true_med <= ci[1]
    log["S3_boot_cover"] = cover / T
    assert 0.88 <= cover / T <= 0.995, cover / T
    P(f"  [S3] 簇自助 95% 区间对已知中位数的覆盖率 {cover / T:.2f}  OK")
    return log
