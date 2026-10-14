#!/usr/bin/env python3
"""两组 λ2 条件检验在**实际证据集几何**上的功效（CLAUDE.md §10.3 的命中方向）。

背景（为什么必须做这一步）
--------------------------
docs/24 §6.2 场景 A（冠状病毒 5 + OMTKY3 单酶 3，共 8 个矩阵）给
λ2 = 0.627、标定后条件零分布 p = 0.2787（**不超出**）。这条**非拒绝**
已被人类裁定升为「第二轴不普适」的主要证据。
docs/24 做过的只有**不误伤**方向（合成「只共享一维」给 p = 0.506，未误报）；
**命中方向在这个 k 上从未做过**。CLAUDE.md §10.3 禁止在没过还原性诊断的
情况下写下「测不出结构」。本脚本补这一侧。

五件事
------
[1] 已知事实复现闸（不过就中止）：原样重跑 docs/24 §6.2 场景 A 的构造。
[2] 合成层的量级校准（§8）：逐矩阵 shape / mask / 载荷尺度 / 估计噪声，
    合成矩阵的秩 2 交互解释率必须与真实矩阵同量级。
[3] **理想条件还原性诊断**：若家族内第二轴**完全共享**（真 ρ = 1），
    这条管线测出来的成对 cos θ2 是多少？这给出噪声地板，
    也是「观测到的 0.146 / 0.105 到底低不低」的唯一参照。
[4] 功效曲线，两个模型都报（不只报有利的那个）：
      模型 F（上界）—— 家族内第二轴完全共享（b_w = 1），家族级第二轴之间
        以已知跨家族共享度 ρ2 植入。这是检验**最有利**的情形，
        所以它给的是功效的**上界**。
      模型 H（实测）—— 家族内离散度二分法标定到观测成对 cos θ2 中位
        （冠状病毒 0.872、OMTKY3 0.146），跨家族仍为 ρ2。
    第一轴在两个模型里都标定到观测的成对 cos θ1 中位
    （冠状病毒内 0.965、OMTKY3 内 0.839、跨家族 0.884）。
[5] 实际大小（ρ2 = 0 那一格的误报率）+ docs/27 §6.2 的方向级 24 种子口径。

口径警告（必须写进文档，不能让读者自己去推）
--------------------------------------------
证据集有 8 个矩阵，但**条件零分布是在 k_eff = 2 上建的**
（`gate_consequences.py`: `keff = max(2, len(fam))`，两个家族 => 2），
而观测 λ2 是 8 个矩阵**先家族内平均再家族等权**算出来的。
docs/27 标定的 k = 9…90、docs/28 讨论的 k_eff = 3 **都不覆盖这一格**。
本脚本的「实际大小」是在这个**具体操作点**上测的。

数值层全部复用、没有重写：`subspace_lib`（extract_subspace / align /
dc_fast）、`subspace_cond_null`（proj / cond_null / _draw）、
`gate_calib_lib`（load_evidence / rankr_expl）、
`gate_consequences` 的同构造（_fam_avg / fam_avg_null）。
**全程未 import `als_complete`**（未保护版曾产生 −337% / −1091% 的伪增益）。
"""
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_calib_lib as G  # noqa: E402
import subspace_cond_null as CN  # noqa: E402
from subspace_lib import (ROOT, align, dc_fast,  # noqa: E402
                          extract_subspace)

NSIM = 4000
ONE1 = CN.ONE[:, 0]

COV = ["starr2020/WH1/huACE2", "animalACE2/Wuhan-Hu-1/bat",
       "PDCoV/PDCoV/gAPN",
       "sarbecovirus/SARS-CoV-1_Urban/Ra9479ACE2",
       "animalACE2/Omicron_BA2/human_monomer"]
OMT3 = ["OMTKY3/chymotrypsin/1CHO", "OMTKY3/HLE-elastase/1PPF",
        "OMTKY3/subtilisin-Carlsberg/1R0R"]
ORDER = COV + OMT3

PUB = {"lam1": 0.839, "lam2": 0.627, "lam3": 0.305,
       "null_p95_1": 0.423, "null_p95_2": 0.347,
       "p_lambda2": 0.2787, "cal_resid": 0.0035,
       "neg_lam2": 0.584, "neg_p": 0.506,
       "loo_abscos_min": 0.911, "loo_abscos_med": 0.999}
TOL, TOL_P = 0.004, 0.010

SEARCH = {"cells": 0, "replicates": 0, "null_draws": 0, "extractions": 0,
          "calib_bisections": 0}


def log(k, n):
    SEARCH[k] = SEARCH.get(k, 0) + n


def require_harness():
    p = os.path.join(ROOT, "cache", "gate_harness_check.json")
    if not os.path.exists(p):
        raise RuntimeError("cache/gate_harness_check.json 不存在")
    if json.load(open(p)).get("ok") is not True:
        raise RuntimeError("gate_harness_check 未通过，拒绝继续（§10）")
    print("[前置] cache/gate_harness_check.json ok = True", flush=True)


# ----------------------------------------------------------------------
# 构造层：与 gate_consequences.py 同构（先家族内平均、再家族等权）
# ----------------------------------------------------------------------
def _fam_avg(fam_Bs):
    Ps = [np.mean([CN.proj(B)[0] for B in v], axis=0) for v in fam_Bs if v]
    w, V = np.linalg.eigh(np.mean(Ps, axis=0))
    return w[::-1], V[:, ::-1]


def fam_avg_null(sizes, rng, nsim=NSIM):
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


def lambda2_test(fam_Bs, nsim=NSIM, seed=4):
    """与 gate_consequences 逐字同口径的两组 λ2 条件检验。k_eff = 家族数。"""
    w, V = _fam_avg(fam_Bs)
    u = V[:, 0]
    keff = max(2, len(fam_Bs))
    cnd, alf = CN.cond_null(keff, u, np.random.default_rng(seed),
                            nsim=nsim, target_l1=float(w[0]))
    log("null_draws", nsim + 28 * 240)
    resid = abs(float(np.median(cnd[:, 0])) - float(w[0]))
    p2 = float(np.mean(cnd[:, 1] >= w[1]))
    return w, u, p2, resid, float(np.degrees(alf))


# ----------------------------------------------------------------------
# [1] 已知事实复现闸
# ----------------------------------------------------------------------
def stage1(by):
    print("\n" + "=" * 96)
    print("[1] 已知事实复现闸：docs/24 §6.2 场景 A（冠状病毒 5 / OMTKY3 单酶 3）")
    print("=" * 96, flush=True)
    fam = [[by[n] for n in COV], [by[n] for n in OMT3]]
    rng = np.random.default_rng(20261012)   # 与 gate_consequences 同种子次序
    w, V = _fam_avg(fam)
    u = V[:, 0]
    p95 = np.percentile(fam_avg_null([5, 3], rng), 95, axis=0)
    _, _, p2, resid, alf = lambda2_test(fam)
    rng2 = np.random.default_rng(99)
    al = CN.cond_null(2, u, rng2, nsim=10, target_l1=float(w[0]))[1]
    synth = float(np.median([CN._draw(2, u, al, rng2)[1] for _ in range(400)]))
    cn2, _ = CN.cond_null(2, u, rng2, nsim=1500, target_l1=float(w[0]))
    pneg = float(np.mean(cn2[:, 1] >= synth))
    loo = []
    for fi in (0, 1):
        for j in range(len(fam[fi])):
            f2 = [list(fam[0]), list(fam[1])]
            f2[fi] = [b for q, b in enumerate(fam[fi]) if q != j]
            loo.append(abs(float(u @ _fam_avg(f2)[1][:, 0])))
    got = {"lam1": float(w[0]), "lam2": float(w[1]), "lam3": float(w[2]),
           "null_p95_1": float(p95[0]), "null_p95_2": float(p95[1]),
           "p_lambda2": p2, "cal_resid": resid, "neg_lam2": synth,
           "neg_p": pneg, "loo_abscos_min": float(min(loo)),
           "loo_abscos_med": float(np.median(loo))}
    print(f"{'量':<22}{'本次':>10}{'docs/24 §6.2':>14}{'差':>9}  判定")
    ok = True
    for k, tgt in PUB.items():
        tol = TOL_P if k in ("p_lambda2", "neg_p") else TOL
        d = abs(got[k] - tgt)
        ok &= d <= tol
        print(f"{k:<22}{got[k]:>10.4f}{tgt:>14.4f}{d:>9.4f}  "
              f"{'OK' if d <= tol else '<< 不符'}")
    print(f"条件零分布抖动角 alpha = {alf:.1f}°（k_eff = 2）")
    print(f"负对照间距比 观测λ2/负对照λ2 = {got['lam2']/got['neg_lam2']:.3f}×")
    if not ok:
        raise RuntimeError("已知事实复现未通过 —— 按 §10 中止")
    print("\n[1] 通过：原构造逐位复现，下面的数字与 docs/24 可直接相比。",
          flush=True)
    return got, u


# ----------------------------------------------------------------------
# [2] 合成层
# ----------------------------------------------------------------------
def fit_geometry(by_mat):
    """逐矩阵：真实 shape / mask / 秩 2 载荷尺度 (s1,s2) / 残差噪声 sigma。"""
    out = {}
    for nm, d in by_mat.items():
        M = np.asarray(d["M"], float)
        mask = np.asarray(d["mask"], bool)
        B, cols, _ = extract_subspace(M, mask)
        keep_r = mask.sum(axis=1) >= 4
        Ms, ms = M[np.ix_(keep_r, cols)], mask[np.ix_(keep_r, cols)]
        R = dc_fast(np.where(ms, Ms, 0.0), ms)
        A = np.zeros((R.shape[0], 2))
        for i in range(R.shape[0]):
            o = ms[i]
            if o.sum() >= 4:
                A[i] = np.linalg.lstsq(B[o], R[i, o], rcond=None)[0]
        resid = (R - A @ B.T)[ms]
        out[nm] = {"shape": (int(R.shape[0]), int(R.shape[1])),
                   "mask": ms.copy(),
                   "s": [float(A[:, 0].std()), float(A[:, 1].std())],
                   "sigma": float(np.sqrt(np.mean(resid ** 2)))}
    return out


def rand_perp(rng, *vs):
    a = rng.normal(size=20)
    a -= ONE1 * (ONE1 @ a)
    for v in vs:
        a -= v * (v @ a)
    return a / np.linalg.norm(a)


def mix(base, rho, rng, *perp):
    """rho*base + sqrt(1-rho^2)*(与 base 及 perp 都正交的随机方向)。"""
    if rho >= 1.0:
        return base
    w = rand_perp(rng, base, *perp)
    v = rho * base + np.sqrt(max(0.0, 1 - rho ** 2)) * w
    return v / np.linalg.norm(v)


def synth_set(geo, par, rng):
    """合成一个 8 成员证据集（层级：全局 -> 家族 -> 矩阵）。

    par: a_c（第一轴跨家族）、a_w（dict，第一轴家族内）、
         b_c（第二轴跨家族 = ρ2）、b_w（dict，第二轴家族内）
    返回 (按家族分组的 B 列表, 名字->B 的 dict)；抽取失败返回 None。
    """
    u = rand_perp(rng)
    v = rand_perp(rng, u)
    fam_Bs, named = [], {}
    for fam_name, members in (("cov", COV), ("omt", OMT3)):
        uf = mix(u, par["a_c"], rng, v)
        vf = mix(v, par["b_c"], rng, u)
        Bs = []
        for nm in members:
            g = geo[nm]
            nr, nc = g["shape"]
            ui = mix(uf, par["a_w"][fam_name], rng)
            vi = mix(vf, par["b_w"][fam_name], rng, ui)
            vi = vi - ui * (ui @ vi)
            vi /= np.linalg.norm(vi)
            S = np.stack([ui, vi], 1)
            A = rng.normal(size=(nr, 2)) * np.array(g["s"])
            M = A @ S.T + rng.normal(scale=g["sigma"], size=(nr, nc))
            B, c, _ = extract_subspace(M, g["mask"])
            log("extractions", 1)
            if B is None or len(c) != nc:
                return None, None
            Bs.append(B)
            named[nm] = (B, c)
        fam_Bs.append(Bs)
    return fam_Bs, named


def pair_meds(named):
    """合成集上的成对 cos θ1 / cos θ2 中位（与观测同一函数 align）。"""
    def m(pairs):
        c1, c2 = [], []
        for a, b in pairs:
            Ba, ca = named[a]
            Bb, cb = named[b]
            r = align(Ba, ca, Bb, cb)
            if r:
                c1.append(r["cos"][0])
                c2.append(r["cos"][1])
        return (float(np.median(c1)) if c1 else np.nan,
                float(np.median(c2)) if c2 else np.nan)
    return {"cov": m(list(itertools.combinations(COV, 2))),
            "omt": m(list(itertools.combinations(OMT3, 2))),
            "cross": m([(a, b) for a in COV for b in OMT3])}


def measure(geo, par, rng, nrep):
    """nrep 次合成，返回成对中位数的中位数 + λ 的中位。"""
    acc = {"cov": [[], []], "omt": [[], []], "cross": [[], []]}
    lams = []
    for _ in range(nrep):
        fb, named = synth_set(geo, par, rng)
        if fb is None:
            continue
        pm = pair_meds(named)
        for k in acc:
            acc[k][0].append(pm[k][0])
            acc[k][1].append(pm[k][1])
        lams.append(_fam_avg(fb)[0][:3])
    return ({k: (float(np.median(v[0])), float(np.median(v[1])))
             for k, v in acc.items()},
            np.median(np.array(lams), axis=0) if lams else None)


def bisect_param(geo, par, key, sub, target, which, rng, nrep=20, iters=11):
    """把 par[key][sub]（或 par[key]）调到使合成的某个成对中位 = 观测值。

    which: ('cov'|'omt'|'cross', 0|1)  —— 0 = cos θ1，1 = cos θ2

    **为什么标定必须有次序（第一版在这里错了）**：`align` 返回的是两个 2 维
    子空间的**两个主角余弦**（cos θ1 ≥ cos θ2），不是「第一轴对第一轴」与
    「第二轴对第二轴」。所以 cos θ1 由两条方向里**更共享**的那一条决定。
    若先按 cos θ1 标第一轴而把第二轴留在 b_w = 1，则 cos θ1 ≡ 1，
    与第一轴参数无关，二分法会把第一轴参数推到 0 —— 第一版正是如此
    （a_w(cov) 收敛到 0.000，λ1 掉到 0.55，与观测 0.839 完全不符）。
    正确次序：**先用 cos θ2 标更弱的那一条（第二轴），再用 cos θ1 标第一轴**，
    并交替一轮以吸收耦合。
    """
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if sub is None:
            par[key] = mid
        else:
            par[key][sub] = mid
        pm, _ = measure(geo, par, rng, nrep)
        val = pm[which[0]][which[1]]
        if val < target:
            lo = mid
        else:
            hi = mid
    val = 0.5 * (lo + hi)
    if sub is None:
        par[key] = val
    else:
        par[key][sub] = val
    log("calib_bisections", 1)
    return val


# ----------------------------------------------------------------------
# 功效曲线
# ----------------------------------------------------------------------
def power_curve(geo, par0, rhos, nrep, obs, label, seed):
    print(f"\n   --- {label} ---", flush=True)
    rows = []
    for rho2 in rhos:
        par = {"a_c": par0["a_c"], "a_w": dict(par0["a_w"]),
               "b_w": dict(par0["b_w"]), "b_c": rho2}
        rng = np.random.default_rng(seed + int(round(rho2 * 1000)))
        t0 = time.time()
        ps, l1s, l2s, xc2 = [], [], [], []
        ndisc = 0
        for _ in range(nrep):
            fb, named = synth_set(geo, par, rng)
            if fb is None:
                ndisc += 1
                continue
            w, _, p2, resid, _ = lambda2_test(fb, seed=int(rng.integers(1e9)))
            if resid >= 0.02:
                ndisc += 1       # 该次条件零分布标定失败，按 docs/24 口径弃用
                continue
            ps.append(p2)
            l1s.append(float(w[0]))
            l2s.append(float(w[1]))
            xc2.append(pair_meds(named)["cross"][1])
        if not ps:
            raise RuntimeError(f"ρ2 = {rho2} 全部重复被弃用（{ndisc} 次）"
                               f"—— 该格不可用")
        ps = np.array(ps)
        rec = {"rho2": rho2, "n": int(len(ps)), "n_discard": ndisc,
               "power05": float(np.mean(ps < 0.05)),
               "p_med": float(np.median(ps)),
               "p_q": [float(np.percentile(ps, 2.5)),
                       float(np.percentile(ps, 97.5))],
               "lam1_med": float(np.median(l1s)),
               "lam2_med": float(np.median(l2s)),
               "lam2_q": [float(np.percentile(l2s, 2.5)),
                          float(np.percentile(l2s, 97.5))],
               "cross_cos2_med": float(np.median(xc2)),
               "secs": time.time() - t0}
        rows.append(rec)
        log("cells", 1)
        log("replicates", len(ps))
        print(f"   ρ2 = {rho2:.1f}  n = {rec['n']:>3}  λ1中位 "
              f"{rec['lam1_med']:.3f}  λ2中位 {rec['lam2_med']:.3f} "
              f"[{rec['lam2_q'][0]:.3f}, {rec['lam2_q'][1]:.3f}]  "
              f"跨家族cos2中位 {rec['cross_cos2_med']:.3f}  "
              f"p中位 {rec['p_med']:.4f}  **检出率 {rec['power05']:.3f}**  "
              f"({rec['secs']:.0f}s)", flush=True)
    return rows


def interp_80(rows):
    xs = [r["rho2"] for r in rows]
    ps = [r["power05"] for r in rows]
    for i in range(len(xs) - 1):
        if ps[i] < 0.8 <= ps[i + 1]:
            t = (0.8 - ps[i]) / (ps[i + 1] - ps[i])
            return xs[i] + t * (xs[i + 1] - xs[i])
    return None


def compat_bound(rows, obs):
    ok_p = [r["rho2"] for r in rows
            if r["p_q"][0] <= obs["p_lambda2"] <= r["p_q"][1]]
    ok_l = [r["rho2"] for r in rows
            if r["lam2_q"][0] <= obs["lam2"] <= r["lam2_q"][1]]
    return (max(ok_p) if ok_p else None), (max(ok_l) if ok_l else None)


# ----------------------------------------------------------------------
# 方向级 24 种子实际大小（docs/27 §6.2 口径）
# ----------------------------------------------------------------------
def size_dirlevel(u, target_l1, nseed=24):
    ps, l2s = [], []
    for s in range(nseed):
        rng = np.random.default_rng(5000 + s)
        al = CN.cond_null(2, u, rng, nsim=10, target_l1=target_l1)[1]
        obs2 = float(np.median([CN._draw(2, u, al, rng)[1]
                                for _ in range(400)]))
        cn, _ = CN.cond_null(2, u, rng, nsim=1500, target_l1=target_l1)
        log("null_draws", 1500 + 400 + 2 * 28 * 240)
        ps.append(float(np.mean(cn[:, 1] >= obs2)))
        l2s.append(obs2)
    ps = np.array(ps)
    return {"n_seed": nseed, "lam2_med": float(np.median(l2s)),
            "p_med": float(np.median(ps)),
            "p_q": [float(np.percentile(ps, 10)),
                    float(np.percentile(ps, 90))],
            "size": float(np.mean(ps < 0.05))}


SEEDS = {"F": 1001, "H": 2002, "H2": 3003}
CALIB_CACHE = "lambda2_power_calib.json"


def build_calib(geo, OBS_PM, verbose=True):
    """标定三个模型的参数。结果缓存到 cache/lambda2_power_calib.json。

    缓存只是为了让 30 个格子能并行跑而不必各自重标一遍 ——
    **并行与串行给出逐位相同的数**，因为每个格子的 rng 只由
    (模型种子, ρ2) 决定（见 `power_curve`）。
    """
    p = os.path.join(ROOT, "cache", CALIB_CACHE)
    if os.path.exists(p):
        d = json.load(open(p))
        if verbose:
            print(f"   [标定复用] {os.path.relpath(p, ROOT)}")
        return d["par_F"], d["par_H"], d["par_H2"]
    par_H = {"a_c": 1.0, "a_w": {"cov": 1.0, "omt": 1.0},
             "b_c": 0.0, "b_w": {"cov": 1.0, "omt": 1.0}}
    rngc = np.random.default_rng(4242)
    for it in range(2):
        for fk in ("cov", "omt"):
            bisect_param(geo, par_H, "b_w", fk, OBS_PM[fk][1], (fk, 1), rngc)
            bisect_param(geo, par_H, "a_w", fk, OBS_PM[fk][0], (fk, 0), rngc)
        if verbose:
            print(f"   轮 {it+1}: b_w = {par_H['b_w']}、a_w = {par_H['a_w']}",
                  flush=True)
    bisect_param(geo, par_H, "a_c", None, OBS_PM["cross"][0], ("cross", 0),
                 rngc)
    par_F = {"a_c": par_H["a_c"], "a_w": dict(par_H["a_w"]), "b_c": 0.0,
             "b_w": {"cov": 1.0, "omt": 1.0}}
    par_H2 = {"a_c": par_H["a_c"], "a_w": dict(par_H["a_w"]), "b_c": 0.0,
              "b_w": {"cov": par_H["b_w"]["cov"], "omt": 0.38}}
    json.dump({"par_F": par_F, "par_H": par_H, "par_H2": par_H2},
              open(p, "w"), ensure_ascii=False, indent=1, default=float)
    return par_F, par_H, par_H2


def _prep():
    """复现闸 + 几何 + 标定，单格模式与汇总模式共用。"""
    require_harness()
    mats = G.load_evidence()
    by, by_mat = {}, {}
    for nm in ORDER:
        d = mats[nm]
        B, c, _ = extract_subspace(d["M"], d["mask"])
        assert B is not None and len(c) == 20, nm
        by[nm] = B
        by_mat[nm] = d
    cb = {}
    for nm in ORDER:
        _, c, _ = extract_subspace(by_mat[nm]["M"], by_mat[nm]["mask"])
        cb[nm] = c
    OBS_PM = pair_meds({nm: (by[nm], cb[nm]) for nm in ORDER})
    geo = fit_geometry(by_mat)
    return by, by_mat, OBS_PM, geo


def run_cell(model, rho2, nrep):
    """只跑一个 (模型, ρ2) 格子，写 cache/cells/。与串行逐位相同。"""
    by, by_mat, OBS_PM, geo = _prep()
    pF, pH, pH2 = build_calib(geo, OBS_PM)
    par = {"F": pF, "H": pH, "H2": pH2}[model]
    w, V = _fam_avg([[by[n] for n in COV], [by[n] for n in OMT3]])
    obs = {"lam1": float(w[0]), "lam2": float(w[1]),
           "p_lambda2": PUB["p_lambda2"]}
    rows = power_curve(geo, par, [rho2], nrep, obs,
                       f"模型 {model}（单格 ρ2 = {rho2}）", SEEDS[model])
    d = os.path.join(ROOT, "cache", "cells")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"lambda2_cell_{model}_{rho2:.1f}.json")
    json.dump({"model": model, "rho2": rho2, "nrep": nrep, "nsim": NSIM,
               "par": par, "row": rows[0], "search": SEARCH},
              open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print("产物:", p)
    return 0


def main():
    require_harness()
    t0 = time.time()
    mats = G.load_evidence()
    by, by_mat = {}, {}
    for nm in ORDER:
        d = mats[nm]
        B, c, _ = extract_subspace(d["M"], d["mask"])
        assert B is not None and len(c) == 20, nm
        by[nm] = B
        by_mat[nm] = d
        log("extractions", 1)

    obs, u_obs = stage1(by)

    # 观测的成对中位（标定目标；与 docs/28 §7 的表同口径）
    cb = {}
    for nm in ORDER:
        _, c, _ = extract_subspace(by_mat[nm]["M"], by_mat[nm]["mask"])
        cb[nm] = c
    named_obs = {nm: (by[nm], cb[nm]) for nm in ORDER}
    OBS_PM = pair_meds(named_obs)
    print("\n观测的成对中位（标定目标）：")
    for k, v in OBS_PM.items():
        print(f"   {k:<6} cos θ1 中位 {v[0]:.3f}   cos θ2 中位 {v[1]:.3f}")

    # ---- [2] 量级校准 ----
    print("\n" + "=" * 96)
    print("[2] 合成层的量级校准（§8）")
    print("=" * 96, flush=True)
    geo = fit_geometry(by_mat)
    cal = json.load(open(os.path.join(ROOT, "cache",
                                      "gate_rank2_calibration.json")))
    expl_real = {r["name"]: r["expl_rank2"] for r in cal["rows"]}
    print(f"{'成员':<44}{'shape':>10}{'s1':>7}{'s2':>7}{'sigma':>8}"
          f"{'expl(真)':>10}{'expl(合成)':>11}")
    magcal, rngm = [], np.random.default_rng(31337)
    for nm in ORDER:
        g = geo[nm]
        ee = []
        for _ in range(8):
            nr, nc = g["shape"]
            uu = rand_perp(rngm)
            vv = rand_perp(rngm, uu)
            A = rngm.normal(size=(nr, 2)) * np.array(g["s"])
            M = A @ np.stack([uu, vv], 1).T \
                + rngm.normal(scale=g["sigma"], size=(nr, nc))
            x, _ = G.rankr_expl(M, g["mask"], rank=2)
            if x is not None:
                ee.append(x)
        es = float(np.median(ee))
        print(f"{nm[:43]:<44}{str(g['shape']):>10}{g['s'][0]:>7.2f}"
              f"{g['s'][1]:>7.2f}{g['sigma']:>8.3f}"
              f"{expl_real[nm]:>10.3f}{es:>11.3f}")
        magcal.append({"name": nm, "shape": g["shape"], "s": g["s"],
                       "sigma": g["sigma"], "expl_real": expl_real[nm],
                       "expl_synth": es})
    dmax = max(abs(z["expl_synth"] - z["expl_real"]) for z in magcal)
    print(f"   秩 2 交互解释率最大偏差 = {dmax:.3f}  "
          f"{'OK（同量级）' if dmax < 0.12 else '<< 量级对不上'}", flush=True)
    if dmax >= 0.12:
        raise RuntimeError("合成层与真实矩阵不同量级（§8）—— 数字不可用")

    # ---- [3] 理想条件还原性诊断：完全共享下的噪声地板 ----
    print("\n" + "=" * 96)
    print("[3] 理想条件还原性诊断（§10.3）：真 ρ = 1 时这条管线测出多少")
    print("=" * 96, flush=True)
    par_ideal = {"a_c": 1.0, "a_w": {"cov": 1.0, "omt": 1.0},
                 "b_c": 1.0, "b_w": {"cov": 1.0, "omt": 1.0}}
    pm_id, lam_id = measure(geo, par_ideal, np.random.default_rng(11), 60)
    print(f"   完全共享（全部 ρ = 1，60 次）下的成对中位：")
    for k in ("cov", "omt", "cross"):
        print(f"      {k:<6} cos θ1 {pm_id[k][0]:.3f}（观测 "
              f"{OBS_PM[k][0]:.3f}）   cos θ2 {pm_id[k][1]:.3f}（观测 "
              f"{OBS_PM[k][1]:.3f}）")
    print(f"   完全共享下的 λ 中位 = {np.round(lam_id, 3).tolist()}"
          f"（观测 [{obs['lam1']:.3f}, {obs['lam2']:.3f}, "
          f"{obs['lam3']:.3f}]）")
    floor_ok = pm_id["omt"][1] > OBS_PM["omt"][1]
    print(f"   判读：完全共享时 OMTKY3 家族内 cos θ2 地板 = "
          f"{pm_id['omt'][1]:.3f}，观测 {OBS_PM['omt'][1]:.3f} "
          f"{'**低于地板 —— 该家族的第二轴确实彼此不同**' if floor_ok else '在地板之内'}",
          flush=True)

    # ---- [4] 标定两个模型的参数 ----
    print("\n" + "=" * 96)
    print("[4] 参数标定（第一轴标到观测成对 cos θ1；第二轴家族内按模型）")
    print("=" * 96, flush=True)
    par_H = {"a_c": 1.0, "a_w": {"cov": 1.0, "omt": 1.0},
             "b_c": 0.0, "b_w": {"cov": 1.0, "omt": 1.0}}
    rngc = np.random.default_rng(4242)
    for it in range(2):
        for fk in ("cov", "omt"):
            bisect_param(geo, par_H, "b_w", fk, OBS_PM[fk][1], (fk, 1), rngc)
            bisect_param(geo, par_H, "a_w", fk, OBS_PM[fk][0], (fk, 0), rngc)
        print(f"   轮 {it+1}: b_w = {par_H['b_w']}、a_w = {par_H['a_w']}",
              flush=True)
    bisect_param(geo, par_H, "a_c", None, OBS_PM["cross"][0], ("cross", 0),
                 rngc)
    print(f"   模型 H 标定结果：a_w = "
          f"{ {k: round(v,3) for k,v in par_H['a_w'].items()} }、b_w = "
          f"{ {k: round(v,3) for k,v in par_H['b_w'].items()} }、"
          f"a_c = {par_H['a_c']:.3f}")
    # 模型 F：家族内第二轴完全共享（b_w = 1），第一轴沿用 H 的标定值，
    # a_c 在 b_w = 1 下重标（因为 cos θ1 的归属会变）。
    par_F = {"a_c": par_H["a_c"], "a_w": dict(par_H["a_w"]), "b_c": 0.0,
             "b_w": {"cov": 1.0, "omt": 1.0}}
    print(f"   模型 F：b_w = 1.000（家族级第二轴完全锐利 —— 功效上界）；"
          f"其余沿用 H")
    # 模型 H2：b_w(omt) 的敏感性。交替标定在 0.03–0.38 之间**来回摆**
    # （两轮分别收到 0.376 与 0.033），而两个值给出的**可观测量**
    # （OMTKY3 家族内 cos θ2 中位）都对得上观测 0.146 —— 即该参数在这个
    # 区间内弱可识别。所以必须把区间另一端也跑出来，不能只报一个。
    par_H2 = {"a_c": par_H["a_c"], "a_w": dict(par_H["a_w"]), "b_c": 0.0,
              "b_w": {"cov": par_H["b_w"]["cov"], "omt": 0.38}}
    print(f"   模型 H2（敏感性）：b_w(omt) = 0.380（交替标定的另一端）")
    for nm_, p_ in (("F", par_F), ("H", par_H), ("H2", par_H2)):
        pm, lam = measure(geo, p_, np.random.default_rng(777), 40)
        print(f"   [模型 {nm_} 在 ρ2 = 0 的标定核对] "
              f"cov cos1/cos2 {pm['cov'][0]:.3f}/{pm['cov'][1]:.3f}、"
              f"omt {pm['omt'][0]:.3f}/{pm['omt'][1]:.3f}、"
              f"cross {pm['cross'][0]:.3f}/{pm['cross'][1]:.3f}；"
              f"λ = {np.round(lam, 3).tolist()}", flush=True)

    # ---- [5] 功效曲线 ----
    rhos = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    nrep = int(os.environ.get("NREP", "200"))
    print("\n" + "=" * 96)
    print(f"[5] 功效曲线（每格 {nrep} 次；λ2 条件零分布每次 {NSIM} 次，"
          f"k_eff = 2，逐次标定到该次自己的 λ1）")
    print("=" * 96, flush=True)
    rows_F = power_curve(geo, par_F, rhos, nrep, obs,
                         "模型 F：家族内第二轴完全共享（功效上界）", 1001)
    rows_H = power_curve(geo, par_H, rhos, nrep, obs,
                         "模型 H：家族内离散度标定到观测（b_w(omt) 低端）", 2002)
    rows_H2 = power_curve(geo, par_H2, rhos, nrep, obs,
                          "模型 H2：同上但 b_w(omt) = 0.38（敏感性）", 3003)

    dl = size_dirlevel(u_obs, obs["lam1"])
    out = {"obs": obs, "pub": PUB, "obs_pair_med": OBS_PM, "magcal": magcal,
           "ideal": {"pair_med": pm_id, "lam": np.asarray(lam_id).tolist()},
           "par_F": par_F, "par_H": par_H, "par_H2": par_H2,
           "nrep": nrep, "nsim": NSIM,
           "rows_F": rows_F, "rows_H": rows_H, "rows_H2": rows_H2,
           "size_dirlevel_keff2": dl,
           "neg_gap_ratio": obs["lam2"] / obs["neg_lam2"],
           "search": SEARCH}
    for nm_, rows in (("F", rows_F), ("H", rows_H), ("H2", rows_H2)):
        r80 = interp_80(rows)
        bp, bl = compat_bound(rows, obs)
        z = [r for r in rows if r["rho2"] == 0.0][0]
        se = np.sqrt(max(z["power05"] * (1 - z["power05"]), 1e-9)
                     / max(1, z["n"]))
        out[f"summary_{nm_}"] = {
            "rho2_80": r80, "compat_max_rho2_p": bp,
            "compat_max_rho2_lam2": bl, "size": z["power05"],
            "size_ci": [max(0.0, z["power05"] - 1.96 * se),
                        z["power05"] + 1.96 * se]}
        print(f"\n[模型 {nm_}] 80% 功效所需最小 ρ2 = "
              + (f"{r80:.2f}" if r80 is not None else "（全区间达不到）")
              + f"；与观测 p 相容的最大 ρ2 = "
              + (f"{bp:.1f}" if bp is not None else "无")
              + f"；与观测 λ2 相容的最大 ρ2 = "
              + (f"{bl:.1f}" if bl is not None else "无")
              + f"；ρ2 = 0 的实际大小 = {z['power05']:.3f} "
                f"[{out[f'summary_{nm_}']['size_ci'][0]:.3f}, "
                f"{out[f'summary_{nm_}']['size_ci'][1]:.3f}]")
    print(f"\n[方向级 24 种子，k_eff = 2，λ1 标到 {obs['lam1']:.3f}]"
          f" 合成 λ2 中位 {dl['lam2_med']:.3f}、p 中位 {dl['p_med']:.3f}、"
          f"10%–90% [{dl['p_q'][0]:.3f}, {dl['p_q'][1]:.3f}]、"
          f"**实际大小 {dl['size']:.3f}**（docs/27 在 k = 9…90 上是 0.08–0.12）")
    print(f"负对照间距比 = {obs['lam2']/obs['neg_lam2']:.3f}×"
          f"（docs/27 折叠一侧 4.3–6.5×）")
    print(f"搜索量（§5）：{SEARCH}")
    print(f"总耗时 {time.time()-t0:.0f}s")
    p = os.path.join(ROOT, "cache", "lambda2_power.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print("产物:", p)
    return 0


def merge(rhos, nrep):
    """把 cache/cells/ 的格子汇总成 cache/lambda2_power.json 并打判词数字。"""
    by, by_mat, OBS_PM, geo = _prep()
    pF, pH, pH2 = build_calib(geo, OBS_PM)
    w, V = _fam_avg([[by[n] for n in COV], [by[n] for n in OMT3]])
    u_obs = V[:, 0]
    obs, _ = stage1(by)
    d = os.path.join(ROOT, "cache", "cells")
    out = {"obs": obs, "pub": PUB, "obs_pair_med": OBS_PM,
           "par_F": pF, "par_H": pH, "par_H2": pH2,
           "nrep": nrep, "nsim": NSIM}
    for model in ("F", "H", "H2"):
        rows = []
        for r in rhos:
            p = os.path.join(d, f"lambda2_cell_{model}_{r:.1f}.json")
            if not os.path.exists(p):
                raise RuntimeError(f"缺格子 {p}")
            rows.append(json.load(open(p))["row"])
        out[f"rows_{model}"] = rows
    print("\n" + "=" * 96)
    print("[5] 功效曲线（汇总 cache/cells/）")
    print("=" * 96)
    for model in ("F", "H", "H2"):
        print(f"\n   --- 模型 {model} ---")
        for r in out[f"rows_{model}"]:
            se = np.sqrt(max(r["power05"] * (1 - r["power05"]), 1e-9) / r["n"])
            print(f"   ρ2 = {r['rho2']:.1f}  n = {r['n']:>3}  λ1中位 "
                  f"{r['lam1_med']:.3f}  λ2中位 {r['lam2_med']:.3f} "
                  f"[{r['lam2_q'][0]:.3f}, {r['lam2_q'][1]:.3f}]  "
                  f"跨家族cos2中位 {r['cross_cos2_med']:.3f}  "
                  f"p中位 {r['p_med']:.4f}  **检出率 {r['power05']:.3f}** "
                  f"[{max(0.0, r['power05']-1.96*se):.3f}, "
                  f"{r['power05']+1.96*se:.3f}]")
    dl = size_dirlevel(u_obs, obs["lam1"])
    out["size_dirlevel_keff2"] = dl
    out["neg_gap_ratio"] = obs["lam2"] / obs["neg_lam2"]
    for nm_ in ("F", "H", "H2"):
        rows = out[f"rows_{nm_}"]
        r80 = interp_80(rows)
        bp, bl = compat_bound(rows, obs)
        z = [r for r in rows if r["rho2"] == 0.0][0]
        se = np.sqrt(max(z["power05"] * (1 - z["power05"]), 1e-9) / z["n"])
        out[f"summary_{nm_}"] = {
            "rho2_80": r80, "compat_max_rho2_p": bp,
            "compat_max_rho2_lam2": bl, "size": z["power05"],
            "size_ci": [max(0.0, z["power05"] - 1.96 * se),
                        z["power05"] + 1.96 * se]}
        print(f"\n[模型 {nm_}] 80% 功效所需最小 ρ2 = "
              + (f"{r80:.2f}" if r80 is not None else "（全区间达不到）")
              + "；与观测 p 相容的最大 ρ2 = "
              + (f"{bp:.1f}" if bp is not None else "无")
              + "；与观测 λ2 相容的最大 ρ2 = "
              + (f"{bl:.1f}" if bl is not None else "无")
              + f"；ρ2 = 0 的实际大小 = {z['power05']:.3f} "
                f"[{out[f'summary_{nm_}']['size_ci'][0]:.3f}, "
                f"{out[f'summary_{nm_}']['size_ci'][1]:.3f}]")
    print(f"\n[方向级 24 种子，k_eff = 2，λ1 标到 {obs['lam1']:.3f}]"
          f" 合成 λ2 中位 {dl['lam2_med']:.3f}、p 中位 {dl['p_med']:.3f}、"
          f"10%–90% [{dl['p_q'][0]:.3f}, {dl['p_q'][1]:.3f}]、"
          f"**实际大小 {dl['size']:.3f}**（docs/27 在 k = 9…90 上是 0.08–0.12）")
    print(f"负对照间距比 = {obs['lam2']/obs['neg_lam2']:.3f}×"
          f"（docs/27 折叠一侧 4.3–6.5×）")
    tot = {"cells": 0, "replicates": 0, "null_draws": 0, "extractions": 0}
    for model in ("F", "H", "H2"):
        for r in rhos:
            s = json.load(open(os.path.join(
                d, f"lambda2_cell_{model}_{r:.1f}.json")))["search"]
            for k in tot:
                tot[k] += s.get(k, 0)
    out["search"] = tot
    print(f"搜索量（§5，30 个格子合计）：{tot}")
    p = os.path.join(ROOT, "cache", "lambda2_power.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print("产物:", p)
    return 0


RHOS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

if __name__ == "__main__":
    nrep_ = int(os.environ.get("NREP", "200"))
    if len(sys.argv) >= 2 and sys.argv[1] == "cell":
        raise SystemExit(run_cell(sys.argv[2], float(sys.argv[3]), nrep_))
    if len(sys.argv) >= 2 and sys.argv[1] == "merge":
        raise SystemExit(merge(RHOS, nrep_))
    raise SystemExit(main())
