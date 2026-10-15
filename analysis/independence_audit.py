#!/usr/bin/env python3
"""docs/41 独立性审计：把「15 个子集」的符号检验换成诚实的独立单元。

先决条件（任一不过即中止，CLAUDE.md §10）：
  P0  预先登记哈希锁（cache/docs41_prereg_lock.json）一致 —— 改一个字节即中止
  P1  rank_core.selfcheck()  与  independence_stats.selfcheck()  全过
  P2  Gen-3 截尾检测器的已知事实自检（无截尾 -> ~0；植入 40% -> ~0.4；n<200 闸报「n 太少」而非 0）
  P3  analysis/subspace_selftest.py 退出码 0（子进程）
  P4  复现 headline：14/15、p=0.00049、+4.53%、+11.39%（来自 cache/censor_recheck.json），
      并从原始数据**重算**，与缓存的 trim 曲线逐位核对（< 1e-9）

用法：python3 analysis/independence_audit.py [--nperm 200] [--nboot 2000] [--quick]
输出：cache/independence_audit.json（全部数字）、cache/independence_tables.md（可直接贴的表）
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")      # 8 个进程 x 多线程 BLAS 会互相抢核，实测慢数倍
import argparse
import collections
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import zlib
import time

import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

import rank_core                       # noqa: E402  已验证的数值层（als_guarded 向量化等价版）
import independence_stats as IS        # noqa: E402
import censor_density as CD            # noqa: E402

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
CV = dict(K=5, restarts=2, iters=150, ridge=0.1)   # 与 censor_recheck 一致
MAXR = 4
SEEDS_PANEL = (1, 2, 3)                # 与缓存一致
SEEDS_SYS = (1, 2, 3, 4, 5)


# ============================================================ P0 预先登记锁
def check_prereg_lock():
    lk = json.load(open(os.path.join(ROOT, "cache", "docs41_prereg_lock.json")))
    p = os.path.join(ROOT, lk["file"])
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert h == lk["sha256"], (
        "预先登记文件被改动，违反 CLAUDE.md §4：\n"
        f"  锁定 {lk['sha256']}\n  现在 {h}")
    return {"prereg_sha256": h, "locked_at": lk["locked_at"]}


# ============================================================ P2 截尾检测器自检
def censor_selfcheck():
    rng = np.random.default_rng(5)
    v0 = CD.synth_censored(rng, 0.0, 0.0)
    q0 = CD.censor_fraction_density(v0)
    assert q0 < 0.02, f"无截尾被报 {q0:.3f}"
    v4 = CD.synth_censored(rng, 0.4, 0.0)
    q4 = CD.censor_fraction_density(v4)
    assert abs(q4 - 0.4) < 0.02, f"植入 40% 硬赋堆积被报 {q4:.3f}"
    v4n = CD.synth_censored(rng, 0.4, 0.05)
    q4n = CD.censor_fraction_density(v4n)
    assert abs(q4n - 0.4) < 0.08, f"植入 40%+噪声被报 {q4n:.3f}"
    _, info = CD.censor_fraction_density(
        np.random.default_rng(1).normal(0, 1, 150), detail=True)
    assert info["gate"] == "n 太少", "n<200 必须由闸报「n 太少」，不得静默读成 0%"
    return f"P2 截尾检测器: 0%->{q0:.3f}, 40%硬赋->{q4:.3f}, 40%+噪声->{q4n:.3f}, n<200->闸报'n 太少'  OK"


def censor_of(raw):
    """Gen-3 检测器，喂**原始值**。n<200 返回 None（n/a），不是 0。"""
    q, info = CD.censor_fraction_density(np.asarray(raw, float), detail=True)
    if info["gate"] == "n 太少":
        return None, info["gate"]
    return float(q), info["gate"]


# ============================================================ 定秩 + 逐格误差
def trim95(e):
    return float(math.sqrt(np.quantile(e, 0.95)))


def cell_errors(M, mask, seeds, max_rank=MAXR):
    """对每个种子调用 rank_core.cv_rank_noleak_errors，并按固定格子序对齐。

    cv_rank_noleak_errors 返回的 errs 按「洗牌后的折序」排列；折序由种子决定。
    这里复现那一步洗牌（同样的 6 行）以拿到列号，并断言行号逐位相等 ——
    不相等就中止，因为那意味着我对格子身份的理解错了。
    返回 dict: cells (n,2)、E[r] (n,) 种子平均的逐格平方误差、curves[seed][r] = trim95。
    """
    cells = np.argwhere(mask)
    key = {(int(i), int(j)): n for n, (i, j) in enumerate(cells)}
    acc = {r: np.zeros(len(cells)) for r in range(max_rank + 1)}
    curves = []
    for sd in seeds:
        rows, errs = rank_core.cv_rank_noleak_errors(
            M, mask, max_rank=max_rank, seed=sd, **CV)
        rng = np.random.default_rng(sd)
        idx = np.argwhere(mask)
        rng.shuffle(idx)
        assert np.array_equal(idx[:, 0], rows), "折序复现失败：格子身份不可信"
        order = np.array([key[(int(i), int(j))] for i, j in idx])
        c = {}
        for r in range(max_rank + 1):
            e = np.empty(len(cells))
            e[order] = errs[r]
            acc[r] += e
            c[r] = trim95(errs[r])
        curves.append(c)
    E = {r: acc[r] / len(seeds) for r in acc}
    return {"cells": cells, "E": E, "curves": curves}


def curve_summary(curves, max_rank=MAXR):
    mean = {r: float(np.mean([c[r] for c in curves])) for r in range(max_rank + 1)}
    bests = [min(c, key=c.get) for c in curves]
    cnt = collections.Counter(bests)
    mode, nm = cnt.most_common(1)[0]
    best_mean = min(mean, key=mean.get)
    g = lambda a, b: (mean[a] - mean[b]) / mean[a]
    return {"mean": mean, "bests": bests, "mode": int(mode), "stab": nm / len(bests),
            "best_of_mean": int(best_mean),
            "g01_best": (mean[0] - mean[best_mean]) / mean[0],
            "g12": g(1, 2), "g23": g(2, 3), "g02": g(0, 2)}


# ============================================================ 加载矩阵
def load_all():
    import subspace_lib as L
    import cr9114_parse as P
    mats = L.load_all_dms()
    out = []
    for d in mats:
        if d["dataset"] == "animalACE2":
            d["system"] = "S1"
            out.append(d)
        elif d["dataset"] == "PDCoV":
            d["system"] = "S2"
            out.append(d)
    for d in L.load_skempi():
        if d["dataset"] == "SKEMPI_OMTKY3":
            d["system"] = "S3"
            d["vote"] = d["ligand"] == "4-proteases-pooled"
            out.append(d)
    for d in L.load_independent():
        if d["dataset"] == "titeseq2016":
            d["system"] = "S4"
            out.append(d)
        elif d["dataset"] == "lehner_ddPCA":
            d["system"] = {"PSD95-PDZ3": "S6", "GRB2-SH3": "S7", "GB1": "S8"}[d["backbone"]]
            out.append(d)
    cr = P.load_all("dmlk", run_selftest=True)     # 44 项已知事实自检
    for nm in ("GL_H1", "WT_H1", "WT_H3", "WT_fluB"):
        d = cr[nm]
        M, mask = d["M"].copy(), d["mask"].copy()
        kc = mask.sum(0) >= L.MIN_PER_COL            # 与 cr9114_gate.prep 同一常规过滤
        M, mask = M[:, kc], mask[:, kc]
        kr = mask.sum(1) >= L.MIN_PER_ROW
        M, mask = M[kr], mask[kr]
        sites = [s for s, k in zip(d["sites"], kr) if k]
        out.append({"M": M, "mask": mask, "sites": sites, "raw": d["raw"],
                    "dataset": "CR9114", "backbone": "CR9114", "ligand": nm,
                    "name": f"CR9114/{nm}", "system": "S5", "n_obs": int(mask.sum())})
    for d in out:
        d.setdefault("vote", True)
        d["nsites"] = int(d["mask"].any(1).sum())
    return out


# ============================================================ §10.3 还原性诊断
def _plant(M, mask, k, rng_seed):
    """以矩阵自己的 mask 为几何：主效应 + 秩 k 曲面 + 高斯噪声。"""
    r, c = rank_core.center_fit(M, mask)
    R = rank_core.residual(M, mask)
    nr, nc = M.shape
    n = int(mask.sum())
    if k == 0:
        P = np.zeros_like(M)
    else:
        U, V, _ = rank_core.fit_loadings(M, mask, rank=k, restarts=4, seed=11)
        P = U @ V.T
    sse = float(((R[mask] - P[mask]) ** 2).sum())
    dof = (nr + nc - 1) + k * (nr + nc - k)
    sig = math.sqrt(sse / max(n - dof, 1))
    rng = np.random.default_rng(rng_seed)
    T = r[:, None] + c[None, :] + P
    Y = T + rng.normal(0, sig, M.shape)
    return np.where(mask, Y, 0.0), sig


def recovery(M, mask, reps=8, seed0=1000):
    out = {}
    for k in (2, 1, 0):
        bests, g12 = [], []
        sig = None
        for t in range(reps):
            Y, sig = _plant(M, mask, k, seed0 + 17 * t + k)
            e = rank_core.cv_rank_noleak(Y, mask, max_rank=MAXR, seed=3, **CV)
            br, trim = rank_core.best_rank(e)
            bests.append(int(br))
            g12.append((trim[1] - trim[2]) / trim[1])
        out[f"plant{k}"] = {"best": bests, "sigma": sig,
                            "p_detect2": float(np.mean([b == 2 for b in bests])),
                            "p_detect0": float(np.mean([b == 0 for b in bests])),
                            "p_r2_beats_r1": float(np.mean([g > 0 for g in g12]))}
    out["powered"] = bool(out["plant2"]["p_detect2"] >= 0.75
                          and out["plant0"]["p_detect0"] >= 0.75)
    return out


def _worker(args):
    name, M, mask, seeds, do_rec = args
    t = time.time()
    ce = cell_errors(M, mask, seeds)
    rec = recovery(M, mask) if do_rec else None
    return name, ce, rec, time.time() - t


# ============================================================ 位点聚类 bootstrap
def site_boot_gains(units, nboot, seed, pairs=((1, 2), (2, 3))):
    """units: [{sites, rows (cell row idx), E}]。位点标签在各矩阵间对齐，联合重抽。

    返回 {pair: (nboot, nunits)} 每个矩阵每次重抽的 (t_a - t_b)/t_a，以及 pair 的 t 值本身不保存。
    """
    labels = sorted({s for u in units for s in u["sites"]}, key=str)
    li = {s: i for i, s in enumerate(labels)}
    rng = np.random.default_rng(seed)
    # 每个矩阵：cell -> 位点标签索引
    cell_lab = [np.array([li[u["sites"][r]] for r in u["rows"]]) for u in units]
    out = {p: np.empty((nboot, len(units))) for p in pairs}
    for b in range(nboot):
        w = np.bincount(rng.integers(0, len(labels), len(labels)), minlength=len(labels))
        for k, u in enumerate(units):
            rep = w[cell_lab[k]]
            t = {}
            for r in {x for p in pairs for x in p}:
                t[r] = math.sqrt(np.quantile(np.repeat(u["E"][r], rep), 0.95))
            for (a, c) in pairs:
                out[(a, c)][b, k] = (t[a] - t[c]) / t[a]
    return out


# ============================================================ 位点内联合置换（panel）
def _perm_worker(args):
    pseed, mats = args
    labels = sorted({s for (_, _, sites) in mats for s in sites}, key=str)
    rng = np.random.default_rng(max(pseed, 0))     # pseed<0 = 恒等置换（自检用）
    perm = {s: (rng.permutation(20) if pseed >= 0 else np.arange(20)) for s in labels}
    g = []
    for (M, mask, sites) in mats:
        M2, m2 = M.copy(), mask.copy()
        for i, s in enumerate(sites):
            M2[i] = M[i][perm[s]]
            m2[i] = mask[i][perm[s]]
        e = rank_core.cv_rank_noleak(M2, m2, max_rank=2, seed=1, **CV)
        t1, t2 = e[1][1], e[2][1]
        g.append((t1 - t2) / t1)
    return pseed, g


# ============================================================ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    R = {"search_volume": {}}
    log = []

    def P(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        log.append(s)

    P("=" * 100)
    P("docs/41 独立性审计")
    P("=" * 100)
    R["prereg"] = check_prereg_lock()
    P(f"P0 预先登记锁一致 sha256 {R['prereg']['prereg_sha256'][:16]}…  OK")
    rank_core.selfcheck(verbose=False)
    P("P1a rank_core.selfcheck 全过  OK")
    IS.selfcheck(verbose=False)
    P("P1b independence_stats.selfcheck 全过  OK")
    P(censor_selfcheck())
    rc = subprocess.run([sys.executable, os.path.join(HERE, "subspace_selftest.py")],
                        capture_output=True, text=True)
    assert rc.returncode == 0, "subspace_selftest 未通过：\n" + rc.stdout[-2000:]
    P("P3 subspace_selftest.py 退出码 0  OK（尾行：" +
      rc.stdout.strip().splitlines()[-1] + "）")
    R["selftests"] = {"rank_core": "pass", "independence_stats": "pass",
                      "censor": "pass", "subspace_selftest": "pass"}

    # ------------------------------------------------ P4 复现 headline
    cache = json.load(open(os.path.join(ROOT, "cache", "censor_recheck.json")))
    A = [u for u in cache["units"] if u["dataset"] == "animalACE2"]
    assert len(A) == 15
    bbs = sorted({u["bg"] for u in A})
    ligs = ["bind_deer", "bind_hamster", "bind_bat", "bind_cat", "bind_human_monomer"]
    lig_names = [l.replace("bind_", "") for l in ligs]
    cc = {}
    for u in A:
        tc = u["trim_curves"]
        cc[(u["bg"], u["lig"])] = {r: float(np.mean([c[str(r)] for c in tc]))
                                   for r in range(4)}
    G12 = np.array([[(cc[(b, l)][1] - cc[(b, l)][2]) / cc[(b, l)][1] for l in ligs]
                    for b in bbs])
    G23 = np.array([[(cc[(b, l)][2] - cc[(b, l)][3]) / cc[(b, l)][2] for l in ligs]
                    for b in bbs])
    G02 = np.array([[(cc[(b, l)][0] - cc[(b, l)][2]) / cc[(b, l)][0] for l in ligs]
                    for b in bbs])
    npos = int((G12 > 0).sum())
    p_head = IS.exact_sign_p(npos, 15)
    med12, med23, med02 = (float(np.median(G12)), float(np.median(G23)),
                           float(np.median(G02)))
    assert npos == 14, f"14/15 未复现: {npos}/15"
    assert round(p_head, 5) == 0.00049, p_head
    assert round(100 * med12, 2) == 4.53, med12
    assert round(100 * med02, 2) == 11.39, med02
    P(f"P4 复现 headline（缓存 cache/censor_recheck.json，animalACE2 15 单元）: "
      f"{npos}/15, 单侧精确 p={p_head:.5f}（双侧 {IS.exact_sign_p(npos,15,'two-sided'):.5f}）, "
      f"秩1→2 中位 {100*med12:+.2f}%, 秩0→2 中位 {100*med02:+.2f}%  OK")
    P(f"   秩2→3 中位（同一口径：(t2−t3)/t2，逐子集先对 3 种子取均值）= {100*med23:+.2f}%  "
      f"—— headline 写的是 −5.54%，**本口径复现不出**（见 docs/41 §1）")
    R["reproduction"] = {"n_pos": npos, "p_one_sided": p_head,
                         "p_two_sided": IS.exact_sign_p(npos, 15, "two-sided"),
                         "median_g12": med12, "median_g02": med02,
                         "median_g23_same_definition": med23,
                         "headline_g23": -0.0554, "g23_reproduced": False,
                         "source": "cache/censor_recheck.json (units, dataset=animalACE2, trim_curves)",
                         "backbones": bbs, "ligands": lig_names,
                         "G12": G12.tolist(), "G23": G23.tolist()}

    # ------------------------------------------------ 加载并跑所有矩阵
    mats = load_all()
    P(f"加载矩阵 {len(mats)} 个: " + collections.Counter(m['system'] for m in mats).__repr__())
    jobs = []
    for m in mats:
        seeds = SEEDS_PANEL if m["system"] == "S1" else SEEDS_SYS
        jobs.append((m["name"], m["M"], m["mask"], seeds, True))
    t0 = time.time()
    nproc = max(1, min(8, mp.cpu_count() - 1))
    pk = os.path.join(ROOT, "cache", "independence_cells.pkl")
    sig = hashlib.sha256(repr([(j[0], float(np.nansum(j[1])), int(j[2].sum()), j[3])
                               for j in jobs] + [CV, MAXR]).encode()).hexdigest()
    res = None
    if a.reuse and os.path.exists(pk):
        import pickle
        blob = pickle.load(open(pk, "rb"))
        if blob["sig"] == sig:
            res = blob["res"]
            P("  复用 cache/independence_cells.pkl（输入签名一致）")
    if res is None:
        with mp.get_context("spawn").Pool(nproc) as pool:
            res = pool.map(_worker, jobs, chunksize=1)
        import pickle
        pickle.dump({"sig": sig, "res": res}, open(pk, "wb"))
    P(f"定秩 + 还原性诊断: {len(jobs)} 个矩阵, {time.time()-t0:.0f}s, {nproc} 进程")
    for m, (nm, ce, rec, dt) in zip(mats, res):
        assert nm == m["name"]
        m["ce"], m["rec"] = ce, rec
        m["sum"] = curve_summary(ce["curves"])
        q, gate = censor_of(m["raw"] if "raw" in m else m["M"][m["mask"]])
        m["censor"], m["censor_gate"] = q, gate

    # 复现：从原始数据重算的 S1 曲线与缓存逐位一致（秩 0–3）
    S1 = {(m["backbone"], "bind_" + m["ligand"]): m for m in mats if m["system"] == "S1"}
    maxdiff = 0.0
    for (b, l), m in S1.items():
        for r in range(4):
            mc = float(np.mean([c[r] for c in m["ce"]["curves"]]))
            maxdiff = max(maxdiff, abs(mc - cc[(b, l)][r]))
    assert maxdiff < 1e-9, f"重算的 trim 曲线与缓存不一致 (max|Δ|={maxdiff:.2e})"
    P(f"P4b 从原始数据重算 15 个子集的 trim95 曲线（秩 0–3）与缓存 max|Δ| = {maxdiff:.1e}  OK")
    R["reproduction"]["recompute_max_abs_diff"] = maxdiff

    # ------------------------------------------------ Step 3：面板（S1）上的各单元
    P("\n" + "=" * 100)
    P("Step 3：RBD–ACE2 面板（animalACE2，3 骨架 × 5 配体）在各候选单元下的统计")
    P("=" * 100)
    panel = {}
    sv = R["search_volume"]
    sv["units_considered"] = 9
    sv["tests_run"] = 0

    def count(n=1):
        sv["tests_run"] += n

    # U0 朴素
    panel["U0"] = {"n": 15, "k_pos": npos, "p_one_sided": p_head}
    count()
    # U1/U2 单向簇层（预先登记的检验；自检显示在交叉设计下仍反保守）
    for tag, G, names, axis in (("U1_backbone", G12, bbs, 1), ("U2_ligand", G12, lig_names, 0)):
        cl = G.mean(axis=axis)                 # 簇内 3（或 5）个子集的平均增益
        clmed = np.median(G, axis=axis)
        k = int((clmed > 0).sum())
        n = len(clmed)
        kall = int((cl > 0).sum())
        panel[tag] = {"names": names, "cluster_median": clmed.tolist(),
                      "cluster_mean": cl.tolist(), "k_pos_median": k, "n": n,
                      "p_formal": IS.exact_sign_p(k, n), "min_attainable_p": IS.min_attainable_p(n),
                      "k_pos_mean": kall}
        count()
        # 块符号翻转（整块同时翻转）
        pf, B, tobs = IS.block_signflip_p(G, axis=(0 if tag == "U1_backbone" else 1), stat="count")
        pm, _, _ = IS.block_signflip_p(G, axis=(0 if tag == "U1_backbone" else 1), stat="median")
        panel[tag]["blockflip_p_count"] = pf
        panel[tag]["blockflip_p_median"] = pm
        panel[tag]["blockflip_B"] = B
        count(2)
    # 留一
    loo_b = [float(np.median(np.delete(G12, i, axis=0))) for i in range(3)]
    loo_l = [float(np.median(np.delete(G12, j, axis=1))) for j in range(5)]
    panel["leave_one_out"] = {"drop_backbone": dict(zip(bbs, loo_b)),
                              "drop_ligand": dict(zip(lig_names, loo_l)),
                              "min_k_pos_drop_backbone": [int((np.delete(G12, i, 0) > 0).sum()) for i in range(3)],
                              "min_k_pos_drop_ligand": [int((np.delete(G12, j, 1) > 0).sum()) for j in range(5)]}
    # U3 交叉：ANOVA + REML + 交叉 bootstrap
    for nm, G in (("g12", G12), ("g23", G23), ("g02", G02)):
        an = IS.two_way_anova(G)
        try:
            rm = IS.reml_crossed(G)
        except Exception as ex:                  # noqa: BLE001
            rm = {"error": repr(ex)}
        bsc = IS.crossed_bootstrap(G, np.median, nboot=4000, seed=7)
        bsr = IS.crossed_bootstrap(G, np.median, nboot=4000, seed=8, resample=("rows",))
        bsl = IS.crossed_bootstrap(G, np.median, nboot=4000, seed=9, resample=("cols",))
        panel.setdefault("U3", {})[nm] = {
            "anova": an, "reml": rm,
            "boot_crossed_ci": IS.pct_ci(bsc), "boot_rows_ci": IS.pct_ci(bsr),
            "boot_cols_ci": IS.pct_ci(bsl),
            "boot_crossed_frac_le0": float((bsc <= 0).mean())}
        count(4)
    # t 单侧 p（g12）
    an = panel["U3"]["g12"]["anova"]
    tstat = an["mean"] / an["se"]
    panel["U3"]["g12"]["t"] = float(tstat)
    panel["U3"]["g12"]["p_one_sided_t"] = float(1 - st.t.cdf(tstat, an["df"]))
    an23 = panel["U3"]["g23"]["anova"]
    panel["U3"]["g23"]["p_one_sided_t_worse"] = float(st.t.cdf(an23["mean"] / an23["se"], an23["df"]))

    # U4 位点：联合位点 bootstrap（15 个矩阵共享位点标签）
    t0 = time.time()
    units15 = []
    order15 = [(b, l) for b in bbs for l in ligs]
    for (b, l) in order15:
        m = S1[(b, l)]
        units15.append({"sites": m["sites"], "rows": m["ce"]["cells"][:, 0], "E": m["ce"]["E"]})
    nb = a.nboot
    sb = site_boot_gains(units15, nb, seed=41, pairs=((1, 2), (2, 3), (0, 2)))
    g12b, g23b, g02b = sb[(1, 2)], sb[(2, 3)], sb[(0, 2)]
    # 点估计（种子平均误差口径），与 headline（种子平均 trim 曲线口径）并列
    pe = {}
    for (a_, c_), key in (((1, 2), "g12"), ((2, 3), "g23"), ((0, 2), "g02")):
        pe[key] = float(np.median([(trim95(u["E"][a_]) - trim95(u["E"][c_])) / trim95(u["E"][a_])
                                   for u in units15]))
    panel["U4_site"] = {
        "nboot": nb, "point_median_errmean_g12": pe["g12"],
        "median_g12_ci": IS.pct_ci(np.median(g12b, axis=1)),
        "median_g23_ci": IS.pct_ci(np.median(g23b, axis=1)),
        "median_g02_ci": IS.pct_ci(np.median(g02b, axis=1)),
        "npos_g12_dist": {"p2.5": float(np.percentile((g12b > 0).sum(1), 2.5)),
                          "median": float(np.median((g12b > 0).sum(1))),
                          "frac_ge_13": float(((g12b > 0).sum(1) >= 13).mean()),
                          "frac_eq_14_or_more": float(((g12b > 0).sum(1) >= 14).mean())},
        "per_subset_g12_ci": {f"{b}/{l}": IS.pct_ci(g12b[:, k]) for k, (b, l) in enumerate(order15)},
        "per_subset_frac_gt0": {f"{b}/{l}": float((g12b[:, k] > 0).mean()) for k, (b, l) in enumerate(order15)},
        "n23_neg_dist_median": float(np.median((g23b < 0).sum(1))),
        "seconds": time.time() - t0}
    count(3)
    P(f"U4 位点联合 bootstrap {nb} 次 {time.time()-t0:.0f}s")

    # 位点共模：在位点联合重抽下，「为正的子集数」的方差 vs 若 15 个子集独立时的方差
    cnt_b = (g12b > 0).sum(1)
    p_k = (g12b > 0).mean(0)
    var_indep = float((p_k * (1 - p_k)).sum())
    deff = float(cnt_b.var() / var_indep) if var_indep > 0 else float("nan")
    panel["U4_site"]["count_var_boot"] = float(cnt_b.var())
    panel["U4_site"]["count_var_if_independent"] = var_indep
    panel["U4_site"]["design_effect_count"] = deff
    panel["U4_site"]["n_eff_count"] = 15 / deff if deff > 0 else None
    # 同时重抽位点与（骨架、配体）水平：骨架、配体、位点三者都随机
    rngc = np.random.default_rng(99)
    comb = {}
    for key, gb in (("g12", g12b), ("g23", g23b), ("g02", g02b)):
        med = np.empty(nb)
        for t in range(nb):
            ri = rngc.integers(0, 3, 3)
            ci = rngc.integers(0, 5, 5)
            med[t] = np.median(gb[t].reshape(3, 5)[np.ix_(ri, ci)])
        comb[key] = {"ci": IS.pct_ci(med), "frac_le0": float((med <= 0).mean()),
                     "frac_ge0": float((med >= 0).mean())}
    panel["U3U4_combined"] = comb
    count(3)

    # 位点内联合置换（保留跨矩阵共享的格子级噪声）
    t0 = time.time()
    mats15 = [(S1[k]["M"], S1[k]["mask"], S1[k]["sites"]) for k in order15]
    nperm = 10 if a.quick else a.nperm
    pj = [(-1, mats15)] + [(100 + s, mats15) for s in range(nperm)]
    with mp.get_context("spawn").Pool(nproc) as pool:
        pres = pool.map(_perm_worker, pj, chunksize=1)
    ident = np.array(dict(pres)[-1])
    ref = np.array([(cc[k][1] - cc[k][2]) / cc[k][1] for k in order15])
    # 自检：恒等置换复现真实数据的 seed=1 增益（用 seed-1 曲线核对，而不是 3 种子平均）
    ref1 = np.array([(S1[k]["ce"]["curves"][0][1] - S1[k]["ce"]["curves"][0][2]) / S1[k]["ce"]["curves"][0][1]
                     for k in order15])
    assert np.abs(ident - ref1).max() < 1e-9, "恒等置换未复现 seed=1 真实增益"
    perm_g = np.array([g for s, g in pres if s != -1])
    perm_cnt = (perm_g > 0).sum(1)
    perm_med = np.median(perm_g, axis=1)
    obs_cnt1 = int((ref1 > 0).sum())
    obs_med1 = float(np.median(ref1))
    panel["perm_null"] = {
        "nperm": nperm, "obs_count_seed1": obs_cnt1, "obs_median_seed1": obs_med1,
        "null_count_mean": float(perm_cnt.mean()), "null_count_max": int(perm_cnt.max()),
        "null_count_hist": dict(collections.Counter(perm_cnt.tolist())),
        "null_median_mean": float(perm_med.mean()), "null_median_max": float(perm_med.max()),
        "p_count": float((1 + (perm_cnt >= obs_cnt1).sum()) / (nperm + 1)),
        "p_median": float((1 + (perm_med >= obs_med1).sum()) / (nperm + 1)),
        "identity_check_max_abs": float(np.abs(ident - ref1).max()),
        "seconds": time.time() - t0}
    count(2)
    P(f"置换零分布 {nperm} 次 {time.time()-t0:.0f}s: 观测 {obs_cnt1}/15 vs 零分布均值 "
      f"{perm_cnt.mean():.2f}（最大 {perm_cnt.max()}）")

    # 依赖结构：矩阵间残差相关、逐格损失差相关
    resid = {}
    for k in order15:
        m = S1[k]
        Rm = rank_core.residual(m["M"], m["mask"])
        resid[k] = {(m["sites"][i], j): Rm[i, j] for i, j in np.argwhere(m["mask"])}
    dloss = {}
    for k in order15:
        m = S1[k]
        d = m["ce"]["E"][1] - m["ce"]["E"][2]
        dloss[k] = {(m["sites"][i], j): d[n] for n, (i, j) in enumerate(m["ce"]["cells"])}
    # 位点层面的损失差（每位点均值）
    dsite = {}
    for k in order15:
        tmp = collections.defaultdict(list)
        for (s, j), v in dloss[k].items():
            tmp[s].append(v)
        dsite[k] = {s: float(np.mean(v)) for s, v in tmp.items()}

    def pair_corr(table):
        rel = collections.defaultdict(list)
        for i, k1 in enumerate(order15):
            for k2 in order15[i + 1:]:
                common = sorted(set(table[k1]) & set(table[k2]), key=str)
                if len(common) < 100:
                    continue
                x = np.array([table[k1][c] for c in common])
                y = np.array([table[k2][c] for c in common])
                r_ = float(np.corrcoef(x, y)[0, 1])
                if k1[0] == k2[0]:
                    rel["same_backbone_diff_ligand"].append(r_)
                elif k1[1] == k2[1]:
                    rel["same_ligand_diff_backbone"].append(r_)
                else:
                    rel["diff_both"].append(r_)
        return {k: {"n_pairs": len(v), "mean": float(np.mean(v)), "min": float(np.min(v)),
                    "max": float(np.max(v))} for k, v in rel.items()}
    panel["dependence"] = {"interaction_residual_corr": pair_corr(resid),
                           "cell_loss_diff_corr": pair_corr(dloss),
                           "site_loss_diff_corr": pair_corr(dsite)}
    # 骨架序列差异（事实，来自 cache/dms）
    panel["dependence"]["backbone_site_differences_of_201"] = {
        "Eta_vs_WH1": 1, "BA2_vs_WH1": 16, "BA2_vs_Eta": 16}

    # 内部极小（面板）
    H23 = -G23                                       # 正 = 秩 3 更差
    both = (G12 > 0) & (H23 > 0)
    panel["interior_min"] = {
        "k_leg1_rank2_beats_rank1": int((G12 > 0).sum()),
        "k_leg2_rank3_worse": int((H23 > 0).sum()),
        "k_both_legs": int(both.sum()),
        "both_legs_by_backbone": {b: int(both[i].sum()) for i, b in enumerate(bbs)},
        "both_legs_by_ligand": {l: int(both[:, j].sum()) for j, l in enumerate(lig_names)},
        "leg2_blockflip_rows_p": IS.block_signflip_p(H23, 0, "count")[0],
        "leg2_blockflip_cols_p": IS.block_signflip_p(H23, 1, "count")[0],
        "median_g23": med23}
    count(2)
    R["panel"] = panel

    # ------------------------------------------------ Step 4：独立体系
    P("\n" + "=" * 100)
    P("Step 4：独立体系 × 检出秩（噪声感知的统一协议）")
    P("=" * 100)
    SYSNAME = {"S1": "SARS-CoV-2 RBD × ACE2 正交体（animalACE2）", "S2": "PD-CoV RBD × APN",
               "S3": "OMTKY3 × 蛋白酶（SKEMPI Ka）", "S4": "4-4-20 scFv × 荧光素（Tite-seq）",
               "S5": "CR9114 重链 × HA（Tite-seq）", "S6": "PSD95-PDZ3 × CRIPT（ddPCA）",
               "S7": "GRB2-SH3 × GAB2（ddPCA）", "S8": "GB1 × IgG-Fc（ddPCA）"}
    rows = []
    for m in mats:
        s = m["sum"]
        rows.append({"system": m["system"], "name": m["name"], "vote_matrix": bool(m["vote"]),
                     "sites": m["nsites"], "n_obs": m["n_obs"], "censor_gen3": m["censor"],
                     "censor_gate": m["censor_gate"], "mode_rank": s["mode"], "stab": s["stab"],
                     "bests": s["bests"], "trim_mean": s["mean"], "g01_best": s["g01_best"],
                     "g12": s["g12"], "g23": s["g23"], "powered": m["rec"]["powered"],
                     "rec": m["rec"]})
    R["matrices"] = rows

    # 体系层：投票矩阵、位点 bootstrap 区间
    def pooled(ms_list, seed):
        """矩阵间中位数在联合位点重抽下的区间。返回点估计与区间（g12/g23/g02）。"""
        units = [{"sites": m["sites"], "rows": m["ce"]["cells"][:, 0], "E": m["ce"]["E"]}
                 for m in ms_list]
        sbx = site_boot_gains(units, a.nboot, seed=seed, pairs=((1, 2), (2, 3), (0, 2)))
        out = {}
        for key, pr in (("g12", (1, 2)), ("g23", (2, 3)), ("g02", (0, 2))):
            med = np.median(sbx[pr], axis=1)
            out[key] = {"point": float(np.median([m["sum"][key] for m in ms_list])),
                        "ci": IS.pct_ci(med)}
        count(3)
        return out

    def classify(o):
        c = o["g12"]["ci"]
        if c[0] > 0:
            return "SUPPORTS_2"
        if c[1] < 0:
            return "SUPPORTS_LE1"
        return "INDETERMINATE"

    sysres = {}
    for sid in sorted({m["system"] for m in mats}):
        ms = [m for m in mats if m["system"] == sid and m["vote"]]
        pw = [m for m in ms if m["rec"]["powered"]]
        info = {"name": SYSNAME[sid], "n_vote_matrices": len(ms), "n_powered": len(pw),
                "in_primary_vote": sid in ("S1", "S2", "S3", "S4", "S5")}
        if not pw:
            info["status"] = "UNINFORMATIVE"
            sysres[sid] = info
            continue
        o = pooled(pw, 500 + int(sid[1:]))
        status = classify(o)
        info.update({"powered_matrices": [m["name"] for m in pw],
                     "g12_point": o["g12"]["point"], "g12_ci": o["g12"]["ci"],
                     "g23_point": o["g23"]["point"], "g23_ci": o["g23"]["ci"],
                     "g02_point": o["g02"]["point"], "g02_ci": o["g02"]["ci"],
                     "status": status, "vote_point_sign": int(np.sign(o["g12"]["point"]))})
        # 内部极小：g12 区间下界>0 且 g23 区间上界<0
        info["interior_min_supported"] = bool(o["g12"]["ci"][0] > 0 and o["g23"]["ci"][1] < 0)
        # 【事后敏感性，未预先登记】更严的口径：秩 2 还必须优于秩 0
        info["posthoc_strict_supports_2"] = bool(o["g12"]["ci"][0] > 0 and o["g02"]["ci"][0] > 0)
        sysres[sid] = info
    # 【事后敏感性，未预先登记；看过上表之后才加，只作旁证，不进主投票】
    sens = {}
    byname = {m["name"]: m for m in mats}
    for tag, names in (("S2_gAPN_only", ["PDCoV/PDCoV/gAPN"]),
                       ("S5_WT_H1_only(docs/25 唯一过截尾闸)", ["CR9114/WT_H1"])):
        o = pooled([byname[n] for n in names], 900 + len(sens))
        sens[tag] = {"matrices": names, "g12": o["g12"], "g23": o["g23"], "g02": o["g02"],
                     "status": classify(o)}
    R["posthoc_sensitivity_systems"] = sens
    # 每个矩阵单独的位点 bootstrap 区间（供表格用）
    for m in mats:
        o = pooled([m], 1500 + zlib.crc32(m["name"].encode()) % 1000)
        m["ci"] = o
    for r_, m in zip(rows, mats):
        r_["ci"] = m["ci"]
        r_["status_matrix"] = classify(m["ci"])
    # CR9114：docs/25 的「滴定窗外」比例（Gen-3 对这种噪声摊开的堆积是盲的）
    try:
        cr = json.load(open(os.path.join(ROOT, "cache", "cr9114_censor.json")))["matrices"]
        for r_ in rows:
            if r_["name"].startswith("CR9114/"):
                r_["f_outside_titration_docs25"] = cr[r_["name"].split("/")[1]]["f_outside_titration"]
    except Exception as ex:                    # noqa: BLE001
        R["cr9114_f_out_error"] = repr(ex)
    R["systems"] = sysres

    # S1 扩展（描述性，来自缓存）：34 单元，Gen-3 截尾分层 × 检出秩（不引切点）
    g3 = json.load(open(os.path.join(ROOT, "cache", "censor_gen3_final.json")))["units"]
    strata = [(0.0, 0.02), (0.02, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 1.01)]
    ext = {}
    for lo, hi in strata:
        sel = [u for u in g3 if lo <= u["g3"] < hi]
        ext[f"{lo:.2f}-{min(hi,1):.2f}"] = {
            "n": len(sel), "ranks": dict(collections.Counter(int(u["rank_noleak"]) for u in sel)),
            "by_dataset": dict(collections.Counter(u["dataset"] for u in sel))}
    R["s1_extended_34_by_gen3"] = ext
    R["s1_extended_note"] = ("34 单元 = animalACE2 15 + sarbecovirus 19（censor_gen3_final.json）；"
                             "二者都是 Starr 酵母展示 RBD–ACE2，同属 S1，不是 34 个独立观测")

    # 主投票与敏感性
    prim = [sid for sid in ("S1", "S2", "S3", "S4", "S5")]
    allv = prim + ["S6", "S7", "S8"]

    def tally(ids, how):
        pos = neg = 0
        for sid in ids:
            s = sysres[sid]
            if s["status"] == "UNINFORMATIVE":
                continue
            if how == "ci":
                if s["status"] == "SUPPORTS_2":
                    pos += 1
                elif s["status"] == "SUPPORTS_LE1":
                    neg += 1
            else:
                if s["vote_point_sign"] > 0:
                    pos += 1
                elif s["vote_point_sign"] < 0:
                    neg += 1
        return pos, neg
    vt = {}
    for nm, ids in (("primary_S1-S5", prim), ("with_ddPCA_S1-S8", allv)):
        for how in ("ci", "point"):
            pos, neg = tally(ids, how)
            n = pos + neg
            vt[f"{nm}|{how}"] = {"k_support2": pos, "k_support_le1": neg, "n_voting": n,
                                 "n_systems": len(ids),
                                 "min_attainable_p": IS.min_attainable_p(n) if n else None,
                                 "p_formal": IS.exact_sign_p(pos, n) if n else None,
                                 "interpretable": bool(n and IS.min_attainable_p(n) <= 0.01)}
            count()
    R["system_votes"] = vt
    # 家族 / 平台单元（仅计数）
    R["coarse_units"] = {
        "U7_family_primary": {"n": 3, "members": {"coronavirus_RBD": ["S1", "S2"],
                                                  "Ig_variable-antigen": ["S4", "S5"],
                                                  "protease-inhibitor": ["S3"]},
                              "min_attainable_p": IS.min_attainable_p(3)},
        "U7_family_with_ddPCA": {"n": 6, "min_attainable_p": IS.min_attainable_p(6)},
        "U8_platform_primary": {"n": 3, "members": {"yeast_display_Kd": ["S1", "S2"],
                                                    "Tite-seq": ["S4", "S5"],
                                                    "enzyme_Ka": ["S3"]},
                                "min_attainable_p": IS.min_attainable_p(3)},
        "U8_platform_with_ddPCA": {"n": 4, "min_attainable_p": IS.min_attainable_p(4)},
        "U5_dataset": {"n": 1, "min_attainable_p": 0.5}}

    # 截尾 × 检出秩的单调关系（不引切点）
    cen = [(r["censor_gen3"], r["mode_rank"]) for r in rows if r["censor_gen3"] is not None]
    if len(cen) > 4:
        rho, pr = st.spearmanr([c for c, _ in cen], [k for _, k in cen])
        R["censor_rank_spearman"] = {"rho": float(rho), "n_matrices_not_independent": len(cen),
                                     "note": "矩阵不是独立的（S1 占 15 个），仅描述单调性，不报 p"}

    R["search_volume"]["matrices_run"] = len(mats)
    R["search_volume"]["recovery_cv_runs"] = len(mats) * 3 * 8
    R["search_volume"]["perm_cv_runs"] = (nperm + 1) * 15 * 3
    R["seconds_total"] = time.time() - t_start
    json.dump(R, open(os.path.join(ROOT, "cache", "independence_audit.json"), "w"),
              indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    write_tables(R, rows, log)
    P(f"\n完成，用时 {R['seconds_total']:.0f}s。cache/independence_audit.json, cache/independence_tables.md")
    return 0


def fm(x, pct=True, d=1):
    if x is None:
        return "n/a"
    return f"{100*x:+.{d}f}%" if pct else f"{x:.{d}f}"


def write_tables(R, rows, log):
    L = []
    pn = R["panel"]
    L.append("## 面板统计（RBD–ACE2，3×5）\n")
    L.append("| 单元 | 独立单元数 | 结果 | 检验 |")
    L.append("|---|---|---|---|")
    L.append(f"| U0 子集（原文） | 15（名义） | {pn['U0']['k_pos']}/15 | 单侧精确 p = {pn['U0']['p_one_sided']:.5f}（形式值） |")
    for tag, nm in (("U1_backbone", "U1 骨架"), ("U2_ligand", "U2 配体")):
        u = pn[tag]
        L.append(f"| {nm} | {u['n']} | {u['k_pos_median']}/{u['n']} 个簇的中位增益>0 | "
                 f"形式精确 p = {u['p_formal']:.4f}（最小可达 {u['min_attainable_p']:.4f}；**不可解释**）；"
                 f"块翻转 p(计数) = {u['blockflip_p_count']:.4f}, p(中位) = {u['blockflip_p_median']:.4f}（{u['blockflip_B']} 块，2^B 全枚举）|")
    a = pn["U3"]["g12"]
    L.append(f"| U3 骨架×配体交叉 | n_eff = {a['anova']['n_eff']:.1f}（ICC 骨架 {a['anova']['icc_row']:.2f}，配体 {a['anova']['icc_col']:.2f}）| "
             f"均值 {fm(a['anova']['mean'],d=2)}，t 区间 [{fm(a['anova']['ci'][0],d=2)}, {fm(a['anova']['ci'][1],d=2)}]（Satterthwaite df = {a['anova']['df']:.1f}）| "
             f"单侧 t p = {a['p_one_sided_t']:.3f}（df≈{a['anova']['df']:.1f}，**勉强**）|")
    L.append(f"| U3 交叉 bootstrap | 3 + 5 水平 | 中位 95% 区间 骨架+配体 {fm(a['boot_crossed_ci'][0],d=2)}…{fm(a['boot_crossed_ci'][1],d=2)}；仅骨架 {fm(a['boot_rows_ci'][0],d=2)}…{fm(a['boot_rows_ci'][1],d=2)}；仅配体 {fm(a['boot_cols_ci'][0],d=2)}…{fm(a['boot_cols_ci'][1],d=2)} | 3 个骨架水平上严重欠覆盖 |")
    u4 = pn["U4_site"]
    L.append(f"| U4 位点（联合） | 201 位点（1 个面板） | 中位增益 95% 区间 [{fm(u4['median_g12_ci'][0],d=2)}, {fm(u4['median_g12_ci'][1],d=2)}]；≥14/15 为正的重抽比例 {u4['npos_g12_dist']['frac_eq_14_or_more']:.2f}；≥13/15 为正 {u4['npos_g12_dist']['frac_ge_13']:.2f} | 只含位点抽样，不含骨架/配体泛化 |")
    pp = pn["perm_null"]
    L.append(f"| 位点内联合置换零分布 | {pp['nperm']} 次 | 观测(seed1) {pp['obs_count_seed1']}/15 vs 零分布均值 {pp['null_count_mean']:.2f}，最大 {pp['null_count_max']} | p(计数) ≤ {pp['p_count']:.4f}（分辨率 1/{pp['nperm']+1}） |")
    L.append("")
    open(os.path.join(ROOT, "cache", "independence_tables.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
