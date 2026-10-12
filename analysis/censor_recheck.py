#!/usr/bin/env python3
"""docs/12 §2 的 r(截尾, 检出秩) = -0.786 复核（修正后的截尾检测器）。

背景：docs/13 §3 抓到第十次工具错误 —— 旧截尾检测器只数「恰好等于最小值」
的格子，在 PD-CoV 上静默失效（hAPN 真实 ~46% 被报成 0.0%）。
`r = -0.786` 的**自变量**因此不可信，必须重算。

本脚本只做复核，不做新搜索：
  1. 精确重建 docs/12 §2 的 34 个单元（analysis/ligand_specific_rank.py 的
     animalACE2 3 骨架 x 5 配体 + sarbecovirus 6 骨架 x 5 配体，去掉被
     「观测不足 / 矩阵太小」跳过的 11 个）。
     重建判据（已知事实核对，不是自洽性）：旧截尾比例 = cens/(cens+n)
     必须复现 docs/12 §2 表里的「秩0 中位 43.2% / 秩1 中位 10.9%」
     以及 21 个 <10% 组合里 76% 给秩 2、7 个 >=25% 组合 7/7 给秩 0。
  2. 用 **docs/13 的修正检测器** subspace_lib.censor_fraction 重测截尾
     （不写第三个检测器；只在窗口宽度上做敏感性扫描）。
  3. 秩用**已验证的数值工具层** rank_core.cv_rank_noleak
     （无泄露中心化 + als_guarded 的向量化等价版 + 裁 95% 的 CV RMSE），
     并与 cache/ligand_rank.json 里的原始秩并列。
  4. bootstrap >= 2000 报区间，不单独报点估计（CLAUDE.md §5）。
  5. 信噪比 vs 截尾的对打（信噪比只在有重复列的单元上可算）。

运行：python3 analysis/censor_recheck.py [--boot 4000] [--seeds 3] [--quick]
产物：cache/censor_recheck.json
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rank_core                                      # noqa: E402
from subspace_lib import CENSOR_WINDOW, censor_fraction  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DMS = os.path.join(ROOT, "cache", "dms")
AA20 = list("ACDEFGHIKLMNPQRSTVWY")

# analysis/ligand_specific_rank.py main() 的 jobs，逐字照抄
JOBS = [
    ("animalACE2", "starr_animalACE2_final_variant_scores.csv", "position",
     ["bind_deer", "bind_hamster", "bind_bat", "bind_cat",
      "bind_human_monomer"]),
    ("sarbecovirus", "starr_SARSr_RBD_final_variant_scores_lib40_41.csv",
     "site", ["huACE2", "Ra787ACE2", "Ra9479ACE2", "RlanACE2", "RalcACE2"]),
]
# 重复测量列（用于信噪比）。sarbecovirus 文件**没有**重复列 —— 见 §SNR。
REPS = {("animalACE2", "bind_deer"): ("deer_rep1", "deer_rep2"),
        ("animalACE2", "bind_hamster"): ("hamster_rep1", "hamster_rep2"),
        ("animalACE2", "bind_bat"): ("bat_rep1", "bat_rep2"),
        ("animalACE2", "bind_cat"): ("cat_rep1", "cat_rep2")}
WINDOWS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00)


# ------------------------------------------------------------------ 自检
def _synth_censored(rng, q, sigma, nr=200, nc=20):
    """合成一个 site x residue 的「真实机制」截尾矩阵。

    真实机制不是「丢掉下限以下的点」（docs/13 §2 的 F 项自检承认这一点
    太弱），而是**把下限以下的真值压平到下限上**，再叠一层测量噪声
    把这个堆积摊开 —— 摊开正是旧检测器失效的原因。
    """
    a = rng.normal(0, 0.8, nr)
    b = rng.normal(0, 0.6, nc)
    v = (a[:, None] + b[None, :] + rng.normal(0, 0.3, (nr, nc))
         + 7.0).ravel()
    if q > 0:
        thr = np.quantile(v, q)
        v = np.where(v < thr, thr, v)
    if sigma > 0:
        v = v + rng.normal(0, sigma, v.shape)
    return v


def censor_selftest(rng, verbose=True, window=CENSOR_WINDOW, reps=6):
    """双向标定（CLAUDE.md §10.2）：既测能命中，也测不误伤，并划定适用域。

    **不是**简单的「通过/不通过」—— 第一次跑这张网格就暴露了修正检测器
    本身的适用域边界，所以判据必须按适用域写，否则就是把一个会失效的工具
    当万能工具用（这正是 D-007「过度敏感的护栏最终会被绕过」的反面）。

    固定窗口 w 对「堆积被 SD = sigma 的噪声摊开」的情形：
      最小值会被摊到堆积中心以下约 3.5 sigma 处（样本量 ~2000-4000 时的
      高斯极值），所以 w 必须覆盖这段，即 **w >~ 3.5 sigma**；
      否则窗口整段落在堆积下方，检测值塌回 ~0（静默欠检）。
    第一次跑这张网格还暴露了相反方向的偏：**sigma = 0（堆积被硬赋成同一个
    值）时窗口会额外吃掉下限上方 w 以内的连续质量，检测值系统性偏高**。
    Starr 的 ACE2 系列正是 sigma ~ 0 的情形，所以这条偏不是理论顾虑。

    判据按三个口径分开，每条都要能单独读：
      (1) **不误伤（无条件）**：q = 0 的干净矩阵在**任何** sigma 下 < 3%。
      (2) **序（本次复核真正依赖的性质）**：适用域内，测得值随 q 单调不降，
          且在整张网格上 Spearman(q, 测得) >= 0.9。
          相关系数与阈值分层只需要序，不需要绝对标度。
      (3) **绝对标度（只在窄带内成立）**：0 < sigma <= 0.7 w / 3.5 时
          |测 - q| <= 8pp。sigma = 0 单列：只要求**保守偏高**
          （q <= 测 <= q + 15pp），不要求无偏。
    适用域外（3.5 sigma > w）只记录、不计入通过判定，但必须打印 ——
    它决定了本次复核能对哪些矩阵下绝对数值的结论。
    """
    rows = []
    hi_sigma = 0.7 * window / 3.5
    ok_fp = ok_mono = ok_abs = ok_bias = True
    qs = (0.0, 0.10, 0.25, 0.40, 0.50)
    grid_q, grid_m = [], []
    for sigma in (0.0, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20):
        in_domain = 3.5 * sigma <= window + 1e-9
        abs_band = 0.0 < sigma <= hi_sigma + 1e-9
        prev = -1.0
        for q in qs:
            m = float(np.mean([censor_fraction(_synth_censored(rng, q, sigma))
                               for _ in range(reps)]))
            if in_domain:
                grid_q.append(q)
                grid_m.append(m)
            if q == 0.0:
                good, crit, kind = m < 0.03, "<3%（不误伤）", "fp"
            elif abs_band:
                good, crit, kind = (abs(m - q) <= 0.08,
                                    f"|.-{q:.0%}|<=8pp", "abs")
            elif sigma == 0.0:
                good, crit, kind = (q - 1e-9 <= m <= q + 0.15,
                                    f"[{q:.0%},{q + 0.15:.0%}] 保守偏高",
                                    "bias")
            else:
                good, crit, kind = (abs(m - q) <= 0.08,
                                    "适用域外（只记录）", "note")
            if kind == "fp":
                ok_fp &= good
            elif kind == "abs":
                ok_abs &= good
            elif kind == "bias":
                ok_bias &= good
            if in_domain:
                ok_mono &= m >= prev - 1e-9
            prev = m
            rows.append({"sigma": sigma, "imposed": q, "measured": m,
                         "in_domain": bool(in_domain), "kind": kind,
                         "crit": crit, "pass": bool(good)})
    rho = spearman(grid_q, grid_m)
    ok_ord = ok_mono and rho >= 0.90
    ok = ok_fp and ok_ord and ok_abs and ok_bias
    if verbose:
        print("=" * 90)
        print(f"截尾检测器双向标定  窗口 w = min + {window}（原生单位）")
        print(f"适用域 3.5σ <= w => σ <= {window / 3.5:.3f}；"
              f"绝对标度窄带 0 < σ <= {hi_sigma:.3f}")
        print("=" * 90)
        print(f"{'摊开噪声σ':>10}{'3.5σ':>7}{'域内':>6}{'施加':>7}"
              f"{'测得':>8}{'判据':>24}{'':>4}")
        for r in rows:
            print(f"{r['sigma']:>10.2f}{3.5 * r['sigma']:>7.2f}"
                  f"{('是' if r['in_domain'] else '否'):>6}"
                  f"{r['imposed']:>7.0%}{r['measured']:>8.1%}"
                  f"{r['crit']:>24}"
                  f"{('  OK' if r['pass'] else '  <<'):>4}")
        print(f"\n(1) 不误伤（q=0，全 σ < 3%）              : "
              f"{'通过' if ok_fp else '** 未通过 **'}")
        print(f"(2) 序（域内单调 + Spearman {rho:+.3f} >= 0.90）: "
              f"{'通过' if ok_ord else '** 未通过 **'}")
        print(f"(3) 绝对标度（0 < σ <= {hi_sigma:.2f} 时 ±8pp）     : "
              f"{'通过' if ok_abs else '** 未通过 **'}")
        print(f"(4) σ = 0 时保守偏高（q .. q+15pp）        : "
              f"{'通过' if ok_bias else '** 未通过 **'}")
        print(f"双向标定: {'通过' if ok else '** 未通过 —— 中止 **'}")
        print("\n标定暴露的两个边界，必须随每个数字一起报：")
        print(f"   a) σ ≈ 0（堆积被**硬赋**成同一个值 —— Starr 的 ACE2 系列、"
              f"也就是本次 34 个单元，正是这种）时检测值**保守偏高**"
              f"约 +7..+10pp。")
        print(f"   b) σ >~ w/3.5 = {window / 3.5:.2f} 时窗口整段落在堆积下方，"
              f"检测值塌回 ~0 —— 静默欠检。")
        print("   => 本次复核的相关系数与分层只用**序**（判据 2 覆盖），"
              "绝对阈值陈述必须带 (a) 的 +10pp 偏作为敏感性。")
    return ok, rows, {"spearman_grid": rho, "hi_sigma": hi_sigma,
                      "ok_fp": ok_fp, "ok_ord": ok_ord,
                      "ok_abs": ok_abs, "ok_bias": ok_bias}


def window_needed_scan(rng, verbose=True):
    """适用域的另一面：给定摊开噪声 σ，多宽的窗口才测得准。

    这是对「窗口宽度」这一个自由参数的标定，不是又一个检测器。
    """
    rows = []
    if verbose:
        print("\n" + "=" * 86)
        print("窗口宽度 x 摊开噪声：施加 40% 截尾时的测得值（选窗的标定）")
        print("=" * 86)
        print("σ\\w   " + "".join(f"{w:>8.2f}" for w in WINDOWS))
    for sigma in (0.0, 0.05, 0.10, 0.20):
        line = []
        for w in WINDOWS:
            v = [_synth_censored(rng, 0.40, sigma) for _ in range(4)]
            m = float(np.mean([max((x <= x.min() + 1e-6).mean(),
                                   (x <= x.min() + w).mean()) for x in v]))
            line.append(m)
            rows.append({"sigma": sigma, "window": w, "measured": m})
        if verbose:
            print(f"{sigma:>5.2f} " + "".join(f"{m:>8.1%}" for m in line))
    if verbose:
        print("读法：对角线以上塌回 ~0（窗口不够宽）；左上角偏高（窗口吃到"
              "连续质量）。w = 0.25 只适合 σ <~ 0.07。")
    return rows


# ------------------------------------------------- 单元重建（照抄原流程）
def build_units(seeds, max_rank=3, quick=False):
    units = []
    for tag, fn, skey, cols in JOBS:
        path = os.path.join(DMS, fn)
        rows = list(csv.DictReader(open(path)))
        bgs = sorted({r["target"] for r in rows})
        for bg in bgs:
            for col in cols:
                u = _one(tag, bg, col, skey, rows, seeds, max_rank, quick)
                if u is not None:
                    units.append(u)
    return units


def _one(tag, bg, col, skey, rows, seeds, max_rank, quick):
    vals = []
    for r in rows:
        if r.get("target") != bg:
            continue
        v = r.get(col, "")
        if v in ("", "NA"):
            continue
        vals.append((r[skey], r["mutant"], float(v)))
    if not vals:
        return None
    arr = np.array([v for _, _, v in vals])
    lo = arr.min()
    cens = int((arr <= lo + 1e-6).sum())
    keep = [(s, a, v) for s, a, v in vals if v > lo + 1e-6]
    if len(keep) < 500:                      # 原脚本的「观测不足」
        return None
    sites = sorted({s for s, _, _ in keep})
    si = {s: i for i, s in enumerate(sites)}
    M = np.full((len(sites), 20), np.nan)
    for s, a, v in keep:
        if a in AA20:
            M[si[s], AA20.index(a)] = v
    mask = np.isfinite(M)
    if mask.sum() < 500 or len(sites) < 30:  # 原脚本的「矩阵太小」
        return None
    n = int(mask.sum())
    u = {"dataset": tag, "bg": bg, "lig": col, "sites": len(sites), "n": n,
         "cens_count": cens,
         # 旧检测器的比例：恰好等于最小值的格子 / 全部原始观测。
         # cens/(cens+n) 是 docs/12 §2 表的口径（已复现 43.2% / 10.9%）。
         "old_frac": cens / (cens + n),
         "old_frac_raw": cens / len(vals),
         "new_frac": censor_fraction(arr),
         # 堆积锐度 = 精确并列 / 窗口内比例。接近 1 = 堆积被**硬赋**成同一个
         # 值（摊开噪声 σ ≈ 0，落在修正检测器的适用域内，偏差方向是偏高）；
         # 接近 0 = 堆积被噪声摊开（PD-CoV 的 hAPN/pAPN 是 0.001）。
         "sharp": (cens / len(vals)) / max(censor_fraction(arr), 1e-9),
         "win_scan": {f"{w:.2f}": float(max((arr <= lo + 1e-6).mean(),
                                            (arr <= lo + w).mean()))
                      for w in WINDOWS}}
    # 秩：已验证的无泄露流程（rank_core），多种子取众数
    if not quick:
        bests, curves = [], []
        for sd in seeds:
            errs = rank_core.cv_rank_noleak(M, mask, max_rank=max_rank,
                                            K=5, restarts=2, seed=sd,
                                            iters=150, ridge=0.1)
            b, trim = rank_core.best_rank(errs)
            bests.append(int(b))
            curves.append({int(k): float(v) for k, v in trim.items()})
        vals_, cnt = np.unique(bests, return_counts=True)
        u["rank_noleak"] = int(vals_[np.argmax(cnt)])
        u["rank_noleak_seeds"] = bests
        u["rank_noleak_stab"] = float(cnt.max() / len(bests))
        u["trim_curves"] = curves
    return u


def attach_cached_rank(units):
    """原始秩（docs/12 §2 的因变量）来自 cache/ligand_rank.json。"""
    cached = json.load(open(os.path.join(ROOT, "cache", "ligand_rank.json")))
    key = {(c["dataset"], c["bg"], c["lig"]): c for c in cached
           if "skip" not in c or c.get("skip") is None}
    miss = []
    for u in units:
        c = key.get((u["dataset"], u["bg"], u["lig"]))
        if c is None:
            miss.append(u)
            continue
        u["rank_orig"] = int(c["best_mode"])
        u["gain_orig"] = float(c["gain"])
        u["cens_count_cached"] = int(c["cens"])
        u["n_cached"] = int(c["n"])
    return miss


# ------------------------------------------------------------------ 信噪比
def attach_snr(units):
    """重复噪声 = 两重复差的 SD / sqrt(2)；信号 = 双向中心化残差 SD。

    与 docs/12 §2 同一口径（那张表是 PD-CoV 的）。sarbecovirus 文件没有
    重复列，animalACE2 的 human_monomer 也没有 —— 所以 34 个单元里
    只有 12 个可算信噪比。这是本次复核里信噪比一侧的硬限制，如实记录。
    """
    cache = {}
    for u in units:
        k = (u["dataset"], u["lig"])
        if k not in REPS:
            u["snr"] = None
            continue
        fn = dict((t, f) for t, f, _, _ in JOBS)[u["dataset"]]
        skey = dict((t, s) for t, _, s, _ in JOBS)[u["dataset"]]
        if fn not in cache:
            cache[fn] = list(csv.DictReader(
                open(os.path.join(DMS, fn))))
        r1c, r2c = REPS[k]
        d, kept = [], []
        for r in cache[fn]:
            if r.get("target") != u["bg"]:
                continue
            a, b = r.get(r1c, ""), r.get(r2c, "")
            v = r.get(u["lig"], "")
            if v in ("", "NA"):
                continue
            if a not in ("", "NA") and b not in ("", "NA"):
                d.append(float(a) - float(b))
            kept.append((r[skey], r["mutant"], float(v)))
        if len(d) < 50:
            u["snr"] = None
            continue
        noise = float(np.std(d, ddof=1) / math.sqrt(2))
        # 信号：与定秩同一个矩阵（剔除精确下限后）的双向中心化残差 SD
        arr = np.array([v for _, _, v in kept])
        lo = arr.min()
        keep = [(s, a, v) for s, a, v in kept if v > lo + 1e-6]
        sites = sorted({s for s, _, _ in keep})
        si = {s: i for i, s in enumerate(sites)}
        M = np.full((len(sites), 20), np.nan)
        for s, a, v in keep:
            if a in AA20:
                M[si[s], AA20.index(a)] = v
        mask = np.isfinite(M)
        R = rank_core.residual(M, mask)
        u["rep_noise"] = noise
        u["coup_sd"] = float(R[mask].std())
        u["snr"] = float(R[mask].std() / noise) if noise > 0 else None


# ------------------------------------------------------------- 统计
def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    def rk(a):
        a = np.asarray(a, float)
        o = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), float)
        r[o] = np.arange(len(a), dtype=float)
        # 平均秩处理并列
        for v in np.unique(a):
            m = a == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    return pearson(rk(x), rk(y))


def boot_corr(x, y, nboot, rng, kind="pearson"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    f = pearson if kind == "pearson" else spearman
    out = []
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        v = f(x[i], y[i])
        if np.isfinite(v):
            out.append(v)
    out = np.array(out)
    return {"point": f(x, y), "n": n, "nboot": len(out),
            "lo": float(np.percentile(out, 2.5)),
            "hi": float(np.percentile(out, 97.5)),
            "median": float(np.median(out))}


def part_corr(x, y, z):
    """x 与 y 的偏相关，控制 z（线性残差法）。"""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    Z = np.column_stack([np.ones(len(z)), z])

    def res(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta
    return pearson(res(x), res(y))


def thresholds(units, fkey, rkey):
    lowf = [u for u in units if u[fkey] < 0.10]
    high = [u for u in units if u[fkey] >= 0.25]
    r2 = sum(1 for u in lowf if u[rkey] == 2)
    r0 = sum(1 for u in high if u[rkey] == 0)
    return {"n_low": len(lowf), "low_rank2": r2,
            "low_rank2_frac": r2 / len(lowf) if lowf else float("nan"),
            "n_high": len(high), "high_rank0": r0,
            "high_rank0_frac": r0 / len(high) if high else float("nan")}


def by_rank(units, fkey, rkey):
    out = {}
    for rk in sorted({u[rkey] for u in units}):
        sub = [u for u in units if u[rkey] == rk]
        out[int(rk)] = {"n": len(sub),
                        "median_frac": float(np.median([u[fkey]
                                                        for u in sub]))}
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true",
                    help="跳过无泄露定秩重算，只用缓存秩")
    ap.add_argument("--reuse-noleak", action="store_true",
                    help="从 cache/censor_recheck.json 复用上一次算好的无泄露秩"
                         "（纯重放，不改变任何数值；用于只重跑窗口扫描）")
    a = ap.parse_args()
    rng = np.random.default_rng(20261011)

    # 0) 工具自检 —— 不通过就中止（CLAUDE.md §10）
    rank_core.selfcheck(verbose=False)
    print("rank_core.selfcheck(): 通过\n")
    ok, cal, calmeta = censor_selftest(rng)
    if not ok:
        print("截尾检测器标定未通过 —— 中止，不看任何结论")
        return 1
    wscan = window_needed_scan(rng)
    print()

    # 1) 重建单元
    seeds = tuple(range(1, a.seeds + 1))
    quick = a.quick or a.reuse_noleak
    units = build_units(seeds, quick=quick)
    if a.reuse_noleak:
        prev = json.load(open(os.path.join(ROOT, "cache",
                                           "censor_recheck.json")))
        pk = {(u["dataset"], u["bg"], u["lig"]): u for u in prev["units"]}
        nmiss = 0
        for u in units:
            p0 = pk.get((u["dataset"], u["bg"], u["lig"]), {})
            if "rank_noleak" in p0:
                for f in ("rank_noleak", "rank_noleak_seeds",
                          "rank_noleak_stab", "trim_curves"):
                    u[f] = p0[f]
            else:
                nmiss += 1
        if nmiss:
            print(f"** {nmiss} 个单元在缓存里没有无泄露秩 —— 中止")
            return 1
        a.quick = False
    miss = attach_cached_rank(units)
    attach_snr(units)
    print(f"重建单元数 = {len(units)}（docs/12 §2 说 34）"
          f"；缓存里找不到秩的 = {len(miss)}")
    units = [u for u in units if "rank_orig" in u]

    # 1b) 重建核对：必须复现 docs/12 §2 的派生数字
    old_by_rank = by_rank(units, "old_frac", "rank_orig")
    old_thr = thresholds(units, "old_frac", "rank_orig")
    checks = [
        ("秩0 的旧截尾中位 = 43.2%",
         abs(old_by_rank.get(0, {}).get("median_frac", 0) - 0.432) < 0.003),
        ("秩1 的旧截尾中位 = 10.9%",
         abs(old_by_rank.get(1, {}).get("median_frac", 0) - 0.109) < 0.003),
        ("<10% 的组合数 = 21", old_thr["n_low"] == 21),
        ("<10% 里秩2 占 76%", abs(old_thr["low_rank2_frac"] - 0.76) < 0.01),
        (">=25% 的组合数 = 7", old_thr["n_high"] == 7),
        (">=25% 里 7/7 给秩 0", old_thr["high_rank0"] == 7),
    ]
    print("\n重建核对（对照 docs/12 §2 的已发表派生量）")
    allok = True
    for nm, good in checks:
        print(f"   {'OK ' if good else '<< 不符'} {nm}")
        allok &= good
    print(f"   => 单元集重建{'精确吻合' if allok else '**未吻合**'}")

    # 2) 表：旧 vs 新
    print("\n" + "=" * 100)
    print("每个单元：旧截尾（恰好等于最小值） vs 新截尾（docs/13 修正检测器）")
    print("=" * 100)
    print(f"{'数据集':<14}{'骨架':<20}{'配体':<14}{'位点':>5}{'观测':>6}"
          f"{'旧截尾':>8}{'新截尾':>8}{'差pp':>7}{'堆积锐度':>9}"
          f"{'原秩':>5}{'无泄露秩':>9}{'信噪比':>8}")
    print("-" * 100)
    for u in sorted(units, key=lambda z: -z["new_frac"]):
        rn = u.get("rank_noleak", "")
        sn = f"{u['snr']:.2f}" if u.get("snr") else "—"
        print(f"{u['dataset']:<14}{u['bg'][:19]:<20}{u['lig'][:13]:<14}"
              f"{u['sites']:>5}{u['n']:>6}{u['old_frac']:>8.1%}"
              f"{u['new_frac']:>8.1%}"
              f"{100 * (u['new_frac'] - u['old_frac']):>+7.1f}"
              f"{u['sharp']:>9.2f}"
              f"{u['rank_orig']:>5}{str(rn):>9}{sn:>8}")

    oldf = [u["old_frac"] for u in units]
    newf = [u["new_frac"] for u in units]
    r_o = [u["rank_orig"] for u in units]
    res = {"n_units": len(units), "calibration": cal,
           "calibration_meta": calmeta, "window_needed": wscan,
           "rebuild_checks": [(n, bool(g)) for n, g in checks],
           "units": units}

    # 3) 相关 + bootstrap
    print("\n" + "=" * 100)
    print(f"相关与 bootstrap 区间（{a.boot} 次重抽，CLAUDE.md §5）")
    print("=" * 100)
    combos = [("旧截尾 vs 原秩（复刻 docs/12 的 -0.786）", oldf, r_o),
              ("新截尾 vs 原秩（**本次主结果**）", newf, r_o)]
    if not a.quick:
        r_n = [u["rank_noleak"] for u in units]
        combos += [("旧截尾 vs 无泄露秩", oldf, r_n),
                   ("新截尾 vs 无泄露秩", newf, r_n)]
    res["corr"] = {}
    for nm, x, y in combos:
        p = boot_corr(x, y, a.boot, rng, "pearson")
        s = boot_corr(x, y, a.boot, rng, "spearman")
        res["corr"][nm] = {"pearson": p, "spearman": s}
        print(f"\n{nm}  (n = {p['n']})")
        print(f"   Pearson  r = {p['point']:+.3f}  "
              f"95% bootstrap [{p['lo']:+.3f}, {p['hi']:+.3f}]")
        print(f"   Spearman ρ = {s['point']:+.3f}  "
              f"95% bootstrap [{s['lo']:+.3f}, {s['hi']:+.3f}]")

    # 3b) 窗口宽度敏感性
    print("\n" + "=" * 100)
    print("敏感性：窗口宽度 -> 相关系数与阈值陈述")
    print("（w >= 0.70 在 log10 Ka 上已吃进真信号，只作为上界敏感性，不是候选口径）")
    print("=" * 100)
    print(f"{'窗口':>8}{'中位截尾':>10}{'Pearson':>10}{'Spearman':>10}"
          f"{'<10% n':>8}{'其中秩2':>9}{'>=25% n':>9}{'其中秩0':>9}")
    res["window_scan"] = {}
    for w in WINDOWS:
        k = f"{w:.2f}"
        f_ = [u["win_scan"][k] for u in units]
        for u in units:
            u["_tmp"] = u["win_scan"][k]
        t = thresholds(units, "_tmp", "rank_orig")
        pr, sp = pearson(f_, r_o), spearman(f_, r_o)
        res["window_scan"][k] = {"pearson": pr, "spearman": sp, **t}
        print(f"{w:>8.2f}{np.median(f_):>10.1%}{pr:>+10.3f}{sp:>+10.3f}"
              f"{t['n_low']:>8}{t['low_rank2_frac']:>9.0%}"
              f"{t['n_high']:>9}{t['high_rank0_frac']:>9.0%}")
    for u in units:
        u.pop("_tmp", None)

    # 4) 阈值陈述与分层
    print("\n" + "=" * 100)
    print("阈值陈述（docs/12 §2）：旧 vs 新")
    print("=" * 100)
    for lbl, fk in (("旧", "old_frac"), ("新", "new_frac")):
        t = thresholds(units, fk, "rank_orig")
        br = by_rank(units, fk, "rank_orig")
        res[f"thr_{fk}"] = t
        res[f"byrank_{fk}"] = br
        print(f"\n[{lbl}截尾]  <10%: {t['n_low']} 个组合, "
              f"{t['low_rank2']}/{t['n_low']} = {t['low_rank2_frac']:.0%} 给秩 2"
              f"   |   >=25%: {t['n_high']} 个, "
              f"{t['high_rank0']}/{t['n_high']} = "
              f"{t['high_rank0_frac']:.0%} 给秩 0")
        print("   检出秩 -> 组合数 / 截尾中位: " +
              "  ".join(f"秩{k}: {v['n']} / {v['median_frac']:.1%}"
                        for k, v in sorted(br.items())))
    if not a.quick:
        for lbl, fk in (("旧", "old_frac"), ("新", "new_frac")):
            t = thresholds(units, fk, "rank_noleak")
            res[f"thr_noleak_{fk}"] = t
            print(f"\n[{lbl}截尾 x 无泄露秩]  <10%: {t['low_rank2']}/"
                  f"{t['n_low']} = {t['low_rank2_frac']:.0%} 秩2   |   "
                  f">=25%: {t['high_rank0']}/{t['n_high']} = "
                  f"{t['high_rank0_frac']:.0%} 秩0")

    # 5) 信噪比 vs 截尾
    print("\n" + "=" * 100)
    print("信噪比 vs 截尾：谁更能预测检出的秩")
    print("=" * 100)
    sub = [u for u in units if u.get("snr")]
    print(f"可算信噪比的单元 = {len(sub)} / {len(units)}"
          f"（sarbecovirus 文件无重复列，animalACE2 的 human_monomer 无重复列）")
    res["snr_subset"] = {"n": len(sub)}
    if len(sub) >= 6:
        s_ = [u["snr"] for u in sub]
        c_ = [u["new_frac"] for u in sub]
        co_ = [u["old_frac"] for u in sub]
        rr = [u["rank_orig"] for u in sub]
        for nm, x in (("信噪比 vs 秩", s_), ("新截尾 vs 秩", c_),
                      ("旧截尾 vs 秩", co_)):
            p = boot_corr(x, rr, a.boot, rng, "spearman")
            res["snr_subset"][nm] = p
            print(f"   {nm:<16} Spearman ρ = {p['point']:+.3f}  "
                  f"[{p['lo']:+.3f}, {p['hi']:+.3f}]")
        res["snr_subset"]["corr_snr_censor"] = pearson(s_, c_)
        print(f"   混淆核对：信噪比 与 新截尾 的 Pearson r = "
              f"{pearson(s_, c_):+.3f}（Spearman {spearman(s_, c_):+.3f}）")
        res["snr_subset"]["partial_censor_given_snr"] = part_corr(
            c_, rr, s_)
        res["snr_subset"]["partial_snr_given_censor"] = part_corr(
            s_, rr, c_)
        print(f"   偏相关：截尾|控信噪比 = "
              f"{res['snr_subset']['partial_censor_given_snr']:+.3f}；"
              f"信噪比|控截尾 = "
              f"{res['snr_subset']['partial_snr_given_censor']:+.3f}")
        print(f"   注：这 12 个单元的检出秩只有 {{1,2}}（10 个秩2、2 个秩1），"
              f"新截尾全部在 "
              f"{100 * min(c_):.1f}–{100 * max(c_):.1f}% —— 自变量与因变量"
              f"方差都太小，任何预测量在这里都不可能有判别力；"
              f"记为**不可判**，不记为「信噪比赢了」（docs/19 §6.1）。")

    # 5b) PD-CoV 扩展：信噪比主张的**原始出处**，不在 34 个单元里
    print("\n" + "=" * 100)
    print("PD-CoV 扩展（docs/12 §2 信噪比表的原始出处；**不**计入 34 个单元）")
    print("=" * 100)
    pd_rank = {"gAPN": (2, "秩 2，12/12 种子，真内部极小"),
               "hAPN": (0, "秩 0，CV 误差随秩单调上升"),
               "pAPN": (None, "不可判（平坦非单调，cache/within_rank 给到上限)")}
    pd_snr = {"gAPN": 1.56, "hAPN": 3.57, "pAPN": 3.04}
    pdrows = []
    print(f"{'配体':<8}{'观测':>6}{'旧截尾':>8}{'新截尾':>8}{'差pp':>7}"
          f"{'堆积锐度':>9}{'信噪比':>8}  秩（docs/12 §2 / cache/within_rank）")
    print("-" * 100)
    for lig in ("gAPN", "hAPN", "pAPN"):
        v = []
        for r in csv.DictReader(open(os.path.join(
                DMS, "starr_PDCoV_RBD_final_variant_scores.csv"))):
            x = r.get(f"bind_{lig}", "")
            if x not in ("", "NA"):
                v.append(float(x))
        arr = np.array(v)
        lo = arr.min()
        old = float((arr <= lo + 1e-6).mean())
        new = censor_fraction(arr)
        rk, note = pd_rank[lig]
        pdrows.append({"lig": lig, "n": len(arr), "old_frac": old,
                       "new_frac": new, "snr": pd_snr[lig],
                       "rank": rk, "note": note,
                       "win_scan": {f"{w:.2f}": float(
                           max(old, (arr <= lo + w).mean()))
                           for w in WINDOWS}})
        print(f"{lig:<8}{len(arr):>6}{old:>8.1%}{new:>8.1%}"
              f"{100 * (new - old):>+7.1f}{old / max(new, 1e-9):>9.3f}"
              f"{pd_snr[lig]:>8.2f}  {note}")
    res["pdcov"] = pdrows
    print("\n这三行是整个复核里信噪比与截尾**唯一真正分开**的地方：")
    print("   截尾（新）0.7% / 45.7% / 47.9%  -> 秩 2 / 0 / 不可判   顺序完全一致")
    print("   信噪比      1.56 / 3.57 / 3.04  -> 秩 2 / 0 / 不可判   顺序完全相反")
    print("   旧检测器在这里报 0.04% / 0.04% / 0.04% —— 对秩零信息量。")
    print("   另记：within_system_rank.py 对 hAPN/pAPN 用的硬编码下限 4.0")
    print("   低于实测最小值（4.155 / 4.235），所以**一个截尾点都没剔除**，")
    print("   整个 46-48% 的堆积原样进了定秩矩阵 —— 秩 0 的直接机制。")

    out = os.path.join(ROOT, "cache", "censor_recheck.json")
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(f"\n写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
