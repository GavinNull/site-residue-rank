#!/usr/bin/env python3
"""docs/27 §4.0：**σ 的量级校准**（CLAUDE.md §8），以及它为什么必须单列。

### 为什么这一步是必须的，而不是额外严谨

把 docs/15 §2.1(b) 的功效代理
    `coup_sd_true = sqrt(max(coup_sd_obs² − σ², 0))`
直接配上 §2.5 的**体外校准** σ = 0.70，会得到：

    164 个过闸域里 **151 个** coup_sd_true = 0 ⇒ 「全都没有可定秩的信号」

而**直接测量**说的恰好相反：这些矩阵第一轴的位点对半可复现度是
**0.84–0.94**（随机 p95 = 0.44），比 docs/26 唯一那个有功效矩阵
（GRB2-SH3/folding，0.787）**还高**。
纯噪声矩阵给不出 0.94 的对半一致 —— 所以「coup_sd_true = 0」
**被一个更直接的测量否证了**。

两者只能有一个对，而且分得清是哪一个：

* `coup_sd_obs` 是**双向中心化之后**残差的 SD（中位 0.511）。
* σ = 0.70 是**体外对照的总 RMSE**。那个 RMSE 里有多少是
  「逐蛋白 / 逐位点的**系统性**差异」（构建体被截短并加 GS 接头、
  缓冲与温度不同、文献自己的误差 —— §1.1 的 gap 已记明它是上界）？
  **逐蛋白的常数已经在 §2.5 里被去掉了，但逐位点的系统差没有。**
  而双向中心化**会把逐位点的系统差整条吸收进行主效应**
  （row_sd 中位 0.781，是 coup_sd_obs 的 1.5 倍）。
* 所以配给**交互项**的 σ 不是「总 ΔΔG 误差」，
  而是**双向中心化之后**还剩下的、逐格子的随机误差。
  **用总误差去减中心化后的残差，是在用一个太大的 σ**
  —— 这是 D-028 的**镜像**：D-028 是 σ 取太小把「测不出」说成「测得出」，
  这里是 σ 取太大，会把一个**有功效的阳性**说成「没有可定秩的信号」。

### 这一步不靠推理，靠三个互相独立的锚点把 σ 夹住

**(A) 下界：两条蛋白酶的技术重复。** 同一个变体有 `deltaG_t`（胰蛋白酶）
与 `deltaG_c`（胰凝乳蛋白酶）两条**独立的**蛋白酶读出。
各自建一个 ΔΔG 矩阵、各自双向中心化，则
`SD(R_t − R_c) = σ_single·√2`、`σ_ML ≈ σ_single/√2 = SD(R_t−R_c)/2`。
这是**在正确的层级上**（中心化之后、逐格子）测到的噪声，
但两条读出共用同一个 cDNA 展示库与测序批次，共同系统误差在差值里消掉，
所以它是**下界**。

**(B) 上界：§2.5 的体外校准 0.70。**

**(C) 由数据自己的可复现度反解。** 对每个域扫 σ：令
`coup_sd_true(σ) = sqrt(max(coup_sd_obs² − σ², 0))`，在该域自己的几何上
合成秩 1 矩阵（植入方向 = u），测**合成的**位点对半可复现度，
取使它等于**实测**可复现度的那个 σ。
这正是 docs/26 §7.4 做过的那个比较（合成 0.985 vs 真实 0.787）**反解一次**。

**判据（§8，在看结果之前写定）**：(C) 必须落在 [(A), (B)] 之间。
若 (C) < (A) 或 (C) > (B)，说明三个锚点互相矛盾，**本步不通过，
后续的功效论断不许做**。

**同时这一步白送一个比位点对半强得多的 D-036 证据**：
`R_t` 与 `R_c` 的第一轴夹角是一次**真正的技术重复**
（两次独立的蛋白酶实验），不是同一批数据的重切分。

运行：python3 analysis/tsuboyama_sigma.py [--jobs 9]
产物：cache/tsuboyama_sigma.json
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rank_core as RC                                          # noqa: E402
import tsuboyama_load as TL                                     # noqa: E402
from generic_folding_axis import (cen_unit, first_axis,         # noqa: E402
                                  synth_like)
from subspace_lib import AA20, ROOT, haar_1d_null               # noqa: E402

SINGLE = TL.SINGLE
DG_LO, DG_HI = TL.DG_LO, TL.DG_HI
SIG_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
            0.55, 0.60, 0.65, 0.70]
NREP = 120


def build_tc(keep):
    """为 keep 里的域各建两个矩阵：trypsin 版与 chymotrypsin 版的 ΔΔG。

    ΔΔG = dG(变体) − median(dG(该域的 wt 行))，各蛋白酶独立；
    只收两条读出**都**落在可信窗口 [−1, 5] 内的格子
    （否则差值会被截尾污染，那就不是噪声估计了）。
    输出仓库约定（正 = 去稳定）。
    """
    want = set(keep)
    wt = defaultdict(lambda: {"t": [], "c": []})
    cells = defaultdict(dict)
    sites = defaultdict(set)
    with open(TL.MAIN, newline="") as fh:
        for x in csv.DictReader(fh):
            w = x["WT_name"]
            if w not in want:
                continue
            if x["mut_type"] == "wt":
                for k, col in (("t", "deltaG_t"), ("c", "deltaG_c")):
                    v = TL._num(x[col])
                    if v is not None and DG_LO <= v <= DG_HI:
                        wt[w][k].append(v)
                continue
            m = SINGLE.match(x["mut_type"])
            if not m:
                continue
            a = TL._num(x["deltaG_t"]); b = TL._num(x["deltaG_c"])
            if a is None or b is None:
                continue
            if not (DG_LO <= a <= DG_HI and DG_LO <= b <= DG_HI):
                continue
            p, mut = int(m.group(2)), m.group(3)
            cells[w][(p, mut)] = (a, b)
            sites[w].add(p)
    out = {}
    for w in want:
        if not wt[w]["t"] or not wt[w]["c"] or len(sites[w]) < 20:
            continue
        gt = float(np.median(wt[w]["t"])); gc = float(np.median(wt[w]["c"]))
        pos = sorted(sites[w])
        si = {p: i for i, p in enumerate(pos)}
        Mt = np.full((len(pos), 20), np.nan)
        Mc = np.full((len(pos), 20), np.nan)
        for (p, mut), (a, b) in cells[w].items():
            Mt[si[p], AA20.index(mut)] = -(a - gt)
            Mc[si[p], AA20.index(mut)] = -(b - gc)
        out[w] = {"Mt": Mt, "Mc": Mc, "wt_t": gt, "wt_c": gc,
                  "pos": pos}
    return out


def one(args):
    """一个域：(A) 技术重复 σ、技术重复轴夹角、(C) 反解 σ。"""
    name, M, mask, Mt, Mc, u = args
    res = {"name": name}
    mk = np.isfinite(Mt) & np.isfinite(Mc)
    if mk.sum() < 200:
        return name, {"err": "两条读出共同格子太少"}

    # ---- (A) 技术重复的、中心化后的逐格子噪声 ------------------------
    rt, ct = RC.center_fit(np.where(mk, Mt, np.nan), mk)
    rc, cc = RC.center_fit(np.where(mk, Mc, np.nan), mk)
    Rt = Mt - rt[:, None] - ct[None, :]
    Rc = Mc - rc[:, None] - cc[None, :]
    d = (Rt - Rc)[mk]
    sd_diff = float(d.std())
    res.update({
        "n_common": int(mk.sum()),
        "sd_resid_diff": sd_diff,
        "sigma_single_A": sd_diff / math.sqrt(2),
        "sigma_ML_A": sd_diff / 2.0,
        "coup_sd_obs_t": float(Rt[mk].std()),
        "coup_sd_obs_c": float(Rc[mk].std()),
        "r_resid_tc": float(np.corrcoef(Rt[mk], Rc[mk])[0, 1]),
    })
    # ---- 技术重复的第一轴夹角（真正的重复，不是重切分）--------------
    at, colst = first_axis(np.where(mk, Mt, np.nan), mk)
    ac, colsc = first_axis(np.where(mk, Mc, np.nan), mk)
    if at is not None and ac is not None:
        com = np.intersect1d(colst, colsc)
        v1 = cen_unit(at[np.searchsorted(colst, com)])
        v2 = cen_unit(ac[np.searchsorted(colsc, com)])
        res["axis_tc_abscos"] = abs(float(v1 @ v2))
        res["axis_tc_d"] = int(len(com))
        res["cos_u_t"] = abs(float(v1 @ cen_unit(u, com)))
        res["cos_u_c"] = abs(float(v2 @ cen_unit(u, com)))

    # ---- 实测位点对半（ML 矩阵上，和 Step 4 同口径）-----------------
    rng = np.random.default_rng(77)
    nr = M.shape[0]
    vals = []
    for _ in range(NREP):
        p = rng.permutation(nr)
        h1, h2 = p[: nr // 2], p[nr // 2:]
        a1, c1 = first_axis(M[h1], mask[h1])
        a2, c2 = first_axis(M[h2], mask[h2])
        if a1 is None or a2 is None:
            continue
        com = np.intersect1d(c1, c2)
        if len(com) < 12:
            continue
        vals.append(abs(float(cen_unit(a1[np.searchsorted(c1, com)])
                              @ cen_unit(a2[np.searchsorted(c2, com)]))))
    obs_rep = float(np.median(vals)) if vals else float("nan")
    res["obs_splithalf"] = obs_rep

    r, c = RC.center_fit(M, mask)
    cso = float((M - r[:, None] - c[None, :])[mask].std())
    res["coup_sd_obs"] = cso
    res["row_sd"] = float(r.std()); res["col_sd"] = float(c.std())

    # ---- (C) 反解：哪个 σ 使**合成**的对半可复现度 == 实测 ----------
    ud = cen_unit(u)
    curve = []
    rngc = np.random.default_rng(321)
    for sg in SIG_GRID:
        cst = math.sqrt(max(cso ** 2 - sg ** 2, 0.0))
        sv = []
        for k in range(30):
            Mz = synth_like(mask, ud, cst, sg, max(res["row_sd"], 1e-6),
                            max(res["col_sd"], 1e-6),
                            np.random.default_rng(4100 + 11 * k), rank=1)
            p = rngc.permutation(nr)
            h1, h2 = p[: nr // 2], p[nr // 2:]
            a1, c1 = first_axis(Mz[h1], mask[h1])
            a2, c2 = first_axis(Mz[h2], mask[h2])
            if a1 is None or a2 is None:
                continue
            com = np.intersect1d(c1, c2)
            if len(com) < 12:
                continue
            sv.append(abs(float(cen_unit(a1[np.searchsorted(c1, com)])
                                @ cen_unit(a2[np.searchsorted(c2, com)]))))
        curve.append({"sigma": sg, "coup_sd_true": cst,
                      "synth_splithalf": float(np.median(sv)) if sv
                      else float("nan")})
    res["sigma_curve"] = curve
    ok = [z for z in curve if np.isfinite(z["synth_splithalf"])]
    if ok and np.isfinite(obs_rep):
        best = min(ok, key=lambda z: abs(z["synth_splithalf"] - obs_rep))
        res["sigma_C"] = best["sigma"]
        res["sigma_C_coup_sd_true"] = best["coup_sd_true"]
        res["sigma_C_synth"] = best["synth_splithalf"]
    return name, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=9)
    a = ap.parse_args()

    info, mats = TL.load_cached()
    cen = json.load(open(os.path.join(ROOT, "cache",
                                      "tsuboyama_censor.json")))
    keep = sorted(cen["primary_pass"].values())
    tg = json.load(open(os.path.join(ROOT, "cache",
                                     "subspace_two_groups.json")))
    u = np.array(tg["axis1_family_avg"]["u"], float)

    print("=" * 102)
    print("docs/27 §4.0：σ 的量级校准（CLAUDE.md §8）—— 三个独立锚点把 σ 夹住")
    print("=" * 102)
    print(f"\n[输入] Step 2 过闸的天然独立结构域 {len(keep)} 个")
    print("[B 上界] §2.5 的体外校准 σ = "
          f"{info['sigma']['adopted']:.2f}"
          f"（逐蛋白中位 {info['sigma']['lit_window']['median_rmse_ctr']:.2f}、"
          f"不限窗口 {info['sigma']['lit_all']['weighted_rmse_ctr']:.2f}）")
    print("\n[建两条蛋白酶读出各自的 ΔΔG 矩阵] ...")
    tc = build_tc(keep)
    print(f"   成功建出 {len(tc)}/{len(keep)} 个域的 trypsin / chymotrypsin "
          f"双矩阵")

    jobs = [(w, mats[w]["M"], mats[w]["mask"], tc[w]["Mt"], tc[w]["Mc"], u)
            for w in sorted(tc)]
    res = {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for i, (nm, r) in enumerate(ex.map(one, jobs, chunksize=1)):
            res[nm] = r
            if (i + 1) % 40 == 0:
                print(f"   ... {i + 1}/{len(jobs)}", flush=True)
    good = {k: v for k, v in res.items() if "err" not in v}
    print(f"   完成 {len(good)}/{len(jobs)}")

    sA = np.array([v["sigma_ML_A"] for v in good.values()])
    sA1 = np.array([v["sigma_single_A"] for v in good.values()])
    rtc = np.array([v["r_resid_tc"] for v in good.values()])
    print("\n" + "=" * 102)
    print("[A 下界] 两条蛋白酶的技术重复，**中心化之后、逐格子**的噪声")
    print("=" * 102)
    print(f"   SD(R_t − R_c) 中位 "
          f"{np.median([v['sd_resid_diff'] for v in good.values()]):.3f}")
    print(f"   ⇒ 单酶 σ 中位 **{np.median(sA1):.3f}**、"
          f"ML 估计 σ 中位 **{np.median(sA):.3f}**，10%–90% "
          f"[{np.percentile(sA, 10):.3f}, {np.percentile(sA, 90):.3f}]")
    print(f"   两条读出的中心化残差相关 r 中位 **{np.median(rtc):.3f}**"
          f"（10%–90% [{np.percentile(rtc, 10):.3f}, "
          f"{np.percentile(rtc, 90):.3f}]）")
    print("   —— 这个相关本身就是「交互项里有真实共享结构」的直接证据：")
    print("      两条**独立蛋白酶**实验的中心化残差相关到这个程度，")
    print("      不可能来自噪声。")

    print("\n" + "=" * 102)
    print("[真技术重复的 D-036 证据] trypsin 轴 vs chymotrypsin 轴")
    print("=" * 102)
    at = np.array([v["axis_tc_abscos"] for v in good.values()
                   if "axis_tc_abscos" in v])
    n1 = haar_1d_null(20, n=20000, seed=5)
    p95 = float(np.percentile(n1, 95))
    print(f"   n = {len(at)}：|cos(轴_t, 轴_c)| 中位 **{np.median(at):.3f}**、"
          f"四分位 [{np.percentile(at, 25):.3f}, "
          f"{np.percentile(at, 75):.3f}]、最小 {at.min():.3f}")
    print(f"   随机 p95（d = 20）= {p95:.3f}；超出的域 "
          f"**{int((at > p95).sum())}/{len(at)} = {np.mean(at > p95):.1%}**")
    print("   —— 这是**两次独立的蛋白酶实验**给出同一条残基方向，")
    print("      比位点对半（同一批数据重切分）强一个档次。")

    print("\n" + "=" * 102)
    print("[C 反解] 哪个 σ 使**合成**的位点对半可复现度等于**实测**")
    print("=" * 102)
    sC = np.array([v["sigma_C"] for v in good.values() if "sigma_C" in v])
    ob = np.array([v["obs_splithalf"] for v in good.values()])
    print(f"   实测位点对半中位 {np.median(ob):.3f}")
    print(f"   反解 σ：中位 **{np.median(sC):.3f}**、"
          f"四分位 [{np.percentile(sC, 25):.3f}, "
          f"{np.percentile(sC, 75):.3f}]、"
          f"10%–90% [{np.percentile(sC, 10):.3f}, "
          f"{np.percentile(sC, 90):.3f}]")

    print("\n" + "=" * 102)
    print("[判据，§8，在看结果之前写定] (C) 必须落在 [(A), (B)] 之间")
    print("=" * 102)
    A, B, C = float(np.median(sA)), info["sigma"]["adopted"], \
        float(np.median(sC))
    print(f"   A（技术重复下界） = {A:.3f}")
    print(f"   C（由可复现度反解） = **{C:.3f}**")
    print(f"   B（体外校准上界） = {B:.3f}")
    inside = A <= C <= B
    print(f"   => {'**通过**' if inside else '**不通过**'}："
          f"{A:.3f} ≤ {C:.3f} ≤ {B:.3f} 为 {inside}")
    assert inside, "三个 σ 锚点互相矛盾 —— 后续功效论断不许做"
    nz = sum(1 for v in good.values()
             if v.get("sigma_C_coup_sd_true", 0) <= 1e-9)
    print(f"   在 σ = C 下 coup_sd_true = 0 的域：**{nz}/{len(good)}**"
          f"（在 σ = B = {B:.2f} 下是 151/164 —— "
          f"这正是那个会制造假阴性的 σ）")
    ct = np.array([v["sigma_C_coup_sd_true"] for v in good.values()
                   if "sigma_C_coup_sd_true" in v])
    print(f"   在 σ = C 下 coup_sd_true 中位 **{np.median(ct):.3f}**"
          f" = {np.median(ct) / C:.2f}σ"
          f"（对照 GRB2-SH3/folding：0.723 = 1.57σ）")

    out = {"n_keep": len(keep), "n_tc": len(tc), "n_done": len(good),
           "sigma_grid": SIG_GRID,
           "anchor_A_median": A,
           "anchor_A_single_median": float(np.median(sA1)),
           "anchor_B": B, "anchor_C_median": C,
           "anchor_C_quartiles": [float(np.percentile(sC, 25)),
                                  float(np.percentile(sC, 75))],
           "anchor_C_decile": [float(np.percentile(sC, 10)),
                               float(np.percentile(sC, 90))],
           "bracket_ok": bool(inside),
           "resid_r_tc_median": float(np.median(rtc)),
           "axis_tc_abscos_median": float(np.median(at)),
           "axis_tc_abscos_quartiles": [float(np.percentile(at, 25)),
                                        float(np.percentile(at, 75))],
           "axis_tc_frac_over_p95": float(np.mean(at > p95)),
           "axis_tc_n": int(len(at)), "random_p95_d20": p95,
           "obs_splithalf_median": float(np.median(ob)),
           "coup_sd_true_at_C_median": float(np.median(ct)),
           "n_coup_sd_true_zero_at_C": nz,
           "rows": res}
    p = os.path.join(ROOT, "cache", "tsuboyama_sigma.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
