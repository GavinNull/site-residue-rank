#!/usr/bin/env python3
"""docs/22：用第三代（密度型）截尾检测器在 docs/19 的**同一** 34 个单元上
裁决 docs/19 §5.2 与 docs/21 §9 两个方向相反的预判。

本脚本不发明新检测器、不重调任何检测器参数（docs/21 §2.4 已记录搜索量并
给出不敏感性扫描；在这里再调一次就会摧毁这次判别）。它只做四件事：

  1. **精确重建** docs/12 §2 / docs/19 §3 的 34 个单元
     （animalACE2 3 骨架 × 5 配体 + sarbecovirus 6 骨架 × 5 配体，
      去掉原脚本「观测不足 < 500 / 矩阵太小」跳掉的 11 个），
     并用 docs/19 已发表的**六个派生量 + 两个相关系数**作为重建的
     已知事实自检。不通过就中止（CLAUDE.md §10）。
  2. 三代检测器在**同一单元集**上并列：
        Gen-1 = 精确并列 / 全部原始观测（ligand_specific_rank.py 的口径）
        Gen-2 = subspace_lib.censor_fraction（max(并列, min+0.25 窗口)）
        Gen-3 = censor_density.censor_fraction_density(+_ci)
     三者**喂同一个数组**：`arr` = 该 (骨架, 配体) 的全部有限原生数值，
     **含下限堆积**。这正是 docs/21 §10 指出必须注意的地方 ——
     `subspace_lib._starr_matrix` 的 `M[mask]` 已经把恰好等于最小值的格子
     丢掉了（sarbecovirus 重度截尾列会丢 83–93% 的堆积）。
     自检 [RAW] 用「masked 数组 vs raw 数组」的对照把这件事量化出来。
  3. 相关：Pearson + Spearman，4000 次 bootstrap，报区间（CLAUDE.md §5）。
     因变量两套：原秩（cache/ligand_rank.json）与无泄露秩
     （rank_core.cv_rank_noleak + rank_test_power.als_guarded，**不是**
      未保护的 als_complete）。
  4. 两条阈值陈述的再裁决，带**组大小**与**到线的间距**；
     每个截尾值带 docs/21 口径的 12× 80% 子抽样 10–90% 区间，
     并标出落在 Gen-3 不可靠区（极值闸边界 / 区间塌陷）的单元。

运行：python3 analysis/censor_gen3_final.py [--boot 4000] [--seeds 1 2 3]
产物：cache/censor_gen3_final.json
前置（本脚本会自己调用，不通过即中止）：
     analysis/censor_density_check.py 的全部判据（由 --skip-calib 跳过时
     必须手工先跑并在文档里贴结果）。
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rank_core                                              # noqa: E402
from censor_density import (censor_fraction_density,           # noqa: E402
                            censor_fraction_density_ci)
from subspace_lib import CENSOR_WINDOW, censor_fraction       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DMS = os.path.join(ROOT, "cache", "dms")
AA20 = list("ACDEFGHIKLMNPQRSTVWY")

# analysis/ligand_specific_rank.py main() 的 jobs，逐字照抄（不许换样本）
JOBS = [
    ("animalACE2", "starr_animalACE2_final_variant_scores.csv", "position",
     ["bind_deer", "bind_hamster", "bind_bat", "bind_cat",
      "bind_human_monomer"]),
    ("sarbecovirus", "starr_SARSr_RBD_final_variant_scores_lib40_41.csv",
     "site", ["huACE2", "Ra787ACE2", "Ra9479ACE2", "RlanACE2", "RalcACE2"]),
]

# docs/19 §3 已发表的六个派生量（已知事实，用于重建自检）
DOCS19_DERIVED = [
    ("秩0 的 Gen-1 截尾中位 = 43.2%", lambda s: abs(s["med_r0"] - 0.432) < 5e-4),
    ("秩1 的 Gen-1 截尾中位 = 10.9%", lambda s: abs(s["med_r1"] - 0.109) < 5e-4),
    ("Gen-1 <10% 的组合数 = 21", lambda s: s["n_lt10"] == 21),
    ("其中给秩 2 的占 76%", lambda s: abs(s["frac_r2_lt10"] - 16 / 21) < 1e-9),
    ("Gen-1 >=25% 的组合数 = 7", lambda s: s["n_ge25"] == 7),
    ("其中 7/7 给秩 0", lambda s: s["n_r0_ge25"] == 7 and s["n_ge25"] == 7),
]
# docs/19 §5.1 的相关系数（三位小数）
DOCS19_CORR = [("Gen-1 × 原秩 Pearson = -0.786", "g1_orig", "pearson", -0.786),
               ("Gen-1 × 原秩 Spearman = -0.779", "g1_orig", "spearman",
                -0.779),
               ("Gen-2 × 原秩 Pearson = -0.777", "g2_orig", "pearson",
                -0.777),
               ("Gen-2 × 原秩 Spearman = -0.769", "g2_orig", "spearman",
                -0.769),
               ("Gen-1 × 无泄露秩 Pearson = -0.789", "g1_noleak", "pearson",
                -0.789),
               ("Gen-2 × 无泄露秩 Pearson = -0.784", "g2_noleak", "pearson",
                -0.784)]


# ----------------------------------------------------------------- 统计
def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x - x.mean()
    y = y - y.mean()
    d = math.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def _rank_avg(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    s = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(x, y):
    return pearson(_rank_avg(x), _rank_avg(y))


def stat_selfcheck():
    """已知事实自检：完美单调 -> ±1；常量 -> nan；平局处理与 scipy 一致。"""
    out = []
    x = np.array([1.0, 2, 3, 4, 5])
    out.append(("Pearson(x, 2x+1) = +1", abs(pearson(x, 2 * x + 1) - 1) < 1e-12))
    out.append(("Pearson(x, -x) = -1", abs(pearson(x, -x) + 1) < 1e-12))
    out.append(("Spearman(x, exp(x)) = +1 而 Pearson < 1",
                abs(spearman(x, np.exp(x)) - 1) < 1e-12
                and pearson(x, np.exp(x)) < 0.95))
    # 平局：秩必须取平均
    out.append(("平局取平均秩 [1,2,2,3] -> [1,2.5,2.5,4]",
                np.allclose(_rank_avg([1, 2, 2, 3]), [1, 2.5, 2.5, 4])))
    try:
        from scipy.stats import pearsonr, spearmanr
        rng = np.random.default_rng(0)
        a = rng.normal(size=40)
        b = np.round(rng.normal(size=40) + 0.4 * a)   # 带大量平局
        out.append(("与 scipy.pearsonr 一致",
                    abs(pearson(a, b) - pearsonr(a, b)[0]) < 1e-10))
        out.append(("与 scipy.spearmanr 一致（含平局）",
                    abs(spearman(a, b) - spearmanr(a, b)[0]) < 1e-10))
    except Exception as e:                            # pragma: no cover
        out.append((f"scipy 对照不可用：{e}", False))
    return out


def boot_corr(x, y, nboot, seed, kind="pearson"):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    f = pearson if kind == "pearson" else spearman
    pt = f(x, y)
    vals = []
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2 or len(np.unique(x[i])) < 2:
            continue
        v = f(x[i], y[i])
        if np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    return {"point": float(pt), "n": int(n), "nboot": int(nboot),
            "n_used": int(len(vals)),
            "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5)),
            "median": float(np.median(vals))}


# ------------------------------------------------------------- 单元重建
def build_units(seeds, max_rank=3):
    """照抄 ligand_specific_rank.analyse 的筛选与建矩阵逻辑，一字不改。"""
    units = []
    for tag, fn, skey, cols in JOBS:
        path = os.path.join(DMS, fn)
        rows = list(csv.DictReader(open(path)))
        bgs = sorted({r["target"] for r in rows})
        for bg in bgs:
            for col in cols:
                vals = []
                for r in rows:
                    if r.get("target") != bg:
                        continue
                    v = r.get(col, "")
                    if v in ("", "NA"):
                        continue
                    vals.append((r[skey], r["mutant"], float(v)))
                if not vals:
                    continue
                arr = np.array([v for _, _, v in vals])      # 原生全集，含堆积
                lo = arr.min()
                cens = int((arr <= lo + 1e-6).sum())
                keep = [(s, a, v) for s, a, v in vals if v > lo + 1e-6]
                if len(keep) < 500:                          # 「观测不足」
                    continue
                sites = sorted({s for s, _, _ in keep})
                si = {s: i for i, s in enumerate(sites)}
                M = np.full((len(sites), 20), np.nan)
                for s, a, v in keep:
                    if a in AA20:
                        M[si[s], AA20.index(a)] = v
                mask = np.isfinite(M)
                if mask.sum() < 500 or len(sites) < 30:      # 「矩阵太小」
                    continue
                n = int(mask.sum())
                u = {"dataset": tag, "bg": bg, "lig": col,
                     "sites": len(sites), "n": n, "cens_count": cens,
                     "n_raw": int(len(vals)),
                     # Gen-1：docs/12 §2 的口径（cens/(cens+n)，docs/19 §3 已核）
                     "g1": cens / (cens + n),
                     "g1_raw": cens / len(vals),
                     # Gen-2：固定窗口
                     "g2": float(censor_fraction(arr)),
                     "sharp": (cens / len(vals))
                     / max(float(censor_fraction(arr)), 1e-9)}
                # Gen-3：密度型，**喂 raw**（含堆积），原样参数
                ci = censor_fraction_density_ci(arr, seed=0)
                u["g3"] = float(ci["q"])
                u["g3_lo"] = float(ci["lo"])
                u["g3_hi"] = float(ci["hi"])
                u["g3_sub"] = ci["sub"]
                inf = ci["info"]
                u["g3_info"] = {k: inf.get(k) for k in
                                ("gate", "why", "q_atom", "q_refl",
                                 "q_trunc", "resid", "sigma_hat",
                                 "sigma_over_s", "ev_over_zstar", "m",
                                 "c_rel", "n")}
                # 对照：若错误地喂 masked（已剔除精确最小值）数组会得到什么
                u["g3_masked_wrong"] = float(
                    censor_fraction_density(M[mask]))
                # 秩：无泄露（rank_core -> als_guarded），多种子取众数
                bests = []
                for sd in seeds:
                    errs = rank_core.cv_rank_noleak(M, mask,
                                                    max_rank=max_rank, K=5,
                                                    restarts=2, seed=sd,
                                                    iters=150, ridge=0.1)
                    b, _ = rank_core.best_rank(errs)
                    bests.append(int(b))
                vv, cc = np.unique(bests, return_counts=True)
                u["rank_noleak"] = int(vv[np.argmax(cc)])
                u["rank_noleak_seeds"] = bests
                u["rank_noleak_stab"] = float(cc.max() / len(bests))
                units.append(u)
                print(f"  {tag:<13}{bg[:20]:<20}{col[:18]:<18}"
                      f"n={n:<6}g1={u['g1']:>6.1%} g2={u['g2']:>6.1%} "
                      f"g3={u['g3']:>6.1%} [{u['g3_lo']:>5.1%},"
                      f"{u['g3_hi']:>5.1%}] rk_nl={u['rank_noleak']}",
                      flush=True)
    return units


def attach_orig_rank(units):
    cached = json.load(open(os.path.join(ROOT, "cache", "ligand_rank.json")))
    key = {(c["dataset"], c["bg"], c["lig"]): c for c in cached
           if "skip" not in c}
    miss = []
    for u in units:
        c = key.get((u["dataset"], u["bg"], u["lig"]))
        if c is None:
            miss.append((u["dataset"], u["bg"], u["lig"]))
            continue
        u["rank_orig"] = int(c["best_mode"])
        u["gain_orig"] = float(c["gain"])
        # 已知事实核对：缓存里的 cens / n 必须与我重建的一致
        u["cache_match"] = (int(c["cens"]) == u["cens_count"]
                            and int(c["n"]) == u["n"])
    return miss


# --------------------------------------------------------- 派生量 / 阈值
def derived(units, fkey, rkey):
    f = np.array([u[fkey] for u in units])
    r = np.array([u[rkey] for u in units])
    med = {}
    for k in (0, 1, 2, 3):
        s = f[r == k]
        med[k] = (float(np.median(s)) if len(s) else None, int(len(s)))
    lt10 = [u for u in units if u[fkey] < 0.10]
    ge25 = [u for u in units if u[fkey] >= 0.25]
    return {"med_r0": med[0][0], "med_r1": med[1][0], "med_r2": med[2][0],
            "med_r3": med[3][0],
            "n_by_rank": {k: med[k][1] for k in med},
            "n_lt10": len(lt10),
            "n_r2_lt10": sum(1 for u in lt10 if u[rkey] == 2),
            "frac_r2_lt10": (sum(1 for u in lt10 if u[rkey] == 2) / len(lt10)
                             if lt10 else None),
            "n_ge25": len(ge25),
            "n_r0_ge25": sum(1 for u in ge25 if u[rkey] == 0),
            "frac_r0_ge25": (sum(1 for u in ge25 if u[rkey] == 0) / len(ge25)
                             if ge25 else None),
            "ge25_units": [(u["bg"], u["lig"], u[fkey], u[rkey])
                           for u in sorted(ge25, key=lambda z: -z[fkey])],
            "margins": sorted([(abs(u[fkey] - 0.25), u[fkey] - 0.25,
                                u["dataset"], u["bg"], u["lig"], u[rkey])
                               for u in units])[:10]}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--skip-calib", action="store_true")
    a = ap.parse_args()

    print("=" * 96)
    print("docs/22  第三代截尾检测器 × docs/19 的同一 34 个单元")
    print("=" * 96)

    # ---- [0] 强制前置自检（CLAUDE.md §10：不通过就中止）
    print("\n[0] 前置工具自检")
    ok = True
    sc = rank_core.selfcheck(verbose=False)
    print(f"    rank_core.selfcheck()（无泄露 CV + als_guarded）: "
          f"{'OK' if sc else '** 失败 **'}")
    ok &= bool(sc)
    for nm, good in stat_selfcheck():
        print(f"    统计实现自检 {nm}: {'OK' if good else '** 失败 **'}")
        ok &= good
    if not a.skip_calib:
        print("    censor_density_check.py（密度检测器双向标定）...",
              end="", flush=True)
        p = subprocess.run([sys.executable,
                            os.path.join(ROOT, "analysis",
                                         "censor_density_check.py")],
                           capture_output=True, text=True)
        calib_ok = p.returncode == 0 and "自检: 通过" in p.stdout
        print(f" {'OK' if calib_ok else '** 失败 **'}")
        ok &= calib_ok
        open(os.path.join(ROOT, "cache", "censor_density_check_out.txt"),
             "w").write(p.stdout)
    if not ok:
        print("\n前置自检未通过 —— 中止，不看任何结论。")
        return 1

    # ---- [1] 重建 34 个单元
    print(f"\n[1] 重建单元（无泄露定秩种子 {a.seeds}）")
    units = build_units(a.seeds)
    print(f"    单元数 = {len(units)}")
    if len(units) != 34:
        print("    ** 不是 34 个 —— 样本变了，中止 **")
        return 1
    miss = attach_orig_rank(units)
    if miss:
        print(f"    ** cache/ligand_rank.json 缺 {miss} —— 中止 **")
        return 1
    bad = [u for u in units if not u["cache_match"]]
    print(f"    与 cache/ligand_rank.json 的 cens/n 逐单元一致: "
          f"{'OK' if not bad else f'** {len(bad)} 个不一致 **'}")
    if bad:
        return 1

    # ---- [2] 重建自检：docs/19 的六个派生量 + 两代相关系数
    print("\n[2] 重建自检（docs/19 已发表的派生量，已知事实）")
    d_g1_orig = derived(units, "g1", "rank_orig")
    rebuild = []
    for nm, fn in DOCS19_DERIVED:
        good = bool(fn(d_g1_orig))
        rebuild.append((nm, good))
        print(f"    {nm:<34}: {'OK' if good else '** 不吻合 **'}")
        ok &= good
    corr = {}
    pairs = {"g1_orig": ("g1", "rank_orig"), "g2_orig": ("g2", "rank_orig"),
             "g3_orig": ("g3", "rank_orig"),
             "g1_noleak": ("g1", "rank_noleak"),
             "g2_noleak": ("g2", "rank_noleak"),
             "g3_noleak": ("g3", "rank_noleak")}
    for k, (fk, rk) in pairs.items():
        x = [u[fk] for u in units]
        y = [u[rk] for u in units]
        corr[k] = {"pearson": boot_corr(x, y, a.boot, 11, "pearson"),
                   "spearman": boot_corr(x, y, a.boot, 11, "spearman")}
    print("\n    docs/19 §5.1 的系数复现（三位小数）")
    for nm, k, kind, want in DOCS19_CORR:
        got = corr[k][kind]["point"]
        good = abs(round(got, 3) - want) < 1e-9
        rebuild.append((nm, good))
        print(f"    {nm:<36} 实测 {got:+.4f}  "
              f"{'OK' if good else '** 不吻合 **'}")
        ok &= good
    if not ok:
        print("\n重建未通过 —— 按任务要求停下并报告，不再往下。")
        json.dump({"rebuild": rebuild, "corr": corr, "units": units},
                  open(os.path.join(ROOT, "cache",
                                    "censor_gen3_final.json"), "w"),
                  indent=1, default=float)
        return 1

    # ---- [3] [RAW] 喂 raw vs 喂 masked
    print("\n[3] [RAW] 检测器必须看到堆积：raw vs masked（docs/21 §10）")
    print(f"    {'骨架/配体':<34}{'raw g3':>9}{'masked g3':>11}{'丢失(pp)':>10}")
    diffs = []
    for u in sorted(units, key=lambda z: -z["g3"])[:8]:
        d = u["g3"] - u["g3_masked_wrong"]
        diffs.append(d)
        print(f"    {u['bg'][:18]+'/'+u['lig'][:14]:<34}"
              f"{u['g3']:>9.1%}{u['g3_masked_wrong']:>11.1%}{100*d:>+10.1f}")
    allmax = max(u["g3"] - u["g3_masked_wrong"] for u in units)
    print(f"    最大丢失 = {100*allmax:+.1f}pp  "
          f"（> 10pp 证明必须喂 raw；本脚本喂的是 raw）")

    # ---- [4] 三代并列表
    print("\n[4] 三代检测器在同一 34 单元上并列")
    hdr = (f"{'数据集':<13}{'骨架':<20}{'配体':<17}{'Gen1':>7}{'Gen2':>7}"
           f"{'Gen3':>7}{'Gen3 区间':>17}{'w_i':>7}{'锐度':>6}"
           f"{'原秩':>5}{'无泄露':>7}")
    print(hdr)
    print("-" * len(hdr))
    for u in sorted(units, key=lambda z: -z["g3"]):
        den = u["g2"] - u["g1"]
        u["w_i"] = (u["g3"] - u["g1"]) / den if den > 0.02 else None
        ws = f"{u['w_i']:>7.2f}" if u["w_i"] is not None else f"{'—':>7}"
        print(f"{u['dataset']:<13}{u['bg'][:18]:<20}{u['lig'][:15]:<17}"
              f"{u['g1']:>7.1%}{u['g2']:>7.1%}{u['g3']:>7.1%}"
              f"{'['+format(u['g3_lo'],'.1%')+','+format(u['g3_hi'],'.1%')+']':>17}"
              f"{ws}{u['sharp']:>6.2f}{u['rank_orig']:>5}"
              f"{u['rank_noleak']:>7}")
    wi = [u["w_i"] for u in units if u["w_i"] is not None]
    n_diff = sum(1 for u in units if abs(u["g3"] - u["g1"]) > 5e-4)
    print(f"\n    有定义的 w_i：{len(wi)} 个（|Gen2−Gen1| > 2pp），"
          f"中位 = {np.median(wi):+.3f}，范围 [{min(wi):+.2f}, {max(wi):+.2f}]")
    print(f"    g3 与 g1 差 > 0.05pp 的单元数 = {n_diff} / 34"
          f"（核对预注册 §1.4：Gen-3 不是把 Gen-1 原样搬过来）")
    print(f"    中位截尾：Gen-1 {np.median([u['g1'] for u in units]):.1%}  "
          f"Gen-2 {np.median([u['g2'] for u in units]):.1%}  "
          f"Gen-3 {np.median([u['g3'] for u in units]):.1%}")

    # ---- [5] 相关
    print(f"\n[5] 相关系数（bootstrap {a.boot} 次，95% 区间；CLAUDE.md §5）")
    print(f"    {'自变量':<8}{'因变量':<10}{'Pearson':>10}{'95% 区间':>22}"
          f"{'Spearman':>10}{'95% 区间':>22}")
    lab = {"g1_orig": ("Gen-1", "原秩"), "g2_orig": ("Gen-2", "原秩"),
           "g3_orig": ("Gen-3", "原秩"), "g1_noleak": ("Gen-1", "无泄露秩"),
           "g2_noleak": ("Gen-2", "无泄露秩"),
           "g3_noleak": ("Gen-3", "无泄露秩")}
    for k in ("g1_orig", "g2_orig", "g3_orig", "g1_noleak", "g2_noleak",
              "g3_noleak"):
        p_, s_ = corr[k]["pearson"], corr[k]["spearman"]
        print(f"    {lab[k][0]:<8}{lab[k][1]:<10}{p_['point']:>+10.3f}"
              f"{'['+format(p_['lo'],'+.3f')+', '+format(p_['hi'],'+.3f')+']':>22}"
              f"{s_['point']:>+10.3f}"
              f"{'['+format(s_['lo'],'+.3f')+', '+format(s_['hi'],'+.3f')+']':>22}")
    agree = sum(1 for u in units if u["rank_orig"] == u["rank_noleak"])
    print(f"    原秩 vs 无泄露秩一致：{agree}/34"
          f"（docs/19 记 31/34）；不一致 = "
          f"{[(u['bg'], u['lig'], u['rank_orig'], u['rank_noleak']) for u in units if u['rank_orig'] != u['rank_noleak']]}")

    # ---- [6] 阈值再裁决
    print("\n[6] 两条阈值陈述的再裁决（组大小必须与比率同时报）")
    thr = {}
    for fk in ("g1", "g2", "g3"):
        for rk in ("rank_orig", "rank_noleak"):
            thr[f"{fk}_{rk}"] = derived(units, fk, rk)
    print(f"    {'口径':<22}{'<10% 的 n':>11}{'其中秩2':>10}{'占比':>8}"
          f"{'>=25% 的 n':>12}{'其中秩0':>10}{'占比':>8}")
    for fk in ("g1", "g2", "g3"):
        for rk in ("rank_orig", "rank_noleak"):
            d = thr[f"{fk}_{rk}"]
            nm = f"{fk.upper().replace('G','Gen-')} × " \
                 f"{'原秩' if rk=='rank_orig' else '无泄露秩'}"
            print(f"    {nm:<22}{d['n_lt10']:>11}{d['n_r2_lt10']:>10}"
                  f"{d['frac_r2_lt10']:>8.0%}{d['n_ge25']:>12}"
                  f"{d['n_r0_ge25']:>10}{d['frac_r0_ge25']:>8.0%}")
    print("\n    Gen-3 下 >=25% 组的成员（到 25% 线的间距）")
    for bg, lig, f, r in thr["g3_rank_orig"]["ge25_units"]:
        print(f"      {bg[:20]:<22}{lig[:16]:<18}{f:>7.1%}"
              f"  线上 {100*(f-0.25):>+6.1f}pp  秩 {r}")
    print("\n    离 25% 线最近的 8 个单元（线的锐度）")
    for m, sgn, ds, bg, lig, r in thr["g3_rank_orig"]["margins"][:8]:
        print(f"      {bg[:20]:<22}{lig[:16]:<18}{sgn+0.25:>7.1%}"
              f"  {100*sgn:>+7.1f}pp  秩 {r}")
    print("\n    按秩分层的截尾中位（单调性，三代并列）")
    print(f"      {'秩':<4}{'n':>4}{'Gen-1':>9}{'Gen-2':>9}{'Gen-3':>9}")
    for k in (0, 1, 2, 3):
        s = [u for u in units if u["rank_orig"] == k]
        if not s:
            continue
        print(f"      {k:<4}{len(s):>4}"
              f"{np.median([u['g1'] for u in s]):>9.1%}"
              f"{np.median([u['g2'] for u in s]):>9.1%}"
              f"{np.median([u['g3'] for u in s]):>9.1%}")

    # ---- [7] Gen-3 不可靠区
    print("\n[7] Gen-3 不可靠区的单元（docs/21 §2.5：区间塌陷 / 极值闸边界）")
    print(f"    {'骨架/配体':<34}{'Gen3':>7}{'区间':>17}{'宽(pp)':>8}"
           f"{'σ̂/s_up':>8}{'ev/z*':>7}{'闸':>8}")
    unrel = []
    for u in sorted(units, key=lambda z: -(z["g3_hi"] - z["g3_lo"])):
        w = u["g3_hi"] - u["g3_lo"]
        i = u["g3_info"]
        ev = i.get("ev_over_zstar")
        so = i.get("sigma_over_s")
        flag = []
        if w > 0.15:
            flag.append("区间宽>15pp")
        if ev is not None and (ev < 0.75 or ev > 1.20):
            flag.append("极值闸边界")
        if so is not None and so > 0.28:
            flag.append("σ̂/s_up 近 γ")
        if flag:
            unrel.append({"bg": u["bg"], "lig": u["lig"], "g3": u["g3"],
                          "lo": u["g3_lo"], "hi": u["g3_hi"],
                          "flags": flag, "ev": ev, "sigma_over_s": so,
                          "rank_orig": u["rank_orig"]})
            print(f"    {u['bg'][:18]+'/'+u['lig'][:14]:<34}{u['g3']:>7.1%}"
                  f"{'['+format(u['g3_lo'],'.1%')+','+format(u['g3_hi'],'.1%')+']':>17}"
                  f"{100*w:>8.1f}"
                  f"{(f'{so:.3f}' if so is not None else '—'):>8}"
                  f"{(f'{ev:.2f}' if ev is not None else '—'):>7}"
                  f"{','.join(flag):>8}")
    if not unrel:
        print("    （无）")
    print(f"    标记数 = {len(unrel)} / 34")

    out = {"n_units": len(units), "boot": a.boot, "seeds": a.seeds,
           "rebuild": rebuild, "units": units, "corr": corr,
           "thresholds": thr, "unreliable": unrel,
           "raw_vs_masked_max_pp": float(100 * allmax),
           "w_i_median": float(np.median(wi)),
           "w_i_n": len(wi),
           "n_g3_ne_g1": int(n_diff),
           "rank_agree": int(agree),
           "median_frac": {"g1": float(np.median([u["g1"] for u in units])),
                           "g2": float(np.median([u["g2"] for u in units])),
                           "g3": float(np.median([u["g3"] for u in units]))},
           "provenance": {
               "units": "analysis/ligand_specific_rank.py JOBS 逐字照抄",
               "gen1": "cens/(cens+n)，cens = 恰好等于最小值的原始观测数",
               "gen2": f"subspace_lib.censor_fraction, 窗口 {CENSOR_WINDOW}",
               "gen3": "censor_density.censor_fraction_density_ci（原样参数）",
               "rank_orig": "cache/ligand_rank.json best_mode",
               "rank_noleak": "rank_core.cv_rank_noleak + "
                              "rank_test_power.als_guarded",
               "data": "cache/dms/starr_animalACE2_final_variant_scores.csv, "
                       "cache/dms/starr_SARSr_RBD_final_variant_scores_"
                       "lib40_41.csv"}}
    json.dump(out, open(os.path.join(ROOT, "cache",
                                     "censor_gen3_final.json"), "w"),
              indent=1, default=float)
    print("\n产物：cache/censor_gen3_final.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
