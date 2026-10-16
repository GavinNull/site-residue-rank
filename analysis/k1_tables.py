#!/usr/bin/env python3
"""docs/44 的表：从 cache/k1_recompute.json 与 cache/k1_reproduce.json 生成 markdown，
`--check docs/44-k1-recompute.md` 机械核对文档里的每一行表行都出现在生成结果里（防抄错）。

用法：python3 analysis/k1_tables.py > cache/k1_tables.md
      python3 analysis/k1_tables.py --check docs/44-k1-recompute.md
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "cache", "k1_recompute.json")))
REP = json.load(open(os.path.join(ROOT, "cache", "k1_reproduce.json")))


def pc(x, d=1):
    return "n/a" if x is None else f"{100 * x:+.{d}f}%"


def ci(c, d=1):
    return "n/a" if c is None else f"[{100 * c[0]:+.{d}f}%, {100 * c[1]:+.{d}f}%]"


def pt_ci(pt, c):
    return f"{pc(pt)} {ci(c)}"


def tables():
    out = []
    A = out.append
    # ---------------- T1 复现
    A("### T1 Step 1 复现 docs/43（生产模块 rank_k1）\n")
    A("| 量 | docs/43 | 本次（rank_k1） | 判 |")
    A("|---|---|---|---|")
    g = REP["gc"]
    A(f"| K1 假秩 2 率 pooled（s/σ=1–9，4 几何 × 6 水平 × 30） | 1/720 | {g['K1_pooled_false_rank2'][0]}/{g['K1_pooled_false_rank2'][1]} | 相同 |")
    A(f"| K1 P(检出=2\\|真2) 范围 | 111–120/120 | {g['K1_recovery_range'][0]}–{g['K1_recovery_range'][1]}/120 | 相同 |")
    A(f"| K1 结构缺失检出 0 | 120/120 | {g['methods']['K1']['null_is0'][0]}/{g['methods']['K1']['null_is0'][1]} | 相同 |")
    b = g["methods"]["baseline"]["levels"]
    A(f"| 现行程序 P(检出≥2\\|真1) s/σ=4 / 6 / 9 | 103 / 120 / 120 /120 | {b['4.0']['FR2'][0]} / {b['6.0']['FR2'][0]} / {b['9.0']['FR2'][0]} /120 | 相同 |")
    A(f"| 与 docs/43 缓存 `overshoot_gc.json` 的 GC 计数逐格相同 | — | {g['all_identical_to_docs43']} | — |")
    ap = REP["apply"]
    A(f"| 18 个矩阵头条口径秩 2 的个数：现行 / K1 | 15 / 12 | {ap['n_rank2_old']} / {ap['n_rank2_k1']} | 相同 |")
    A(f"| 18 个矩阵 K1 与现行均值曲线与 docs/43 缓存 max\\|Δ\\| | — | {ap['max_abs_curve_diff']:.1e} | 逐位 |")
    bo, bk = REP["boot"]["base"], REP["boot"]["K1"]
    A(f"| S1(15) g12 现行（逐格口径，位点重抽 2000，seed 44） | +3.5% [+0.7%, +7.2%] | {pt_ci(bo['g12'][0], bo['g12'][1:])} | 相同 |")
    A(f"| S1(15) g12 K1 | +1.3% [−1.4%, +4.4%] | {pt_ci(bk['g12'][0], bk['g12'][1:])} | 相同 |")
    A(f"| S1(15) g01 K1 | +6.6% [+4.0%, +9.7%] | {pt_ci(bk['g01'][0], bk['g01'][1:])} | 相同 |")
    A(f"| S1(15) g23 K1 | −11.6% [−15.3%, −8.9%] | {pt_ci(bk['g23'][0], bk['g23'][1:])} | 相同 |")
    A("")
    # ---------------- T2 面板
    A("### T2 RBD×ACE2 面板（15 个子集）：现行 vs K1\n")
    A("| 腿 | 程序 | 点估计（中位） | 只重抽位点 | 骨架+配体+位点同时重抽 | 单独区间 >0 / 跨 0 / <0（15 个） | 点估计为正的矩阵数 |")
    A("|---|---|---|---|---|---|---|")
    names = {"g01": "秩 0→1", "g12": "秩 1→2", "g23": "秩 2→3", "g02": "秩 0→2（固定秩 2 对可加）", "g03": "秩 0→3", "g0star": "最优秩 vs 可加（每次重抽内取 max(g01,g02,g03)）"}
    for k in ("g01", "g12", "g23", "g02", "g03", "g0star"):
        for arm, lab in (("old", "现行"), ("k1", "K1")):
            p = R["panel"][arm]
            c = p["subset_interval_counts"].get(k)
            cc = f"{c['above0']} / {c['span0']} / {c['below0']}" if c else "—"
            pm = p["pos_matrices"].get(k)
            A(f"| {names[k]} | {lab} | {pc(p['point_median'][k])} | {ci(p['site_ci'][k])} | {ci(p['combined_ci'][k])}"
              f"（≤0 的比例 {p['combined_frac_le0'][k]:.3f}） | {cc} | {pm if pm is not None else '—'}/15 |")
    A("")
    A("逐子集（K1，逐格口径；区间 = 单矩阵位点重抽 2000 次）：\n")
    A("| 子集 | 现行 g12 | K1 g12 | K1 g01 | K1 g23 |")
    A("|---|---|---|---|---|")
    for nm in R["panel"]["k1"]["per_subset"]:
        o, k = R["panel"]["old"]["per_subset"][nm], R["panel"]["k1"]["per_subset"][nm]
        A(f"| {nm.replace('animalACE2/', '')} | {pt_ci(o['g12'][0], o['g12'][1:])} | {pt_ci(k['g12'][0], k['g12'][1:])} | "
          f"{pt_ci(k['g01'][0], k['g01'][1:])} | {pt_ci(k['g23'][0], k['g23'][1:])} |")
    A("")
    # ---------------- T3 34 单元
    A("### T3 34 个单元：Gen-3 截尾（喂原始值）与检出秩（头条口径 / 种子众数 / 逐格口径）\n")
    A("| 单元 | n | Gen-3 | 现行 | K1 | 现行(众数) | K1(众数) | 现行(逐格) | K1(逐格) |")
    A("|---|---|---|---|---|---|---|---|---|")
    for u in R["units34"]:
        A(f"| {u['name'].replace('animalACE2/', 'A/').replace('sarbecovirus/', 'S/')} | {u['n']} | {100 * u['g3']:.1f}% | {u['rank_old']} | {u['rank_k1']} | "
          f"{u['rank_old_mode']} | {u['rank_k1_mode']} | {u['rank_old_cell']} | {u['rank_k1_cell']} |")
    A("")
    A("改判交叉（现行 → K1，头条口径）：" + "；".join(f"{k}: {v}" for k, v in R["reshuffle_old_to_k1"].items()) + "\n")
    # ---------------- T4 分层
    A("### T4 按检出秩的 Gen-3 截尾分层\n")
    A("| 口径 | 程序 | 秩 0：n，中位 [区间] | 秩 1 | 秩 2 | 秩 3 | Spearman [区间] | Pearson [区间] | 单调不减（n≥3 的组） | g3<10% 里秩 2 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    lab = {"rank_old": ("头条", "现行"), "rank_k1": ("头条", "K1"), "rank_old_mode": ("种子众数", "现行"),
           "rank_k1_mode": ("种子众数", "K1"), "rank_old_cell": ("逐格", "现行"), "rank_k1_cell": ("逐格", "K1")}
    for key, (cv, pr) in lab.items():
        s = R["strat"][key]
        cells = []
        for k in (0, 1, 2, 3):
            m, n = s["median_by_rank"][str(k)], s["n_by_rank"][str(k)]
            cells.append("—" if m is None else f"n={n}，{100 * m:.1f}% {ci(s['median_ci'][str(k)]) if s['median_ci'][str(k)] else ''}".replace("+", ""))
        sp, pe = s["spearman"], s["pearson"]
        A(f"| {cv} | {pr} | " + " | ".join(cells) +
          f" | {sp['point']:+.2f} [{sp['lo']:+.2f}, {sp['hi']:+.2f}] | {pe['point']:+.2f} [{pe['lo']:+.2f}, {pe['hi']:+.2f}]"
          f" | {s['monotone_nondecreasing']}（组 {s['monotone_groups']}） | {s['lt10']['rank2']}/{s['lt10']['n']} |")
    A("")
    A("秩 0 对秩 ≥1 的 g3 中位之差（单元重抽区间）：\n")
    A("| 程序（头条） | n0 / n≥1 | 秩 0 中位 | 秩 ≥1 中位 | 差 [区间] |")
    A("|---|---|---|---|---|")
    for key, pr in (("rank_old", "现行"), ("rank_k1", "K1")):
        d = R["strat"][key]["rank0_vs_ge1"]
        A(f"| {pr} | {d['n0']} / {d['n_ge1']} | {100 * d['median0']:.1f}% | {100 * d['median_ge1']:.1f}% | "
          f"{100 * d['diff']:.1f} pp [{100 * d['diff_ci'][0]:.1f}, {100 * d['diff_ci'][1]:.1f}] |")
    A("")
    A("截尾区间 × 检出秩（头条口径）：\n")
    A("| Gen-3 区间 | 单元数 | 现行 | K1 |")
    A("|---|---|---|---|")
    for b_ in R["strat"]["rank_k1"]["crosstab"]:
        o, k = R["strat"]["rank_old"]["crosstab"][b_], R["strat"]["rank_k1"]["crosstab"][b_]
        f = lambda d: str({int(a): c for a, c in sorted(d["ranks"].items())})
        A(f"| {b_} | {k['n']} | {f(o)} | {f(k)} |")
    A("")
    # ---------------- T5 Fig1
    A("### T5 Figure 1 的数据（秩 0 归一的中位曲线；层 = cens/(cens+n) < 10%）\n")
    A("| 数据 | 层 | n | 中位曲线（秩 0,1,2,3） | argmin | m1−m2 [区间] | m3−m2 [区间] |")
    A("|---|---|---|---|---|---|---|")
    F = R["fig1"]
    for arm, lb in (("old_as_drawn", "图上现用（ligand_rank.json，全数据中心化，有泄露）"), ("old_noleak", "现行 noleak 同协议重算"), ("k1", "K1")):
        for lay, ln in (("low", F["n_low"]), ("high", F["n_high"])):
            d = F[arm][lay]
            A(f"| {lb} | {'<10%' if lay == 'low' else '≥10%'} | {ln} | {', '.join(f'{x:.3f}' for x in d['median_curve'])} | {d['argmin']} | "
              f"{d['m1_minus_m2']:+.3f} [{d['m1_minus_m2_ci'][0]:+.3f}, {d['m1_minus_m2_ci'][1]:+.3f}] | "
              f"{d['m3_minus_m2']:+.3f} [{d['m3_minus_m2_ci'][0]:+.3f}, {d['m3_minus_m2_ci'][1]:+.3f}] |")
    A("")
    # ---------------- T6 恢复性
    A("### T6 K1 的恢复性诊断（矩阵自己的 mask 与样本内拟合残差；植入 2/1/0 各 8 次）\n")
    A("| 矩阵 | K1 命中 2 | K1 保 0 | K1 植入 1 被报 ≥2 | K1 有功效 | 现行有功效（docs/41） |")
    A("|---|---|---|---|---|---|")
    for nm, o in R["recovery_k1"].items():
        old = R["recovery_old_docs41"][nm]
        A(f"| {nm} | {o['plant2']['p_detect2']:.2f} | {o['plant0']['p_detect0']:.2f} | {o['plant1']['p_ge2']:.2f} | "
          f"{'是' if o['powered'] else '**否**'} | {'是' if old['powered'] else '否'} |")
    A("")
    # ---------------- T7 体系表
    A("### T7 独立体系 × 检出秩（体系层；有功效矩阵的增益中位，位点联合重抽 2000 次）\n")
    A("| 体系 | 程序 | 有功效/矩阵 | 秩 0→1 | 秩 1→2 | 秩 2→3 | 秩 0→2 | 状态 | 交互 | 秩 3 更差 | 内部极小 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    snm = {"S1": "S1 RBD×ACE2", "S2": "S2 PD-CoV×APN", "S3": "S3 OMTKY3×蛋白酶", "S4": "S4 4-4-20×荧光素"}
    for sid in ("S1", "S2", "S3", "S4"):
        for arm, lb in (("old", "现行"), ("k1", "K1")):
            s = R["systems"][arm][sid]
            if s.get("status") == "UNINFORMATIVE":
                A(f"| {snm[sid]} | {lb} | 0/{len(s['matrices'])} | — | — | — | — | UNINFORMATIVE（无功效，不外推） | — | — | — |")
                continue
            A(f"| {snm[sid]} | {lb} | {s['n_powered']}/{len(s['matrices'])} | {pt_ci(s['g01']['point'], s['g01']['ci'])} | "
              f"{pt_ci(s['g12']['point'], s['g12']['ci'])} | {pt_ci(s['g23']['point'], s['g23']['ci'])} | "
              f"{pt_ci(s['g02']['point'], s['g02']['ci'])} | {s['g12_status']} | {'是' if s['interaction'] else '否'} | "
              f"{'是' if s['rank3_worse'] else '否'} | {'是' if s['interior_min_supported'] else '否'} |")
    A("")
    # ---------------- T8 范围内矩阵的秩
    A("### T8 范围内矩阵的头条口径检出秩（现行 → K1；括号 = 3 种子 argmin 众数的稳定度，K1）\n")
    A("| 矩阵 | 现行 | K1 | K1 稳定度 | K1 逐格口径 |")
    A("|---|---|---|---|---|")
    for nm, o in R["ranks_scope"].items():
        A(f"| {nm} | {o['old']['rank']} | {o['k1']['rank']} | {o['k1']['stab']:.2f} | {o['k1']['cell']} |")
    A("")
    n17 = R["n_inscope"]
    A(f"范围内 {n17} 个矩阵（S1 的 15 + gAPN + OMTKY3 pooled）里头条口径检出秩 2 的个数：现行 {R['n_rank2_inscope17']['old']}/{n17}，K1 {R['n_rank2_inscope17']['k1']}/{n17}。\n")
    A("补充 D（ddPCA，描述性）：" + "；".join(f"{nm} 现行 {o['rank_old']} → K1 {o['rank_k1']}" for nm, o in R["ddpca"].items()) + "\n")
    m = R["magnitude"]
    A(f"量级核对：K1 与现行的 rank-0 trim95 之比 min/median/max = {m['rank0_trim95_ratio_k1_over_old'][0]:.4f} / "
      f"{m['rank0_trim95_ratio_k1_over_old'][1]:.4f} / {m['rank0_trim95_ratio_k1_over_old'][2]:.4f}；"
      f"K1 trim95 曲线取值范围 {m['trim95_range_k1'][0]:.3f}–{m['trim95_range_k1'][1]:.3f}。\n")
    return "\n".join(out)


if __name__ == "__main__":
    md = tables()
    if len(sys.argv) >= 3 and sys.argv[1] == "--check":
        doc = open(os.path.join(ROOT, sys.argv[2]), encoding="utf-8").read()
        bad = [l for l in md.splitlines() if l.startswith("|") and not l.startswith("|---") and l not in doc]
        print(f"表行 {sum(1 for l in md.splitlines() if l.startswith('|'))} 行；文档里缺失 {len(bad)} 行")
        for l in bad[:10]:
            print("  缺:", l[:140])
        sys.exit(1 if bad else 0)
    print(md)
