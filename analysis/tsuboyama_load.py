#!/usr/bin/env python3
"""docs/27 Step 1：Tsuboyama 2023 的解析器 + **用已知事实做的自检**。

CLAUDE.md §10 是硬规则：这个解析器在被用来得出任何结论之前，必须有一组
用**已知事实**（不是自洽性）的自检，强制运行，不通过就中止。
本仓库在这一条上烧过六次，其中 D-023（AAindex parser：残基顺序交错 +
缺失标记读成 0）**完全静默**。

**本数据集真实存在同一类陷阱，已被本脚本抓到并在 docs/27 §2 记录：**
  * `ddG_ML` 列有 98 846 / 487 914 个单取代行的值是**字面字符串 `-`**；
  * `dG_ML`  列有字面字符串 **`<-1`** 与 **`>5`** —— 这是**动态范围标记**，
    不是数值。`pandas.read_csv` 会把整列读成 object，一个
    `df.ddG_ML.astype(float)` 会抛异常，而
    `pd.to_numeric(..., errors="coerce")` 会把它们变成 NaN 而**不告警**，
    于是「这个变体超出动态范围」与「这个变体没测」被合并，
    §2 的截尾口径就无从建立。
  本脚本把这三个标记当成**一等公民**分类计数，并且有一项自检专门断言
  「朴素 float() 必须在这些标记上失败」（T5 的正控制）。

符号约定（T6，**用两条独立的已知事实钉死，不假设**）：
  (i) 恒等式 `ddG_ML == dG_ML(variant) − dG_ML(WT)`，逐域 sd = 0.000000；
 (ii) Pro 取代在本仓库已建立为「20 个残基中最去稳定」（docs/15 V7，3/3 域）。
  两条一致地给出 **ddG_ML 负 = 去稳定**（与 Olson2014 Table S4 同向，
  与 ddPCA 权重表反向 —— docs/26 §2 正是因为这一类约定差点把一次成功的
  校准读成解析错误）。
  本模块**对外一律输出仓库约定**：`M = -ddG_ML`，即**正 = 去稳定**，
  与 `ddpca_load.matrix` 同向，这样下游脚本不需要知道上游约定。

σ 的来源（D-028 硬约束）：**不得**用 `deltaG_95CI`
（中位 0.130 kcal/mol ⇒ SE ≈ 0.033，与 ddPCA 的 bootstrap SE 同型，
低估真实误差一个量级）。用体外文献 ΔG 对照算出的 RMSE（`lit_sigma()`）。

运行：python3 analysis/tsuboyama_load.py
产物：cache/tsuboyama_parse.json（清单 + 全部自检结果）
      cache/tsuboyama_matrices.npz（逐域矩阵，供下游脚本复用）
"""
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import AA20, ROOT                              # noqa: E402

DDIR = os.path.join(ROOT, "cache", "dms", "tsuboyama2023")
MAIN = os.path.join(DDIR, "Tsuboyama2023_Dataset2_Dataset3_20230416.csv")
LIST = os.path.join(DDIR, "Single_DMS_list.csv")
EXTD = os.path.join(DDIR, "dG_extdG_data_Fig1.csv")

# 作者的可信动态范围（由数据本身给出：float 的 dG_ML 严格落在开区间内，
# 越界的被换成字面标记 '<-1' / '>5'）。T10 断言这一点。
DG_LO, DG_HI = -1.0, 5.0
OUT_MARK = {"<-1": "lo", ">5": "hi"}
MISS = "-"

SINGLE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")
NAT = re.compile(r"^([0-9][A-Za-z0-9]{3})\.pdb(?:_(.+))?$")

# 纳入判据的「规模」部分，在看任何轴之前写定（CLAUDE.md §4）
MIN_SITES = 30
MIN_OBS = 400


# ----------------------------------------------------------------------
def _num(s):
    """严格数值化：标记不是数值，返回 None 而**不是** 0 / nan 静默合并。"""
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def scan(path=MAIN, verbose=True):
    """单次流式扫描，返回逐域记录。只碰单取代行与 wt 行。"""
    dom = {}
    wt_rows = defaultdict(list)
    n_row = n_single = 0
    mark_count = Counter()
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        for x in rd:
            n_row += 1
            w = x["WT_name"]
            mt = x["mut_type"]
            if mt == "wt":
                wt_rows[w].append({"aa_seq": x["aa_seq"], "dG_ML": x["dG_ML"],
                                   "deltaG": x["deltaG"]})
                continue
            m = SINGLE.match(mt)
            if not m:
                continue
            n_single += 1
            d = dom.setdefault(w, {
                "name": w, "cells": {}, "wt_at": defaultdict(Counter),
                "n": 0, "ok": 0, "hi": 0, "lo": 0, "nan": 0,
                "ci": [], "raw": [], "dg_in": [], "selfmut": 0})
            wtr, pos, mut = m.group(1), int(m.group(2)), m.group(3)
            d["n"] += 1
            d["wt_at"][pos][wtr] += 1
            if mut == wtr:
                d["selfmut"] += 1
            g, dd = x["dG_ML"], x["ddG_ML"]
            if g in OUT_MARK:
                mark_count[g] += 1
                d[OUT_MARK[g]] += 1
            elif g == MISS:
                mark_count["dG-"] += 1
                d["nan"] += 1
            v = _num(dd)
            if v is None:
                if dd == MISS:
                    mark_count["ddG-"] += 1
                continue
            d["ok"] += 1
            d["cells"][(pos, mut)] = v
            d["raw"].append(v)
            c = _num(x["deltaG_95CI"])
            if c is not None:
                d["ci"].append(c)
            gg = _num(g)
            if gg is not None:
                d["dg_in"].append(gg)
    if verbose:
        print(f"   扫描 {n_row} 行，其中单取代 {n_single} 行，"
              f"{len(dom)} 个 WT 背景")
        print(f"   字面标记计数: ddG_ML '-' = {mark_count['ddG-']}, "
              f"dG_ML '<-1' = {mark_count['<-1']}, "
              f"dG_ML '>5' = {mark_count['>5']}, "
              f"dG_ML '-' = {mark_count['dG-']}")
    return dom, wt_rows, {"n_row": n_row, "n_single": n_single,
                          "marks": dict(mark_count)}


# ----------------------------------------------------------------------
def lit_sigma(path=EXTD, window=True):
    """体外文献 ΔG 对照 → ΔΔG 尺度的 RMSE（D-028 要求的「体外校准 σ」）。

    口径：**逐蛋白**先去掉该蛋白自己的偏移（文献与本 assay 的缓冲/温度差异
    是逐蛋白的常数，而双向中心化本来就会移除逐矩阵常数），再算
      RMSE_ctr   = 去偏移后逐变体残差的 RMSE  —— 这是 ΔΔG 级的 σ
      RMSE_pair  = 同蛋白内两两差值的 RMSE     —— 应 ≈ sqrt(2) × RMSE_ctr
    `window=True` 时只用 assay ΔG 落在 [-1, 5] 的变体（我们的矩阵只用这些）。
    """
    g = defaultdict(list)
    for x in csv.DictReader(open(path)):
        u, v = _num(x["deltaG"]), _num(x["prev_dG"])
        if u is None or v is None:
            continue
        if window and not (DG_LO <= u <= DG_HI):
            continue
        g[x["PDB_ID"]].append((u, v))
    per, wn = [], 0
    for k, v in sorted(g.items(), key=lambda z: -len(z[1])):
        if len(v) < 6:
            continue
        u = np.array([p[0] for p in v]); w = np.array([p[1] for p in v])
        ctr = float(np.sqrt((((u - u.mean()) - (w - w.mean())) ** 2).mean()))
        du, dw = u[:, None] - u[None, :], w[:, None] - w[None, :]
        iu = np.triu_indices(len(u), 1)
        pr = float(np.sqrt(((du[iu] - dw[iu]) ** 2).mean()))
        per.append({"pdb": k, "n": int(len(u)),
                    "r": float(np.corrcoef(u, w)[0, 1]),
                    "rmse_ctr": ctr, "rmse_pair": pr})
        wn += len(u)
    return {"per_protein": per, "n_total": wn, "window": bool(window),
            "weighted_rmse_ctr":
                float(np.sqrt(sum(p["n"] * p["rmse_ctr"] ** 2
                                  for p in per) / wn)),
            "weighted_rmse_pair":
                float(np.sqrt(sum(p["n"] * p["rmse_pair"] ** 2
                                  for p in per) / wn)),
            "median_rmse_ctr": float(np.median([p["rmse_ctr"] for p in per]))}


def internal_sigma(path=MAIN, cap=400000):
    """内部一致性 σ：胰蛋白酶 vs 胰凝乳蛋白酶两条**独立**蛋白酶读出。

    这**不是**体外校准（两条读出共用同一个 cDNA 展示库与测序批次，
    共同系统误差不会出现在差值里），所以它只作**乐观下界**报告，
    主分析的 σ 一律用 `lit_sigma()`。
    """
    a, b = [], []
    with open(path, newline="") as fh:
        for x in csv.DictReader(fh):
            if not SINGLE.match(x["mut_type"]):
                continue
            if x["ddG_ML"] == MISS:
                continue
            u, v = _num(x["deltaG_t"]), _num(x["deltaG_c"])
            if u is None or v is None:
                continue
            if not (DG_LO <= u <= DG_HI and DG_LO <= v <= DG_HI):
                continue
            a.append(u); b.append(v)
            if len(a) >= cap:
                break
    a, b = np.array(a), np.array(b)
    rm = float(np.sqrt(((a - b) ** 2).mean()))
    return {"n": int(len(a)), "r": float(np.corrcoef(a, b)[0, 1]),
            "rmse_t_vs_c": rm, "sigma_per_assay": rm / math.sqrt(2),
            "sigma_of_ML": rm / 2.0}


# ----------------------------------------------------------------------
def build(d, wtseq=""):
    """逐域 site x 20 矩阵。**输出仓库约定：正 = 去稳定**（= -ddG_ML）。

    `wtseq` 是该域 wt 行的 `aa_seq`（完整、连续）。必须带着它，
    因为「有取代的位点」可能不连续（412 个候选里有 4 个域恰好缺一个位点），
    用「按位点拼出来的串」去做外部序列对账会因为一个缺失位造成移码，
    把一次正确的对账读成失败 —— 这正是 §10「区分工具自身的错与
    被检查对象的错」的一次实际应用，已在 docs/27 §2 留痕。
    """
    pos = sorted(d["wt_at"])
    si = {p: i for i, p in enumerate(pos)}
    M = np.full((len(pos), 20), np.nan)
    for (p, a), v in d["cells"].items():
        M[si[p], AA20.index(a)] = -v
    mask = np.isfinite(M)
    design = len(pos) * 19
    nout = d["hi"] + d["lo"]
    subs = np.array([sum(d["wt_at"][p].values()) for p in pos])
    return {
        "name": d["name"], "M": M, "mask": mask,
        "raw": np.array([-v for v in d["raw"]], float),
        "pos": pos,
        "wtseq": wtseq,
        "wtseq_from_muts": "".join(max(d["wt_at"][p].items(),
                                       key=lambda z: z[1])[0] for p in pos),
        "pos_contiguous": bool(len(pos) == (pos[-1] - pos[0] + 1)),
        "sites": len(pos), "design": design, "n_obs": int(mask.sum()),
        "n_var": d["n"], "n_hi": d["hi"], "n_lo": d["lo"], "n_nan": d["nan"],
        "f_out": nout / d["n"] if d["n"] else float("nan"),
        "f_out_hi": d["hi"] / d["n"] if d["n"] else float("nan"),
        "f_out_lo": d["lo"] / d["n"] if d["n"] else float("nan"),
        "miss_frac": 1.0 - mask.sum() / design,
        "subs_min": int(subs.min()), "subs_med": float(np.median(subs)),
        "subs_max": int(subs.max()), "subs_mean": float(subs.mean()),
        "subs_hist": {int(k): int(v) for k, v in
                      sorted(Counter(subs.tolist()).items())},
        "ci_med": float(np.median(d["ci"])) if d["ci"] else float("nan"),
        "dg_in_med": float(np.median(d["dg_in"])) if d["dg_in"] else
        float("nan"),
        "selfmut": d["selfmut"],
    }


# ----------------------------------------------------------------------
def selfcheck(dom, wt_rows, meta, verbose=True):
    """T0–T10：全部用已知事实，任一不过抛 AssertionError。"""
    res = {}
    P = print if verbose else (lambda *a, **k: None)
    P("\n" + "=" * 92)
    P("Step 1 已知事实自检（CLAUDE.md §10；任一不通过即中止）")
    P("=" * 92)

    # --- T5 正控制：朴素 float() 必须在三个字面标记上失败 ---------------
    bad = []
    for s in ("-", "<-1", ">5"):
        try:
            float(s); bad.append(s)
        except ValueError:
            pass
    assert not bad, f"这些标记居然能被 float() 吃下去: {bad}"
    assert _num("-") is None and _num("<-1") is None and _num(">5") is None
    assert meta["marks"].get("ddG-", 0) > 0 and meta["marks"].get(">5", 0) > 0
    res["T5_markers"] = meta["marks"]
    P(f"   [T5] 缺失/越界标记是**字面字符串**，朴素 float() 在三者上全部失败；"
      f"本解析器分类计数 ddG_ML '-' = {meta['marks']['ddG-']}、"
      f"dG_ML '<-1' = {meta['marks']['<-1']}、'>5' = {meta['marks']['>5']}"
      f"  —— D-023 型静默归零**不可能发生**  OK")

    # --- T1 野生型残基在每个位点单值 -----------------------------------
    amb = [w for w, d in dom.items()
           if any(len(c) > 1 for c in d["wt_at"].values())]
    assert not amb, f"{len(amb)} 个域的某位点野生型残基不单值: {amb[:5]}"
    res["T1_wt_single_valued"] = {"n_domains": len(dom), "n_ambiguous": 0}
    P(f"   [T1] 野生型残基在每个位点单值：{len(dom)}/{len(dom)} 个背景通过  OK")

    # --- T2 野生型残基 == 该域 wt 行序列的第 pos 个字符 ------------------
    tot = ok = nodom = 0
    mism = []
    for w, d in dom.items():
        seqs = {r["aa_seq"] for r in wt_rows.get(w, [])}
        if len(seqs) != 1:
            nodom += 1
            continue
        s = seqs.pop()
        for p, c in d["wt_at"].items():
            tot += 1
            a = next(iter(c))
            if 1 <= p <= len(s) and s[p - 1] == a:
                ok += 1
            else:
                mism.append((w, p, a, s[p - 1] if 1 <= p <= len(s) else "?"))
    assert tot > 0 and ok == tot, f"位点↔序列错配 {len(mism)}: {mism[:8]}"
    res["T2_wt_matches_seq"] = {"sites_checked": tot, "ok": ok,
                                "domains_without_wt_row": nodom}
    P(f"   [T2] mut_type 的位点编号是 1-based on 该域 wt 行的 `aa_seq`："
      f"{ok}/{tot} 个位点逐残基吻合（{nodom} 个背景无 wt 行，已跳过）  OK")

    # --- T3 野生型从不作为突变出现在自己的位点 --------------------------
    sm = sum(d["selfmut"] for d in dom.values())
    assert sm == 0, f"有 {sm} 个「野生型→野生型」行"
    res["T3_no_selfmut"] = 0
    P("   [T3] 野生型残基从不出现在自己位点的突变列表里：0 例  OK")

    # --- T4 每位点 <= 19 个取代 ----------------------------------------
    mx = max(max(sum(c.values()) for c in d["wt_at"].values())
             for d in dom.values())
    assert mx <= 19, f"某位点取代数 {mx} > 19"
    res["T4_max_subs_per_site"] = int(mx)
    P(f"   [T4] 每位点取代数 ≤ 19（完整设计上限）：实测最大 {mx}  OK")

    # --- T6 符号约定：两条独立已知事实 ---------------------------------
    # (i) 恒等式 ddG_ML == dG_ML(var) - dG_ML(WT)
    ident = []
    for w, d in list(dom.items()):
        pairs = [(g, d["cells"][k]) for k, g in []]  # 占位，下面重算
    # 重新流式取一小批做恒等式核对（只读前若干域，避免再全表扫）
    probe = {}
    with open(MAIN, newline="") as fh:
        for x in csv.DictReader(fh):
            w = x["WT_name"]
            if len(probe) >= 12 and w not in probe:
                continue
            if not SINGLE.match(x["mut_type"]):
                continue
            g, dd = _num(x["dG_ML"]), _num(x["ddG_ML"])
            if g is None or dd is None:
                continue
            probe.setdefault(w, []).append(g - dd)
    for w, v in probe.items():
        if len(v) >= 50:
            ident.append({"domain": w, "n": len(v),
                          "implied_wt_dG": float(np.mean(v)),
                          "sd": float(np.std(v))})
    assert ident, "恒等式核对取不到样本"
    worst = max(i["sd"] for i in ident)
    assert worst < 1e-6, f"ddG != dG - dG_wt，最大 sd = {worst}"
    res["T6a_identity"] = {"domains": ident, "max_sd": worst}
    P(f"   [T6a] 恒等式 ddG_ML == dG_ML(变体) − dG_ML(野生型)："
      f"{len(ident)} 个域逐域 sd ≤ {worst:.2e}，"
      f"隐含的野生型 dG 全部为正（"
      f"{min(i['implied_wt_dG'] for i in ident):.2f}–"
      f"{max(i['implied_wt_dG'] for i in ident):.2f} kcal/mol）"
      f"  => ΔG 大 = 更稳定，故 **ddG_ML 负 = 去稳定**  OK")

    # (ii) Pro 锚点（docs/15 V7 建立的已知事实，逐残基中位数口径）
    pro = []
    for w, d in dom.items():
        if d["ok"] < 400:
            continue
        by = defaultdict(list)
        for (p, a), v in d["cells"].items():
            by[a].append(v)
        med = {a: float(np.median(v)) for a, v in by.items() if len(v) >= 8}
        if len(med) < 18 or "P" not in med:
            continue
        order = sorted(med, key=lambda a: med[a])     # 最负（最去稳定）在前
        pro.append({"domain": w, "pro_rank": order.index("P") + 1,
                    "pro_med": med["P"], "n_res": len(med),
                    "frac_neg": float(np.mean(
                        [v < 0 for vs in by.values() for v in vs]))})
    assert len(pro) >= 100, f"Pro 锚点样本太少 {len(pro)}"
    r1 = sum(1 for z in pro if z["pro_rank"] == 1)
    r3 = sum(1 for z in pro if z["pro_rank"] <= 3)
    frac_neg_med = float(np.median([z["frac_neg"] for z in pro]))
    assert r3 / len(pro) >= 0.80, \
        f"Pro 只在 {r3}/{len(pro)} 个域排进最去稳定前三 —— 符号或解析可疑"
    assert frac_neg_med > 0.5, "多数取代不是去稳定方向 —— 符号可疑"
    res["T6b_pro_anchor"] = {
        "n_domains": len(pro), "pro_rank1": r1, "pro_rank_le3": r3,
        "pro_rank_hist": {int(k): int(v) for k, v in
                          sorted(Counter(z["pro_rank"]
                                         for z in pro).items())},
        "median_frac_destabilizing": frac_neg_med,
        "median_pro_median_ddG": float(np.median([z["pro_med"]
                                                  for z in pro]))}
    P(f"   [T6b] Pro 锚点（docs/15 V7 的逐残基中位数口径）："
      f"{len(pro)} 个 ≥400 观测的域里，Pro 排最去稳定**第 1 位 {r1} 个 "
      f"({r1 / len(pro):.0%})、前三位 {r3} 个 ({r3 / len(pro):.0%})**；"
      f"去稳定方向的取代占比中位 {frac_neg_med:.0%}  "
      f"=> 与 T6a 同向，**ddG_ML 负 = 去稳定**  OK")
    P("        （本模块对外输出 M = -ddG_ML，即**正 = 去稳定**，"
      "与 ddpca_load.matrix 同向）")

    # --- T10 动态范围窗口由数据自己给出 --------------------------------
    allv = np.concatenate([np.array(d["dg_in"]) for d in dom.values()
                           if d["dg_in"]])
    assert allv.min() > DG_LO - 1e-3 and allv.max() < DG_HI + 1e-3
    at_lo = float((allv <= DG_LO + 1e-4).mean())
    at_hi = float((allv >= DG_HI - 1e-4).mean())
    assert at_lo < 0.001 and at_hi < 0.001, \
        f"边界上有堆积 lo {at_lo:.4f} hi {at_hi:.4f} —— 机制判断要改"
    res["T10_window"] = {"lo": DG_LO, "hi": DG_HI, "n": int(len(allv)),
                         "min": float(allv.min()), "max": float(allv.max()),
                         "mass_at_lo": at_lo, "mass_at_hi": at_hi}
    P(f"   [T10] 数值型 dG_ML 严格落在 ({DG_LO}, {DG_HI}) 内"
      f"（实测 {allv.min():.4f}…{allv.max():.4f}，n = {len(allv)}）；"
      f"**边界上没有堆积**（= {DG_LO} 的质量 {at_lo:.5%}、"
      f"= {DG_HI} 的 {at_hi:.5%}）"
      f"  => 越界变体的值是**被扣留**（换成标记）而不是被压到边界  OK")

    # --- T7 量级 ------------------------------------------------------
    raw = np.concatenate([np.array(d["raw"]) for d in dom.values()
                          if d["raw"]])
    q = {f"q{p}": float(np.percentile(raw, p))
         for p in (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)}
    assert abs(raw).max() < 12.0, "ΔΔG 量级超出物理可能"
    res["T7_magnitude"] = {"n": int(len(raw)), "min": float(raw.min()),
                           "max": float(raw.max()),
                           "mean": float(raw.mean()), **q}
    P(f"   [T7] ΔΔG 量级（仓库约定 正 = 去稳定）：n = {len(raw)}，"
      f"范围 [{raw.min():.2f}, {raw.max():.2f}]，中位 {q['q50']:+.2f}，"
      f"1–99% 分位 [{q['q1']:+.2f}, {q['q99']:+.2f}] kcal/mol"
      f"  —— 单点取代的物理合理区间  OK")

    # --- T8 fitting CI 绝不作为噪声模型（D-028 的机制化留痕） -----------
    cis = np.concatenate([np.array(d["ci"]) for d in dom.values()
                          if d["ci"]])
    res["T8_fitting_ci"] = {"median_95CI": float(np.median(cis)),
                            "implied_SE": float(np.median(cis) / 3.92),
                            "max_95CI": float(cis.max())}
    P(f"   [T8] `deltaG_95CI` 中位 {np.median(cis):.3f} kcal/mol "
      f"⇒ 隐含 SE ≈ {np.median(cis) / 3.92:.3f} —— 与 ddPCA 的 bootstrap SE "
      f"同型（D-028）。**记录但不作为噪声模型使用。**")
    return res


# ----------------------------------------------------------------------
def main():
    print("=" * 92)
    print("docs/27 Step 1：Tsuboyama 2023 解析 + 已知事实自检")
    print("=" * 92)
    prov = os.path.join(DDIR, "tsuboyama_PROVENANCE.txt")
    assert os.path.exists(prov), "先跑 analysis/tsuboyama_acquire.sh"
    print(open(prov).read().strip())
    print("\n[扫描]")
    dom, wt_rows, meta = scan()
    chk = selfcheck(dom, wt_rows, meta)

    def _wtseq(w):
        ss = {r["aa_seq"] for r in wt_rows.get(w, [])}
        return ss.pop() if len(ss) == 1 else ""
    mats = {w: build(d, _wtseq(w)) for w, d in dom.items()}

    # ---- 纳入判据的「规模」部分（在看任何轴之前写定） -----------------
    print("\n" + "=" * 92)
    print("候选集（规模判据，在看任何轴之前写定；截尾闸与可分辨闸在 Step 2/3）")
    print("=" * 92)
    cand = {w: m for w, m in mats.items()
            if m["sites"] >= MIN_SITES and m["n_obs"] >= MIN_OBS}
    zero = [w for w, m in mats.items() if m["n_obs"] == 0]
    print(f"   全部 WT 背景 {len(mats)} 个")
    print(f"   其中 ΔΔG 全不可得（n_obs = 0）的 {len(zero)} 个"
          f" —— 机制：这些背景**自己的**野生型 dG_ML 就是 '>5'（或 '<-1'），")
    print(f"      所以 ddG 没有定义。Tsuboyama 对这些蛋白另做了去稳定背景"
          f"（如 1UBQ.pdb_I3A），本工作用那些背景。")
    print(f"   规模判据（位点 ≥ {MIN_SITES} 且观测 ≥ {MIN_OBS}）后："
          f"**{len(cand)} 个候选矩阵**")

    nat, des = {}, {}
    for w, m in cand.items():
        mm = NAT.match(w)
        m["parent"] = mm.group(1) if mm else None
        m["bg_mut"] = mm.group(2) if mm else None
        (nat if mm else des)[w] = m
    par = defaultdict(list)
    for w, m in nat.items():
        par[m["parent"]].append(w)
    print(f"   分层：天然结构域（PDB 编号命名）**{len(nat)}** 个背景 / "
          f"**{len(par)}** 个不同母体 PDB；"
          f"设计蛋白 {len(des)} 个背景")

    # 逐母体 PDB 取一个背景：f_out 最低者（判据与轴无关，确定性）
    primary = {}
    for p, ws in par.items():
        ws = sorted(ws, key=lambda w: (nat[w]["f_out"], -nat[w]["n_obs"], w))
        primary[p] = ws[0]
    print(f"   逐母体 PDB 取 f_out 最低的那个背景（确定性规则，与轴无关）"
          f" => **{len(primary)} 个独立天然结构域**")

    sub = np.array([cand[w]["subs_med"] for w in cand])
    print(f"\n   每位点取代数（候选集 {len(cand)} 个矩阵的逐矩阵中位数）："
          f"min {sub.min():.0f}、中位 {np.median(sub):.0f}、max {sub.max():.0f}")
    hist = Counter()
    for w in cand:
        for k, v in cand[w]["subs_hist"].items():
            hist[k] += v
    print(f"   全候选集逐位点取代数直方图（位点计）: "
          + ", ".join(f"{k}:{v}" for k, v in sorted(hist.items())
                      if v > 0))

    # ---- σ 校准 -------------------------------------------------------
    print("\n" + "=" * 92)
    print("σ 校准（D-028：绝不用 fitting CI；用体外文献 ΔG 对照）")
    print("=" * 92)
    lw = lit_sigma(window=True)
    la = lit_sigma(window=False)
    isig = internal_sigma()
    print(f"   {'蛋白':8}{'n':>6}{'r':>8}{'RMSE_ctr':>10}{'RMSE_pair':>11}")
    for p in lw["per_protein"]:
        print(f"   {p['pdb']:8}{p['n']:>6}{p['r']:>+8.3f}"
              f"{p['rmse_ctr']:>10.3f}{p['rmse_pair']:>11.3f}")
    print(f"   窗口内（dG ∈ [{DG_LO}, {DG_HI}]，n = {lw['n_total']}）："
          f"n 加权 RMSE_ctr = **{lw['weighted_rmse_ctr']:.3f}**、"
          f"逐蛋白中位 {lw['median_rmse_ctr']:.3f}、"
          f"两两差值 RMSE = {lw['weighted_rmse_pair']:.3f}"
          f"（/√2 = {lw['weighted_rmse_pair'] / math.sqrt(2):.3f}，自洽）")
    print(f"   不限窗口（n = {la['n_total']}）："
          f"n 加权 RMSE_ctr = {la['weighted_rmse_ctr']:.3f}")
    print(f"   内部一致性（胰蛋白酶 vs 胰凝乳蛋白酶，n = {isig['n']}，"
          f"r = {isig['r']:.3f}）：RMSE = {isig['rmse_t_vs_c']:.3f} ⇒ "
          f"单酶 σ ≈ {isig['sigma_per_assay']:.3f}、ML σ ≈ "
          f"{isig['sigma_of_ML']:.3f} —— **乐观下界，不作主 σ**")
    SIGMA = round(lw["weighted_rmse_ctr"], 2)
    print(f"\n   => 本工作采用的体外校准 **σ = {SIGMA:.2f} kcal/mol**"
          f"（落在 ddPCA 的 0.46–0.89 带内，docs/15 §1.1 V5）；"
          f"敏感性在 {lw['median_rmse_ctr']:.2f}（乐观）与 "
          f"{la['weighted_rmse_ctr']:.2f}（保守）各报一次。")

    out = {
        "provenance_file": os.path.relpath(prov, ROOT),
        "scan_meta": meta, "selfcheck": chk,
        "sigma": {"adopted": SIGMA, "lit_window": lw, "lit_all": la,
                  "internal": isig},
        "inclusion_scale": {"min_sites": MIN_SITES, "min_obs": MIN_OBS},
        "n_backgrounds": len(mats), "n_zero_ddG": len(zero),
        "zero_ddG_examples": sorted(zero)[:20],
        "n_candidates": len(cand), "n_natural_bg": len(nat),
        "n_natural_parents": len(par), "n_designed": len(des),
        "primary_natural": primary,
        "matrices": {w: {k: v for k, v in m.items()
                         if k not in ("M", "mask", "raw")}
                     for w, m in cand.items()},
    }
    p = os.path.join(ROOT, "cache", "tsuboyama_parse.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    np.savez_compressed(
        os.path.join(ROOT, "cache", "tsuboyama_matrices.npz"),
        **{f"{w}|M": m["M"] for w, m in cand.items()},
        **{f"{w}|raw": m["raw"] for w, m in cand.items()},
        **{f"{w}|pos": np.array(m["pos"]) for w, m in cand.items()})
    print(f"\n写入 {os.path.relpath(p, ROOT)} 与 cache/tsuboyama_matrices.npz")
    return 0


def load_cached():
    """下游脚本用：返回 (info_dict, {name: {M, mask, raw, pos, ...}})。"""
    info = json.load(open(os.path.join(ROOT, "cache",
                                       "tsuboyama_parse.json")))
    z = np.load(os.path.join(ROOT, "cache", "tsuboyama_matrices.npz"))
    mats = {}
    for w, m in info["matrices"].items():
        M = z[f"{w}|M"]
        mats[w] = dict(m)
        mats[w].update({"M": M, "mask": np.isfinite(M),
                        "raw": z[f"{w}|raw"], "pos": z[f"{w}|pos"]})
    return info, mats


if __name__ == "__main__":
    raise SystemExit(main())
