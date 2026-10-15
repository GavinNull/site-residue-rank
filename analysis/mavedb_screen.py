#!/usr/bin/env python3
"""MaveDB 系统筛查（docs/39 §1）。

闸的顺序与 docs/23 §1 / docs/25 一致，**统计之前先过闸**：
  F1 全量已发表 score set
  F2 蛋白质靶标（protein_coding）
  F3 结合相关（控制词 'Binding assay' ∪ 结合词表，召回优先）
  F4 读出分类（滴定型逐变体 Kd / 富集或分选 / 生长适应度 / 折叠稳定性 / 其他）
  F5 G5 家族独立（非丝氨酸蛋白酶-抑制剂、非冠状病毒 RBD/ACE2）
  F6 G1+G2 矩阵结构（位点数、每位点取代数、保护后能否抽 2 维子空间）

F4 的判读是人工的（用 methodText 原文），所以它以**数据表**的形式写在
本脚本里（`READOUT_CALLS`），每条带引用的 methodText 片段，便于复核。
自动分类器只用来**保证没有漏掉**：它把 F3 的全部条目打上标签，
任何被打成「可能是滴定型」的条目都必须出现在 READOUT_CALLS 里，否则中止。

自检（CLAUDE.md §10，不通过中止）：
  B1 已知事实：urn:mavedb:00000044-a-1（Starr 2020 SARS-CoV-2 RBD ACE2 结合）
     必须解析出 **201 个位点**（本仓库 `starr2020/WH1` 的已知位点数，
     docs/24 §3.4 / §3.5 引用的那个几何），每位点 20 个取代。
  B2 不误伤：一个合成的完备 12×20 秩 2 矩阵必须能过 G2
     （D-007：过敏护栏最终会被绕过）。
  B3 缺失不许读成 0：合成稀疏矩阵的观测格数必须等于植入数。
  B4 hgvs_pro 三字母 → 单字母映射必须对 20 种标准残基逐一命中，
     且必须拒绝 `Ter` / `=`（同义）/ 多点串。
  B5 分类器不误伤：对一条已知的**非**滴定型条目（00000090-a-1，
     bacterial two-hybrid）分类器不得给出「滴定型」。
"""
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import AA20, extract_subspace            # noqa: E402

API = "https://api.mavedb.org/api/v1"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MDB = os.path.join(ROOT, "cache", "mavedb")
SCORES = os.path.join(MDB, "scores")

THREE2ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}

BIND_VOCAB = re.compile(
    r"affinit|\bbinding\b|\bbinds\b|dissociation|titrat|tite-?seq|sort-?seq|"
    r"\bk_?d\b|\bspr\b|plasmon|biolayer|\bitc\b|two-hybrid|\bppi\b|display",
    re.I)

# 「可能是滴定型逐变体亲和力」的自动标记词：多浓度 + 拟合
TITR_VOCAB = re.compile(
    r"titration|range of concentration|concentration series|"
    r"apparent k_?d|\bk_?d,?app|equilibrium dissociation|binding curve|"
    r"binding isotherm|hill (?:coefficient|equation)|"
    r"surface plasmon|biolayer interferometr|isothermal titration",
    re.I)

# ---------------------------------------------------------------- F4 人工判读
# readout: titration_kd | enrichment_sort | growth_fitness | folding | other
# 每条给 methodText 的判据片段（原文），便于复核。
READOUT_CALLS = {
    "urn:mavedb:00000044-a": (
        "titration_kd",
        "Tite-seq 式：'binding score' 由多个 ACE2 浓度的滴定曲线拟合 "
        "log10(KD,app)；本仓库已有该数据（starr2020/WH1）"),
    "urn:mavedb:00000044-b": (
        "other", "同一实验的表达量（RBD 表面展示水平），不是结合量"),
    "urn:mavedb:00000668-a": (
        "titration_kd", "Starr 系列 Omicron BA.1 RBD，同一 Tite-seq 流程"),
    "urn:mavedb:00000670-a": (
        "titration_kd", "Starr 系列 Omicron BA.2 RBD，同一 Tite-seq 流程"),
    "urn:mavedb:00000069-a": (
        "enrichment_sort",
        "ACE2 变体对 spike 的结合，FACS 两档（high/low affinity）分选富集，"
        "不是逐变体滴定"),
    "urn:mavedb:00000077-a": (
        "enrichment_sort",
        "'Cells in the top 5% by fluorescence were used as the selected "
        "population for score calculation' —— 单门分选富集比"),
    "urn:mavedb:00000077-b": (
        "enrichment_sort", "同上（同一论文的第二个 cohesin 骨架）"),
    "urn:mavedb:00000090-a": (
        "enrichment_sort",
        "'selected for binding to CRIPT by bacteria two-hybrid' —— "
        "选择富集，不是 Kd"),
    "urn:mavedb:00000090-b": (
        "enrichment_sort", "同上（配体换成 T-2F）"),
    "urn:mavedb:00000047-b": (
        "enrichment_sort", "抗体 2D7 结合的 FACS 分选富集"),
    "urn:mavedb:00000047-c": (
        "enrichment_sort", "gp120-CD4 结合的 FACS 分选富集"),
    "urn:mavedb:00000048-b": (
        "enrichment_sort", "CXCL12-sfGFP 结合的 FACS 分选富集"),
    "urn:mavedb:00000048-c": (
        "enrichment_sort", "抗体 12G5 结合的 FACS 分选富集"),
    "urn:mavedb:00000038-a": (
        "enrichment_sort",
        "'deep sequencing was used to quantify enrichment or depletion of "
        "cells in the bound population'"),
    "urn:mavedb:00000038-b": ("enrichment_sort", "同上（excess E1）"),
    "urn:mavedb:00001239-a": (
        "enrichment_sort",
        "四档分选 + 'predefined scoring equation'；insulin / mAb 83-07 / "
        "mAb 83-14 三个配体 + 表面表达对照，设计正确但读出是功能分"),
    "urn:mavedb:00000115-a": (
        "growth_fitness",
        "doubledeepPCA（酵母蛋白互补），读出是生长适应度；其 "
        "'binding free energy changes' 列是三状态反解量 —— "
        "docs/28 / docs/29 已判定该类交互项不可用"),
    "urn:mavedb:00000105-a": (
        "enrichment_sort",
        "GB1 对 IgG-Fc 的 mRNA display 富集；本仓库曾有该矩阵，"
        "已按人类决定剔除（docs/21 §4）"),
    "urn:mavedb:00000002-a": ("enrichment_sort", "WW 域噬菌体/酵母展示富集"),
    "urn:mavedb:00000003-b": ("enrichment_sort", "BRCA1 酵母双杂交富集"),
    "urn:mavedb:00000004-a": ("enrichment_sort", "E4B U-box 自泛素化展示富集"),
    "urn:mavedb:00000010-a": ("growth_fitness", "Pab1 RRM 酵母生长"),
    "urn:mavedb:00000012-a": ("enrichment_sort", "Gal4 报告基因选择"),
    "urn:mavedb:00000036-a": ("enrichment_sort", "Y2H 富集（imputed/refined）"),
    "urn:mavedb:00000046-a": (
        "enrichment_sort", "CD86 跨膜域 ZsGreen 分选；'titrated' 指病毒滴度"),
    "urn:mavedb:00000053-a": ("enrichment_sort", "PSD95 PDZ3 成对突变富集"),
    "urn:mavedb:00000082-a": (
        "enrichment_sort",
        "403 个单点（全序列 1–418，≈每位点 1 个取代）的 mCherry 结合读数"),
    "urn:mavedb:00001247-a": ("enrichment_sort", "GCK 酵母双杂交"),
    # 以下三条是自动标记器的假阳性，逐条读过原文后归类（CLAUDE.md §10.2：
    # 既要测能命中，也要测不误伤 —— 这三条正是「不误伤」的实测）
    "urn:mavedb:00000071-a": (
        "growth_fitness",
        "'titrations for negative selection, positive selection' 指的是"
        "转化量滴定，不是配体浓度滴定；读出是正/负选择"),
    "urn:mavedb:00000071-b": ("growth_fitness", "同上（负选择）"),
    # PYL1（Lehner lab GluePCA，Nat Commun 2026 10.1038/s41467-026-70341-2）：
    # 12 个 (+)-ABA 浓度的 dose-response + 2 个丰度臂。
    # **控制词 `Binding assay` 的 14 个 experiment 全是它。**
    # 读出是**生长速率**（GluePCA 的 DHFR 互补），score 列是
    # `lib_tile_normalised_growth_rate` / `raw_growth_rate` ——
    # 按任务判据，这是「convolves stability with binding 的生长表型」，
    # 论文自己也因此单列了两个 homodimer abundance 臂。
    "urn:mavedb:00001287-a": (
        "growth_fitness",
        "GluePCA 生长速率（dose-response 的一个 (+)-ABA 浓度点）"),
    "urn:mavedb:00001288-a": (
        "growth_fitness",
        "GluePCA 生长速率（dose-response 的一个 (+)-ABA 浓度点）"),
    "urn:mavedb:00001289-a": (
        "growth_fitness",
        "GluePCA 生长速率（dose-response 的一个 (+)-ABA 浓度点）"),
    "urn:mavedb:00001290-a": (
        "growth_fitness",
        "GluePCA 生长速率（dose-response 的一个 (+)-ABA 浓度点）"),
    "urn:mavedb:00001291-a": (
        "growth_fitness",
        "GluePCA 生长速率（dose-response 的一个 (+)-ABA 浓度点）"),
    "urn:mavedb:00001292-a": (
        "growth_fitness",
        "GluePCA 生长速率（dose-response 的一个 (+)-ABA 浓度点）"),
    "urn:mavedb:00001293-a": (
        "growth_fitness",
        "GluePCA 生长速率（dose-response 的一个 (+)-ABA 浓度点）"),
    "urn:mavedb:00001294-a": (
        "growth_fitness",
        "GluePCA 生长速率（dose-response 的一个 (+)-ABA 浓度点）"),
    "urn:mavedb:00001295-a": (
        "growth_fitness",
        "GluePCA 生长速率（dose-response 的一个 (+)-ABA 浓度点）"),
    "urn:mavedb:00001296-a": (
        "growth_fitness",
        "GluePCA 生长速率（dose-response 的一个 (+)-ABA 浓度点）"),
    "urn:mavedb:00001297-a": (
        "growth_fitness",
        "GluePCA 生长速率（dose-response 的一个 (+)-ABA 浓度点）"),
    "urn:mavedb:00001298-a": (
        "growth_fitness",
        "GluePCA 生长速率（dose-response 的一个 (+)-ABA 浓度点）"),
    "urn:mavedb:00001299-a": (
        "other", "PYL1 同二聚体丰度臂（0 nM ABA）—— 丰度，不是结合"),
    "urn:mavedb:00001300-a": (
        "other", "PYL1 同二聚体丰度臂（250 µM ABA）"),
    # Tier A 的 4 条残留（标题带结合措辞但读出不是结合量），逐条读过后归类
    "urn:mavedb:00000072-a": (
        "folding", "IgG1 Fc **stability**（热/蛋白酶稳定性），不是结合量"),
    "urn:mavedb:00000665-a": (
        "growth_fitness", "PAX6 geneticin 选择生长（±geneticin 四个臂）"),
    "urn:mavedb:00000665-b": ("growth_fitness", "同上"),
    "urn:mavedb:00000665-c": ("growth_fitness", "同上（无 geneticin 对照）"),
    "urn:mavedb:00000665-d": ("growth_fitness", "同上"),
    "urn:mavedb:00000672-c": (
        "other", "PLpro leaky expression —— 表达量，不是结合量"),
    "urn:mavedb:00001286-a": (
        "growth_fitness", "TINF2 saturation genome editing，细胞存活"),
    "urn:mavedb:00001204-a": (
        "growth_fitness",
        "'drug titration assays' 用于定药物 IC50 浓度，读出是细胞增殖"),
    "urn:mavedb:00001206-a": ("growth_fitness", "同上"),
}

# F5：家族排除（G5）
EXCLUDE_FAMILY = {
    "urn:mavedb:00000044-a": "冠状病毒 RBD/ACE2（且本仓库已含）",
    "urn:mavedb:00000044-b": "冠状病毒 RBD/ACE2",
    "urn:mavedb:00000668-a": "冠状病毒 RBD/ACE2（Omicron BA.1）",
    "urn:mavedb:00000670-a": "冠状病毒 RBD/ACE2（Omicron BA.2）",
    "urn:mavedb:00000069-a": "冠状病毒 RBD/ACE2（ACE2 一侧）",
}


def _get(path, timeout=240, tries=10, raw=False):
    for i in range(tries):
        try:
            req = urllib.request.Request(
                API + path, headers={"Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                return fh.read()
        except Exception as exc:                              # noqa: BLE001
            print(f"    retry {i + 1}/{tries} {exc}", file=sys.stderr)
            time.sleep(min(30, 4 * (i + 1)))
    raise RuntimeError("GET failed: " + path)


# ------------------------------------------------------------------ 解析层
def parse_hgvs_pro(s):
    """返回 (site:int, wt:str, mut:str) 或 None（不是单点 missense）。

    必须拒绝：同义（`p.Xxx12=` 或 wt == mut）、终止（Ter / *）、
    多点（含 ';'）、插入/删除（ins/del/dup/fs）。
    """
    if not s or s in ("NA", ""):
        return None
    s = s.strip()
    if ";" in s or "[" in s:
        return None
    if re.search(r"ins|del|dup|fs|ext|\?", s):
        return None
    m = re.fullmatch(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|=|\*|Ter)", s)
    if not m:
        return None
    wt3, pos, mu3 = m.group(1), int(m.group(2)), m.group(3)
    if mu3 in ("=", "*", "Ter"):
        return None
    wt, mu = THREE2ONE.get(wt3), THREE2ONE.get(mu3)
    if wt is None or mu is None or wt == mu:
        return None
    return pos, wt, mu


def matrix_from_scores(rows, score_col="score"):
    """rows: csv DictReader 的行。返回 (M, mask, sites, wt_by_site, stats)。

    同一 (site, mut) 多行（多个 library / barcode）取均值。
    缺失**不填 0**，由 mask 携带（D-023）。
    """
    acc, wt_by_site = {}, {}
    n_rows = n_parsed = 0
    for r in rows:
        n_rows += 1
        p = parse_hgvs_pro(r.get("hgvs_pro"))
        if p is None:
            continue
        site, wt, mu = p
        try:
            v = float(r[score_col])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        n_parsed += 1
        wt_by_site.setdefault(site, set()).add(wt)
        acc.setdefault((site, mu), []).append(v)
    if not acc:
        return None, None, [], {}, {"rows": n_rows, "parsed": 0}
    sites = sorted({s for s, _ in acc})
    idx = {s: i for i, s in enumerate(sites)}
    M = np.zeros((len(sites), 20))
    mask = np.zeros((len(sites), 20), bool)
    for (s, mu), vals in acc.items():
        j = AA20.index(mu)
        M[idx[s], j] = float(np.mean(vals))
        mask[idx[s], j] = True
    per_site = mask.sum(1)
    multi_wt = {s: sorted(v) for s, v in wt_by_site.items() if len(v) > 1}
    stats = {
        "rows": n_rows, "parsed": n_parsed,
        "n_sites": len(sites), "n_obs": int(mask.sum()),
        "subs_per_site_min": int(per_site.min()),
        "subs_per_site_median": float(np.median(per_site)),
        "subs_per_site_max": int(per_site.max()),
        "subs_per_site_mean": round(float(per_site.mean()), 2),
        "sites_with_ge4": int((per_site >= 4).sum()),
        "n_residue_types_used": int(mask.any(0).sum()),
        "wt_multivalued_sites": len(multi_wt),
    }
    return M, mask, sites, {s: sorted(v) for s, v in wt_by_site.items()}, stats


# ------------------------------------------------------------------- 自检
def selftest():
    fails, notes = [], {}

    # B4 三字母映射 + 拒绝规则
    for i, (three, one) in enumerate(THREE2ONE.items()):
        got = parse_hgvs_pro(f"p.Gly{i + 1}{three}")
        if one == "G":
            if got is not None:
                fails.append("B4 同义串未被拒绝：Gly->Gly")
        elif got != (i + 1, "G", one):
            fails.append(f"B4 映射错：p.Gly{i + 1}{three} -> {got}")
    for bad in ["p.Gly12Ter", "p.Gly12*", "p.Gly12=", "p.Gly12Ala;p.Ala13Gly",
                "p.Gly12_Ala13del", "NA", "", "c.36G>A"]:
        if parse_hgvs_pro(bad) is not None:
            fails.append(f"B4 应被拒绝但没被拒绝：{bad!r}")
    if parse_hgvs_pro("p.Gly12Ala") != (12, "G", "A"):
        fails.append("B4 正例失败：p.Gly12Ala")
    notes["B4"] = "ok" if not fails else "FAIL"

    # B3 缺失不许读成 0
    rows = [{"hgvs_pro": "p.Gly1Trp", "score": "1.0"},
            {"hgvs_pro": "p.Ala2Gly", "score": "-1.0"},
            {"hgvs_pro": "p.Ala2Trp", "score": "2.0"},
            {"hgvs_pro": "p.Ala3Ter", "score": "9.0"}]
    M, mask, sites, _, st = matrix_from_scores(rows)
    if st["n_obs"] != 3:
        fails.append(f"B3 观测格数应为 3，实际 {st['n_obs']}")
    if mask[0, AA20.index("W")] is np.False_ or not mask[0, AA20.index("W")]:
        fails.append("B3 残基列定位错（W）")
    if mask.sum() != 3 or M[mask].tolist().count(9.0):
        fails.append("B3 终止变体被读进了矩阵")
    notes["B3"] = "ok"

    # B2 不误伤：完备 12x20 秩 2 必须过 G2
    rng = np.random.default_rng(0)
    U = rng.normal(size=(12, 2))
    V = rng.normal(size=(20, 2))
    Mc, mc = U @ V.T, np.ones((12, 20), bool)
    B, cols, err = extract_subspace(Mc, mc, rank=2)
    if B is None:
        fails.append(f"B2 不误伤失败：完备秩 2 矩阵过不了 G2（{err}）")
    notes["B2"] = "ok"

    # B5 分类器不误伤
    neg = ("Deep mutational scan of Rat DLG4 PDZ3 domain selected for binding "
           "to CRIPT by bacteria two-hybrid in E. coli.")
    if TITR_VOCAB.search(neg):
        fails.append("B5 分类器把 two-hybrid 误标为滴定型")
    pos = ("Binding scores were determined by fitting a titration curve of "
           "apparent KD across a range of concentrations.")
    if not TITR_VOCAB.search(pos):
        fails.append("B5 分类器漏掉了明确的滴定型描述")
    notes["B5"] = "ok"

    if fails:
        raise SystemExit("自检不通过：\n  " + "\n  ".join(fails))
    return notes


def selftest_known_matrix():
    """B1：用已知事实（Starr 2020 RBD = 201 位点 × 20 取代）校解析层。"""
    urn = "urn:mavedb:00000044-a-1"
    raw = fetch_scores(urn)
    M, mask, sites, wt, st = matrix_from_scores(
        csv.DictReader(io.StringIO(raw.decode())))
    fails = []
    if st["n_sites"] != 201:
        fails.append(f"B1 位点数 {st['n_sites']} != 已知 201"
                     f"（docs/24 §3.4 的 starr2020/WH1 几何）")
    if st["subs_per_site_median"] != 19.0:
        fails.append(f"B1 每位点取代数中位 {st['subs_per_site_median']}"
                     f" != 19（20 种残基里扣掉野生型）")
    if st["wt_multivalued_sites"] != 0:
        fails.append(f"B1 野生型残基在 {st['wt_multivalued_sites']} 个位点"
                     f"多值 —— 编号或解析错位")
    if fails:
        raise SystemExit("B1 自检不通过：\n  " + "\n  ".join(fails))
    return st


def fetch_scores(urn):
    os.makedirs(SCORES, exist_ok=True)
    fn = os.path.join(SCORES, urn.replace(":", "_") + ".csv")
    if os.path.exists(fn):
        with open(fn, "rb") as fh:
            return fh.read()
    raw = _get(f"/score-sets/{urn}/scores")
    with open(fn, "wb") as fh:
        fh.write(raw)
    time.sleep(0.3)
    return raw


def main():
    notes = selftest()
    print("解析层自检：", json.dumps(notes, ensure_ascii=False))
    b1 = selftest_known_matrix()
    print("B1 已知事实（Starr RBD）：", json.dumps(b1, ensure_ascii=False))

    R = json.load(open(os.path.join(MDB, "mavedb_scoresets.json")))
    E = {e["urn"]: e for e in json.load(
        open(os.path.join(MDB, "mavedb_experiments_full.json")))}

    funnel = {}
    funnel["F1_all_published_score_sets"] = len(R)
    funnel["F1_experiments"] = len(set(r["experiment"]["urn"] for r in R))
    funnel["F1_experiment_sets"] = len(
        set(r["experiment"]["experimentSetUrn"] for r in R))

    # F2 蛋白靶标
    F2 = [r for r in R
          if any(g["category"] == "protein_coding" for g in r["targetGenes"])]
    funnel["F2_protein_coding_score_sets"] = len(F2)

    # F3 结合相关
    def etext(eu):
        e = E.get(eu, {})
        return " ".join(str(e.get(k) or "") for k in
                        ["title", "shortDescription", "abstractText",
                         "methodText"])

    def kw_binding(eu):
        for k in (E.get(eu, {}).get("keywords") or []):
            if k["keyword"]["label"] == "Binding assay":
                return True
        return False

    F3, F3_kw = [], set()
    for r in F2:
        eu = r["experiment"]["urn"]
        own = " ".join(str(r.get(k) or "") for k in
                       ["title", "shortDescription"])
        if kw_binding(eu):
            F3_kw.add(eu)
        if kw_binding(eu) or BIND_VOCAB.search(own) or BIND_VOCAB.search(
                etext(eu)):
            F3.append(r)
    funnel["F3_binding_vocab_score_sets"] = len(F3)
    funnel["F3_binding_vocab_experiments"] = len(
        set(r["experiment"]["urn"] for r in F3))
    funnel["F3_controlled_keyword_Binding_assay_experiments"] = len(F3_kw)
    # 硬断言：控制词是本库里精度最高的那个种子，**不许留在 unreviewed**。
    # 第一版脚本没有这条断言，于是 14 个 `Binding assay` 的 experiment
    # （全是 PYL1 的 ABA dose-response）静悄悄落进了 678 个未读里 ——
    # 见 docs/39 §2.5。
    kw_missing = sorted(u for u in F3_kw if u not in READOUT_CALLS)
    if kw_missing:
        raise SystemExit(
            "控制词 `Binding assay` 的 experiment 必须逐条读过 methodText：\n  "
            + "\n  ".join(kw_missing))

    # F3b 滴定型自动标记（保证 F4 没有漏）—— 两条独立通路
    #  (i) experiment 级 abstractText + methodText
    #  (ii) score set 级 abstractText + methodText（search 响应里没有这两个
    #       字段，必须逐条取全记录；不取就等于 D-014 那种「程序欠功效」）
    titr_auto = sorted({r["experiment"]["urn"] for r in F3
                        if TITR_VOCAB.search(etext(r["experiment"]["urn"]))})
    funnel["F3b_titration_vocab_experiments_via_experiment_text"] = len(
        titr_auto)

    ss_full, titr_ss = {}, []
    os.makedirs(os.path.join(MDB, "full"), exist_ok=True)
    for i, r in enumerate(F3):
        fn = os.path.join(MDB, "full", r["urn"].replace(":", "_") + ".json")
        if os.path.exists(fn):
            with open(fn, "rb") as fh:
                raw = fh.read()
        else:
            raw = _get(f"/score-sets/{r['urn']}")
            with open(fn, "wb") as fh:
                fh.write(raw)
            time.sleep(0.15)
        d = json.loads(raw)
        ss_full[r["urn"]] = d
        own = " ".join(str(d.get(k) or "") for k in
                       ["title", "shortDescription", "abstractText",
                        "methodText"])
        if TITR_VOCAB.search(own):
            titr_ss.append(r["urn"])
        if (i + 1) % 200 == 0:
            print(f"    full score-set records {i + 1}/{len(F3)}", flush=True)
    funnel["F3b_titration_vocab_score_sets_via_scoreset_text"] = len(titr_ss)
    funnel["F3b_titration_vocab_score_sets_list"] = sorted(titr_ss)

    titr_all = sorted(set(titr_auto) | {
        s["experiment"]["urn"] for s in F3 if s["urn"] in titr_ss})
    funnel["F3b_titration_vocab_experiments_union"] = len(titr_all)
    missing = [u for u in titr_all if u not in READOUT_CALLS]
    if missing:
        raise SystemExit(
            "自动标记为「可能滴定型」但没有人工判读的条目（必须逐条读过）：\n  "
            + "\n  ".join(missing))

    # F3c Tier A：score set **自己**的标题/描述带结合措辞的，必须逐条交代。
    #     剩下的按显式规则归到折叠侧（蛋白酶解 / domainome aPCA），
    #     规则之外若还有残留就中止 —— 不许默默留白。
    OWN_BIND = re.compile(r"affinit|\bbinding\b|\bbinds\b|dissociation|"
                          r"titrat|\bk_?d\b|two-hybrid|display", re.I)
    FOLD_RULE = re.compile(r"trypsin digestion|chymotrypsin digestion|"
                           r"combined scores|domainome", re.I)
    tierA, tierA_fold, tierA_left = set(), set(), set()
    for r in F3:
        own = " ".join(str(r.get(k) or "") for k in
                       ["title", "shortDescription"])
        eown = " ".join(str(E[r["experiment"]["urn"]].get(k) or "") for k in
                        ["title", "shortDescription"])
        if not (OWN_BIND.search(own) or OWN_BIND.search(eown)):
            continue
        tierA.add(r["urn"])
        if r["experiment"]["urn"] in READOUT_CALLS:
            continue
        if FOLD_RULE.search(own):
            tierA_fold.add(r["urn"])
        else:
            tierA_left.add(r["urn"])
    funnel["F3c_tierA_score_sets_own_binding_wording"] = len(tierA)
    funnel["F3c_tierA_folding_by_rule"] = len(tierA_fold)
    funnel["F3c_tierA_unaccounted"] = sorted(tierA_left)
    if tierA_left:
        print("Tier A 未交代的条目（必须逐条读）：", file=sys.stderr)
        for u in sorted(tierA_left):
            print("   ", u, "|", next(x["title"] for x in F3
                                      if x["urn"] == u), file=sys.stderr)

    # F4 读出分类（人工判读，按 experiment 级）
    by_readout = {}
    for r in F3:
        eu = r["experiment"]["urn"]
        call = READOUT_CALLS.get(eu)
        label = call[0] if call else "unreviewed"
        by_readout.setdefault(label, set()).add(eu)
    funnel["F4_readout_classes_experiments"] = {
        k: len(v) for k, v in sorted(by_readout.items())}
    funnel["F4_titration_kd_experiments"] = sorted(
        by_readout.get("titration_kd", []))

    # F5 家族独立
    F5 = [u for u in by_readout.get("titration_kd", [])
          if u not in EXCLUDE_FAMILY]
    funnel["F5_titration_kd_after_family_exclusion"] = F5
    funnel["F5_excluded"] = {u: EXCLUDE_FAMILY[u]
                             for u in by_readout.get("titration_kd", [])
                             if u in EXCLUDE_FAMILY}

    # F6 对「设计上接近、只在读出闸失败」的一组实测矩阵几何
    #    （任务要求：每个候选都要报每位点取代数分布）
    probe = ["urn:mavedb:00000077-a-1", "urn:mavedb:00000077-b-1",
             "urn:mavedb:00000090-a-1", "urn:mavedb:00000090-b-1",
             "urn:mavedb:00000047-b-1", "urn:mavedb:00000047-c-1",
             "urn:mavedb:00000048-b-1", "urn:mavedb:00000048-c-1",
             "urn:mavedb:00000038-a-1", "urn:mavedb:00000082-a-1",
             "urn:mavedb:00001239-a-2", "urn:mavedb:00001239-a-3",
             "urn:mavedb:00001239-a-6", "urn:mavedb:00000105-a-1",
             "urn:mavedb:00000069-a-1", "urn:mavedb:00000115-a-22",
             "urn:mavedb:00000115-a-13", "urn:mavedb:00000002-a-2",
             "urn:mavedb:00000053-a-2",
             "urn:mavedb:00001292-a-1", "urn:mavedb:00001287-a-1",
             "urn:mavedb:00001299-a-1"]
    geom = {}
    for urn in probe:
        try:
            raw = fetch_scores(urn)
        except Exception as exc:                              # noqa: BLE001
            geom[urn] = {"error": str(exc)}
            continue
        M, mask, sites, wt, st = matrix_from_scores(
            csv.DictReader(io.StringIO(raw.decode())))
        st["sha256"] = hashlib.sha256(raw).hexdigest()[:16]
        st["bytes"] = len(raw)
        if M is None:
            st["G2"] = "无可解析的单点 missense"
        else:
            B, cols, err = extract_subspace(M, mask, rank=2)
            st["G2"] = "过" if B is not None else f"不过：{err}"
            st["G2_cols_kept"] = int(len(cols)) if cols is not None else 0
        geom[urn] = st
        print(f"  {urn}: sites={st['n_sites']} obs={st['n_obs']} "
              f"subs/site {st['subs_per_site_min']}/"
              f"{st['subs_per_site_median']}/{st['subs_per_site_max']} "
              f"G2={st['G2']}", flush=True)

    out = {
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selftest": notes, "selftest_known_matrix_starr_rbd": b1,
        "funnel": funnel,
        "readout_calls": {k: list(v) for k, v in READOUT_CALLS.items()},
        "geometry": geom,
        "script": "analysis/mavedb_screen.py",
    }
    with open(os.path.join(ROOT, "cache", "mavedb_screen.json"), "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(json.dumps(funnel, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
