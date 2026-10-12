#!/usr/bin/env python3
"""第三个独立家族的穷尽本地筛查（docs/23）。

docs/21 §9.4 把「找第三个独立家族」列为比扩 OMTKY3 蛋白酶数更重要的动作。
本脚本**只做筛查**，不改任何已落盘的论证，不把外部数据引入证据集
（CLAUDE.md §9.1 的人类决策点）。

数值层全部复用，没有重写：
  subspace_lib.extract_subspace / align / bootstrap_subspaces / load_scales
                / load_skempi / load_independent / dedup_by_content
  censor_density.censor_fraction_density_ci   （第三代，唯一通过双向标定的）
  omtky3_panel.parse_mut / ddg_from_row       （SKEMPI v2 原生解析）
  rank_test_power.als_guarded（经 subspace_lib 的 method="als" 分支）

判据（与 docs/13 §4.3 一致，不新造）：
  G1 结构    : 必须是 site × residue 的结合量矩阵（不是组合变体、不是位点标量）
  G2 采样    : MIN_PER_ROW=4 / MIN_PER_COL=6 的保护后能抽出子空间
  G3 截尾    : 第三代密度型 < 25%（docs/13 §4.1 的剔除判据）
  G4 可分辨  : 对半分自对齐 cos1（docs/13 §4.3 的保守口径），
               算不出来时退回 bootstrap stab cos2 >= 0.60
  G5 家族独立: 不是丝氨酸蛋白酶/抑制剂，也不是冠状病毒 RBD/ACE2

每一个候选都进表，**失败的理由必须写下来**（CLAUDE.md §5：
一个只报成功项的筛查会制造出一个第三家族）。
"""
import collections
import csv
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import (AA20, ROOT, MIN_PER_COL, MIN_PER_ROW,  # noqa: E402
                          align, bootstrap_subspaces, ci, dedup_by_content,
                          extract_subspace, haar_pair_null, load_independent,
                          load_scales, load_skempi, pick_primary)
from censor_density import censor_fraction_density_ci  # noqa: E402
from omtky3_panel import ddg_from_row, parse_mut  # noqa: E402

DMS = os.path.join(ROOT, "cache", "dms")
AAI = {a: i for i, a in enumerate(AA20)}
CENSOR_MAX = 0.25
STAB_MIN = 0.60
N_BOOT = 2000
N_SPLIT = 400


# ----------------------------------------------------------------------
# 0. 工具自检（CLAUDE.md §10：先自验，不过不许看下面的结论）
# ----------------------------------------------------------------------
def gate_selfcheck():
    """本脚本新增的唯一解析逻辑是 `mat_from_cells` 与各候选的突变串解析。
    两者都必须用**已知事实**自检，不用自洽性。"""
    print("=" * 96)
    print("[自检] CLAUDE.md §10 —— 不过不许读下面任何数字")
    print("=" * 96)
    ok = True

    # [M1] mat_from_cells 的残基顺序：植入一个只有 Trp 列有值的矩阵，
    #      必须落在 AA20.index('W')，而不是别的列（D-023 是残基顺序交错）。
    cells = {(f"s{i}", "W"): 1.0 for i in range(8)}
    cells.update({(f"s{i}", "G"): -1.0 for i in range(8)})
    M, mask, sites = mat_from_cells(cells)
    hit = sorted(np.where(mask.any(axis=0))[0])
    want = sorted([AAI["W"], AAI["G"]])
    print(f"  [M1] 残基列定位 植入 W/G -> 列 {hit}，期望 {want}  "
          f"{'OK' if hit == want else 'FAIL'}")
    ok &= hit == want
    v_ok = (M[:, AAI["W"]][mask[:, AAI["W"]]] == 1.0).all()
    print(f"  [M1b] 数值未被转置/错位 (W 列全为 +1.0)  "
          f"{'OK' if v_ok else 'FAIL'}")
    ok &= bool(v_ok)

    # [M2] 缺失不许读成 0（D-023 的第二个错）。上面矩阵只有 2 列有值，
    #      其余 18 列必须 mask=False 且**不参与**任何统计。
    nz = int(mask.sum())
    print(f"  [M2] 缺失标记 观测格子 {nz}，期望 16；"
          f"M 中非观测位置是否被当成 0 参与: "
          f"{'否 OK' if nz == 16 else '是 FAIL'}")
    ok &= nz == 16

    # [M3] 不误伤：一个**真的**有 20 列、12 位点的完备矩阵必须能抽出子空间
    #      （过度敏感的保护最终会被绕过，D-007）。
    rng = np.random.default_rng(0)
    cells = {}
    u = rng.normal(size=(12, 2)); v = rng.normal(size=(20, 2))
    for i in range(12):
        for j, a in enumerate(AA20):
            cells[(f"t{i}", a)] = float(u[i] @ v[j])
    M2, mk2, _ = mat_from_cells(cells)
    B, cols, err = extract_subspace(M2, mk2)
    print(f"  [M3] 不误伤 完备 12x20 矩阵能抽子空间: "
          f"{'OK' if B is not None else 'FAIL ' + str(err)}")
    ok &= B is not None

    # [M4] 突变串解析器：用已知事实（手写的四个串）核对，
    #      并**故意**给两个必须被拒绝的串（同义、多点）。
    cases = [("222-1C06>VH:D110Q-CAG", [("VH", "D", 110, "Q")]),
             ("319-345>VL:A9A-GCC;V20V-GTC", []),
             ("x>VH:C22F-TTC|VL:T5W-TGG", [("VH", "C", 22, "F"),
                                           ("VL", "T", 5, "W")]),
             ("Q21G", [(None, "Q", 21, "G")])]
    for s, want_ in cases:
        got = parse_variant(s)
        good = got == want_
        print(f"  [M4] {s[:34]:<36} -> {got}  {'OK' if good else 'FAIL'}")
        ok &= good

    # [M5] aaindex 的第三次独立核验（docs/21 §6 的同一条；烧过两次）
    sc = load_scales(verbose=False)
    mr = sc["molar_refractivity"]
    amin, amax = AA20[int(np.nanargmin(mr))], AA20[int(np.nanargmax(mr))]
    print(f"  [M5] 摩尔折射度 argmin = {amin} ({np.nanmin(mr):.1f}), "
          f"argmax = {amax} ({np.nanmax(mr):.1f})  "
          f"{'OK' if (amin, amax) == ('G', 'W') else 'FAIL'}")
    ok &= (amin, amax) == ("G", "W")

    print("-" * 96)
    print(f"自检总判定: {'全部通过' if ok else '失败 —— 中止'}")
    if not ok:
        raise SystemExit("自检未通过，按 CLAUDE.md §10 中止")
    print()
    return ok


def mat_from_cells(cells):
    """{(site, mut_res): value} -> (M, mask, sites)。残基顺序严格按 AA20。"""
    sites = sorted({s for s, _ in cells})
    idx = {s: i for i, s in enumerate(sites)}
    M = np.zeros((len(sites), 20))
    mask = np.zeros((len(sites), 20), bool)
    for (s, a), v in cells.items():
        if a not in AAI:
            continue
        M[idx[s], AAI[a]] = v
        mask[idx[s], AAI[a]] = True
    return M, mask, sites


_VSEG = re.compile(r"^(?:(V[HL]):)?([A-Z])(\d+)([A-Z*])")


def parse_variant(s):
    """magma 的变体串 -> 非同义单点突变列表 [(chain, wt, pos, mut), ...]。
    同义（wt == mut）与 '*' 一律丢弃。多点突变保留全部，由调用方决定。"""
    body = s.split(">", 1)[1] if ">" in s else s
    out, chain = [], None
    for seg in re.split(r"[;|]", body):
        m = _VSEG.match(seg.strip())
        if not m:
            continue
        c, wt, pos, mut = m.groups()
        if c:
            chain = c
        if mut == "*" or mut == wt:
            continue
        out.append((c if c else chain, wt, int(pos), mut))
    return out


# ----------------------------------------------------------------------
# 1. 候选加载器（每个都返回 0..n 个 {M, mask, sites, ...} 记录 + 结构说明）
# ----------------------------------------------------------------------
def cand_pafa():
    """HT-MEK PafA（Markin et al. 2021）的两个小分子抑制剂（VO4 / WO4）。

    这是任务里最被看好的一个：PafA 是磷酸酶（既非蛋白酶/抑制剂复合物，
    也非病毒 RBD），VO4 与 WO4 是**同骨架 × 两个小分子配体**。
    所以必须先量它的结构，而不是先量它的截尾。
    """
    out = []
    for lig in ("VO4", "WO4"):
        p = os.path.join(DMS, f"htmek_PafA_{lig}_inhibition.csv")
        rows = list(csv.DictReader(open(p)))
        per_site = collections.defaultdict(lambda: collections.defaultdict(list))
        aa_seen = collections.Counter()
        for r in rows:
            m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", r["MutantID"].strip())
            if not m:
                continue
            wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
            if wt == mut or mut not in AAI:
                continue
            try:
                v = float(r["KiNormMedian"])
            except (ValueError, KeyError):
                continue
            if not (v > 0):
                continue
            # ΔΔG_bind = RT ln(Ki_mut / Ki_wt)；KiNorm 已是 Ki_mut/Ki_wt
            per_site[(wt, pos)][mut].append(1.9872041e-3 * 298.15 * math.log(v))
            aa_seen[mut] += 1
        cells = {}
        for (wt, pos), d in per_site.items():
            for mut, vs in d.items():
                cells[(f"{wt}{pos}", mut)] = float(np.mean(vs))
        M, mask, sites = mat_from_cells(cells)
        subs = collections.Counter(int(x) for x in mask.sum(axis=1))
        out.append({"name": f"htmek_PafA/{lig}", "M": M, "mask": mask,
                    "sites": sites, "n_obs": int(mask.sum()),
                    "family": "alkaline phosphatase (PafA) / oxyanion",
                    "design": "same backbone x 2 small-molecule ligands",
                    "note": (f"替换到的残基只有 {len(aa_seen)} 种 "
                             f"{dict(aa_seen)}；每位点取代数分布 {dict(sorted(subs.items()))}"),
                    "n_distinct_mut_aa": len(aa_seen),
                    "max_subs_per_site": int(mask.sum(axis=1).max())})
    return out


def cand_magma():
    """MAGMA（Kowalsky/… 抗体组合变体）。docs/15 判为「不是 site×residue 矩阵」，
    本函数不继承这个判断，逐 sheet 逐抗体地量一遍。

    MOESM4 / MOESM5 先看结构：前者是 7 个位点各 2 个等位（germline vs mature）
    的 2^7 组合设计，后者是 LASSO 系数表。两者都不是 site × 20residue。
    MOESM6 才有逐变体 Kd / ΔΔG，并且带 'Ab' 列（多个抗体骨架）。
    """
    import openpyxl
    out, notes = [], []
    # --- MOESM4：组合设计的结构核对 ---
    wb = openpyxl.load_workbook(os.path.join(DMS, "magma_MOESM4.xlsx"),
                                read_only=True)
    ws = wb.worksheets[0]
    rr = [r for r in ws.iter_rows(min_row=1, max_row=8, values_only=True)]
    hdr = [str(x) for x in rr[2][:7]]
    alle = collections.defaultdict(set)
    for r in ws.iter_rows(min_row=4, values_only=True):
        if r[0] is None:
            continue
        for j in range(1, 7):
            if r[j] is not None:
                alle[hdr[j]].add(str(r[j]))
    wb.close()
    notes.append(("magma_MOESM4", "组合变体表，不是 site×residue",
                  f"{len(alle)} 个位点 {hdr[1:7]}，每位点等位数 "
                  f"{[len(alle[h]) for h in hdr[1:7]]}（2 = germline/mature 二选一）"))
    notes.append(("magma_MOESM5", "LASSO 回归系数表，不是矩阵",
                  "每个 sheet 是模型阶数的 MSE/RMSE/MAE 或单变体系数权重"))

    # --- MOESM6：逐变体 Kd / ΔΔG，按 Ab 拆 ---
    wb = openpyxl.load_workbook(os.path.join(DMS, "magma_MOESM6.xlsx"),
                                read_only=True)
    for sheet, antigen in (("topbinHA_rep2", "influenza HA"),
                           ("all_barcodes_222-1C06", "influenza HA"),
                           ("all_barcodes_319-345", "influenza HA"),
                           ("4A8_CC121_combined", "SARS-CoV-2 S")):
        ws = wb[sheet]
        it = ws.iter_rows(values_only=True)
        h = [str(x) for x in next(it)]
        iv, ia = h.index("Variant"), h.index("Ab")
        ik = h.index("Kd")
        isu = h.index("Success") if "Success" in h else None
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        wtkd = collections.defaultdict(list)
        for r in it:
            if len(r) <= max(iv, ia, ik):
                continue
            v, ab, kd = r[iv], r[ia], r[ik]
            if v is None or ab is None:
                continue
            if isu is not None and str(r[isu]).strip().lower() == "false":
                continue
            try:
                kd = float(kd)
            except (TypeError, ValueError):
                continue
            if not (kd > 0):
                continue
            mus = parse_variant(str(v))
            if not mus:
                wtkd[str(ab)].append(kd)
            elif len(mus) == 1:
                _, wt, pos, mut = mus[0]
                by[str(ab)][(f"{wt}{pos}", mut)].append(kd)
        for ab, d in sorted(by.items()):
            ref = (float(np.median(wtkd[ab])) if wtkd[ab]
                   else float(np.median([np.median(x) for x in d.values()])))
            cells = {k: 1.9872041e-3 * 298.15 * math.log(float(np.median(v)) / ref)
                     for k, v in d.items()}
            M, mask, sites = mat_from_cells(cells)
            subs = collections.Counter(int(x) for x in mask.sum(axis=1))
            out.append({"name": f"magma/{sheet}/{ab}", "M": M, "mask": mask,
                        "sites": sites, "n_obs": int(mask.sum()),
                        "family": f"antibody {ab} / {antigen}",
                        "design": "one backbone x one antigen",
                        "note": (f"单点变体 {int(mask.sum())} 个，"
                                 f"每位点取代数分布 {dict(sorted(subs.items()))}"
                                 f"；WT 参考 Kd "
                                 f"{'有' if wtkd[ab] else '缺（用中位代）'}"),
                        "n_distinct_mut_aa": int((mask.sum(axis=0) > 0).sum()),
                        "max_subs_per_site": int(mask.sum(axis=1).max())})
    wb.close()
    return out, notes


def cand_titeseq():
    """4-4-20 scFv vs 荧光素。docs/13 §4.3 记 stab cos2 = 0.496、
    对半分自对齐 cos1 = 0.496（未过 G4）。本函数在**第三代**截尾口径与
    当前流程下重测，并把测到的值写下来（不许只说「仍然失败」）。
    直接复用 load_independent()，不重写它的加载。"""
    return [d for d in load_independent() if "titeseq" in d["name"]
            or "4-4-20" in d["name"]]


def cand_platinum():
    """Platinum（蛋白-小分子亲和力突变数据库）。
    按 (pdb, chain, ligand) 分组 —— 同一个复合物的多位点多取代才可能成矩阵。"""
    p = os.path.join(ROOT, "cache", "platinum", "platinum_flat_file.csv")
    rows = list(csv.DictReader(open(p)))
    grp = collections.defaultdict(dict)
    meta = {}
    for r in rows:
        if r.get("mut.is_single_point", "").strip().upper() != "YES":
            continue
        m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", r["mutation"].strip())
        if not m:
            continue
        wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
        if wt == mut or mut not in AAI:
            continue
        try:
            dk = float(r["affin.delta_k"])
        except (ValueError, KeyError):
            continue
        key = (r["affin.pdb_id"], r["affin.chain"], r["affin.lig_id"])
        # delta_k 是 log 单位的变化（见 affin.unit）；统一成 ΔΔG 方向无关，
        # 子空间只看中心化后的结构，单位的整体缩放不影响主角度。
        grp[key][(f"{wt}{pos}", mut)] = dk
        meta[key] = (r.get("prot.molecule_name", ""), r.get("prot.organism", ""),
                     r.get("affin.lig_name", "")[:40], r.get("affin.unit", ""))
    out = []
    for key, cells in grp.items():
        M, mask, sites = mat_from_cells(cells)
        if mask.sum() < 20:
            continue
        subs = collections.Counter(int(x) for x in mask.sum(axis=1))
        out.append({"name": "platinum/%s_%s/%s" % key, "M": M, "mask": mask,
                    "sites": sites, "n_obs": int(mask.sum()),
                    "family": f"{meta[key][0][:34]} / {meta[key][2]}",
                    "design": "one backbone x one small-molecule ligand",
                    "note": f"每位点取代数分布 {dict(sorted(subs.items()))}",
                    "n_distinct_mut_aa": int((mask.sum(axis=0) > 0).sum()),
                    "max_subs_per_site": int(mask.sum(axis=1).max())})
    out.sort(key=lambda d: -d["n_obs"])
    return out, len(grp)


def cand_skempi_single():
    """SKEMPI v2 的**新判据**扫法。

    docs/20 扫的是「同骨架 × k ≥ 4 个配体」的面板设计，结论是只有 OMTKY3。
    本函数换判据：**单配体**体系，只要位点 × 取代足够自己撑起一个 2 维
    子空间，就仍然是成对跨家族统计里的一个第三家族。
    分组键 = (被突变的蛋白骨架, 伙伴蛋白)；跨 PDB 合并同一对蛋白。
    """
    rows = [r for r in csv.DictReader(open(os.path.join(DMS, "skempi_v2.csv")),
                                      delimiter=";")]
    grp = collections.defaultdict(dict)
    n_single = 0
    for r in rows:
        toks = r["Mutation(s)_PDB"].split(",")
        if len(toks) != 1:
            continue
        m = parse_mut(toks[0])
        if m is None:
            continue
        wt, chain, pos, mut = m
        parts = r["#Pdb"].split("_")
        if len(parts) != 3:
            continue
        pdb, c1, c2 = parts
        if chain in c1:
            bb, pt = r["Protein 1"], r["Protein 2"]
        elif chain in c2:
            bb, pt = r["Protein 2"], r["Protein 1"]
        else:
            continue
        ddg, T = ddg_from_row(r)
        if ddg is None:
            continue
        n_single += 1
        grp[(bb, pt)][(f"{chain}{pos}", mut)] = ddg
    out = []
    for (bb, pt), cells in grp.items():
        M, mask, sites = mat_from_cells(cells)
        subs = collections.Counter(int(x) for x in mask.sum(axis=1))
        out.append({"name": f"skempiV2/{bb[:28]}/{pt[:24]}", "M": M,
                    "mask": mask, "sites": sites, "n_obs": int(mask.sum()),
                    "family": f"{bb[:34]} vs {pt[:30]}",
                    "design": "one backbone x one protein partner",
                    "note": f"每位点取代数分布 {dict(sorted(subs.items()))}",
                    "n_distinct_mut_aa": int((mask.sum(axis=0) > 0).sum()),
                    "max_subs_per_site": int(mask.sum(axis=1).max()),
                    "backbone": bb, "partner": pt})
    out.sort(key=lambda d: -d["n_obs"])
    return out, n_single, len(grp)


# ----------------------------------------------------------------------
# 2. 统一的筛查闸（每个候选走同一条路，闸的顺序固定）
# ----------------------------------------------------------------------
def screen(d, do_resolve=True, nboot=N_BOOT, nsplit=N_SPLIT):
    """返回 d 的筛查结果。闸的顺序：G2 采样 -> G3 截尾 -> G4 可分辨。
    任何一道没过就停在那道，并把**测到的值**写进 result。"""
    res = {"name": d["name"], "sites": len(d["sites"]), "n_obs": d["n_obs"],
           "max_subs": d.get("max_subs_per_site"),
           "n_mut_aa": d.get("n_distinct_mut_aa"),
           "family": d.get("family", ""), "design": d.get("design", ""),
           "note": d.get("note", ""), "gate": None, "why": "",
           "censor": None, "censor_ci": None, "censor_gate": None,
           "stab_cos1": None, "stab_cos2": None,
           "split_cos1": None, "split_cos1_ci": None,
           "split_cos2": None, "d_cols": None}

    # ---- G2 采样：保护后能不能抽出子空间 ----
    B, cols, err = extract_subspace(d["M"], d["mask"])
    if B is None:
        res["gate"] = "G2 采样"
        res["why"] = f"保护后抽不出子空间：{err}（MIN_PER_ROW={MIN_PER_ROW}, " \
                     f"MIN_PER_COL={MIN_PER_COL}）"
        return res
    res["d_cols"] = len(cols)
    mkr = d["mask"].sum(axis=1) >= MIN_PER_ROW
    res["sites_kept"] = int(mkr.sum())

    # ---- G3 截尾：第三代密度型（唯一通过双向标定的） ----
    raw = d["M"][d["mask"]]
    cd = censor_fraction_density_ci(raw)
    res["censor"] = cd["q"]
    res["censor_ci"] = (cd["lo"], cd["hi"])
    res["censor_gate"] = cd["info"].get("gate")
    if cd["q"] >= CENSOR_MAX:
        res["gate"] = "G3 截尾"
        res["why"] = f"密度型截尾 {cd['q']:.1%} >= {CENSOR_MAX:.0%}"
        return res

    if not do_resolve:
        res["gate"] = "未测可分辨性"
        res["why"] = "本轮不跑（见上游闸）"
        return res

    # ---- G4 可分辨：对半分自对齐（保守）+ bootstrap 稳定性 ----
    rng = np.random.default_rng(42)
    nr = d["M"].shape[0]
    c1, c2 = [], []
    for _ in range(nsplit):
        p = rng.permutation(nr)
        h1, h2 = p[: nr // 2], p[nr // 2:]
        B1, k1, _ = extract_subspace(d["M"][h1], d["mask"][h1])
        B2, k2, _ = extract_subspace(d["M"][h2], d["mask"][h2])
        if B1 is None or B2 is None:
            continue
        r = align(B1, k1, B2, k2)
        if r:
            c1.append(r["cos"][0]); c2.append(r["cos"][1])
    if c1:
        res["split_cos1"] = float(np.median(c1))
        res["split_cos1_ci"] = ci(c1)
        res["split_cos2"] = float(np.median(c2))
    boots = bootstrap_subspaces(d["M"], d["mask"], n=nboot, seed=7)
    s1, s2 = [], []
    for x in boots:
        if not x:
            continue
        r = align(B, cols, x[0], x[1])
        if r:
            s1.append(r["cos"][0]); s2.append(r["cos"][1])
    if s1:
        res["stab_cos1"] = float(np.median(s1))
        res["stab_cos2"] = float(np.median(s2))

    res["B"] = B.tolist()
    res["cols"] = cols.tolist()
    if res["stab_cos2"] is None:
        res["gate"] = "G4 可分辨"
        res["why"] = "bootstrap 全部失败，连第一维都测不出"
    elif res["stab_cos2"] >= STAB_MIN:
        res["gate"] = "通过"
        res["why"] = f"stab cos2 = {res['stab_cos2']:.3f} >= {STAB_MIN}"
    elif res["stab_cos1"] is not None and res["stab_cos1"] >= 0.80:
        res["gate"] = "仅第一维"
        res["why"] = (f"stab cos1 = {res['stab_cos1']:.3f} 但 "
                      f"stab cos2 = {res['stab_cos2']:.3f} < {STAB_MIN}")
    else:
        res["gate"] = "G4 可分辨"
        res["why"] = (f"stab cos1 = {res['stab_cos1']:.3f}、"
                      f"cos2 = {res['stab_cos2']:.3f} —— 连自己的第一轴都定不出")
    return res


def main():
    gate_selfcheck()
    null = haar_pair_null(20, n=4000, seed=0)
    p95 = {"cos1": float(np.percentile(null["cos1"], 95)),
           "cos2": float(np.percentile(null["cos2"], 95))}
    print(f"随机零分布 d=20：cos1 p95 = {p95['cos1']:.3f}、"
          f"cos2 p95 = {p95['cos2']:.3f}（量级校准，CLAUDE.md §8）\n")

    report = {"p95": p95, "screened": [], "structural_rejects": [],
              "counts": {}}

    # ---------- A. PafA（最被看好的一个） ----------
    print("=" * 96)
    print("[A] HT-MEK PafA（磷酸酶 × 两个小分子抑制剂）")
    print("=" * 96)
    pafa = cand_pafa()
    for d in pafa:
        print(f"  {d['name']}: {len(d['sites'])} 位点 / {d['n_obs']} 观测 / "
              f"最大取代数 {d['max_subs_per_site']} / "
              f"替换残基种类 {d['n_distinct_mut_aa']}")
        print(f"     {d['note']}")
    # 结构闸 G1 必须先判：残基维只有几种残基时，20 维残基载荷无从定义
    for d in pafa:
        if d["n_distinct_mut_aa"] < 6:
            report["structural_rejects"].append({
                "name": d["name"], "gate": "G1 结构",
                "why": (f"只扫了 {d['n_distinct_mut_aa']} 种替换残基"
                        f"（Gly/Val/Ala 扫），残基维 ≤ "
                        f"{d['n_distinct_mut_aa']}，中心化后 ≤ "
                        f"{d['n_distinct_mut_aa'] - 1} 维，撑不起 2 维残基载荷面"),
                "sites": len(d["sites"]), "n_obs": d["n_obs"],
                "max_subs": d["max_subs_per_site"],
                "family": d["family"], "design": d["design"],
                "note": d["note"]})
            print(f"  -> {d['name']} 在 G1 结构闸就停："
                  f"替换残基只有 {d['n_distinct_mut_aa']} 种")
        else:
            report["screened"].append(screen(d))
    # 即便如此也把 G2 的实测结果记下来（不许只给一句定性话）
    for d in pafa:
        r = screen(d, do_resolve=False)
        print(f"     [实测 G2] {d['name']}: {r['gate']} —— {r['why']}")
        for e in report["structural_rejects"]:
            if e["name"] == d["name"]:
                e["g2_measured"] = f"{r['gate']}: {r['why']}"

    # ---------- B. magma ----------
    print("\n" + "=" * 96)
    print("[B] MAGMA 抗体数据（docs/15 的判断不继承，逐 sheet 重量）")
    print("=" * 96)
    mg, notes = cand_magma()
    for n, g, w in notes:
        print(f"  {n}: G1 结构闸 —— {g}；{w}")
        report["structural_rejects"].append({
            "name": n, "gate": "G1 结构", "why": g, "note": w,
            "family": "antibody combinatorial / LASSO", "design": "-"})
    for d in mg:
        print(f"  {d['name']}: {len(d['sites'])} 位点 / {d['n_obs']} 观测 / "
              f"最大取代数 {d['max_subs_per_site']} / 残基列 "
              f"{d['n_distinct_mut_aa']}")
        r = screen(d, nboot=500, nsplit=200)
        print(f"     -> {r['gate']}: {r['why']}")
        report["screened"].append(r)

    # ---------- C. Tite-seq 4-4-20 ----------
    print("\n" + "=" * 96)
    print("[C] Tite-seq 4-4-20 scFv vs 荧光素（docs/13 已失败，重测并报数值）")
    print("=" * 96)
    for d in cand_titeseq():
        print(f"  {d['name']}: {len(d['sites'])} 位点 / {d['n_obs']} 观测")
        r = screen(d)
        r["family"] = "4-4-20 scFv / fluorescein (small molecule)"
        r["design"] = "one backbone x one small-molecule ligand"
        print(f"     -> {r['gate']}: {r['why']}")
        print(f"        密度型截尾 {r['censor']:.1%} "
              f"[{r['censor_ci'][0]:.1%}, {r['censor_ci'][1]:.1%}]"
              f"（旧口径 docs/13 记 4.7%）")
        print(f"        对半分自对齐 cos1 = {r['split_cos1']}, "
              f"stab cos1/cos2 = {r['stab_cos1']:.3f}/{r['stab_cos2']:.3f}")
        report["screened"].append(r)

    # ---------- D. Platinum ----------
    print("\n" + "=" * 96)
    print("[D] Platinum（蛋白-小分子）")
    print("=" * 96)
    pl, n_grp = cand_platinum()
    print(f"  扫过 {n_grp} 个 (pdb, chain, ligand) 分组，"
          f"其中 ≥20 个单点观测的 {len(pl)} 个")
    report["counts"]["platinum_groups"] = n_grp
    for d in pl[:12]:
        r = screen(d, nboot=400, nsplit=120)
        print(f"  {d['name'][:46]:<48}{len(d['sites']):>4} 位点"
              f"{d['n_obs']:>5} 观测  最大取代 {d['max_subs_per_site']:>2}  "
              f"-> {r['gate']}: {r['why'][:48]}")
        report["screened"].append(r)
    for d in pl[12:]:
        report["screened"].append(screen(d, do_resolve=False))

    # ---------- E. SKEMPI v2 单配体扫法 ----------
    print("\n" + "=" * 96)
    print("[E] SKEMPI v2：**单配体**体系，按位点×取代能否自撑 2 维子空间")
    print("=" * 96)
    sk, n_single, n_grp = cand_skempi_single()
    print(f"  扫过全部 {n_single} 行单点突变（可算 ΔΔG 的），"
          f"落在 {n_grp} 个 (骨架, 伙伴) 分组里")
    report["counts"]["skempi_single_rows"] = n_single
    report["counts"]["skempi_groups"] = n_grp
    print(f"  {'体系':<52}{'位点':>5}{'观测':>6}{'最大取代':>8}{'残基列':>7}  判决")
    print("  " + "-" * 92)
    nshow = 0
    for d in sk:
        quick = d["n_obs"] < 40 or len(d["sites"]) < 6
        r = screen(d, do_resolve=not quick, nboot=400, nsplit=120)
        is_omt = ("ovomucoid" in d["backbone"].lower()
                  or "ovomucoid" in d["partner"].lower())
        r["is_omtky3"] = is_omt
        report["screened"].append(r)
        if d["n_obs"] >= 40 or len(d["sites"]) >= 6:
            nshow += 1
            if nshow <= 25:
                tag = "  [= 已有的 OMTKY3]" if is_omt else ""
                print(f"  {d['name'][:51]:<52}{len(d['sites']):>5}"
                      f"{d['n_obs']:>6}{d['max_subs_per_site']:>8}"
                      f"{d['n_distinct_mut_aa']:>7}  {r['gate']}: "
                      f"{r['why'][:34]}{tag}")
    print(f"  （≥40 观测或 ≥6 位点的分组共 {nshow} 个，上面列前 25）")
    passed_sk = [r for r in report["screened"]
                 if r["name"].startswith("skempiV2/") and r["gate"] == "通过"
                 and not r.get("is_omtky3")]
    print(f"  非 OMTKY3 且四道闸全过的 SKEMPI 体系：{len(passed_sk)} 个")
    for r in passed_sk:
        print(f"     {r['name']}  {r['sites']} 位点  {r['n_obs']} 观测  "
              f"stab cos2 = {r['stab_cos2']:.3f}")

    # ---------- F. SKEMPI v2：按骨架合并（跨全部伙伴）的第三种扫法 ----------
    # 上面 [E] 按 (骨架, 伙伴) 分组。但一个骨架的多个伙伴合并之后
    # 位点×取代会变密（OMTKY3 的 pooled 矩阵就是这样来的）。
    # 所以必须再扫一遍：**任何**骨架，合并它的全部伙伴，能不能抽出子空间。
    print("\n" + "=" * 96)
    print("[F] SKEMPI v2：按**骨架**合并（跨全部伙伴）—— 第三种扫法")
    print("=" * 96)
    pooled = collections.defaultdict(dict)
    for d in sk:
        pooled[d["backbone"]].update(
            {k: d["M"][i, j] for (k, (i, j)) in
             [((d["sites"][i], AA20[j]), (i, j))
              for i in range(d["M"].shape[0])
              for j in range(20) if d["mask"][i, j]]})
    okbb = []
    for bb, cells in pooled.items():
        M, mask, sites = mat_from_cells(cells)
        B, cols, err = extract_subspace(M, mask)
        if B is not None:
            okbb.append((bb, len(sites), int(mask.sum()), len(cols)))
    print(f"  扫过 {len(pooled)} 个骨架（合并它们的全部蛋白伙伴）")
    print(f"  能抽出 2 维残基载荷子空间的骨架：{len(okbb)} 个")
    for bb, ns, no, nc in okbb:
        print(f"     {bb[:50]:<52}{ns:>4} 位点{no:>5} 观测  残基列 {nc}")
    report["counts"]["skempi_backbones_pooled"] = len(pooled)
    report["counts"]["skempi_backbones_extractable"] = [b[0] for b in okbb]

    # ---------- 汇总 ----------
    print("\n" + "=" * 96)
    print("[汇总] 搜索量与判决（CLAUDE.md §5）")
    print("=" * 96)
    byg = collections.Counter(r["gate"] for r in report["screened"])
    print(f"  结构闸 G1 直接否掉：{len(report['structural_rejects'])} 个候选")
    print(f"  进入数值筛查的矩阵：{len(report['screened'])} 个")
    for g, n in byg.most_common():
        print(f"     {g:<16}{n:>5}")
    winners = [r for r in report["screened"]
               if r["gate"] in ("通过", "仅第一维") and not r.get("is_omtky3")]
    print(f"\n  **非既有家族 × 过闸**的候选：{len(winners)} 个")
    for r in winners:
        print(f"     {r['name']}  [{r['gate']}]  {r['family'][:40]}  "
              f"stab cos1/cos2 = {r['stab_cos1']:.3f}/{r['stab_cos2']:.3f}  "
              f"截尾 {r['censor']:.1%}")

    report["counts"]["structural_rejects"] = len(report["structural_rejects"])
    report["counts"]["numeric_screened"] = len(report["screened"])
    report["counts"]["by_gate"] = dict(byg)
    out = os.path.join(ROOT, "cache", "third_family_screen.json")
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=1,
              default=float)
    print(f"\n产物: {out}")


if __name__ == "__main__":
    main()
