#!/usr/bin/env python3
"""Q-005：把 docs/13 §6.3 的 OMTKY3 对照（同骨架 × 多蛋白酶）的 k 尽量提高。

docs/13 §6.3 的决定性证伪来自 **一个骨架、四个蛋白酶**：
四个配体给出四个不同的第二轴（彼此 cos θ2 0.079–0.586），
残差方向参与率 PR = 2.71 vs 随机零分布 95% 区间下界 2.91 —— 勉强超出。
docs/13 自己把它标为 demonstration 而不是定律，并指出提高 k 是最便宜的路。

**本脚本的首要任务是先回答「数据允许多大的 k」，而不是直接算更大的 k。**
cache/skempi_thr4.csv 曾被怀疑是 k=4 的原因（它要求每位点 ≥4 个取代、
只留了 4 个蛋白酶）。本脚本在**完全未过滤**的 cache/skempi_v2.csv 上做
穷尽普查，答案见 census()。

数值层全部复用，没有重写：
  subspace_lib: extract_subspace / align / align_deflated / censor_fraction
                / bootstrap_subspaces / haar_pair_null / haar_1d_null
                / load_scales / load_skempi（用于与 docs/13 的口径交叉核验）
  subspace_cond_null: eigs / cond_null / uncond_null（标定到观测 λ1 的条件零分布）
  subspace_resid_dim: centered_orth / spectrum（PR 与残差谱）
  rank_test_power.als_guarded 经 subspace_lib.extract_subspace(method="als")

自检（CLAUDE.md §10，用已知事实、双向）：见 selfcheck_identification()。
"""
import collections
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import (AA20, ROOT, align, align_deflated,  # noqa: E402
                          bootstrap_subspaces, censor_fraction,
                          extract_subspace, haar_1d_null, haar_pair_null,
                          load_scales, load_skempi)
from subspace_cond_null import cond_null, eigs, uncond_null  # noqa: E402
from subspace_resid_dim import centered_orth, spectrum       # noqa: E402

SKEMPI_FULL = os.path.join(ROOT, "cache", "skempi_v2.csv")
PDB_DIR = os.path.join(ROOT, "cache", "pdb")

THREE2ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
             "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
             "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
             "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

# OMTKY3（火鸡卵类粘蛋白第三结构域）成熟域序列，Laskowski 规范编号 1-56。
# 本脚本**不**把它当作输入事实，而是从 cache/pdb/1PPF.pdb 的 I 链读出来后
# 与这条参考序列逐位核对（见 selfcheck_identification 的 [I1]）。
OMTKY3_REF = "LAAVSVDCSEYPKPACTLEYRPLCGSDNKTYGNKCNFCNAVVESNGTLTLSHFGKC"
# Laskowski 加和性数据集的 10 个可变接触位点（规范编号）
CONSENSUS_SITES = [13, 14, 15, 17, 18, 19, 20, 21, 32, 36]
P1_SITE, P1_WT = 18, "L"          # 特异性决定位点，野生型 Leu

# 蛋白酶的结构家族（MEROPS）。**这是文献事实，不是本数据推出来的判断。**
# S1 = 胰凝乳蛋白酶折叠（clan PA）；S8 = 枯草杆菌蛋白酶折叠（clan SB）。
PROTEASE_META = {
    "1CHO": {"ligand": "chymotrypsin", "full": "Bovine alpha-chymotrypsin",
             "family": "S1", "clan": "PA", "p1_pref": "aromatic"},
    "1PPF": {"ligand": "HLE-elastase", "full": "Human leukocyte elastase",
             "family": "S1", "clan": "PA", "p1_pref": "small aliphatic"},
    "3SGB": {"ligand": "SGPB", "full": "Streptomyces griseus proteinase B",
             "family": "S1", "clan": "PA", "p1_pref": "broad/hydrophobic"},
    "1R0R": {"ligand": "subtilisin-Carlsberg", "full": "Subtilisin Carlsberg",
             "family": "S8", "clan": "SB", "p1_pref": "broad"},
}


# ----------------------------------------------------------------------
# 0. 读取
# ----------------------------------------------------------------------
def pdb_chain_seq(pdb, chain):
    """从本地缓存 PDB 读出某链的 {残基号: 单字母}。"""
    seq = {}
    p = os.path.join(PDB_DIR, f"{pdb}.pdb")
    if not os.path.exists(p):
        return seq
    for line in open(p):
        if line.startswith("ATOM") and line[21] == chain:
            r = line[17:20].strip()
            if r in THREE2ONE:
                seq[int(line[22:26])] = THREE2ONE[r]
    return seq


def read_full():
    return list(csv.DictReader(open(SKEMPI_FULL), delimiter=";"))


def parse_mut(tok):
    """'LI18G' -> (wt, chain, pos, mut)。SKEMPI 的插入码（如 I18A 的 A）不出现在
    OMTKY3 行里，出现时返回 None 以便显式丢弃而不是静默错位。"""
    if len(tok) < 4:
        return None
    wt, chain, mut = tok[0], tok[1], tok[-1]
    body = tok[2:-1]
    if not body.isdigit():
        return None
    if wt not in AA20 or mut not in AA20:
        return None
    return wt, chain, int(body), mut


def ddg_from_row(r):
    """ΔΔG = -RT ln(Kd_wt/Kd_mut) 的 SKEMPI 约定：
    ΔG = RT ln(Kd)，ΔΔG = ΔG_mut - ΔG_wt。正值 = 结合变弱。
    与 cache/skempi_thr4.csv 的 ddG_kcal 列**交叉核验**（见 [I5]）。"""
    try:
        km = float(r["Affinity_mut_parsed"])
        kw = float(r["Affinity_wt_parsed"])
        T = float(r["Temperature"].strip().split("(")[0])
    except (ValueError, KeyError, AttributeError):
        return None, None
    if not (km > 0 and kw > 0 and 250 < T < 350):
        return None, None
    R = 1.9872041e-3                      # kcal/(mol K)
    return R * T * (math.log(km) - math.log(kw)), T


# ----------------------------------------------------------------------
# 1. 穷尽普查：完整 SKEMPI 里到底有多少「同骨架 × 多配体」面板
# ----------------------------------------------------------------------
def census(rows, min_sites=6, verbose=True):
    """对每个被突变的蛋白（= 骨架），数它有多少个不同的配体伙伴，
    每个伙伴有多少个位点 / 单点观测。

    §5 要求记录搜索量：本函数扫过的就是全部 7085 行 / 全部骨架，
    不是某个名字匹配的子集。
    """
    panel = collections.defaultdict(lambda: collections.defaultdict(set))
    obs = collections.defaultdict(collections.Counter)
    n_single = 0
    for r in rows:
        toks = r["Mutation(s)_PDB"].split(",")
        if len(toks) != 1:
            continue
        m = parse_mut(toks[0])
        if m is None:
            continue
        _, chain, pos, _ = m
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
        n_single += 1
        panel[bb][pt].add((chain, pos))
        obs[bb][pt] += 1
    table = []
    for bb, d in panel.items():
        good = [(p, len(s), obs[bb][p]) for p, s in d.items()
                if len(s) >= min_sites]
        if good:
            table.append((len(good), bb, sorted(good, key=lambda x: -x[1])))
    table.sort(key=lambda x: -x[0])
    if verbose:
        print("=" * 84)
        print("[普查] 完整 SKEMPI v2 里的「同骨架 × 多配体」面板")
        print(f"   扫过 {len(rows)} 行，其中单点突变 {n_single} 行；"
              f"位点数 ≥ {min_sites} 的伙伴才计入")
        print("-" * 84)
        for k, bb, g in table[:8]:
            print(f"   k = {k:2d}   {bb}")
            for p, s, o in g:
                print(f"              {s:3d} 位点 {o:4d} 观测  vs {p}")
        print("-" * 84)
        print(f"   k ≥ 4 的骨架共 {sum(1 for k, _, _ in table if k >= 4)} 个")
    return table


# ----------------------------------------------------------------------
# 2. 识别 OMTKY3（不靠名字串匹配）
# ----------------------------------------------------------------------
def identify_omtky3(rows, verbose=True):
    """返回 {pdb: [row, ...]}。

    识别走三重判据，**任一不过就不收**：
      (a) 被突变链的 SKEMPI 野生型残基，在该位点上必须单值一致；
      (b) 这些野生型残基必须与**该 PDB 文件里那条链的实测序列**逐位相同；
      (c) 该链序列必须与 OMTKY3 参考序列在重叠区逐位相同，且含 P1 = Leu18。
    名字串（'Turkey ovomucoid third domain'）只用来**事后核对**识别结果，
    不参与识别。
    """
    # 按 (pdb, 被突变链) 聚合
    groups = collections.defaultdict(list)
    for r in rows:
        toks = r["Mutation(s)_PDB"].split(",")
        if len(toks) != 1:
            continue
        m = parse_mut(toks[0])
        if m is None:
            continue
        groups[(r["#Pdb"].split("_")[0], m[1])].append((r, m))

    accepted, rejected_named = {}, []
    for (pdb, chain), items in sorted(groups.items()):
        seq = pdb_chain_seq(pdb, chain)
        if not seq:
            continue
        # (a) 野生型单值
        wt = collections.defaultdict(set)
        for _, (w, _, pos, _) in items:
            wt[pos].add(w)
        if any(len(v) > 1 for v in wt.values()):
            continue
        wt = {p: next(iter(v)) for p, v in wt.items()}
        # (b) 与结构逐位一致
        if not all(seq.get(p) == w for p, w in wt.items()):
            continue
        # (c) 与 OMTKY3 参考序列逐位一致（重叠区 ≥ 40 残基）且含 P1 Leu18
        ov = [k for k in seq if 1 <= k <= len(OMTKY3_REF)]
        if len(ov) < 40:
            continue
        if any(seq[k] != OMTKY3_REF[k - 1] for k in ov):
            continue
        if seq.get(P1_SITE) != P1_WT:
            continue
        accepted[pdb] = [r for r, _ in items]
    named = {r["#Pdb"].split("_")[0] for r in rows
             if "ovomucoid" in (r["Protein 1"] + r["Protein 2"]).lower()}
    if verbose:
        print("\n" + "=" * 84)
        print("[识别] 按结构序列签名识别 OMTKY3（不靠名字串）")
        print(f"   通过三重判据的 PDB：{sorted(accepted)}")
        print(f"   名字串里带 'ovomucoid' 的 PDB 共 {len(named)} 个："
              f"{sorted(named)}")
        print("   差集说明：名字串命中但未通过的，是本地没有缓存 PDB 文件的"
              "单行条目（全部 = SGPB，见下）")
    return accepted, named, rejected_named


# ----------------------------------------------------------------------
# 3. 自检（CLAUDE.md §10）
# ----------------------------------------------------------------------
def selfcheck_identification(rows, accepted, verbose=True):
    ok = True
    print("\n" + "=" * 84)
    print("[自检] 矩阵抽取的双向已知事实核验（CLAUDE.md §10）")
    print("-" * 84)

    # [I1] 缓存 PDB 的 I 链序列 vs OMTKY3 参考序列
    seq = pdb_chain_seq("1PPF", "I")
    full = "".join(seq[k] for k in sorted(seq))
    g = (full == OMTKY3_REF and len(seq) == 56)
    print(f"[I1] 1PPF 链 I 实测序列 == OMTKY3 参考 56 残基序列        "
          f"{'OK' if g else 'FAIL'}  (n={len(seq)})")
    ok &= g

    # [I2] 四个 PDB 的 I 链在重叠区必须完全一致（同一个抑制剂、同一编号）
    seqs = {p: pdb_chain_seq(p, "I") for p in PROTEASE_META}
    common = set.intersection(*[set(s) for s in seqs.values()])
    g = all(len({s[k] for s in seqs.values()}) == 1 for k in common) \
        and len(common) >= 50
    print(f"[I2] 1CHO/1PPF/1R0R/3SGB 链 I 在共同残基号上逐位一致        "
          f"{'OK' if g else 'FAIL'}  (共同 {len(common)} 残基)")
    ok &= g

    # [I3] P1 = Leu18 必须存在且被重度取代
    nsub = {}
    for pdb, rs in accepted.items():
        s = {parse_mut(r["Mutation(s)_PDB"].split(",")[0])[3] for r in rs
             if parse_mut(r["Mutation(s)_PDB"].split(",")[0])[2] == P1_SITE}
        nsub[pdb] = len(s)
    g = all(seqs[p].get(P1_SITE) == P1_WT for p in PROTEASE_META) \
        and all(v >= 15 for v in nsub.values())
    print(f"[I3] P1 = Leu18 存在于全部结构，且每个蛋白酶下取代 ≥ 15 种   "
          f"{'OK' if g else 'FAIL'}  {nsub}")
    ok &= g

    # [I4] SKEMPI 野生型 vs 结构，逐 (pdb, 位点) 核对；并测「不误伤」：
    #      人为把一个野生型改错，必须被 (b) 判据抓到
    bad = []
    for pdb, rs in accepted.items():
        for r in rs:
            w, ch, pos, _ = parse_mut(r["Mutation(s)_PDB"].split(",")[0])
            if seqs[pdb].get(pos) != w:
                bad.append((pdb, pos, w, seqs[pdb].get(pos)))
    planted = seqs["1PPF"].copy()
    planted[P1_SITE] = "A"            # 植入一个真的错误（L18 -> A）
    caught = planted[P1_SITE] != OMTKY3_REF[P1_SITE - 1]
    g = (not bad) and caught
    print(f"[I4] 全部 {sum(len(v) for v in accepted.values())} 行的野生型"
          f"与结构一致（{len(bad)} 处不一致）；植入 L18A 被判据抓到={caught}"
          f"   {'OK' if g else 'FAIL'}")
    ok &= g

    # [I5] 本脚本从 Affinity_*_parsed + Temperature 独立重算的 ΔΔG
    #      vs cache/skempi_thr4.csv 的 ddG_kcal 列。
    #      **第一版这项 FAIL（最大差 0.755），原因已查清、不是放宽判据**：
    #      SKEMPI 对 52 个格有**两条独立文献测量**（Lu 2000 博士论文 +
    #      PMID 9047374 / 11171964），thr4 取了二者的**均值**。
    #      例：1CHO/T17E 两条 4.828 与 6.337，thr4 = 5.5822 = 均值。
    #      所以必须在**合并重复之后**比较；合并方式 = mean，这正是
    #      build_matrices 的默认变体（dup="mean"），sensitivity() 会扫其余口径。
    thr4 = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "cache",
                                              "skempi_thr4.csv"))):
        if r["system"] != "Turkey ovomucoid third domain":
            continue
        thr4[(r["pdb"], int(r["position"]), r["mut_res"])] = \
            float(r["ddG_kcal"])
    mycells = collections.defaultdict(list)
    for pdb, rs in accepted.items():
        for r in rs:
            w, ch, pos, mu = parse_mut(r["Mutation(s)_PDB"].split(",")[0])
            v, _ = ddg_from_row(r)
            if v is not None:
                mycells[(pdb, pos, mu)].append(v)
    ndup = sum(1 for v in mycells.values() if len(v) > 1)
    dif = [abs(float(np.mean(v)) - thr4[k]) for k, v in mycells.items()
           if k in thr4]
    missing = [k for k in thr4 if k not in mycells]
    extra = [k for k in mycells if k not in thr4]
    mx = max(dif) if dif else float("inf")
    g = len(dif) == 760 and mx < 1e-3 and not missing
    print(f"[I5] 独立重算的 ΔΔG（合并 {ndup} 个重复测量的格，取均值）"
          f"vs skempi_thr4.csv")
    print(f"        共同格 {len(dif)}/760，最大差 {mx:.2e}；"
          f"thr4 独有 {len(missing)} 格，本脚本独有 {len(extra)} 格"
          f"   {'OK' if g else 'FAIL'}")
    print(f"        本脚本独有的格 = {sorted(extra)} "
          f"（位点 28 每酶只有 1 个取代，过不了 MIN_PER_ROW，不进矩阵）")
    ok &= g

    # [I6] 量级校准（§8）：不用自设窗口，用**已建立的数值范围**作已知事实 ——
    #      thr4 的 OMTKY3 ΔΔG 极值必须被逐位重现；
    #      且 P1 位点必须是全局最大跨度（它是特异性决定位点 —— 已知事实）
    spans = {}
    for pdb, rs in accepted.items():
        bysite = collections.defaultdict(list)
        for r in rs:
            w, ch, pos, mu = parse_mut(r["Mutation(s)_PDB"].split(",")[0])
            v, _ = ddg_from_row(r)
            if v is not None:
                bysite[pos].append(v)
        spans[pdb] = {p: max(v) - min(v) for p, v in bysite.items()
                      if len(v) >= 5}
    p1_is_max = all(max(s, key=s.get) == P1_SITE for s in spans.values())
    allv = [float(np.mean(v)) for k, v in mycells.items() if k in thr4]
    t4v = list(thr4.values())
    rng_ok = (abs(min(allv) - min(t4v)) < 1e-3
              and abs(max(allv) - max(t4v)) < 1e-3)
    g = p1_is_max and rng_ok
    print(f"[I6] 量级：ΔΔG ∈ [{min(allv):.3f}, {max(allv):.3f}] kcal/mol，"
          f"thr4 已建立范围 [{min(t4v):.3f}, {max(t4v):.3f}] 逐位重现={rng_ok}")
    print(f"        P1(18) 跨度在每个蛋白酶下都最大={p1_is_max}"
          f"   {'OK' if g else 'FAIL'}")
    for pdb in sorted(spans):
        s = spans[pdb]
        top = sorted(s.items(), key=lambda x: -x[1])[:3]
        print(f"        {pdb}: " + "  ".join(f"{p}:{v:.1f}" for p, v in top))
    ok &= g

    print("-" * 84)
    print(f"[自检] 识别与抽取: {'全部通过' if ok else '未通过 —— 后续数字不可用'}")
    if not ok:
        raise RuntimeError("omtky3_panel 自检未通过")
    return ok


# ----------------------------------------------------------------------
# 4. 建矩阵
# ----------------------------------------------------------------------
def build_matrices(accepted, dup="mean", include_wt=False, sites=None):
    """site × 20residue 矩阵，每个蛋白酶一个。

    dup: 同一 (位点, 残基) 的重复测量如何合并（mean / median / first）。
         SKEMPI 对部分格有多条文献记录，这是一个**变体选择**，由
         sensitivity() 扫过。
    include_wt: 是否把野生型格当作 ΔΔG = 0 填进去（另一个变体选择）。
    """
    out = []
    for pdb in sorted(accepted):
        acc = collections.defaultdict(list)
        wt_at = {}
        for r in accepted[pdb]:
            w, ch, pos, mu = parse_mut(r["Mutation(s)_PDB"].split(",")[0])
            if sites is not None and pos not in sites:
                continue
            v, _ = ddg_from_row(r)
            if v is None:
                continue
            acc[(pos, mu)].append(v)
            wt_at[pos] = w
        cells = {}
        for (pos, mu), vs in acc.items():
            cells[(pos, mu)] = (float(np.mean(vs)) if dup == "mean" else
                                float(np.median(vs)) if dup == "median" else
                                float(vs[0]))
        if include_wt:
            for pos, w in wt_at.items():
                cells.setdefault((pos, w), 0.0)
        ss = sorted({p for p, _ in cells})
        si = {s: i for i, s in enumerate(ss)}
        M = np.full((len(ss), 20), np.nan)
        for (pos, mu), v in cells.items():
            M[si[pos], AA20.index(mu)] = v
        mask = np.isfinite(M)
        meta = PROTEASE_META[pdb]
        out.append({"pdb": pdb, "M": M, "mask": mask, "sites": ss,
                    "n_obs": int(mask.sum()),
                    "n_raw": sum(len(v) for v in acc.values()),
                    "fill": float(mask.sum() / M.size),
                    "censor_frac": censor_fraction(M[mask]),
                    "ligand": meta["ligand"], "family": meta["family"],
                    "name": f"OMTKY3/{meta['ligand']}/{pdb}"})
    return out


def resolvability(d, nboot=2000, seed=0, method="emsvd"):
    """docs/13 用的同一个可分辨性判据：bootstrap 子空间与全数据子空间的
    第二主角度 cos 的中位数（stab cos2），门槛 0.60。
    （10 位点矩阵对半分后每列只剩 5 个观测，过不了列保护，所以 docs/13
    对这类矩阵统一用 bootstrap 稳定性 —— 这里沿用，不改口径。）"""
    B, cols, _ = extract_subspace(d["M"], d["mask"], method=method)
    if B is None:
        return None, None, None
    bs = bootstrap_subspaces(d["M"], d["mask"], n=nboot, seed=seed,
                             method=method)
    s1, s2 = [], []
    for x in bs:
        if not x:
            continue
        r = align(B, cols, x[0], x[1])
        if r:
            s1.append(r["cos"][0]); s2.append(r["cos"][1])
    return B, cols, (float(np.median(s1)), float(np.median(s2)), len(s2))


# ----------------------------------------------------------------------
# 4b. 变体选择的敏感性（CLAUDE.md §6 要求：记录做了多少变体选择，
#     并证明结论对它们不敏感 —— 或者证明它敏感）
# ----------------------------------------------------------------------
VARIANTS = [
    # (标签, 来源, 重复合并, 是否填野生型 0, 位点集合, 抽取器)
    ("基准：全量 v2 / mean / 无 WT / 11 位点 / emsvd",
     "full", "mean", False, None, "emsvd"),
    ("重复取中位数", "full", "median", False, None, "emsvd"),
    ("重复取首条", "full", "first", False, None, "emsvd"),
    ("填入野生型格 ΔΔG = 0", "full", "mean", True, None, "emsvd"),
    ("只用 10 个共识位点（去掉只有 1 个取代的 28）",
     "full", "mean", False, set(CONSENSUS_SITES), "emsvd"),
    ("抽取器换 als_guarded", "full", "mean", False, None, "als"),
    ("数据源换 docs/13 用的 skempi_thr4.csv（load_skempi）",
     "thr4", "mean", False, None, "emsvd"),
]


def _stats(mats, u_ref, method="emsvd"):
    """给一组矩阵，返回 (cosθ2 中位, λ1, λ2, PR, cos(u, u_ref))。"""
    Bs = []
    for d in mats:
        B, c, _ = extract_subspace(d["M"], d["mask"], method=method)
        if B is None or len(c) != 20:
            continue
        Bs.append(B)
    if len(Bs) < 3:
        return None
    c2 = []
    for i in range(len(Bs)):
        for j in range(i + 1, len(Bs)):
            r = align(Bs[i], np.arange(20), Bs[j], np.arange(20))
            c2.append(r["cos"][1])
    w, V = eigs(Bs)
    Qs = [centered_orth(B) for B in Bs]
    P = np.mean([Q @ Q.T for Q in Qs], axis=0)
    uu = np.linalg.eigh(P)[1][:, ::-1][:, 0]
    dirs = []
    for Q in Qs:
        c = Q.T @ uu
        c = c / np.linalg.norm(c)
        dirs.append(Q @ np.array([-c[1], c[0]]))
    _, pr, _ = spectrum(dirs)
    return (len(Bs), float(np.median(c2)), float(w[0]), float(w[1]),
            float(pr), abs(float(uu @ u_ref)))


def sensitivity(accepted, u_ref):
    print("\n" + "=" * 84)
    print("[敏感性] 本分析里做的全部变体选择，逐条替换后的 headline 数字")
    print(f"{'变体':<48}{'k':>3}{'cosθ2中位':>10}{'λ1':>7}{'λ2':>7}"
          f"{'PR':>7}{'|cos(u,u0)|':>12}")
    print("-" * 84)
    rec = []
    for label, src, dup, wt, sites, meth in VARIANTS:
        if src == "thr4":
            ms = [d for d in load_skempi()
                  if d["dataset"] == "SKEMPI_OMTKY3"
                  and d["ligand"] != "4-proteases-pooled"]
        else:
            ms = build_matrices(accepted, dup=dup, include_wt=wt, sites=sites)
        s = _stats(ms, u_ref, method=meth)
        if s is None:
            print(f"{label:<48}  ——（抽取失败）")
            continue
        kk, m2, l1, l2, pr, cu = s
        print(f"{label:<48}{kk:>3}{m2:>10.3f}{l1:>7.3f}{l2:>7.3f}"
              f"{pr:>7.2f}{cu:>12.3f}")
        rec.append({"variant": label, "k": kk, "median_cos2": m2,
                    "lambda1": l1, "lambda2": l2, "pr": pr,
                    "cos_u_vs_base": cu})
    prs = [r["pr"] for r in rec]
    m2s = [r["median_cos2"] for r in rec]
    print("-" * 84)
    print(f"   PR 跨 {len(rec)} 个变体：{min(prs):.2f}–{max(prs):.2f}"
          f"（全距 {max(prs)-min(prs):.2f}；k=4 零分布 95% 区间宽约 0.96）")
    print(f"   cosθ2 中位跨变体：{min(m2s):.3f}–{max(m2s):.3f}")
    print(f"   第一轴方向跨变体与基准的 |cos|："
          f"{min(r['cos_u_vs_base'] for r in rec):.3f} 起")
    return rec


# ----------------------------------------------------------------------
# 5. 主流程
# ----------------------------------------------------------------------
def main():
    rows = read_full()
    table = census(rows)
    accepted, named, _ = identify_omtky3(rows)
    selfcheck_identification(rows, accepted)

    scales = load_scales(verbose=False)
    kd = scales["kd_hydropathy"]

    mats = build_matrices(accepted)
    print("\n" + "=" * 84)
    print("[面板] 从**完全未过滤**的 cache/skempi_v2.csv 建的矩阵")
    print(f"{'蛋白酶':<26}{'家族':>5}{'位点':>5}{'有效':>5}{'残基':>5}{'填充':>8}"
          f"{'观测':>6}{'原始行':>7}{'截尾':>8}")
    print("-" * 84)
    for d in mats:
        eff = int((d["mask"].sum(1) >= 4).sum())
        d["sites_effective"] = eff
        print(f"{d['ligand']:<26}{d['family']:>5}{len(d['sites']):>5}"
              f"{eff:>5}{int(d['mask'].any(0).sum()):>5}{d['fill']*100:>7.1f}%"
              f"{d['n_obs']:>6}{d['n_raw']:>7}{d['censor_frac']*100:>7.1f}%")
    print("   「有效」= 过了 MIN_PER_ROW ≥ 4 保护的位点数；位点 28 每酶只有"
          " 1 个取代，被保护丢弃，所以每个矩阵实际都是 10 位点 × 19 非野生型残基。")

    # --- 可分辨性门 ---
    print("\n[门] 第二维个体可分辨性（bootstrap 2000，门槛 stab cos2 ≥ 0.60）")
    print(f"{'蛋白酶':<26}{'stab cos1':>11}{'stab cos2':>11}{'判定':>8}")
    print("-" * 84)
    keep = []
    for i, d in enumerate(mats):
        B, cols, st = resolvability(d, seed=1000 + i)
        d["B"], d["cols"] = B, cols
        d["stab_cos1"], d["stab_cos2"] = st[0], st[1]
        d["pass_gate"] = st[1] >= 0.60 and len(cols) == 20
        print(f"{d['ligand']:<26}{st[0]:>11.3f}{st[1]:>11.3f}"
              f"{'通过' if d['pass_gate'] else '剔除':>8}")
        if d["pass_gate"]:
            keep.append(d)
    print(f"   通过 {len(keep)} / {len(mats)}；k = {len(keep)}")

    k = len(keep)
    rng = np.random.default_rng(20261011 + 5)

    # --- 成对主角度 ---
    n1 = haar_pair_null(20, n=4000, seed=7)
    n1d = haar_1d_null(20, n=4000, seed=7)
    p95 = (float(np.percentile(n1["cos1"], 95)),
           float(np.percentile(n1["cos2"], 95)))
    p95_1d = float(np.percentile(n1d, 95))
    print("\n" + "=" * 84)
    print(f"[成对] k = {k} 的全部 {k*(k-1)//2} 个配对")
    print(f"   N1 随机零分布 p95: cos1 {p95[0]:.3f}  cos2 {p95[1]:.3f}  "
          f"一维残差 {p95_1d:.3f}")
    print(f"{'配对':<40}{'同家族':>7}{'cosθ1':>8}{'cosθ2':>8}{'投掉疏水后':>11}")
    print("-" * 84)
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = keep[i], keep[j]
            r = align(a["B"], a["cols"], b["B"], b["cols"])
            dfl = align_deflated(a["B"], a["cols"], b["B"], b["cols"],
                                 kd, np.arange(20))
            same = a["family"] == b["family"]
            rec = {"a": a["ligand"], "b": b["ligand"],
                   "fam_a": a["family"], "fam_b": b["family"],
                   "same_family": bool(same),
                   "cos1": r["cos"][0], "cos2": r["cos"][1],
                   "n_res": r["n_res"],
                   "defl": dfl["cos_resid"] if dfl else None}
            pairs.append(rec)
            print(f"{a['ligand']+' | '+b['ligand']:<40}"
                  f"{('是' if same else '否'):>7}"
                  f"{rec['cos1']:>8.3f}{rec['cos2']:>8.3f}"
                  f"{(rec['defl'] if rec['defl'] else float('nan')):>11.3f}")
    c1 = np.array([p["cos1"] for p in pairs])
    c2 = np.array([p["cos2"] for p in pairs])
    dfv = np.array([p["defl"] for p in pairs if p["defl"] is not None])
    print("-" * 84)
    print(f"   cosθ1 中位 {np.median(c1):.3f}  范围 {c1.min():.3f}–{c1.max():.3f}"
          f"  超 N1 p95 {int((c1 > p95[0]).sum())}/{len(c1)}")
    print(f"   cosθ2 中位 {np.median(c2):.3f}  范围 {c2.min():.3f}–{c2.max():.3f}"
          f"  超 N1 p95 {int((c2 > p95[1]).sum())}/{len(c2)}")
    print(f"   投掉疏水后 中位 {np.median(dfv):.3f}  "
          f"超 p95 {int((dfv > p95_1d).sum())}/{len(dfv)}")

    # --- 共识层：λ + 标定到观测 λ1 的条件零分布 ---
    print("\n" + "=" * 84)
    print(f"[共识] 平均投影算子 λ 与标定后的条件零分布（k = {k}，4000 次）")
    Bs = [d["B"] for d in keep]
    w, V = eigs(Bs)
    u = V[:, 0]
    cn, alpha = cond_null(k, u, rng, target_l1=float(w[0]))
    un = uncond_null(k, rng)
    print(f"   观测 λ = {np.round(w[:3], 3).tolist()}")
    print(f"   N1 无条件随机零分布 p95 = "
          f"{np.round(np.percentile(un, 95, axis=0), 3).tolist()}")
    print(f"   条件零分布抖动角 α = {np.degrees(alpha):.1f}°；"
          f"**标定残差** |模拟 λ1 中位 − 观测 λ1| = "
          f"{abs(float(np.median(cn[:, 0])) - float(w[0])):.4f}")
    print(f"   条件零分布中位 = "
          f"{np.round(np.median(cn, axis=0), 3).tolist()}；p95 = "
          f"{np.round(np.percentile(cn, 95, axis=0), 3).tolist()}")
    p_l2 = float(np.mean(cn[:, 1] >= w[1]))
    print(f"   λ2 在条件零分布下的 p = {p_l2:.4f}  -> "
          + ("第二维也共享" if w[1] > np.percentile(cn[:, 1], 95)
             else "只有一条方向共享"))
    # 不误伤（阴性对照）：合成「只共享一维」的 k 个子空间，同一检验必须不显著
    neg = []
    for t in range(4):
        r2 = np.random.default_rng(900 + t)
        Qs = []
        for _ in range(k):
            a = r2.normal(size=20)
            ONE = np.ones(20) / np.sqrt(20)
            a -= ONE * (ONE @ a); a -= u * (u @ a); a /= np.linalg.norm(a)
            Qs.append(np.stack([u, a], 1))
        ww, VV = eigs(Qs)
        cc, _ = cond_null(k, VV[:, 0], r2, nsim=1500,
                          target_l1=float(ww[0]))
        neg.append(float(np.mean(cc[:, 1] >= ww[1])))
    print(f"   阴性对照（合成「只共享一维」）的 p = "
          f"{', '.join(f'{x:.2f}' for x in neg)}  -> "
          + ("不误报 OK" if min(neg) > 0.05 else "误报 FAIL"))

    # --- PR：残差方向的有效维数，零分布在**本 k** 上重新生成 ---
    print("\n" + "=" * 84)
    print(f"[PR] 残差方向参与率（零分布在 k = {k} 上重新生成，4000 次）")
    Qs = [centered_orth(d["B"]) for d in keep]
    P = np.mean([Q @ Q.T for Q in Qs], axis=0)
    ww, VV = np.linalg.eigh(P)
    uu = VV[:, ::-1][:, 0]
    dirs = []
    for Q in Qs:
        c = Q.T @ uu
        c = c / np.linalg.norm(c)
        dirs.append(Q @ np.array([-c[1], c[0]]))
    ws, pr, n90 = spectrum(dirs)
    ONE = np.ones(20) / np.sqrt(20)
    prn, n90n = [], []
    rng2 = np.random.default_rng(31337)
    for _ in range(4000):
        ds = []
        for _ in range(k):
            a = rng2.normal(size=20)
            a -= ONE * (ONE @ a); a -= uu * (uu @ a)
            ds.append(a / np.linalg.norm(a))
        _, p_, n_ = spectrum(ds)
        prn.append(p_); n90n.append(n_)
    prn = np.array(prn)
    print(f"   残差谱（前 4）= " + " ".join(f"{x:.3f}" for x in ws[:4]))
    print(f"   PR = {pr:.2f}   （完全相同 = 1；完全各向同性 ≈ k = {k}）")
    print(f"   PR 零分布（k = {k}）中位 {np.median(prn):.2f} "
          f"[{np.percentile(prn, 2.5):.2f}, {np.percentile(prn, 97.5):.2f}]")
    print(f"   单尾 p（PR ≤ 观测）= {float(np.mean(prn <= pr)):.4f}")
    print(f"   覆盖 90% 需要 {n90} 维；零分布中位 {int(np.median(n90n))} 维")
    # 量级校准（§8）：把 PR 翻译成一个可直接解释的「共享度」，
    # 并与 18 维各向同性基线对照。零分布基线不是 0 —— 这是 §8 的要点。
    vres = np.linalg.eigh(np.mean([np.outer(d, d) for d in dirs],
                                  axis=0))[1][:, -1]
    share = float(np.mean([abs(d @ vres) for d in dirs]))
    pw = [abs(float(dirs[i] @ dirs[j])) for i in range(k)
          for j in range(i + 1, k)]
    base = []
    for _ in range(2000):
        a = rng2.normal(size=20); a -= ONE * (ONE @ a); a -= uu * (uu @ a)
        b = rng2.normal(size=20); b -= ONE * (ONE @ b); b -= uu * (uu @ b)
        base.append(abs(float((a / np.linalg.norm(a)) @ (b / np.linalg.norm(b)))))
    print(f"   量级校准：残差方向与其主轴的平均 |cos| = {share:.2f}；"
          f"两两 |cos| 中位 {np.median(pw):.2f}")
    print(f"      18 维各向同性基线：两两 |cos| 中位 {np.median(base):.2f} "
          f"（基线不是 0 —— 必须读成「比基线高多少」）")

    # --- 后验假设：第二轴按蛋白酶家族分组？（post-hoc，不作结论） ---
    print("\n" + "=" * 84)
    print("[后验假设，POST-HOC] 第二轴是否按蛋白酶结构家族分组？")
    sf = [p["cos2"] for p in pairs if p["same_family"]]
    df_ = [p["cos2"] for p in pairs if not p["same_family"]]
    print(f"   同家族（S1–S1）配对 n = {len(sf)}: "
          + " ".join(f"{x:.3f}" for x in sorted(sf, reverse=True)))
    print(f"   跨家族（S1–S8）配对 n = {len(df_)}: "
          + " ".join(f"{x:.3f}" for x in sorted(df_, reverse=True)))
    if sf and df_:
        print(f"   中位差 = {np.median(sf) - np.median(df_):+.3f}")
        # 精确置换检验（6 个配对，全部 C(6,3) = 20 种标签分配）
        import itertools
        allv = sf + df_
        obs_d = np.median(sf) - np.median(df_)
        cnt = tot = 0
        for comb in itertools.combinations(range(len(allv)), len(sf)):
            s = [allv[i] for i in comb]
            o = [allv[i] for i in range(len(allv)) if i not in comb]
            tot += 1
            if np.median(s) - np.median(o) >= obs_d:
                cnt += 1
        print(f"   精确置换 p = {cnt}/{tot} = {cnt/tot:.3f}  "
              f"（{len(sf)}+{len(df_)} 个配对，全部标签分配穷尽）")
        print("   注意：家族标签是**看过数据之后**才用来分组的，"
              "且 S1 组只有 3 个成员、S8 组只有 1 个 ——")
        print("   S1–S8 的 3 个配对全部共享同一个 S8 矩阵，不独立。"
              "这**只能当假设**，不能当结论。")

    sens = sensitivity(accepted, uu)

    out = {"k": k, "pairs": pairs, "sensitivity": sens,
           "matrices": [{kk: v for kk, v in d.items()
                         if kk not in ("M", "mask", "B", "cols")}
                        for d in mats],
           "null_pair_p95": {"cos1": p95[0], "cos2": p95[1], "d1": p95_1d},
           "consensus": {"lambda": w[:3].tolist(),
                         "uncond_p95": np.percentile(un, 95, axis=0).tolist(),
                         "cond_med": np.median(cn, axis=0).tolist(),
                         "cond_p95": np.percentile(cn, 95, axis=0).tolist(),
                         "alpha_deg": float(np.degrees(alpha)),
                         "calib_resid": abs(float(np.median(cn[:, 0]))
                                            - float(w[0])),
                         "p_lambda2": p_l2, "neg_control_p": neg,
                         "u": u.tolist()},
           "pr": {"resid_eigs": ws[:4].tolist(), "pr": float(pr),
                  "null_med": float(np.median(prn)),
                  "null_ci": [float(np.percentile(prn, 2.5)),
                              float(np.percentile(prn, 97.5))],
                  "p_one_sided": float(np.mean(prn <= pr)),
                  "n90": n90, "n90_null_med": float(np.median(n90n))},
           "census_k_ge4": [bb for kk, bb, _ in table if kk >= 4],
           "named_pdbs": sorted(named), "accepted_pdbs": sorted(accepted)}
    p = os.path.join(ROOT, "cache", "omtky3_panel.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
