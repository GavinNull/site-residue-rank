#!/usr/bin/env python3
"""docs/26 第 1 层：构造「一般性去稳定度」(generic disruptiveness) 的残基标度。

要回答的问题（docs/25 §6 命名的替代解释）：
    每个 site×residue 矩阵都带一条「这个取代一般地不利」的方向，
    它是**氨基酸的性质**而不是界面的性质，因此会在无关家族间
    **按构造**一致。

为什么它能进交互项（这一步不能含糊）：
    一般性去稳定度本身是**主效应**，双向中心化把它完全移除，
    所以它**不能**直接进交互项。但若「一般地更不利的取代」在
    **结构更敏感的位点**上更不利，就产生一条
    (残基去稳定度) × (位点敏感度) 的秩 1 交互 —— 形式与所谓普适轴相同。
    所以要检验的对象是：u（普适轴的残基载荷）与去稳定度标度的关系。

每条标度的「无配体」(ligand-free) 论证必须逐条写明，
因为一条偷偷编码了界面效应的标度会把问题变成循环论证。

本脚本只**构造并自检**标度，不做任何与 u 的比较（那在
analysis/generic_axis_test.py）。CLAUDE.md §10：工具先自验。

自检（全部用已知事实，不用自洽性）：
  [B1] BLOSUM62 对称、对角最大、W/C 对角 = 11/9、已知高可换对为正
  [B2] BLOSUM62 的「平均可换度」必须把 W/C/P 排在最不可换的一端
  [F1] ddPCA folding 的符号约定：Pro 取代最去稳定（docs/15 V7 的已知事实）
  [F2] 列主效应（center_fit）与朴素列均值在完备子集上一致
  [E1] Starr 表达量：Pro 取代降低表达量（折叠缺陷的已知方向）
  [R*] 每条标度自己的可复现性（D-032：不可复现的方向不能与任何东西对齐）

运行：python3 analysis/generic_disrupt_scales.py
产物：cache/generic_disrupt_scales.json
"""
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddpca_load as DL                                   # noqa: E402
import rank_core as RC                                    # noqa: E402
from subspace_lib import AA20, ROOT                       # noqa: E402

DMS = os.path.join(ROOT, "cache", "dms")

# ----------------------------------------------------------------------
# BLOSUM62（Henikoff & Henikoff 1992, PNAS 89:10915）
# 顺序 ARNDCQEGHILKMFPSTWYV（标准发布顺序，不是本仓库的 AA20 顺序）
# —— 顺序交错正是 D-023 事故的形态，所以下面立刻做已知事实自检。
# ----------------------------------------------------------------------
B62_ORDER = "ARNDCQEGHILKMFPSTWYV"
B62_ROWS = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""


def blosum62():
    rows = [r for r in B62_ROWS.strip().splitlines() if r.strip()]
    assert len(rows) == 20, "BLOSUM62 行数不是 20"
    B = np.array([[float(x) for x in r.split()] for r in rows])
    assert B.shape == (20, 20)
    # [B1] 已知事实自检
    assert np.allclose(B, B.T), "BLOSUM62 不对称"
    idx = {a: i for i, a in enumerate(B62_ORDER)}
    known = {("W", "W"): 11, ("C", "C"): 9, ("H", "H"): 8, ("P", "P"): 7,
             ("Y", "Y"): 7, ("I", "V"): 3, ("K", "R"): 2, ("D", "E"): 2,
             ("F", "Y"): 3, ("S", "T"): 1, ("L", "M"): 2, ("A", "S"): 1,
             ("W", "F"): 1, ("H", "Y"): 2, ("G", "A"): 0, ("L", "I"): 2}
    for (a, b), v in known.items():
        got = B[idx[a], idx[b]]
        assert got == v, f"BLOSUM62[{a}][{b}] = {got}，已知值 {v}"
    for i in range(20):
        assert B[i, i] == B[i].max(), f"{B62_ORDER[i]} 行的最大值不在对角"
    # 换到本仓库的 AA20 顺序
    P = np.array([[B[idx[a], idx[b]] for b in AA20] for a in AA20])
    return P


def blosum_exchangeability():
    """返回 (exch, disrupt)：exch = 对其余 19 个残基的平均 BLOSUM62 分，
    disrupt = -exch（越不可换 = 一般地越不利）。

    **无配体论证**：BLOSUM62 由 BLOCKS 数据库里**各类蛋白**的保守区段
    比对块统计得来，没有任何界面 / 配体 / 结合位点的注释参与。
    它编码的是「进化上这两个残基互换得起吗」这一跨家族的平均容忍度。
    它**不是**界面无关的（进化也约束界面），但它对**任一特定配体或界面
    不条件化**，所以不可能编码本证据集里那些具体界面的效应。
    """
    P = blosum62()
    off = P.copy().astype(float)
    np.fill_diagonal(off, np.nan)
    exch = np.nanmean(off, axis=1)
    # [B2] 已知事实：W / C / P 必须落在最不可换的一端（它们在任何
    # 取代矩阵里都是替换代价最高的三类：巨大芳香、二硫、骨架受限）
    order = [AA20[i] for i in np.argsort(exch)]
    bottom5 = set(order[:5])
    assert {"W", "C", "P"} <= bottom5, \
        f"BLOSUM 平均可换度最低 5 位 = {order[:5]}，应含 W/C/P"
    assert order[-1] in "ST A" .replace(" ", "") or order[-1] in "AST", \
        f"最可换的残基是 {order[-1]}，预期在 A/S/T 中"
    return exch, -exch, order


# ----------------------------------------------------------------------
# D1：ddPCA folding ΔΔG 的残基主效应
# ----------------------------------------------------------------------
def _col_effect(M, mask):
    """列主效应（缺失下用 center_fit 的交替估计，不是朴素列均值）。"""
    r, c = RC.center_fit(M, mask)
    return c


def ddpca_folding_scales(verbose=True):
    """返回 {domain: {'col': 20向量, 'mask_n': 20向量, 'M':…}}。

    **无配体论证**：folding ΔΔG 是该结构域**自身的解折叠自由能**，
    由 abundance / stabilityPCA 测定 —— 该 assay 里**没有任何配体**，
    读出的是蛋白在细胞内的丰度（正确折叠的量）。
    GRB2-SH3 与 PSD95-PDZ3 有**独立的** stabilityPCA
    （ddpca_load.HAS_ABUNDANCE_ASSAY，由 zip 索引核对），
    所以它们的 folding 系数不是从结合数据里反推的。
    **GB1 没有独立 abundance assay**（docs/15 §6.1）：它的 folding 与
    binding 系数都是从**同一组 binding-PCA** 观测联合反推的，
    所以 GB1 folding **不是无配体的**，只作污染对照单列，不进主标度。
    """
    out = {}
    for dom in DL.DOMAINS:
        D = DL.load_domain(dom)
        DL.verify(dom, D, verbose=False)
        M, mask, cinfo = DL.matrix(D, "folding")
        c = _col_effect(M, mask)
        med = np.array([np.median(M[:, j][mask[:, j]])
                        if mask[:, j].sum() else np.nan for j in range(20)])
        pi = AA20.index("P")
        # [F1] 已知事实，**按 docs/15 V7 自己用的那个口径**（逐残基中位数）：
        #      Pro 取代是 20 个残基里最去稳定的，正 = 去稳定。
        #      三个域都必须通过。
        assert int(np.nanargmax(med)) == pi, \
            f"{dom}: folding 逐残基中位数最大的不是 Pro" \
            f"（{AA20[int(np.nanargmax(med))]}）"
        # [F1b] 列主效应（center_fit）是本脚本实际使用的量，它与中位数
        #      口径不同（它在缺失下做交替估计），所以另立一条较弱的判据：
        #      Pro 必须在列主效应的前两位，且两个口径必须高度相关。
        rank_p = int(np.where(np.argsort(-c) == pi)[0][0]) + 1
        assert rank_p <= 2, \
            f"{dom}: Pro 在 folding 列主效应里只排第 {rank_p}"
        r_med = float(np.corrcoef(c, med)[0, 1])
        assert r_med > 0.80, f"{dom}: 列主效应与中位数口径 r = {r_med:.3f} 过低"
        # [F2] 列主效应 vs 朴素列均值
        naive = np.array([np.nanmean(M[:, j][mask[:, j]])
                          if mask[:, j].sum() else np.nan
                          for j in range(20)])
        rr = float(np.corrcoef(c, naive)[0, 1])
        assert rr > 0.95, f"{dom}: 列主效应与朴素列均值 r = {rr:.3f} 过低"
        out[dom] = {"col": c, "naive": naive, "median": med,
                    "n_per_col": mask.sum(0).tolist(), "censor": cinfo,
                    "M": M, "mask": mask,
                    "independent_abundance":
                        DL.HAS_ABUNDANCE_ASSAY[dom],
                    "pro_rank_coleff": rank_p,
                    "r_col_vs_median": r_med,
                    "r_col_vs_naive": rr}
        if verbose:
            print(f"  [F1/F2] {dom:<12} 中位数口径 Pro 最大 "
                  f"({med[pi]:+.2f})；列主效应里 Pro 排第 {rank_p} "
                  f"({c[pi]:+.2f})；r(列主效应, 中位数) = {r_med:.3f}；"
                  f"r(列主效应, 列均值) = {rr:.3f}；"
                  f"独立 abundance assay = "
                  f"{DL.HAS_ABUNDANCE_ASSAY[dom]}")
    return out


def ddpca_splithalf(dd, nrep=400, seed=7):
    """[R1] 每个域的列主效应自己可复现吗：位点随机对半，两半各估列主效应。"""
    rep = {}
    for dom, z in dd.items():
        M, mask = z["M"], z["mask"]
        rng = np.random.default_rng(seed)
        nr = M.shape[0]
        rs = []
        for _ in range(nrep):
            p = rng.permutation(nr)
            h1, h2 = p[: nr // 2], p[nr // 2:]
            c1 = _col_effect(M[h1], mask[h1])
            c2 = _col_effect(M[h2], mask[h2])
            if np.std(c1) == 0 or np.std(c2) == 0:
                continue
            rs.append(float(np.corrcoef(c1, c2)[0, 1]))
        rep[dom] = {"n": len(rs), "median": float(np.median(rs)),
                    "ci": [float(np.percentile(rs, 2.5)),
                           float(np.percentile(rs, 97.5))]}
    return rep


# ----------------------------------------------------------------------
# D3：Starr 酵母展示的**表达量**（无配体）
# ----------------------------------------------------------------------
EXPR_JOBS = [
    # (标签, 文件, 位点列, 表达量主列, [重复列], target 列)
    ("starr2020/WH1", "starr2020_rbd_single_mut_effects.csv", "site_SARS2",
     "expr_avg", ["expr_lib1", "expr_lib2"], None),
    ("sarbecovirus", "starr_SARSr_RBD_final_variant_scores_lib40_41.csv",
     "site", "expression", [], "target"),
    ("PDCoV", "starr_PDCoV_RBD_final_variant_scores.csv", "position",
     "expr", ["expr_rep1", "expr_rep2"], "target"),
]


def _starr_expr_matrix(path, site_key, col, target=None, tcol="target"):
    cells = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if target is not None and r.get(tcol) != target:
                continue
            v = r.get(col, "")
            if v in ("", "NA", "nan", None):
                continue
            try:
                cells[(r[site_key], r["mutant"])] = float(v)
            except ValueError:
                continue
    if len(cells) < 200:
        return None
    sites = sorted({s for s, _ in cells})
    si = {s: i for i, s in enumerate(sites)}
    M = np.full((len(sites), 20), np.nan)
    for (s, a), v in cells.items():
        if a in AA20:
            M[si[s], AA20.index(a)] = v
    return M, np.isfinite(M)


def starr_expr_scales(verbose=True):
    """返回 [{label, col, reps:[col,…]}]。

    **无配体论证**：Starr 的酵母表面展示里，表达量通道用**抗标签抗体**
    （anti-MYC）染 RBD 的 C 端标签，分选时培养基里**没有 ACE2 或任何
    RBD 配体**。它度量的是「这个取代让 RBD 还能不能被正确折叠并展示」
    —— 一个纯粹的折叠/展示量，与结合通道是同一批文库的**不同一次分选**。
    **必须声明的限定**：它与本证据集里的 binding 矩阵是**同源**
    （同一个文库、同一批实验），所以它与 u 的任何一致既可能是
    「一般去稳定度」也可能是「同实验共享的技术方向」。
    因此 D-032 要求的「同源 vs 跨源」分列在这条标度上是必须的。
    """
    out = []
    for label, fn, skey, col, reps, tcol in EXPR_JOBS:
        p = os.path.join(DMS, fn)
        if not os.path.exists(p):
            continue
        targets = [None]
        if tcol:
            ts = sorted({r[tcol] for r in csv.DictReader(open(p))})
            targets = ts
        for tg in targets:
            z = _starr_expr_matrix(p, skey, col, tg, tcol or "target")
            if z is None:
                continue
            M, mask = z
            c = _col_effect(M, mask)
            # [E1] 已知事实：Pro 取代降低表达量（折叠缺陷的方向）
            pi = AA20.index("P")
            ok = int(np.argmin(c)) == pi
            rc = []
            for rcol in reps:
                zz = _starr_expr_matrix(p, skey, rcol, tg, tcol or "target")
                if zz is not None:
                    rc.append(_col_effect(zz[0], zz[1]))
            nm = f"{label}/{tg}" if tg else label
            out.append({"label": nm, "col": c, "reps": rc,
                        "n_obs": int(mask.sum()),
                        "pro_is_min": bool(ok),
                        "M": M, "mask": mask, "source_file": fn})
            if verbose:
                print(f"  [E1] {nm[:34]:<36} n={int(mask.sum()):>5}  "
                      f"Pro 列效应 {c[pi]:+.3f}  "
                      f"最小是 Pro？{'是' if ok else '否 ('
                                                  + AA20[int(np.argmin(c))] + ')'}"
                      f"  重复 {len(rc)} 条")
    assert out, "没有任何可用的表达量矩阵"
    # [E1] 聚合判据：多数矩阵必须把 Pro 排在最低（已知事实）
    frac = np.mean([z["pro_is_min"] for z in out])
    assert frac >= 0.5, f"只有 {frac:.0%} 的表达量矩阵把 Pro 排最低"
    if verbose:
        print(f"  [E1] 聚合：{frac:.0%} 的表达量矩阵把 Pro 列效应排最低 —— 通过")
    return out


# ----------------------------------------------------------------------
# D4：骨架几何破坏指示量（纯结构、无任何实验数据）
# ----------------------------------------------------------------------
def backbone_disruption():
    """Pro = 1.0、Gly = 0.5、其余 0。

    **无配体论证**：完全不含实验数据，只用两条教科书结构事实 ——
    Pro 去掉主链 NH 并锁死 φ、Gly 失去 Cβ 而放开 ψ/φ，这两者在**任何**
    结构环境里都破坏骨架几何。权重（1 vs 0.5）在看任何相关之前写定，
    依据是 docs/15 V7 的独立事实（三个域里 Pro 取代中位 +2.82/+3.47/+3.68
    最去稳定，Gly 排 16–19 位 —— 即 Pro > Gly）。

    **它的局限必须写明**：这是一条只有两个非零值的 20 点指示量，
    它单独的相关系数几乎完全由 P、G 两格决定，
    不足以承载任何「解释了多少」的论证，只用来看 u 的负端里
    P/G 占多少。
    """
    v = np.zeros(20)
    v[AA20.index("P")] = 1.0
    v[AA20.index("G")] = 0.5
    return v


# ----------------------------------------------------------------------
def z(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x)


def main():
    print("=" * 92)
    print("generic disruptiveness 标度构造 + 自检（CLAUDE.md §10）")
    print("=" * 92)

    print("\n[B1/B2] BLOSUM62 解析与已知事实自检")
    exch, b_disrupt, order = blosum_exchangeability()
    print("   对称 / 对角最大 / 16 个已知值 全部通过")
    print("   平均可换度（最不可换 -> 最可换）: " + " ".join(order))
    print(f"   W {exch[AA20.index('W')]:+.2f}  C {exch[AA20.index('C')]:+.2f}"
          f"  P {exch[AA20.index('P')]:+.2f}  "
          f"A {exch[AA20.index('A')]:+.2f}  S {exch[AA20.index('S')]:+.2f}")

    print("\n[F1/F2] ddPCA folding 列主效应")
    dd = ddpca_folding_scales()

    print("\n[R1] ddPCA 列主效应的可复现性（位点随机对半，400 次）")
    sh = ddpca_splithalf(dd)
    for dom, r in sh.items():
        print(f"   {dom:<12} 对半 r = {r['median']:.3f} "
              f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")

    print("\n[R1b] 跨域一致性（**跨源**配对，这才是关键的那一列）")
    doms = list(dd)
    cross = []
    for i in range(len(doms)):
        for j in range(i + 1, len(doms)):
            a, b = doms[i], doms[j]
            r = float(np.corrcoef(dd[a]["col"], dd[b]["col"])[0, 1])
            cross.append({"a": a, "b": b, "r": r})
            print(f"   {a:<12} × {b:<12} r = {r:+.3f}")

    print("\n[E1] Starr 表达量（无配体通道）列主效应")
    ex = starr_expr_scales()

    print("\n[R2] 表达量标度的可复现性")
    same_src, cross_src = [], []
    for zz in ex:
        if len(zz["reps"]) >= 2:
            r = float(np.corrcoef(zz["reps"][0], zz["reps"][1])[0, 1])
            same_src.append({"label": zz["label"], "r": r})
            print(f"   [同源] {zz['label'][:34]:<36} 两个重复 r = {r:+.3f}")
    for i in range(len(ex)):
        for j in range(i + 1, len(ex)):
            if ex[i]["source_file"] == ex[j]["source_file"]:
                continue
            r = float(np.corrcoef(ex[i]["col"], ex[j]["col"])[0, 1])
            cross_src.append({"a": ex[i]["label"], "b": ex[j]["label"],
                              "r": r})
    cr = [x["r"] for x in cross_src]
    print(f"   [跨源] {len(cross_src)} 个跨文件配对 r 中位 "
          f"{np.median(cr):+.3f}（{min(cr):+.3f}–{max(cr):+.3f}）")

    # 表达量共识标度：跨文件等权（每个源文件内部先平均）
    byfile = {}
    for zz in ex:
        byfile.setdefault(zz["source_file"], []).append(z(zz["col"]))
    expr_cons = np.mean([np.mean(v, axis=0) for v in byfile.values()], axis=0)
    print(f"   表达量共识 = {len(byfile)} 个源文件等权（文件内先平均）")

    # ddPCA 共识（只用有独立 abundance assay 的两个域）
    ok_doms = [d for d in dd if dd[d]["independent_abundance"]]
    fold_cons = np.mean([z(dd[d]["col"]) for d in ok_doms], axis=0)
    print(f"\n   folding 共识 = {ok_doms}（GB1 因无独立 abundance assay 排除）")

    # 三条经验标度的合成方向（第一主成分）
    S = np.stack([z(fold_cons), z(b_disrupt), -z(expr_cons)], 1)
    Sc = S - S.mean(0)
    _, sv, Vt = np.linalg.svd(Sc, full_matrices=False)
    comp = Sc @ Vt[0]
    if np.corrcoef(comp, z(fold_cons))[0, 1] < 0:
        comp = -comp
    pc1_frac = float(sv[0] ** 2 / (sv ** 2).sum())
    print(f"\n   合成方向 DISRUPT_PC1：三条经验标度的 PC1，"
          f"解释 {pc1_frac:.1%} 方差")
    print("   三条标度彼此的相关（都已转成「越大越去稳定」）:")
    nmz = ["folding_ddG", "blosum_nonexch", "expr_loss"]
    for i in range(3):
        for j in range(i + 1, 3):
            print(f"     {nmz[i]:<16} × {nmz[j]:<16} "
                  f"r = {np.corrcoef(S[:, i], S[:, j])[0, 1]:+.3f}")

    scales = {
        "D1_folding_ddG_ddPCA": fold_cons,
        "D1a_folding_GRB2": dd["GRB2-SH3"]["col"],
        "D1b_folding_PSD95": dd["PSD95-PDZ3"]["col"],
        "D1x_folding_GB1_NOT_ligand_free": dd["GB1"]["col"],
        "D2_blosum62_nonexchangeability": b_disrupt,
        "D3_starr_expression_loss": -expr_cons,
        "D4_backbone_disruption_PG": backbone_disruption(),
        "D5_disrupt_PC1": comp,
    }
    print("\n标度载荷（z 分，越正 = 一般地越去稳定）:")
    for nm, v in scales.items():
        o = np.argsort(-z(v))
        print(f"   {nm:<34} " + " ".join(f"{AA20[i]}" for i in o))

    out = {"aa_order": AA20,
           "scales": {k: np.asarray(v, float).tolist()
                      for k, v in scales.items()},
           "blosum_exchangeability": exch.tolist(),
           "blosum_order_least_to_most": order,
           "ddpca": {d: {"col": dd[d]["col"].tolist(),
                         "median": dd[d]["median"].tolist(),
                         "pro_rank_coleff": dd[d]["pro_rank_coleff"],
                         "r_col_vs_median": dd[d]["r_col_vs_median"],
                         "r_col_vs_naive": dd[d]["r_col_vs_naive"],
                         "independent_abundance":
                             dd[d]["independent_abundance"],
                         "censor_frac": dd[d]["censor"]["censor_frac"],
                         "n_obs": dd[d]["censor"]["n_obs"]}
                     for d in dd},
           "ddpca_splithalf": sh,
           "ddpca_cross_domain": cross,
           "expr": [{"label": zz["label"], "col": zz["col"].tolist(),
                     "n_obs": zz["n_obs"], "pro_is_min": zz["pro_is_min"],
                     "source_file": zz["source_file"]} for zz in ex],
           "expr_repro_same_source": same_src,
           "expr_repro_cross_source": cross_src,
           "disrupt_pc1_var_frac": pc1_frac,
           "folding_consensus_domains": ok_doms}
    p = os.path.join(ROOT, "cache", "generic_disrupt_scales.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
