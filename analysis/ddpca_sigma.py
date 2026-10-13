#!/usr/bin/env python3
"""docs/28 Step 1/2：ddPCA 六个矩阵的 σ 口径与 `coup_sd_true` 重算。

执行 D-042 留下的那个 §9.1 待决项：docs/15 §2.2 判三个 **binding** 矩阵
`coup_sd_true = 0`（于是「没有可定秩的方差」），而那个 σ 取的是**体外文献
ΔΔG 对照的总 RMSE**。docs/27 §4.1 在另一个平台上证明这个配法会制造假阴性。

**判据在看任何 `coup_sd_true` 之前写定并落盘**：
`cache/ddpca_sigma_predecl.json`（CLAUDE.md §4）。本脚本不改它。

四个 σ 锚点（全部**自己测**，没有从 docs/27 搬 0.45）
-----------------------------------------------------
A  下界 = 三次独立生物学重复在**双向中心化之后**的噪声（`ddpca_replicates`）
B' = 体外 RMSE 扣掉被行/列主效应吸收的那部分（docs/27 的机制，带置换零分布）
C  主用 = 逐矩阵反解（使合成秩 1 的位点对半可复现度 == 实测）
B  上界 = 体外文献 ΔΔG 对照的总 RMSE（docs/15 §1.1 V5 原值）

数值层全部复用，没有重写：`ddpca_load`（matrix / load_domain / verify /
literature_check）、`rank_core.center_fit`、`gate_calib_lib.split_half`
（经 `subspace_lib.extract_subspace`，内部是 `rank_test_power.als_guarded`，
**不是**未保护的 `als_complete`）、`generic_folding_axis.synth_like /
first_axis`。对既有脚本的改动：零。
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddpca_load as DL                                       # noqa: E402
import ddpca_replicates as RP                                 # noqa: E402
import gate_calib_lib as G                                    # noqa: E402
import rank_core as RC                                        # noqa: E402
from generic_folding_axis import synth_like                   # noqa: E402
from subspace_lib import AA20, ROOT                           # noqa: E402

PREDECL = os.path.join(ROOT, "cache", "ddpca_sigma_predecl.json")
OUTJSON = os.path.join(ROOT, "cache", "ddpca_sigma.json")

MATS = [("GB1", "folding"), ("GB1", "binding"),
        ("GRB2-SH3", "folding"), ("GRB2-SH3", "binding"),
        ("PSD95-PDZ3", "folding"), ("PSD95-PDZ3", "binding")]

# ddPCA 的哪个 assay 提供该表型的生物学重复。
# GB1 **没有** abundance assay（docs/15 §6.1），所以 GB1/folding 的重复把手
# 与 GB1/binding 是**同一组** binding-PCA 观测 —— 每个 GB1 数字都带这条。
REP_ASSAY = {("GB1", "folding"): "GB1/binding",
             ("GB1", "binding"): "GB1/binding",
             ("GRB2-SH3", "folding"): "GRB2-SH3/abundance",
             ("GRB2-SH3", "binding"): "GRB2-SH3/binding",
             ("PSD95-PDZ3", "folding"): "PSD95-PDZ3/abundance",
             ("PSD95-PDZ3", "binding"): "PSD95-PDZ3/binding"}


def predecl():
    with open(PREDECL) as f:
        return json.load(f)


# ----------------------------------------------------------------------
# 0. σ 分解机器的**双向**自检（CLAUDE.md §10：先跑，不过即中止）
# ----------------------------------------------------------------------
def df_factor(mask):
    """双向中心化消耗 (nr + nc − 1) 个自由度，所以观测格子上的残差 SD
    被**系统性压低** sqrt((n_obs − nr − nc + 1)/n_obs)。

    这是本脚本的自检实际抓到的一处估计量错误（docs/28 §3.3）：
    SD 上只有 ~3.6%，但 `sqrt(coup_sd_obs² − σ²)` 把它放大到低 SNR 下
    **几十个百分点**，并且在 coup/σ ≈ 0.17 时直接给出 0.000。
    方向是**偏低**，即偏向「没有信号」—— 正是本任务要查的那个方向。
    """
    nr, nc = mask.shape
    n = int(mask.sum())
    dof = n - nr - nc + 1
    return math.sqrt(n / dof) if dof > 0 else float("nan")


def machinery_selfcheck(verbose=True):
    """四条已知事实（§10.2：既测能命中，也测不误伤，两个方向都测）：
    (i)  已知 coup_sd_true + 已知 σ 的合成矩阵，**自由度校正后**的公式
         必须在 coup/σ ≥ 0.33 的全格上还原到 ≈ 真值；
    (ii) 无结构矩阵（只有主效应 + 噪声）必须给出 ≈ 0（不误伤）；
    (iii) 未校正的公式必须被测到**偏低**（这是 docs/15 用的那一版，
         其偏向是假阴性 —— 必须机器化留痕而不是写在注释里）；
    (iv) coup_sd_obs 自己必须 ≈ sqrt(coup_true² + σ²) × 1/df_factor，
         即偏差确实来自自由度而不是别的东西。
    """
    rng = np.random.default_rng(2028)
    mask = np.ones((56, 20), bool)
    for i in range(56):           # 制造与真实矩阵同量级的缺失
        mask[i, rng.integers(0, 20)] = False
    k = df_factor(mask)
    log, bad = [], []
    log.append(f"[df] n_obs = {int(mask.sum())}，nr+nc−1 = "
               f"{mask.shape[0] + mask.shape[1] - 1}，"
               f"自由度校正因子 = {k:.4f}")
    n_lower = 0
    for true_coup in (0.70, 0.40, 0.20, 0.10):
        for sg in (0.30, 0.60):
            est = []
            for t in range(12):
                r = np.random.default_rng(900 + t)
                M = synth_like(mask, _rand_dir(r, 20), true_coup, sg,
                               1.0, 0.8, r, rank=1)
                est.append(_coup_sd_obs(M, mask))
            obs = float(np.median(est))
            exp = math.sqrt(true_coup ** 2 + sg ** 2)
            raw = math.sqrt(max(obs ** 2 - sg ** 2, 0.0))
            cor = math.sqrt(max((obs * k) ** 2 - sg ** 2, 0.0))
            rel = abs(cor - true_coup) / true_coup
            # (iv) 偏差来自自由度
            ok4 = abs(obs * k - exp) < 0.03 * exp
            # (i) 只在实际使用区间（coup/σ ≥ 0.33）上断言精度
            use = true_coup / sg >= 0.33
            ok1 = (rel < 0.25) or not use
            # (iii) 未校正版本必须偏低
            if raw < true_coup:
                n_lower += 1
            log.append(
                f"[i/iii/iv] 真 coup {true_coup:.2f} σ {sg:.2f} "
                f"(coup/σ = {true_coup / sg:.2f}) -> obs {obs:.4f} "
                f"(应 {exp:.4f}，obs×df {obs * k:.4f}) | 未校正还原 "
                f"{raw:.4f}（相对差 {(raw - true_coup) / true_coup:+.0%}）"
                f" | **校正后 {cor:.4f}**（{(cor - true_coup) / true_coup:+.0%}）"
                f"  {'OK' if (ok1 and ok4) else 'FAIL'}"
                + ("" if use else " [coup/σ < 0.33，只报不断言]"))
            if not (ok1 and ok4):
                bad.append(log[-1])
    if n_lower < 6:
        bad.append(f"[iii] 未校正公式只在 {n_lower}/8 格上偏低 —— "
                   f"「偏向假阴性」这条没被测到")
    else:
        log.append(f"[iii] 未校正公式在 **{n_lower}/8** 格上偏低 "
                   f"—— 偏向**假阴性**，已机器化留痕  OK")
    for sg in (0.30, 0.60, 0.90):
        est = []
        for t in range(12):
            r = np.random.default_rng(1700 + t)
            M = (r.normal(0, 1.0, 56)[:, None] + r.normal(0, 0.8, 20)[None, :]
                 + r.normal(0, sg, (56, 20)))
            M = np.where(mask, M, np.nan)
            est.append(_coup_sd_obs(M, mask))
        obs = float(np.median(est))
        cor = math.sqrt(max((obs * k) ** 2 - sg ** 2, 0.0))
        ok = cor < 0.12 * sg
        log.append(f"[ii] 无结构 σ {sg:.2f} -> obs {obs:.4f}, 校正后还原 "
                   f"{cor:.4f}（判据 < {0.12 * sg:.3f}）  "
                   f"{'OK' if ok else 'FAIL'}")
        if not ok:
            bad.append(log[-1])
    if verbose:
        print("σ 分解机器的双向自检（§10：既测能命中，也测不误伤）")
        for l in log:
            print("   " + l)
    assert not bad, "σ 分解机器自检不通过:\n" + "\n".join(bad)
    return log


def _rand_dir(rng, n):
    v = rng.normal(size=n)
    v -= v.mean()
    return v / np.linalg.norm(v)


def _coup_sd_obs(M, mask):
    """双向中心化残差在观测格子上的 SD —— 与 docs/15 §2.2 同口径。"""
    r, c = RC.center_fit(M, mask)
    R = M - r[:, None] - c[None, :]
    return float(R[mask].std())


def decompose(M, mask):
    """主效应吸收分解：行 / 列 / 交互各占多少。"""
    r, c = RC.center_fit(M, mask)
    R = M - r[:, None] - c[None, :]
    rr = r - r.mean()
    cc = c - c.mean()
    tot = float(M[mask].std())
    cs = float(R[mask].std())
    k = df_factor(mask)
    return {"row_sd": float(rr.std()), "col_sd": float(cc.std()),
            "coup_sd_obs": cs, "coup_sd_obs_dfcor": cs * k,
            "df_factor": k, "total_sd": tot,
            "row_over_coup": float(rr.std() / cs)}


# ----------------------------------------------------------------------
# 1. 锚点 A：生物学重复在中心化之后的噪声
# ----------------------------------------------------------------------
def rep_matrices(assay, wt, pos):
    """用三个重复的 fitness 各建一个 site×residue 单取代矩阵（同一 mask）。"""
    D = RP.load(assay)
    sg = D["singles"]
    av = D["allvar"]
    # singles 表只有合并 fitness；重复列在 all_variants 里，按 aa_seq 连接
    idx = {str(s): i for i, s in enumerate(av["aa_seq"])}
    F = np.column_stack([av[f"fitness{i}_uncorr"] for i in (1, 2, 3)])
    si = {p: i for i, p in enumerate(pos)}
    Ms = [np.full((len(pos), 20), np.nan) for _ in range(3)]
    Mm = np.full((len(pos), 20), np.nan)
    fm = av["fitness"]
    n = 0
    # **已知事实自检（D-023 同型陷阱：编号偏移 / 残基顺序）**：
    # DiMSum 的 `singles` 表的 `Pos` 必须与 ddPCA 权重表的位点编号同基准，
    # 判据是 `WT_AA` 必须等于 `ddpca_load` 解析出的该位点野生型残基。
    # 若编号偏移一位，这里会**大面积**不符。实测六个组合 0 处不符。
    nm_ok = nm_bad = 0
    for k in range(len(sg["Pos"])):
        p = int(sg["Pos"][k])
        if p not in wt:
            continue
        if wt[p] == str(sg["WT_AA"][k]):
            nm_ok += 1
        else:
            nm_bad += 1
    assert nm_ok > 0 and nm_bad == 0, (
        f"{assay}: singles 表的位点编号与 ddPCA 权重表不同基准 "
        f"（WT_AA 吻合 {nm_ok}、不符 {nm_bad}）—— 疑似编号偏移")
    for k in range(len(sg["Pos"])):
        p, mu = int(sg["Pos"][k]), str(sg["Mut"][k])
        if p not in si or mu not in AA20 or mu == wt.get(p):
            continue
        j = idx.get(str(sg["aa_seq"][k]))
        if j is None:
            continue
        f3 = F[j]
        if not np.all(np.isfinite(f3)):
            continue
        for t in range(3):
            Ms[t][si[p], AA20.index(mu)] = f3[t]
        Mm[si[p], AA20.index(mu)] = fm[j]
        n += 1
    return Ms, Mm, n


def anchor_A(dom, ph, wt, pos, coup_ddg, verbose=True):
    """-> dict(sigma_A, ratio, ...)。单位换算用**比例**而不是绝对值。"""
    assay = REP_ASSAY[(dom, ph)]
    Ms, Mm, n = rep_matrices(assay, wt, pos)
    mk = np.all([np.isfinite(X) for X in Ms], axis=0) & np.isfinite(Mm)
    if mk.sum() < 200:
        return {"ok": False, "why": f"三重复齐全的格子只有 {int(mk.sum())}"}
    Rs = []
    for X in Ms:
        r, c = RC.center_fit(np.where(mk, X, np.nan), mk)
        Rs.append(np.where(mk, X, np.nan) - r[:, None] - c[None, :])
    # 逐格子、逐对差：中心化**之后**的噪声
    d = [float(np.std((Rs[i] - Rs[j])[mk], ddof=1))
         for i, j in ((0, 1), (0, 2), (1, 2))]
    sig_rep = float(np.mean(d) / math.sqrt(2.0))       # 单个重复
    sig_ml = sig_rep / math.sqrt(3.0)                  # 三重复合并
    rc, cc = RC.center_fit(np.where(mk, Mm, np.nan), mk)
    coup_fit = float((np.where(mk, Mm, np.nan) - rc[:, None]
                      - cc[None, :])[mk].std())
    # 同一批重复矩阵上的主效应吸收（与 ΔΔG 侧对读）
    dm = decompose(np.where(mk, Mm, np.nan), mk)
    ratio = sig_ml / coup_fit
    rr = [float(np.corrcoef(Rs[i][mk], Rs[j][mk])[0, 1])
          for i, j in ((0, 1), (0, 2), (1, 2))]
    out = {"ok": True, "assay": assay, "n_cells": int(mk.sum()),
           "sigma_rep_centered_fitness": sig_rep,
           "sigma_ml_centered_fitness": sig_ml,
           "coup_sd_obs_fitness": coup_fit,
           "ratio": ratio, "sigma_A": ratio * coup_ddg,
           "resid_corr_pairs": rr, "resid_corr_med": float(np.median(rr)),
           "fitness_decomp": dm}
    if verbose:
        print(f"   锚点 A（{assay}，{int(mk.sum())} 格）: 中心化后单重复 σ "
              f"{sig_rep:.4f}、三重复 ML σ {sig_ml:.4f}，"
              f"fitness 侧 coup_sd_obs {coup_fit:.4f} -> 比例 {ratio:.3f}"
              f" -> σ_A = {ratio * coup_ddg:.3f} kcal/mol")
        print(f"      三对重复的**中心化残差**相关 r = "
              + ", ".join(f"{x:.3f}" for x in rr))
    return out


# ----------------------------------------------------------------------
# 2. 锚点 B'：体外 RMSE 里被行/列主效应吸收的那部分（带置换零分布）
# ----------------------------------------------------------------------
def site_icc(pairs, nperm=2000, seed=5):
    """**主用**的吸收口径：体外残差在**位点**层的组内相关 ICC。

    为什么不是加性最小二乘：体外对照只有 11–80 个变体，而位点数与之同阶
    （Troilo2018 是 15 个点落在 15 个位点上），加性模型**饱和**，
    观测吸收与置换零分布都恒为 1.000，excess 恒为 0 —— 那不是「没有吸收」，
    是**这个估计量在这里没有自由度**（D-014 那一类：欠功效看起来像无结构）。
    ICC 只用「同一位点内的残差对」，所以只需要有重复位点，不需要位点数 ≪ n。

    ICC = 位点间方差 / 总方差，用单因素随机效应的矩估计；
    零分布 = 在观测设计下打乱残差 2000 次（这就是自由度基线）。
    返回 excess = (ICC_obs − ICC_null中位) / (1 − ICC_null中位)，即
    **真正会被行主效应吸收**的那一份。
    """
    if len(pairs) < 12:
        return None
    y = np.array([v for _, _, v in pairs], float)
    site = np.array([p for p, _, _ in pairs])
    grp = {}
    for i, s in enumerate(site):
        grp.setdefault(s, []).append(i)
    rep = {s: ix for s, ix in grp.items() if len(ix) >= 2}
    if not rep:
        return {"n": len(y), "n_sites": len(grp), "n_rep_sites": 0,
                "why": "没有任何位点有 ≥2 个体外变体 —— ICC 不可估"}

    def icc(yy):
        yy = yy - yy.mean()
        k = len(rep)
        ni = np.array([len(ix) for ix in rep.values()], float)
        mi = np.array([yy[ix].mean() for ix in rep.values()])
        N = ni.sum()
        ssb = float((ni * (mi - (ni * mi).sum() / N) ** 2).sum())
        ssw = float(sum(((yy[ix] - yy[ix].mean()) ** 2).sum()
                        for ix in rep.values()))
        dfb, dfw = k - 1, N - k
        if dfb <= 0 or dfw <= 0:
            return float("nan")
        msb, msw = ssb / dfb, ssw / dfw
        n0 = (N - (ni ** 2).sum() / N) / dfb
        va = (msb - msw) / n0 if n0 > 0 else 0.0
        tot = va + msw
        return 0.0 if tot <= 0 else max(0.0, min(1.0, va / tot))

    obs = float(icc(y))
    rng = np.random.default_rng(seed)
    nl = np.array([icc(rng.permutation(y)) for _ in range(nperm)])
    nl = nl[np.isfinite(nl)]
    med = float(np.median(nl))
    excess = 0.0 if med >= 1 else max(0.0, min(1.0, (obs - med) / (1 - med)))
    return {"n": len(y), "n_sites": len(grp), "n_rep_sites": len(rep),
            "n_in_rep_sites": int(sum(len(ix) for ix in rep.values())),
            "icc_obs": obs, "icc_null_med": med,
            "icc_null_p95": float(np.percentile(nl, 95)),
            "excess": excess, "p_one_sided": float(np.mean(nl >= obs))}


def absorption(pairs, nperm=2000, seed=5):
    """pairs: [(site, residue, residual)]。返回被加性 site+residue 吸收的
    方差比例、置换零分布、以及**超出零分布**的部分（自由度已扣）。

    **只作对照报告**：这些体外集上它基本是饱和的（见 `site_icc` 的说明）。
    `n_params / n` 一起报出来，让饱和看得见。
    """
    if len(pairs) < 12:
        return None
    sites = sorted({p for p, _, _ in pairs})
    res = sorted({a for _, a, _ in pairs})
    si = {s: i for i, s in enumerate(sites)}
    ri = {a: i for i, a in enumerate(res)}
    I = np.array([si[p] for p, _, _ in pairs])
    J = np.array([ri[a] for _, a, _ in pairs])
    y = np.array([v for _, _, v in pairs], float)
    X = np.zeros((len(y), len(sites) + len(res)))
    X[np.arange(len(y)), I] = 1.0
    X[np.arange(len(y)), len(sites) + J] = 1.0

    def frac(yy):
        yy = yy - yy.mean()
        beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        resid = yy - X @ beta
        ss = (yy ** 2).sum()
        return 0.0 if ss <= 0 else 1.0 - (resid ** 2).sum() / ss

    obs = float(frac(y))
    rng = np.random.default_rng(seed)
    nl = np.array([frac(rng.permutation(y)) for _ in range(nperm)])
    med = float(np.median(nl))
    excess = 0.0 if med >= 1 else max(0.0, min(1.0, (obs - med) / (1 - med)))
    return {"n": len(y), "n_sites": len(sites), "n_res": len(res),
            "n_params": len(sites) + len(res),
            "saturated": bool(len(sites) + len(res) >= 0.8 * len(y)),
            "absorbed_obs": obs, "absorbed_null_med": med,
            "absorbed_null_p95": float(np.percentile(nl, 95)),
            "excess": excess, "p_one_sided": float(np.mean(nl >= obs))}


def invitro_pairs(dom, ph, D, se_max=0.5, abs_max=10.0):
    """重跑 ddpca_load.literature_check 的配对，但把**逐变体残差**留下来。

    口径与 docs/15 §1.1 V5 完全一致（同一 SE / |ΔΔG| 过滤、同一符号锚点），
    只是额外返回 (site, residue, ddPCA − 文献) 以便做 §4.2 的吸收分解。
    """
    import csv
    base = DL._ensure_extracted()
    iv = os.path.join(base, "Data", "in_vitro")
    off = D["pdb_offset"]
    V = {p: {k: v for k, v in D["vals"][p].items()
             if np.isfinite(D["se"][p].get(k, math.nan))
             and D["se"][p][k] <= se_max and abs(v) <= abs_max}
         for p in ("folding", "binding")}

    def rows(path, sep="\t"):
        with open(path) as f:
            return list(csv.DictReader(f, delimiter=sep))

    def key(vid, numbering):
        m = DL.ID_RE.match(vid.strip())
        if not m:
            return None
        p = int(m.group(2))
        if numbering == "pdb":
            p -= off
        return (p, m.group(3)), m.group(1)

    got = []
    if dom == "GB1" and ph == "folding":
        pr = []
        for r0 in rows(os.path.join(iv, "GB1_literature_free_energies",
                                    "Olson2014_TableS4.txt")):
            k = key(r0["Mutant"], "pdb")
            if not k:
                continue
            (kk, wtl) = k
            if kk in V["folding"] and D["wt"].get(kk[0]) == wtl:
                pr.append((kk, float(r0["ddG"]), V["folding"][kk]))
        # 符号由该来源内部的 Pro/Gly 锚点定出（docs/15 §1.1 的事故 1）
        pg = [x for (kk, x, _) in pr if kk[1] in ("P", "G")]
        sign = -1.0 if (len(pg) >= 4 and np.mean(pg) < 0) else 1.0
        got.append(("Olson2014_TableS4",
                    [(kk, sign * a, b) for kk, a, b in pr]))
    if dom == "GRB2-SH3":
        rr = rows(os.path.join(iv, "GRB2_literature_free_energies.txt"))
        if ph == "folding":
            tro = {r0["id"].strip(): r0 for r0 in rr
                   if r0["Dataset"] == "Troilo_2018"}
            if "WT" in tro and tro["WT"]["Kf"] and tro["WT"]["Ku"]:
                g0 = -DL.RT * math.log(float(tro["WT"]["Kf"])
                                       / float(tro["WT"]["Ku"]))
                pr = []
                for vid, r0 in tro.items():
                    k = key(vid, "data")
                    if not k or not r0["Kf"] or not r0["Ku"]:
                        continue
                    (kk, wtl) = k
                    if kk in V["folding"] and D["wt"].get(kk[0]) == wtl:
                        g = -DL.RT * math.log(float(r0["Kf"])
                                              / float(r0["Ku"]))
                        pr.append((kk, g - g0, V["folding"][kk]))
                got.append(("Troilo2018(Kf/Ku)", pr))
        else:
            mal = {r0["id"].strip(): r0 for r0 in rr
                   if r0["Dataset"] == "Malagrino_2019"}
            if "WT" in mal and mal["WT"]["Kd"]:
                kd0 = float(mal["WT"]["Kd"])
                pr = []
                for vid, r0 in mal.items():
                    k = key(vid, "data")
                    if not k or not r0["Kd"]:
                        continue
                    (kk, wtl) = k
                    if kk in V["binding"] and D["wt"].get(kk[0]) == wtl:
                        pr.append((kk, DL.RT * math.log(float(r0["Kd"]) / kd0),
                                   V["binding"][kk]))
                got.append(("Malagrino2019(Kd)", pr))
    if dom == "PSD95-PDZ3":
        if ph == "folding":
            pr = []
            for r0 in rows(os.path.join(
                    iv, "PDZ_literature_free_energies",
                    "Calosci2008_invitro_ddGs_PDZ.txt")):
                k = key(r0["id"], "pdb")
                if not k or not r0["f_ddg"]:
                    continue
                (kk, wtl) = k
                if kk in V["folding"] and D["wt"].get(kk[0]) == wtl:
                    pr.append((kk, float(r0["f_ddg"]), V["folding"][kk]))
            got.append(("Calosci2008", pr))
        else:
            pr = []
            for r0 in rows(os.path.join(
                    iv, "PDZ_literature_free_energies",
                    "Chi2008_invitro_kd_ddGf_ddGb_PDZ_CRIPTpeptide.txt"), " "):
                k = key(r0["id"], "pdb")
                if not k or not r0.get("b_ddg"):
                    continue
                (kk, wtl) = k
                if kk in V["binding"] and D["wt"].get(kk[0]) == wtl:
                    pr.append((kk, float(r0["b_ddg"]), V["binding"][kk]))
            got.append(("Chi2008(b_ddg)", pr))
    return got


# ----------------------------------------------------------------------
# 3. 锚点 C：逐矩阵反解（位点对半可复现度）
# ----------------------------------------------------------------------
def observed_repro(M, mask, nsplit, seed=42):
    v = G.split_half(M, mask, rank=1, nsplit=nsplit, seed=seed)
    return (float(np.median(v)) if v else float("nan")), len(v)


def synth_repro(mask, coup_obs, sigma, main_row, main_col, nrep, nsplit,
                seed=0):
    """同 mask / 同主效应尺度 / coup_sd_true = sqrt(coup_obs²−σ²) 的合成秩 1
    矩阵，走**完全相同**的位点对半流程。"""
    # ct = 0 时**仍然要生成并跑完整流程** —— 那时矩阵只有主效应 + 噪声，
    # 位点对半的 |cos| 应落在碰巧水平（d = 20 时中位 ≈ 0.28），**不是 0**。
    # 第一版在这里直接 return 0.0，于是「实测可复现度很低」会被错配到
    # 「σ 大到把信号杀光」那一格。这是本脚本自捉的第二处估计量错误。
    ct = math.sqrt(max(coup_obs ** 2 - sigma ** 2, 0.0))
    vals = []
    for t in range(nrep):
        rng = np.random.default_rng(seed + 977 * t + int(sigma * 1e4))
        if ct > 1e-9:
            X = synth_like(mask, _rand_dir(rng, mask.shape[1]), ct, sigma,
                           main_row, main_col, rng, rank=1)
        else:
            nr, nc = mask.shape
            X = np.where(mask,
                         rng.normal(0, main_row, nr)[:, None]
                         + rng.normal(0, main_col, nc)[None, :]
                         + rng.normal(0, sigma, (nr, nc)), np.nan)
        v = G.split_half(np.where(mask, X, np.nan), mask, rank=1,
                         nsplit=nsplit, seed=7 + t)
        if v:
            vals.append(float(np.median(v)))
    return (float(np.median(vals)) if vals else float("nan")), ct


def backsolve_C(mask, coup_obs, main_row, main_col, obs_rep, grid,
                nrep, nsplit, verbose=True):
    tab = []
    for sg in grid:
        s, ct = synth_repro(mask, coup_obs, sg, main_row, main_col,
                            nrep, nsplit)
        tab.append({"sigma": float(sg), "synth_repro": s,
                    "coup_sd_true": ct})
        if verbose:
            print(f"      σ={sg:.2f} -> coup_true {ct:.3f}, 合成可复现 "
                  f"{s:.3f}（实测 {obs_rep:.3f}）")
    ok = [t for t in tab if np.isfinite(t["synth_repro"])]
    if not ok:
        return None, tab
    best = min(ok, key=lambda t: abs(t["synth_repro"] - obs_rep))
    return best["sigma"], tab


# ----------------------------------------------------------------------
# 4. §10.3 还原性诊断（逐矩阵、逐 σ）
# ----------------------------------------------------------------------
def recovery(mask, coup_obs, sigma, main_row, main_col, u, nrep, d_null,
             seed=0):
    """在矩阵自己的几何 / 缺失 / 主效应下把残基方向植入为 u，再用与真实
    数据**完全相同**的流程取第一轴，看能还原到多少、检出率多少。"""
    from generic_folding_axis import first_axis
    from subspace_lib import haar_1d_null
    ct = math.sqrt(max(coup_obs ** 2 - sigma ** 2, 0.0))
    p95 = float(np.percentile(haar_1d_null(d_null, n=20000, seed=3), 95))
    if ct <= 1e-9:
        return {"coup_sd_true": ct, "median_cos": 0.0, "detect": 0.0,
                "null_p95": p95, "n": 0,
                "note": "coup_sd_true = 0，没有可植入的结构"}
    cs = []
    for t in range(nrep):
        rng = np.random.default_rng(seed + 613 * t + int(sigma * 1e4))
        X = synth_like(mask, u, ct, sigma, main_row, main_col, rng, rank=1)
        w, cols = first_axis(np.where(mask, X, np.nan), mask)
        if w is None or len(cols) != len(u):
            continue
        cs.append(abs(float(w @ u[cols] / (np.linalg.norm(w)
                                           * np.linalg.norm(u[cols])))))
    if not cs:
        return {"coup_sd_true": ct, "median_cos": float("nan"),
                "detect": float("nan"), "null_p95": p95, "n": 0}
    cs = np.array(cs)
    return {"coup_sd_true": ct, "median_cos": float(np.median(cs)),
            "q10": float(np.percentile(cs, 10)),
            "q90": float(np.percentile(cs, 90)),
            "detect": float(np.mean(cs > p95)), "null_p95": p95,
            "n": int(len(cs))}


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrep", type=int, default=24)
    ap.add_argument("--nsplit", type=int, default=120)
    ap.add_argument("--nrec", type=int, default=60)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.nrep, a.nsplit, a.nrec = 6, 40, 20

    pd = predecl()
    print("=" * 96)
    print("docs/28 Step 1/2：σ 口径与 coup_sd_true 重算")
    print("判据已在看任何 coup_sd_true 之前落盘:", PREDECL)
    print("=" * 96)

    # §10：先跑自检，不过即中止
    RC.selfcheck(verbose=False)
    print("[自检] rank_core C1–C8 通过")
    RP.selfcheck(verbose=True)
    machinery_selfcheck()
    hz = json.load(open(os.path.join(ROOT, "cache",
                                     "gate_harness_check.json")))
    assert hz.get("ok") is True, "gate_harness_check.json 的 ok 不为 true"
    print("[自检] cache/gate_harness_check.json ok = true（docs/24 硬前置）")

    SIG_B = {f"{k}": v for k, v in
             pd["anchors"]["B_upper"]["values"].items()}
    grid = pd["anchors"]["C_main_backsolved"]["grid"]

    out = {"predecl": pd, "rows": []}
    for dom, ph in MATS:
        D = DL.load_domain(dom)
        DL.verify(dom, D, verbose=False)
        M, mask, cen = DL.matrix(D, ph)
        pos = sorted(D["wt"])
        name = f"{dom}/{ph}"
        print("\n" + "=" * 96)
        print(f"[{name}]  {cen['sites']}×19 设计 {cen['design']}，"
              f"观测 {cen['n_obs']}，截尾 {cen['censor_frac']:.1%}")
        dc = decompose(M, mask)
        print(f"   主效应吸收分解: row_sd {dc['row_sd']:.3f}、col_sd "
              f"{dc['col_sd']:.3f}、coup_sd_obs {dc['coup_sd_obs']:.3f}"
              f"（total {dc['total_sd']:.3f}）；row/coup = "
              f"{dc['row_over_coup']:.2f}")
        coup = dc["coup_sd_obs"]
        r, c = RC.center_fit(M, mask)
        main_row = float((r - r.mean()).std())
        main_col = float((c - c.mean()).std())

        A = anchor_A(dom, ph, D["wt"], pos, coup)
        # B'：体外残差的吸收
        absorb = []
        for src, pr in invitro_pairs(dom, ph, D):
            if len(pr) < 12:
                absorb.append({"src": src, "n": len(pr),
                               "why": "n < 12，不做分解"})
                continue
            resid = [(kk[0], kk[1], b - aa) for kk, aa, b in pr]
            z = {"src": src}
            z["rmse"] = float(np.sqrt(np.mean(
                [(b - aa) ** 2 for _, aa, b in pr])))
            z["additive"] = absorption(resid)
            z["icc"] = site_icc(resid)
            ad, ic = z["additive"], z["icc"]
            print(f"   体外残差吸收（{src}, n={ad['n']}, "
                  f"{ad['n_sites']} 位点 × {ad['n_res']} 残基，RMSE "
                  f"{z['rmse']:.3f}）")
            print(f"      [对照] 加性 site+residue: 观测吸收 "
                  f"{ad['absorbed_obs']:.3f}，置换零分布中位 "
                  f"{ad['absorbed_null_med']:.3f}，超出 {ad['excess']:.3f}"
                  f"（参数 {ad['n_params']}/{ad['n']}"
                  f"{'，**饱和，无自由度**' if ad['saturated'] else ''}）")
            if "icc_obs" in ic:
                z["excess"] = ic["excess"]
                print(f"      [主用] 位点层 ICC: 观测 {ic['icc_obs']:.3f}"
                      f"（{ic['n_rep_sites']} 个有重复的位点，"
                      f"{ic['n_in_rep_sites']} 个变体），置换零分布中位 "
                      f"{ic['icc_null_med']:.3f}（p95 "
                      f"{ic['icc_null_p95']:.3f}），**超出 "
                      f"{ic['excess']:.3f}**，单尾 p = "
                      f"{ic['p_one_sided']:.4f}")
            else:
                print(f"      [主用] 位点层 ICC 不可估: {ic.get('why')}")
            absorb.append(z)
        sB = float(SIG_B[name])
        exc = [z["excess"] for z in absorb if "excess" in z]
        excess = float(np.mean(exc)) if exc else None
        sBp = sB * math.sqrt(1 - excess) if excess is not None else None

        obs_rep, nv = observed_repro(M, mask, a.nsplit)
        print(f"   实测位点对半第一轴可复现度（n={nv} 次切分）: "
              f"{obs_rep:.3f}")

        u = _rand_dir(np.random.default_rng(7), 20)
        convs = {}
        # 两个 coup_sd_obs 口径都算：
        #   docs15 = 原口径（与 docs/15 §2.2 的表可直接对读，且偏保守）
        #   dfcor  = 自由度校正后（自检证明这一版才无偏，见 §3.3）
        for conv, cval in (("docs15", coup),
                           ("dfcor", dc["coup_sd_obs_dfcor"])):
            print(f"   --- coup_sd_obs 口径 = {conv}"
                  f"（{cval:.3f} kcal/mol）---")
            sC, ctab = backsolve_C(mask, cval, main_row, main_col, obs_rep,
                                   grid, a.nrep, a.nsplit, verbose=False)
            sA = (A["ratio"] * cval) if A.get("ok") else None
            anchors = {"A": sA, "C": sC, "Bp": sBp, "B": sB}
            print("      σ 锚点: " + ", ".join(
                f"{kk} = {v:.3f}" if v is not None else f"{kk} = n/a"
                for kk, v in anchors.items()))
            aA, aC = anchors["A"], anchors["C"]
            gate = (aA is not None and aC is not None
                    and aA <= aC + 1e-9 and aC <= sB + 1e-9)
            print(f"      **预先写定的判据 A ≤ C ≤ B**: "
                  f"{'n/a' if aA is None else round(aA, 3)} ≤ {aC} ≤ {sB}"
                  f"  -> {'通过' if gate else '**不通过**'}")
            ct = {kk: (math.sqrt(max(cval ** 2 - v ** 2, 0.0))
                       if v is not None else None)
                  for kk, v in anchors.items()}
            snr = {kk: (ct[kk] / anchors[kk] if anchors[kk] else None)
                   for kk in anchors}
            rec = {}
            for kk in ("A", "C", "Bp", "B"):
                if anchors[kk] is None:
                    continue
                rec[kk] = recovery(mask, cval, anchors[kk], main_row,
                                   main_col, u, a.nrec, 20)
                print(f"      σ_{kk} = {anchors[kk]:.3f} -> coup_sd_true "
                      f"{ct[kk]:.3f} = {snr[kk]:.2f}σ | §10.3 还原 |cos| "
                      f"中位 {rec[kk]['median_cos']:.3f}，检出率 "
                      f"{rec[kk]['detect']:.0%}"
                      f"（零分布 p95 {rec[kk]['null_p95']:.3f}）")
            carries = (sC is not None and snr["C"] is not None
                       and snr["C"] >= 1.1
                       and rec.get("C", {}).get("detect", 0) >= 0.80)
            print(f"      **预先写定的判决（σ_C，1.1σ + 还原检出 ≥ 80%）**: "
                  f"{'有可定秩的信号' if carries else '没有'}")
            convs[conv] = {"coup_sd_obs": cval, "sigma": anchors,
                           "bracket_pass": gate, "coup_sd_true": ct,
                           "snr": snr, "recovery": rec,
                           "backsolve_table": ctab,
                           "carries_signal": bool(carries)}

        out["rows"].append({
            "name": name, "domain": dom, "phenotype": ph,
            "censor": cen, "decomp": dc,
            "main_row": main_row, "main_col": main_col,
            "anchorA": A, "absorption": absorb, "excess": excess,
            "obs_repro": obs_rep, "n_splits": nv,
            "conventions": convs,
            "carries_signal": convs["docs15"]["carries_signal"],
            "carries_signal_dfcor": convs["dfcor"]["carries_signal"]})

    json.dump(out, open(OUTJSON, "w"), ensure_ascii=False, indent=1,
              default=float)
    print("\n产物:", OUTJSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
