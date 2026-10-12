#!/usr/bin/env python3
"""folding ΔΔG 的 site x residue 交互也是秩 2 吗？—— 配对对照实验。

问题（docs/15-folding-vs-binding.md）
  已确立：protein-ligand **binding** ΔΔG 的 site x residue 交互（双向中心化后）
  是秩 2（docs/12-rank-two.md）。本脚本回答它是否**结合特异**：
  若 folding 稳定性 ΔΔG 也是秩 2 且残基载荷**子空间相同**，
  则秩 2 是氨基酸取代的通性，不是关于配体结合的陈述。

数据：ddPCA（Faure et al. 2022）三个域，同一变体同时给出 folding 与 binding
  ΔΔG —— 相同位点、相同残基、相同 assay 噪声结构，两个表型。
  加载与解析自检见 analysis/ddpca_load.py。

执行顺序是**有意的**（CLAUDE.md §10.3）：
  step power  : 在**真实矩阵尺寸/稀疏/噪声**下测定秩程序的检出概率。
                这一步 gate 后面所有步骤。测不出来就是测不出来。
  step rank   : 每域每表型的无泄露嵌套 CV 定秩。
  step sub    : 残基载荷**子空间**的主角 + 两种零分布 + bootstrap 区间。

用法:
  python3 analysis/folding_vs_binding.py power
  python3 analysis/folding_vs_binding.py rank
  python3 analysis/folding_vs_binding.py sub
  python3 analysis/folding_vs_binding.py all
"""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddpca_load as DL                                    # noqa: E402
import rank_core as RC                                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "cache", "folding_vs_binding")
AA20 = DL.AA20
PHENO = ("folding", "binding")

# 定秩参数：全局固定一套，control / power / rank / 零分布全部用同一套。
# ridge=0.1 与 within_system_rank.py 一致，以便与 docs/12 的数字可比。
#
# 我先前把「理想条件下秩2 被判成秩4」归因于 ridge，那是**错的**，在此
# 更正并留痕（CLAUDE.md §10：区分工具自身的错与被检查对象的错）：
# 扫 ridge ∈ {0, 0.1, 0.3} x sigma ∈ {0.64, 0.35}，每格 8 个种子，
# 选出的秩**与 ridge 完全无关**（0.64 全给 2，0.35 全给 3）。
# 真正的原因是无泄露中心化的吸收偏差（见 rank_core 的 C6b）：
# 每折训练集估主效应时会吃掉一部分耦合，耦合/噪声比越高吃得越多，
# 在留出格子上留下类似额外低秩成分的偏移 -> 定秩**偏高**。
# 方向是偏高，所以本文档里出现的「秩 0 / 秩 1」不可能由它造成。
CV = dict(max_rank=4, K=5, restarts=3, iters=250, ridge=0.1)
SE_MAX, ABS_MAX = 0.5, 10.0

# 每个矩阵的**现实**噪声：ddPCA ΔΔG 与体外文献 ΔΔG 的 RMSE。
# 由 analysis/ddpca_load.py 的 V5 输出，provenance 在那里（zip 内
# Data/in_vitro/*）。这是能拿到的最接近「真实误差」的量 —— ddPCA 的
# bootstrap SE 只度量重拟合波动，低估一个量级以上（0.01–0.19 vs 0.46–0.89）。
LIT_RMSE = {
    ("GB1", "folding"): 0.67,          # Olson2014 Table S4, n=80
    ("GRB2-SH3", "folding"): 0.46,     # Troilo2018 Kf/Ku, n=15
    ("PSD95-PDZ3", "folding"): 0.89,   # Calosci2008, n=30
    ("PSD95-PDZ3", "binding"): 0.64,   # Chi2008 b_ddg, n=11
    ("GRB2-SH3", "binding"): 1.22,     # Malagrino2019 Kd, n=12（动态范围极小）
    # ("GB1", "binding") 无体外对照 -> 用可得值的中位 0.67
}


def _mkdir():
    os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------ 矩阵构造
def build(dom, paired=True):
    """返回 {phenotype: (M, mask, censor_info)} 以及配对后的公共 mask。

    paired=True 时两个表型用**同一个** mask（两边都可用的格子），
    这样 folding/binding 的秩差异不可能来自缺失模式差异。
    """
    D = DL.load_domain(dom)
    DL.verify(dom, D, verbose=False)        # 自检强制运行（§10）
    res = {}
    for ph in PHENO:
        M, mk, c = DL.matrix(D, ph, se_max=SE_MAX, abs_max=ABS_MAX)
        res[ph] = [M, mk, c]
    if paired:
        both = res["folding"][1] & res["binding"][1]
        # 丢掉观测过少的位点。理由不是「让数字好看」：带保护的 ALS 对
        # 训练观测数 < rank+1 的行直接把因子置 0，这样的行既不贡献信息
        # 又会在留出集上制造爆点。实测 GB1 配对后有 1 个位点 0 观测、
        # 3 个 < 5 观测。门槛取 max_rank + 3 = 7（留出一折后仍可定秩）。
        keep = both.sum(axis=1) >= CV["max_rank"] + 3
        n_drop = int((~keep).sum())
        both = both & keep[:, None]
        for ph in PHENO:
            res[ph][1] = both
            res[ph][0] = np.where(both, res[ph][0], np.nan)
            res[ph][2] = dict(res[ph][2], paired=True,
                              n_obs=int(both.sum()),
                              sites_kept=int(keep.sum()),
                              sites_dropped=n_drop,
                              censor_frac=1 - both.sum()
                              / res[ph][2]["design"])
    # 噪声估计：bootstrap SE（乐观）与体外文献 RMSE（现实）
    D["pdb_offset"] = D.get("pdb_offset", 0)
    for ph in PHENO:
        ses = [D["se"][ph][(p, a)] for p in sorted(D["wt"])
               for a in AA20
               if (p, a) in D["se"][ph]
               and np.isfinite(D["se"][ph][(p, a)])
               and D["se"][ph][(p, a)] <= SE_MAX]
        res[ph].append(float(np.median(ses)))
    return res, D


def stats_of(M, mask):
    """实测主效应尺度与耦合残差尺度（都在观测格子上）。"""
    r, c = RC.center_fit(M, mask)
    R = M - r[:, None] - c[None, :]
    return {
        "total_sd": float(M[mask].std()),
        "row_sd": float(r.std()), "col_sd": float(c.std()),
        "coup_sd": float(R[mask].std()),
        "coup_frac": float((R[mask].std() / M[mask].std()) ** 2),
    }


# ------------------------------------------------------------ step power
def _synth_like(mask, r_true, coup_sd, sigma, main_row, main_col, rng):
    """在给定 mask / 噪声 / 主效应尺度下合成真实秩 r_true 的矩阵。

    耦合项按 (lam1, lam2) 比例 1.5:0.8（rank_power.py 的设计点），
    整体缩放到逐元素 SD = coup_sd。
    """
    nr, nc = mask.shape
    M = (rng.normal(0, main_row, nr)[:, None]
         + rng.normal(0, main_col, nc)[None, :])
    if r_true > 0:
        base = [1.5, 0.8, 0.4][:r_true]
        C = np.zeros((nr, nc))
        for lam in base:
            phi = rng.normal(0, 1, nr)
            psi = rng.normal(0, 1, nc)
            phi -= phi.mean()
            psi -= psi.mean()
            phi /= np.linalg.norm(phi) or 1
            psi /= np.linalg.norm(psi) or 1
            C += lam * math.sqrt(nr * nc) * np.outer(phi, psi)
        s = C[mask].std()
        if s > 0:
            M = M + C * (coup_sd / s)
    M = M + rng.normal(0, sigma, (nr, nc))
    return np.where(mask, M, np.nan)


def step_power(nrep=50, seed=101, r_grid=(0, 1, 2), paired=True,
               only=None):
    """在真实矩阵尺寸/稀疏/噪声下，定秩程序能检出秩 2 吗？

    这是 CLAUDE.md §10.3 要求的理想条件还原性诊断的**真实条件版本**：
    不是问「方法在无限数据下对不对」，是问「在我手上这个矩阵上有没有功效」。
    """
    _mkdir()
    rows = []
    t0 = time.time()
    for dom in DL.DOMAINS:
        res, D = build(dom, paired=paired)
        for ph in PHENO:
            if only and (dom, ph) not in only:
                continue
            M, mask, cinfo, sig_boot = res[ph]
            st = stats_of(M, mask)
            # ---- 噪声水平 ----
            # sigma_boot = bootstrap SE 中位。**不能用**，只报告：它只含
            #   mochi 重拟合的不确定性，不含模型偏倚。实测 0.01–0.19，
            #   而同一批 ΔΔG 与体外文献的 RMSE 是 0.46–0.89 kcal/mol
            #   —— 低估一个量级以上。按 sigma_boot 做功效分析会把
            #   「本来就测不出」说成「测得出」。
            # sigma_lit = 与体外 ΔΔG 的 RMSE（LIT_RMSE），这是能拿到的
            #   最接近真实误差的量。缺测的格用可得值的中位 0.67。
            # ---- 耦合强度（这里修了一个量级错误）----
            # 实测的中心化残差 SD 已经**包含噪声**：
            #     coup_sd_obs^2 = coup_sd_true^2 + sigma^2
            # 先前版本把 coup_sd_obs 当真实耦合再叠加 sigma，合成数据的
            # 耦合比真实数据可能达到的最大值还强约 35%，功效被系统性高估。
            # 正确做法：coup_sd_true = sqrt(max(coup_sd_obs^2 - sigma^2, 0))。
            # 若该值为 0，说明观测到的全部耦合方差都能被噪声解释，
            # 这本身就是答案（没有可供定秩的结构）。
            # 两个 sigma：体外校准值（主）与 0.35（本仓库 evidence/
            # assay-noise.md 的「同源单次测量」下限，**乐观**上界情形）。
            # sigma_boot 只记录在 json 里，不用于合成（见上）。
            for signame, sigma in (("lit", LIT_RMSE.get((dom, ph), 0.67)),
                                   ("opt0.35", 0.35)):
                c_true = math.sqrt(max(st["coup_sd"] ** 2 - sigma ** 2, 0.0))
                for mult, mname in ((1.0, "x1"),):
                    cs = c_true * mult
                    for r_true in r_grid:
                        det = []
                        for k in range(nrep):
                            rng = np.random.default_rng(
                                seed + 1000 * r_true + k)
                            Ms = _synth_like(mask, r_true, cs, sigma,
                                             max(st["row_sd"], 1e-6),
                                             max(st["col_sd"], 1e-6), rng)
                            e = RC.cv_rank_noleak(np.nan_to_num(Ms), mask,
                                                  seed=7 + k, **CV)
                            det.append(RC.best_rank(e)[0])
                        det = np.array(det)
                        rows.append({
                            "domain": dom, "phenotype": ph,
                            "paired": paired, "sigma_src": signame,
                            "sigma": sigma, "coup_mult": mname,
                            "coup_sd_true": cs,
                            "r_true": r_true,
                            "n_obs": cinfo["n_obs"],
                            "censor_frac": cinfo["censor_frac"],
                            "coup_sd_obs": st["coup_sd"],
                            "coup_frac": st["coup_frac"],
                            "se_boot_median": sig_boot,
                            "p_exact": float((det == r_true).mean()),
                            "p_ge2": float((det >= 2).mean()),
                            "p_zero": float((det == 0).mean()),
                            "mean_det": float(det.mean()),
                            "hist": {int(k): int((det == k).sum())
                                     for k in range(5)},
                            "nrep": nrep,
                        })
                        print(f"  {dom:<11}{ph:<8}σ={signame}({sigma:.2f}) "
                              f"耦合{mname}({cs:.2f}) 真实秩{r_true} -> "
                              f"{dict(rows[-1]['hist'])} "
                              f"P(=真)={rows[-1]['p_exact']:.2f} "
                              f"P(>=2)={rows[-1]['p_ge2']:.2f}  "
                              f"[{time.time() - t0:.0f}s]")
    tag = "power" if paired else "power_unpaired"
    json.dump(rows, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
    return rows


# ------------------------------------------------------- step rankboot
def step_rankboot(nboot=2000, paired=False, seeds=(3, 11, 23, 37, 53)):
    """秩间比较的区间（CLAUDE.md §5：点估计一律不单独汇报）。

    按**位点**聚类重抽留出平方误差 nboot 次，给出
      相对秩0 的 trim95 增益，以及 秩1->秩2 的相对改善
    的 95% 区间。判据是**区间是否包含 0**，不是均值大小。
    """
    _mkdir()
    rows = []
    for dom in DL.DOMAINS:
        res, _ = build(dom, paired=paired)
        for ph in PHENO:
            M, mask, cinfo, _ = res[ph]
            rr, errs = RC.cv_rank_noleak_errors(np.nan_to_num(M), mask,
                                                **CV)
            b = RC.cluster_bootstrap_gains(rr, errs, nboot=nboot)
            pt = {r: (RC.trim95(errs[0]) - RC.trim95(errs[r]))
                  / RC.trim95(errs[0]) for r in (1, 2, 3)}
            d = {"domain": dom, "phenotype": ph, "paired": paired,
                 "n_obs": cinfo["n_obs"], "nboot": nboot,
                 "censor_frac": cinfo["censor_frac"],
                 "trim95_rank0": RC.trim95(errs[0]),
                 "gain_point": pt,
                 "gain_ci": {r: [float(np.quantile(b[r], .025)),
                                 float(np.quantile(b[r], .975))]
                             for r in (1, 2, 3)},
                 "gain_1to2_point": (RC.trim95(errs[1]) - RC.trim95(errs[2]))
                 / RC.trim95(errs[1]),
                 "gain_1to2_ci": [float(np.quantile(b["1to2"], .025)),
                                  float(np.quantile(b["1to2"], .975))]}
            rows.append(d)
            print(f"  {dom:<11}{ph:<8}秩0 trim95={d['trim95_rank0']:.3f}  "
                  + "  ".join(
                      f"秩{r}:{pt[r]:+.1%}[{d['gain_ci'][r][0]:+.1%},"
                      f"{d['gain_ci'][r][1]:+.1%}]" for r in (1, 2, 3))
                  + f"  秩1->2:{d['gain_1to2_point']:+.1%}"
                  f"[{d['gain_1to2_ci'][0]:+.1%},"
                  f"{d['gain_1to2_ci'][1]:+.1%}]")
    tag = "rankboot" if paired else "rankboot_unpaired"
    json.dump(rows, open(os.path.join(OUT, tag + ".json"), "w"), indent=1,
              default=float)
    return rows


# ------------------------------------------------------- step threshold
def step_threshold(nrep=30, seed=777, paired=False,
                   grid=(0.20, 0.35, 0.50, 0.70, 1.00)):
    """可检出阈值：多强的秩 2 耦合才能在**这个**矩阵上被检出？

    为什么需要这一步：三个 ddPCA **binding** 矩阵的
    coup_sd_true = sqrt(coup_sd_obs^2 - sigma^2) 全部为 0
    —— 观测到的耦合残差整体小于独立校准的测量误差。
    那里无法「按真实强度合成信号」，因为没有信号可合成。
    于是改问另一个有意义的问题：检出秩 2 需要多大的耦合，
    再和该矩阵**实际拥有**的 coup_sd_obs 比。
    """
    _mkdir()
    rows = []
    t0 = time.time()
    for dom in DL.DOMAINS:
        res, _ = build(dom, paired=paired)
        for ph in PHENO:
            M, mask, cinfo, sig_boot = res[ph]
            st = stats_of(M, mask)
            sigma = LIT_RMSE.get((dom, ph), 0.67)
            for cs in grid:
                det = []
                for k in range(nrep):
                    rng = np.random.default_rng(seed + k)
                    Ms = _synth_like(mask, 2, cs, sigma,
                                     max(st["row_sd"], 1e-6),
                                     max(st["col_sd"], 1e-6), rng)
                    e = RC.cv_rank_noleak(np.nan_to_num(Ms), mask,
                                          seed=7 + k, **CV)
                    det.append(RC.best_rank(e)[0])
                det = np.array(det)
                rows.append({
                    "domain": dom, "phenotype": ph, "paired": paired,
                    "sigma": sigma, "coup_sd_synth": cs,
                    "coup_sd_obs": st["coup_sd"],
                    "coup_sd_true": math.sqrt(
                        max(st["coup_sd"] ** 2 - sigma ** 2, 0.0)),
                    "n_obs": cinfo["n_obs"],
                    "censor_frac": cinfo["censor_frac"],
                    "p_detect2": float((det == 2).mean()),
                    "p_ge2": float((det >= 2).mean()),
                    "hist": {int(k): int((det == k).sum())
                             for k in range(5)}, "nrep": nrep})
                print(f"  {dom:<11}{ph:<8}σ={sigma:.2f} 合成耦合SD={cs:.2f} "
                      f"-> {dict(rows[-1]['hist'])} "
                      f"P(检出2)={rows[-1]['p_detect2']:.2f}  "
                      f"[实际 coup_sd_obs={st['coup_sd']:.2f}] "
                      f"[{time.time() - t0:.0f}s]")
    json.dump(rows, open(os.path.join(OUT, "threshold.json"), "w"), indent=1)
    return rows


# ------------------------------------------------------------ step control
def _starr_matrix(path, col, lower_bound, site_key="position",
                  target_key="target"):
    """Starr 系列：剔除堆在检测下限的截尾值，返回 M, mask。"""
    import csv
    cells, cens, tot = {}, 0, 0
    with open(path) as f:
        for r in csv.DictReader(f):
            v = r.get(col, "")
            if v in ("", "NA"):
                continue
            tot += 1
            x = float(v)
            if x <= lower_bound + 1e-9:
                cens += 1
                continue
            tgt = r.get(target_key, "")
            cells[(f"{tgt}_{r[site_key]}", r["mutant"])] = x
    sites = sorted({s for s, _ in cells})
    si = {s: i for i, s in enumerate(sites)}
    M = np.full((len(sites), len(AA20)), np.nan)
    for (s, a), v in cells.items():
        if a in AA20:
            M[si[s], AA20.index(a)] = v
    mask = np.isfinite(M)
    keep = mask.sum(axis=1) >= CV["max_rank"] + 3
    mask = mask & keep[:, None]
    return np.where(mask, M, np.nan), mask, cens / max(tot, 1)


def step_control():
    """**真实数据正控制**：同一套实现必须在已给出秩 2 的结合矩阵上重现秩 2。

    为什么必须做：合成数据上的 C5/C6 只证明算法在我的生成模型下工作。
    如果这套实现在 docs/12 已确立的矩阵上拿不到秩 2，那么 ddPCA 上的
    「秩 0」就无法区分「folding 没结构」与「我这版实现没功效」
    —— 正是 CLAUDE.md §10.3 要排除的混淆。
    """
    _mkdir()
    DMS = os.path.join(ROOT, "cache", "dms")
    jobs = [
        ("SARS-CoV-2 RBD x huACE2",
         os.path.join(DMS, "starr2020_rbd_single_mut_effects.csv"),
         "bind_avg", -np.inf, "site_SARS2", None),
        ("PD-CoV RBD x gAPN",
         os.path.join(DMS, "starr_PDCoV_RBD_final_variant_scores.csv"),
         "bind_gAPN", 5.0, "position", "target"),
    ]
    rows = []
    for name, path, col, lb, sk, tk in jobs:
        if not os.path.exists(path):
            print(f"  {name}: 文件缺失，跳过")
            continue
        M, mask, cfrac = _starr_matrix(path, col, lb, sk,
                                       tk if tk else "__none__")
        st = stats_of(M, mask)
        bests = []
        for ridge in (0.0, 0.1):
            bs = []
            for sd in (3, 11, 23):
                e = RC.cv_rank_noleak(np.nan_to_num(M), mask, seed=sd,
                                      **dict(CV, ridge=ridge))
                bs.append(RC.best_rank(e)[0])
            tm = RC.best_rank(e)[1]
            bests.append((ridge, bs, tm))
            print(f"  {name:<26} ridge={ridge:<4} n={int(mask.sum()):>5} "
                  f"位点={int(mask.any(1).sum()):>4} 截尾={cfrac:.1%} "
                  f"耦合占比={st['coup_frac']:.1%} 最优秩={bs}  trim95="
                  + " ".join(f"{tm[r]:.3f}" for r in sorted(tm)))
        rows.append({"name": name, "n_obs": int(mask.sum()),
                     "sites": int(mask.any(1).sum()),
                     "censor_frac": cfrac, **st,
                     "by_ridge": [{"ridge": r, "best_ranks": b,
                                   "trim95": t} for r, b, t in bests]})
    json.dump(rows, open(os.path.join(OUT, "control.json"), "w"), indent=1)
    return rows


# ------------------------------------------------------------ step rank
def step_rank(seeds=(3, 11, 23, 37, 53), extra_ridge=(0.0,), paired=True):
    """paired=True 用两表型共同可用的格子（配对，但截尾更重）；
    paired=False 用每个表型自己的 mask（破坏配对，但功效最高）。

    两个都要跑。原因：配对取交集后截尾从 3.0%（GRB2 folding 自己）
    升到 44.1%，而本仓库已确立截尾 >= 25% 时绝大多数体系给秩 0
    （docs/19 把 docs/12 的「7/7」更正为 8/9，Spearman rho = -0.77
     [-0.89, -0.54]；方向不变，不要当标定好的定量预测量用）
    —— 配对本身可能把结构洗掉，那样配对设计的优势就被自己毁了。
    """
    _mkdir()
    rows = []
    for dom in DL.DOMAINS:
        res, D = build(dom, paired=paired)
        for ph in PHENO:
            M, mask, cinfo, sig = res[ph]
            st = stats_of(M, mask)
            for ridge in (CV["ridge"],) + tuple(extra_ridge):
                cv = dict(CV, ridge=ridge)
                bests, trims = [], []
                for sd in seeds:
                    e = RC.cv_rank_noleak(np.nan_to_num(M), mask, seed=sd,
                                          **cv)
                    b, t = RC.best_rank(e)
                    bests.append(b)
                    trims.append(t)
                tm = {r: float(np.median([t[r] for t in trims]))
                      for r in trims[0]}
                g = {r: (tm[0] - tm[r]) / tm[0] for r in tm}
                rows.append({
                    "domain": dom, "phenotype": ph, "ridge": ridge,
                    "paired": paired,
                    "n_obs": cinfo["n_obs"], "sites": cinfo["sites"],
                    "censor_frac": cinfo["censor_frac"],
                    "se_median": sig, **st,
                    "best_ranks": bests,
                    "best_rank_mode": int(max(set(bests), key=bests.count)),
                    "trim95_median": tm, "gain_vs_rank0": g,
                })
                print(f"  {dom:<11}{ph:<8}ridge={ridge:<4} "
                      f"n={cinfo['n_obs']:>5} 截尾={cinfo['censor_frac']:.1%} "
                      f"耦合占比={st['coup_frac']:.1%}  "
                      f"trim95=" + " ".join(f"{tm[r]:.3f}"
                                            for r in sorted(tm))
                      + f"  最优秩={bests}")
    tag = "rank" if paired else "rank_unpaired"
    json.dump(rows, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
    return rows


# ------------------------------------------------------------ step sub
def _V(M, mask, rank=2, seed=11, restarts=8):
    _, V, _ = RC.fit_loadings(np.nan_to_num(M), mask, rank=rank,
                              restarts=restarts, seed=seed)
    return V


def _haar_angles(nc, k, nsamp, rng, constrain_ones=True):
    """Haar 随机 k 维子空间两两主角的零分布。

    constrain_ones：双向中心化使残差的列均值为 0，拟合出的残基载荷因此
    被推向与全 1 向量正交的子空间。零分布必须带同样的约束，否则
    零分布偏大、显著性被高估。两种都算，报告更保守的那个。
    """
    out = []
    B = None
    if constrain_ones:
        # 全 1 的正交补的一组正交基 (nc, nc-1)
        A = np.eye(nc) - np.ones((nc, nc)) / nc
        U, s, _ = np.linalg.svd(A)
        B = U[:, : nc - 1]
    for _ in range(nsamp):
        if B is None:
            X, Y = rng.normal(size=(nc, k)), rng.normal(size=(nc, k))
        else:
            X = B @ rng.normal(size=(nc - 1, k))
            Y = B @ rng.normal(size=(nc - 1, k))
        out.append(RC.principal_angles(X, Y))
    return np.array(out)


def _perm_within_site(M, mask, rng):
    """位点内置换残基标签：摧毁 site x residue 耦合，保留两侧边缘分布。

    与 docs/12 §4 的置换零分布同构。
    """
    M2 = np.full_like(M, np.nan)
    for i in range(M.shape[0]):
        j = np.where(mask[i])[0]
        if len(j) == 0:
            continue
        M2[i, rng.permutation(j)] = M[i, j]
    return M2, np.isfinite(M2)


def step_sub(nboot=2000, nperm=500, nhaar=20000, rank=2, seed=5):
    _mkdir()
    rng = np.random.default_rng(seed)
    rows = []
    for dom in DL.DOMAINS:
        res, D = build(dom, paired=True)
        Mf, mask, cf, _ = res["folding"]
        Mb, _, cb, _ = res["binding"]
        assert (mask == res["binding"][1]).all(), "配对 mask 不一致"
        Vf, Vb = _V(Mf, mask, rank), _V(Mb, mask, rank)
        ang = RC.principal_angles(Vf, Vb)
        nc = Mf.shape[1]

        # (a) Haar 随机子空间零分布
        hz = _haar_angles(nc, rank, nhaar, rng, constrain_ones=True)
        hz_free = _haar_angles(nc, rank, nhaar, rng, constrain_ones=False)

        # (b) 位点内置换零分布：置换 binding 后重新提取载荷，与**原始**
        #     folding 载荷比主角。这保留边缘、摧毁耦合。
        pz = []
        for _ in range(nperm):
            Mp, mkp = _perm_within_site(Mb, mask, rng)
            pz.append(RC.principal_angles(Vf, _V(Mp, mkp, rank,
                                                 seed=int(rng.integers(1e6)),
                                                 restarts=3)))
        pz = np.array(pz)

        # (c) bootstrap over sites (>=2000)：两个表型用**同一组**重抽位点，
        #     这才是配对设计该有的做法。
        bz = []
        nr = Mf.shape[0]
        for _ in range(nboot):
            idx = rng.integers(0, nr, nr)
            mk2 = mask[idx]
            if mk2.sum(axis=0).min() < rank + 1:
                continue
            try:
                a = RC.principal_angles(
                    _V(Mf[idx], mk2, rank, seed=int(rng.integers(1e6)),
                       restarts=2),
                    _V(Mb[idx], mk2, rank, seed=int(rng.integers(1e6)),
                       restarts=2))
            except np.linalg.LinAlgError:
                continue
            if len(a) == rank:
                bz.append(a)
        bz = np.array(bz)

        row = {"domain": dom, "rank": rank, "n_obs": cf["n_obs"],
               "censor_frac": cf["censor_frac"],
               "angles_deg": [math.degrees(a) for a in ang],
               "boot_n": len(bz),
               "boot_ci_deg": [[math.degrees(np.quantile(bz[:, k], .025)),
                                math.degrees(np.quantile(bz[:, k], .975))]
                               for k in range(rank)] if len(bz) else None,
               "haar_ones_ci_deg": [
                   [math.degrees(np.quantile(hz[:, k], .025)),
                    math.degrees(np.quantile(hz[:, k], .975))]
                   for k in range(rank)],
               "haar_ones_median_deg": [math.degrees(np.median(hz[:, k]))
                                        for k in range(rank)],
               "haar_free_ci_deg": [
                   [math.degrees(np.quantile(hz_free[:, k], .025)),
                    math.degrees(np.quantile(hz_free[:, k], .975))]
                   for k in range(rank)],
               "haar_p": [float((hz[:, k] <= ang[k]).mean())
                          for k in range(rank)],
               "perm_ci_deg": [
                   [math.degrees(np.quantile(pz[:, k], .025)),
                    math.degrees(np.quantile(pz[:, k], .975))]
                   for k in range(rank)],
               "perm_median_deg": [math.degrees(np.median(pz[:, k]))
                                   for k in range(rank)],
               "perm_p": [float((pz[:, k] <= ang[k]).mean())
                          for k in range(rank)],
               "nboot_req": nboot, "nperm": nperm, "nhaar": nhaar}
        rows.append(row)
        print(f"  {dom}: 主角 "
              + ", ".join(f"{math.degrees(a):.1f}°" for a in ang)
              + f"  bootstrap 95%CI " + str(
                  [[round(x, 1) for x in ci] for ci in row["boot_ci_deg"]])
              + f"\n      Haar(⊥1) 95%CI "
              + str([[round(x, 1) for x in ci]
                     for ci in row["haar_ones_ci_deg"]])
              + f" p={row['haar_p']}"
              + f"\n      置换 95%CI "
              + str([[round(x, 1) for x in ci] for ci in row["perm_ci_deg"]])
              + f" p={row['perm_p']}")
    json.dump(rows, open(os.path.join(OUT, "subspace.json"), "w"), indent=1)
    return rows


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("先跑数值工具层自检（CLAUDE.md §10，不通过则中止）")
    RC.selfcheck(verbose=False)
    print("  rank_core 自检通过")
    for dom in DL.DOMAINS:
        DL.verify(dom, DL.load_domain(dom), verbose=False)
    print("  ddpca_load 解析自检通过（3 个域）\n")
    if what in ("rankboot", "all"):
        print("== step rankboot：秩间比较的 bootstrap 区间 ==")
        step_rankboot(paired=False)
    if what in ("threshold", "all"):
        print("== step threshold：可检出的最小秩2 耦合强度 ==")
        step_threshold()
    if what in ("control", "all"):
        print("== step control：真实数据正控制（必须重现 docs/12 的秩 2）==")
        step_control()
    if what in ("power", "all"):
        print("== step power（配对 mask）：真实尺寸下的检出概率 ==")
        step_power(paired=True)
    if what in ("power_unpaired", "all"):
        print("== step power（各表型自己的 mask）==")
        step_power(paired=False)
    if what in ("rank", "all"):
        print("\n== step rank（配对 mask）：每域每表型定秩 ==")
        step_rank(paired=True)
    if what in ("rank_unpaired", "all"):
        print("\n== step rank（各表型自己的 mask，功效最高）==")
        step_rank(paired=False)
    if what in ("sub", "all"):
        print("\n== step sub：残基载荷子空间主角 ==")
        step_sub()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
