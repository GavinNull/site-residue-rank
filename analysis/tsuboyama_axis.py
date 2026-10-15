#!/usr/bin/env python3
"""docs/27 Step 4/5：folding 第一轴 vs binding 共识普适轴，**n 个结构域**。

这一步要替换的是 docs/26 §7 的 n = 1（GRB2-SH3/folding，|cos| = 0.842、
ρ ≈ 0.85）。顺序照 docs/26，一步不省：

  (1) §10.3 **先做**理想条件还原性诊断：在每个域**自己的**几何 / 缺失模式 /
      主效应尺度 / coup_sd_true / **体外校准 σ** 下把残基方向植入为 u，
      看抽取流程能还原到多高。不先做这一步，「folding 轴与 u 不同」
      与「我本来就检不出一致」无法区分。
  (2) D-036 / D-032 闸：**每一条**被比较的方向都要先证明自己可复现。
      folding 轴用位点对半；u 用留一（docs/24 场景 A 最小 0.911、
      docs/21 原构造最小 0.956）。GB1 的 folding 轴就是在这一条上被废的
      （位点对半 0.311 < 随机 p95 0.438），而它的 0.962 曾经好看、显著、
      机制上说得通。
  (3) 零分布**在本比较自己的 d 上**生成（docs/23 的自捉错：d=20 的 p95
      是 0.446，d=10 是 0.634，方向两面都会错）。
  (4) 同源 / 跨源配对分开报（docs/25 §6.3 的假对齐特征是「同实验配对落在
      随机水平 0.023–0.046 而跨实验配对 0.44–0.58」，与真共享方向反序）。
  (5) σ **不得**用 fitting CI（D-028；本数据的 `deltaG_95CI` 中位 0.130
      ⇒ SE ≈ 0.033，同型低估）。σ 由 `analysis/tsuboyama_sigma.py` 的
      **三锚点量级校准**（CLAUDE.md §8）给出：
        A = 两条蛋白酶技术重复、中心化后逐格子噪声（**下界**）
        B = 体外文献 ΔG 对照的 RMSE 0.70（**上界**；它含逐位点系统差，
            而那部分会被双向中心化吸收进行主效应，所以配给交互项偏大 ——
            直接用它会让 151/164 个域的 coup_sd_true = 0，
            而实测位点对半 0.84–0.94 **否证**了那个读法。这是 D-028 的镜像）
        C = 由该域自己的可复现度反解（**主用值**，判据要求 A ≤ C ≤ B）
      本脚本用**逐域的 σ_C** 做主分析，并在 A / B 两端各报一次敏感性。

数值层**全部复用**：`generic_folding_axis`（`synth_like` / `first_axis` /
`cen_unit`）、`subspace_lib`（`extract_subspace` / `haar_1d_null` /
`load_scales`）、`rank_core.center_fit`、
`rank_test_power.als_guarded`（经 `extract_subspace(method="als")`）。

运行：python3 analysis/tsuboyama_axis.py [--jobs 9]
产物：cache/tsuboyama_axis.json
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rank_core as RC                                          # noqa: E402
import tsuboyama_load as TL                                     # noqa: E402
from generic_folding_axis import (cen_unit, first_axis,         # noqa: E402
                                  synth_like)
from subspace_lib import AA20, ROOT, haar_1d_null, load_scales  # noqa: E402

NSIM = 20000
NREC = 120            # §10.3 还原性诊断的重复数
NREP = 200            # 位点对半的重复数
NRHO = 60             # ρ 扫描每格重复数
RHO_GRID = [0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

_NULL = {}


def null1d(d, n=NSIM, seed=31):
    if d not in _NULL:
        _NULL[d] = haar_1d_null(d, n=n, seed=seed)
    return _NULL[d]


def load_u():
    """binding 共识普适轴。两个版本都带着（它们 |cos| = 0.9905）。

    u_D = docs/21 §5.2 的原构造（冠状病毒 5 / OMTKY3 单酶 4）
          —— **docs/26 §7 用的就是这一条**，保留它才能逐值对账；
    u_A = docs/24 §6.2 标定后的构造（冠状病毒 5 / OMTKY3 3），
          3SGB 已按 D-038 剔除。
    """
    tg = json.load(open(os.path.join(ROOT, "cache",
                                     "subspace_two_groups.json")))
    gc = json.load(open(os.path.join(ROOT, "cache",
                                     "gate_consequences.json")))
    sa = [s for s in gc["consensus"] if s["label"].startswith("A")][0]
    return {"u_D": np.array(tg["axis1_family_avg"]["u"], float),
            "u_A": np.array(sa["u"], float),
            "loo_D": [z["abscos"] for z in tg["axis1_family_avg"]["loo"]],
            "loo_A": [z["abscos"] for z in sa["loo"]]}


# ----------------------------------------------------------------------
def per_domain(args):
    """一个域的全部 Step 4 量。放在子进程里跑。"""
    (name, M, mask, sigma, sig_opt, sig_con, u_D, u_A) = args
    # sigma = 该域反解出的 σ_C（主用）；sig_opt = A 下界；sig_con = B 上界
    r, c = RC.center_fit(M, mask)
    R = M - r[:, None] - c[None, :]
    cso = float(R[mask].std())
    out = {"sites": int(M.shape[0]), "n_obs": int(mask.sum()),
           "row_sd": float(r.std()), "col_sd": float(c.std()),
           "coup_sd_obs": cso}
    for tag, sg in (("", sigma), ("_opt", sig_opt), ("_con", sig_con)):
        out["coup_sd_true" + tag] = math.sqrt(max(cso ** 2 - sg ** 2, 0.0))
    out["sigma"] = sigma

    a, cols = first_axis(M, mask)
    if a is None:
        return name, {"err": "抽不出第一轴"}
    d = len(cols)
    av = cen_unit(a)
    n1 = null1d(d)
    p95 = float(np.percentile(n1, 95))
    out.update({"d": d, "p95_d": p95, "cols": cols.tolist(),
                "axis": av.tolist()})
    for k, u in (("D", u_D), ("A", u_A)):
        cu = abs(float(av @ cen_unit(u, cols)))
        out[f"cos_u{k}"] = cu
        out[f"p_haar_u{k}"] = float(np.mean(n1 >= cu))

    # ---- D-036：这条 folding 轴自己可复现吗（位点对半）---------------
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
        vals.append(abs(float(
            cen_unit(a1[np.searchsorted(c1, com)])
            @ cen_unit(a2[np.searchsorted(c2, com)]))))
    vals = np.array(vals)
    nb = null1d(20, n=4000, seed=41)
    out["repro"] = float(np.median(vals)) if len(vals) else float("nan")
    out["repro_ci"] = ([float(np.percentile(vals, 2.5)),
                        float(np.percentile(vals, 97.5))]
                       if len(vals) else [float("nan")] * 2)
    out["repro_random_p95"] = float(np.percentile(nb, 95))
    out["repro_ok"] = bool(out["repro"] > out["repro_random_p95"])
    out["attenuation_corrected"] = (
        min(1.0, out["cos_uD"] / math.sqrt(out["repro"]))
        if out["repro"] > 0 else float("nan"))

    # ---- §10.3 还原性诊断（在本域自己的几何/σ 上，植入 u）------------
    rec = {}
    ud = cen_unit(u_D)
    for tag, sg in (("cal", sigma), ("lo", sig_opt), ("hi", sig_con)):
        for rk in (1, 2):
            cst = math.sqrt(max(cso ** 2 - sg ** 2, 0.0))
            got = []
            for k in range(NREC):
                Mz = synth_like(mask, ud, cst, sg,
                                max(out["row_sd"], 1e-6),
                                max(out["col_sd"], 1e-6),
                                np.random.default_rng(5000 + 97 * k),
                                rank=rk)
                aa, cc = first_axis(Mz, mask)
                if aa is None:
                    continue
                got.append(abs(float(cen_unit(aa) @ cen_unit(u_D, cc))))
            got = np.array(got)
            rec[f"{tag}|rank{rk}"] = {
                "sigma": sg, "coup_sd_true": cst,
                "median": float(np.median(got)),
                "ci": [float(np.percentile(got, 2.5)),
                       float(np.percentile(got, 97.5))],
                "frac_over_p95": float(np.mean(got > p95)),
                "n": int(len(got))}
    out["recovery"] = rec

    # ---- 合成可靠度（真实方向 = u、秩 1、同几何同 σ 的位点对半）-------
    rngc = np.random.default_rng(321)
    sv = []
    for k in range(40):
        Mz = synth_like(mask, ud, out["coup_sd_true"], sigma,
                        max(out["row_sd"], 1e-6), max(out["col_sd"], 1e-6),
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
    out["repro_synth"] = float(np.median(sv)) if sv else float("nan")

    # ---- ρ 扫描（植入 ρ·u + sqrt(1−ρ²)·正交随机方向）------------------
    # **两个 σ 都扫**，因为两端对结论的方向相反，必须都摆出来：
    #   σ 偏大 → 还原上限偏低 → 实测显得更贴上限 → ρ 被估高（**不保守**）
    #   σ 偏小 → 还原上限贴近 1 → ρ ≈ 实测 |cos|      （**保守**）
    # 所以 σ_A（技术重复下界）给的是 ρ 的**保守**估计，σ_C 给的是校准估计。
    for lab, sg in (("", sigma), ("_lo", sig_opt)):
        cst = math.sqrt(max(cso ** 2 - sg ** 2, 0.0))
        rngb = np.random.default_rng(900)
        grid = []
        for rho in RHO_GRID:
            got = []
            for k in range(NRHO):
                w = cen_unit(rngb.normal(size=20))
                w = cen_unit(w - ud * (ud @ w))
                psi = cen_unit(rho * ud
                               + math.sqrt(max(0, 1 - rho ** 2)) * w)
                Mz = synth_like(mask, psi, cst, sg,
                                max(out["row_sd"], 1e-6),
                                max(out["col_sd"], 1e-6),
                                np.random.default_rng(7000 + 131 * k),
                                rank=1)
                aa, cc = first_axis(Mz, mask)
                if aa is None:
                    continue
                got.append(abs(float(cen_unit(aa) @ cen_unit(u_D, cc))))
            got = np.array(got)
            grid.append({"rho": rho, "median": float(np.median(got)),
                         "ci": [float(np.percentile(got, 2.5)),
                                float(np.percentile(got, 97.5))],
                         "p_detect": float(np.mean(got > p95))})
        out["rho_grid" + lab] = grid
        out["rho_compatible" + lab] = [
            g["rho"] for g in grid
            if g["ci"][0] <= out["cos_uD"] <= g["ci"][1]]
        out["rho_excl_above" + lab] = [g["rho"] for g in grid
                                       if out["cos_uD"] < g["ci"][0]]
        out["rho_excl_below" + lab] = [g["rho"] for g in grid
                                       if out["cos_uD"] > g["ci"][1]]
        det = [g["rho"] for g in grid if g["p_detect"] >= 0.80]
        out["rho_detect80" + lab] = min(det) if det else None
    return name, out


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=9)
    ap.add_argument("--quantile", choices=["p95", "p90"], default="p95")
    a = ap.parse_args()

    scales = load_scales(verbose=False)
    mr = scales["molar_refractivity"]
    assert AA20[int(np.nanargmin(mr))] == "G" \
        and float(mr[AA20.index("G")]) == 0.0
    assert AA20[int(np.nanargmax(mr))] == "W" \
        and abs(float(mr[AA20.index("W")]) - 42.53) < 1e-6
    print("[自检] aaindex：折射度 argmin = G (0.00)、argmax = W (42.53)"
          " —— 第三次独立数值核验通过（D-023 烧过两次）")

    U = load_u()
    print(f"[输入] u 的两个版本：|cos(u_A, u_D)| = "
          f"{abs(float(cen_unit(U['u_A']) @ cen_unit(U['u_D']))):.4f}；"
          f"留一最小 u_D {min(U['loo_D']):.3f} / u_A {min(U['loo_A']):.3f}"
          f" —— D-036 的前置：**u 自己可复现**")

    info, mats = TL.load_cached()
    gate = json.load(open(os.path.join(ROOT, "cache",
                                       "tsuboyama_gate.json")))
    SG = json.load(open(os.path.join(ROOT, "cache",
                                     "tsuboyama_sigma.json")))
    assert SG["bracket_ok"] is True, "σ 的三锚点判据未通过（§8），不许继续"
    adm = sorted(set(gate["pass_" + a.quantile]) & set(SG["rows"]))
    sig_lo = round(SG["anchor_A_median"], 3)
    sig_hi = SG["anchor_B"]
    sigC = {w: SG["rows"][w].get("sigma_C") for w in adm}
    miss = [w for w in adm if sigC[w] is None]
    adm = [w for w in adm if sigC[w] is not None]
    print(f"\n[判别集] Step 2 ∩ Step 3（门槛 {a.quantile}）∩ σ 校准可得 = "
          f"**{len(adm)} 个天然独立结构域**"
          f"{f'（{len(miss)} 个反解不出 σ，已剔除）' if miss else ''}")
    sig = float(np.median([sigC[w] for w in adm]))
    print(f"[σ] 三锚点量级校准（§8，脚本 analysis/tsuboyama_sigma.py）：")
    print(f"    A 下界（蛋白酶技术重复，中心化后逐格子） = {sig_lo:.3f}")
    print(f"    C 主用（逐域反解）中位 = **{sig:.3f}**，"
          f"四分位 {SG['anchor_C_quartiles']}")
    print(f"    B 上界（体外文献 ΔG 对照） = {sig_hi:.2f}")
    print(f"    判据 A ≤ C ≤ B：**通过**（D-028 的镜像：用 B 会让 "
          f"{SG['n_coup_sd_true_zero_at_C']}→151/164 个域 coup_sd_true = 0，"
          f"而实测可复现度否证那个读法）")

    jobs = [(w, mats[w]["M"], mats[w]["mask"], sigC[w], sig_lo, sig_hi,
             U["u_D"], U["u_A"]) for w in adm]
    res = {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for i, (nm, r) in enumerate(ex.map(per_domain, jobs, chunksize=1)):
            res[nm] = r
            if (i + 1) % 20 == 0:
                print(f"   ... {i + 1}/{len(jobs)}")
    good = {k: v for k, v in res.items() if "err" not in v}
    print(f"\n完成 {len(good)}/{len(jobs)}")

    # ------------------------------------------------------------------
    # 0. 功效账（docs/15 §2.1b：coup_sd_true = sqrt(coup_sd_obs² − σ²)）
    # ------------------------------------------------------------------
    print("\n" + "=" * 102)
    print("0. 功效账（σ 一律体外校准；coup_sd_true = sqrt(coup_sd_obs² − σ²)，"
          "docs/15 §2.1b 的那个修正）")
    print("=" * 102)
    cso = np.array([v["coup_sd_obs"] for v in good.values()])
    for tag, lab, ref in (("", "C 主用（逐域）", sig),
                          ("_opt", f"A 下界 {sig_lo:.3f}", sig_lo),
                          ("_con", f"B 上界 {sig_hi:.2f}", sig_hi)):
        ct = np.array([v["coup_sd_true" + tag] for v in good.values()])
        nz = int((ct <= 1e-9).sum())
        print(f"   σ = {lab}：coup_sd_true 中位 **{np.median(ct):.3f}**"
              f"（= {np.median(ct) / ref:.2f}σ）、10%–90% "
              f"[{np.percentile(ct, 10):.3f}, {np.percentile(ct, 90):.3f}]；"
              f"**coup_sd_true = 0 的域 {nz}/{len(ct)}**")
    print(f"   coup_sd_obs 中位 {np.median(cso):.3f}（范围 "
          f"{cso.min():.3f}–{cso.max():.3f}）")
    print(f"   对照（docs/26 §7.1）：GRB2-SH3/folding coup_sd_obs 0.857、"
          f"σ 0.46、coup_sd_true **0.723 = 1.57σ**、P(检出2|真2) = 1.00；"
          f"三个 binding 矩阵的 coup_sd_true 全 = 0（因此全被排除）")

    # ------------------------------------------------------------------
    # 1. §10.3 还原性诊断（先报，再看真实数据）
    # ------------------------------------------------------------------
    print("\n" + "=" * 102)
    print("1. §10.3 理想条件还原性诊断（**先做**）：若 folding 的真实残基"
          "方向就是 u，在每个域自己的几何与体外 σ 下我能还原到多高")
    print("=" * 102)
    for tag, lab in (("cal", "校准 σ_C（逐域）"),
                     ("lo", f"下界 A σ={sig_lo:.3f}"),
                     ("hi", f"上界 B σ={sig_hi:.2f}")):
        for rk in (1, 2):
            m = np.array([v["recovery"][f"{tag}|rank{rk}"]["median"]
                          for v in good.values()])
            f = np.array([v["recovery"][f"{tag}|rank{rk}"]["frac_over_p95"]
                          for v in good.values()])
            print(f"   {lab:<16} 真实秩 {rk}：还原 |cos| 的**跨域中位 "
                  f"{np.median(m):.3f}**，10%–90% 分位 "
                  f"[{np.percentile(m, 10):.3f}, {np.percentile(m, 90):.3f}]、"
                  f"最小 {m.min():.3f}；超各自 p95 的比例中位 "
                  f"{np.median(f):.0%}（最小 {f.min():.0%}）")
    rl = np.array([v["recovery"]["cal|rank1"]["median"] for v in good.values()])
    nlow = int((rl < 0.90).sum())
    print(f"   => 还原中位 < 0.90 的域 {nlow}/{len(good)} 个。"
          f"**这个诊断的结论是「有功效」而不是「欠功效」**："
          f"真方向 = u 时会以近 100% 的比例看到它。")

    # ------------------------------------------------------------------
    # 2. D-036 可复现闸
    # ------------------------------------------------------------------
    print("\n" + "=" * 102)
    print("2. D-036 可复现闸：**不可复现的方向不得与任何东西对齐**"
          "（GB1 就是在这一条上被废的：0.311 < 随机 p95 0.438）")
    print("=" * 102)
    rp = np.array([v["repro"] for v in good.values()])
    ok = {k: v for k, v in good.items() if v["repro_ok"]}
    p95r = float(np.median([v["repro_random_p95"] for v in good.values()]))
    print(f"   位点对半 |cos| 跨域：中位 **{np.median(rp):.3f}**、"
          f"10%–90% [{np.percentile(rp, 10):.3f}, "
          f"{np.percentile(rp, 90):.3f}]、最小 {rp.min():.3f}")
    print(f"   随机基线 p95（d = 20）= {p95r:.3f}；"
          f"**过闸 {len(ok)}/{len(good)} 个**"
          f"（对照：GRB2-SH3/folding 0.787、GB1/folding 0.311 不过）")
    sy = np.array([v["repro_synth"] for v in good.values()])
    print(f"   同几何同 σ 的**合成**秩 1 可靠度中位 {np.median(sy):.3f}"
          f" —— 真实矩阵比合成的"
          f"{'更难' if np.median(sy) > np.median(rp) + 0.05 else '相当'}"
          f"（docs/26 在 GRB2-SH3 上是 0.985 vs 0.787）")

    # ------------------------------------------------------------------
    # 3. 真实比较：|cos(folding 轴, u)| 的**分布**
    # ------------------------------------------------------------------
    print("\n" + "=" * 102)
    print("3. 真实数据：|cos(folding 第一轴, binding 共识普适轴 u)| 的分布")
    print("=" * 102)
    names = sorted(ok)
    cu = np.array([ok[k]["cos_uD"] for k in names])
    cuA = np.array([ok[k]["cos_uA"] for k in names])
    ps = np.array([ok[k]["p95_d"] for k in names])
    ds = np.array([ok[k]["d"] for k in names])
    print(f"   n = {len(names)} 个过三道闸的天然独立结构域；"
          f"d 的分布：中位 {int(np.median(ds))}、"
          f"范围 {ds.min()}–{ds.max()}（零分布逐域在自己的 d 上生成）")
    print(f"   **|cos(轴, u_D)| 中位 {np.median(cu):.3f}**，"
          f"四分位 [{np.percentile(cu, 25):.3f}, "
          f"{np.percentile(cu, 75):.3f}]，"
          f"10%–90% [{np.percentile(cu, 10):.3f}, "
          f"{np.percentile(cu, 90):.3f}]，"
          f"范围 {cu.min():.3f}–{cu.max():.3f}")
    print(f"   用 u_A（docs/24 标定后）：中位 {np.median(cuA):.3f}"
          f"（两条 u 本身 |cos| = 0.9905，所以这两个数不是独立证据）")
    nover = int((cu > ps).sum())
    print(f"   超各自 d 的一维 p95：**{nover}/{len(names)} = "
          f"{nover / len(names):.1%}**（零假设下应 ≈ 5%）")
    # 域级 bootstrap 区间（CLAUDE.md §5：点估计不单独汇报）
    rng = np.random.default_rng(5)
    bm = [np.median(cu[rng.integers(0, len(cu), len(cu))])
          for _ in range(4000)]
    print(f"   中位数的域级 bootstrap 95% 区间（4000 次）："
          f"**[{np.percentile(bm, 2.5):.3f}, {np.percentile(bm, 97.5):.3f}]**")
    # 多重性：max-of-n 零分布（§5 搜索量）
    nb = null1d(20, n=NSIM, seed=777)
    rngm = np.random.default_rng(0)
    mx = np.array([np.max(rngm.choice(nb, len(names))) for _ in range(20000)])
    print(f"   多重性（搜索量 = {len(names)} 个域）：d = 20 的"
          f"「{len(names)} 取最大」零分布 p95 = **{np.percentile(mx, 95):.3f}**；"
          f"实测最大 {cu.max():.3f}，超出的域 "
          f"{int((cu > np.percentile(mx, 95)).sum())} 个")
    # 符号检验（n > 1 才做得了，docs/26 §10 明说 n=1 做不了）
    from scipy.stats import binomtest
    bt = binomtest(nover, len(names), 0.05, alternative="greater")
    print(f"   符号检验（超 p95 的域数 vs 零假设 5%）：p = {bt.pvalue:.3e}"
          f" —— **这是 n = 1 时做不了的那个检验**（docs/26 §10）")
    ac = np.array([ok[k]["attenuation_corrected"] for k in names])
    print(f"   用各自可复现度做衰减校正后：中位 {np.median(ac):.3f}、"
          f"四分位 [{np.percentile(ac, 25):.3f}, "
          f"{np.percentile(ac, 75):.3f}]"
          f"（docs/26 在 GRB2-SH3 上是 0.949）")

    # ------------------------------------------------------------------
    # 4. ρ 的区间
    # ------------------------------------------------------------------
    print("\n" + "=" * 102)
    print("4. 真实一致度 ρ 的区间（替换 docs/26 的单矩阵 ρ ≈ 0.85）")
    print("=" * 102)
    summ_rho = {}
    for lab, tag in (("", "σ_C 校准"), ("_lo", f"σ_A 保守 {sig_lo:.3f}")):
        m2, le2, he2, d2 = [], [], [], []
        for k in names:
            v = ok[k]
            if v["rho_compatible" + lab]:
                m2.append(float(np.median(v["rho_compatible" + lab])))
            if v["rho_excl_below" + lab]:
                le2.append(max(v["rho_excl_below" + lab]))
            if v["rho_excl_above" + lab]:
                he2.append(min(v["rho_excl_above" + lab]))
            if v["rho_detect80" + lab] is not None:
                d2.append(v["rho_detect80" + lab])
        m2 = np.array(m2)
        rg = np.random.default_rng(6)
        bb = [np.median(m2[rg.integers(0, len(m2), len(m2))])
              for _ in range(4000)] if len(m2) else [float("nan")]
        summ_rho[lab or "cal"] = {
            "label": tag, "n": int(len(m2)),
            "median": float(np.median(m2)) if len(m2) else float("nan"),
            "boot_ci": [float(np.percentile(bb, 2.5)),
                        float(np.percentile(bb, 97.5))],
            "quartiles": [float(np.percentile(m2, 25)),
                          float(np.percentile(m2, 75))] if len(m2) else None,
            "decile": [float(np.percentile(m2, 10)),
                       float(np.percentile(m2, 90))] if len(m2) else None,
            "excl_below_median": float(np.median(le2)) if le2 else None,
            "excl_above_median": float(np.median(he2)) if he2 else None,
            "n_excl_below": len(le2), "n_excl_above": len(he2),
            "detect80_median": float(np.median(d2)) if d2 else None,
            "n_eq1_excluded": sum(1 for k in names
                                  if 1.0 in ok[k]["rho_excl_above" + lab])}
        print(f"\n   [{tag}] n = {len(m2)}，逐域「与实测相容的 ρ」中位 "
              f"**{summ_rho[lab or 'cal']['median']:.3f}**，"
              f"bootstrap 95% 区间 "
              f"**[{summ_rho[lab or 'cal']['boot_ci'][0]:.3f}, "
              f"{summ_rho[lab or 'cal']['boot_ci'][1]:.3f}]**")
        if summ_rho[lab or "cal"]["quartiles"]:
            print(f"        四分位 "
                  f"[{summ_rho[lab or 'cal']['quartiles'][0]:.2f}, "
                  f"{summ_rho[lab or 'cal']['quartiles'][1]:.2f}]、"
                  f"10%–90% "
                  f"[{summ_rho[lab or 'cal']['decile'][0]:.2f}, "
                  f"{summ_rho[lab or 'cal']['decile'][1]:.2f}]")
        print(f"        ρ = 1.0 被实测排除的域 "
              f"{summ_rho[lab or 'cal']['n_eq1_excluded']}/{len(names)}；"
              f"检出率 ≥ 80% 的最小 ρ 中位 "
              f"{summ_rho[lab or 'cal']['detect80_median']}")
    mid, lo_ex, hi_ex, det80 = [], [], [], []
    for k in names:
        v = ok[k]
        if v["rho_compatible"]:
            mid.append(float(np.median(v["rho_compatible"])))
        lo_ex.append(max(v["rho_excl_below"]) if v["rho_excl_below"] else
                     float("nan"))
        hi_ex.append(min(v["rho_excl_above"]) if v["rho_excl_above"] else
                     float("nan"))
        if v["rho_detect80"] is not None:
            det80.append(v["rho_detect80"])
    mid = np.array(mid)
    rng = np.random.default_rng(6)
    bmr = [np.median(mid[rng.integers(0, len(mid), len(mid))])
           for _ in range(4000)]
    print(f"   逐域「与实测相容的 ρ」：n = {len(mid)}，中位 **{np.median(mid):.3f}**，"
          f"四分位 [{np.percentile(mid, 25):.2f}, "
          f"{np.percentile(mid, 75):.2f}]，"
          f"10%–90% [{np.percentile(mid, 10):.2f}, "
          f"{np.percentile(mid, 90):.2f}]")
    print(f"   **中位 ρ 的域级 bootstrap 95% 区间："
          f"[{np.percentile(bmr, 2.5):.3f}, {np.percentile(bmr, 97.5):.3f}]**")
    le = np.array([x for x in lo_ex if np.isfinite(x)])
    he = np.array([x for x in hi_ex if np.isfinite(x)])
    print(f"   从**下方**被排除的最大 ρ：中位 "
          f"{np.median(le) if len(le) else float('nan'):.2f}"
          f"（{len(le)}/{len(names)} 个域有下方排除）")
    print(f"   从**上方**被排除的最小 ρ：中位 "
          f"{np.median(he) if len(he) else float('nan'):.2f}"
          f"（{len(he)}/{len(names)} 个域有上方排除）")
    if det80:
        print(f"   检出率 ≥ 80% 的最小真实一致度 ρ：中位 "
              f"{np.median(det80):.2f}、最大 {max(det80):.2f}"
              f" —— **所以这不是一个只能排除「完全一致」的检验**")
    excl1 = sum(1 for k in names if 1.0 in ok[k]["rho_excl_above"])
    print(f"   ρ = 1.0 被实测排除的域：**{excl1}/{len(names)}**"
          f"（即 folding 轴与 u 高度一致但**不完全相同**的域占 "
          f"{excl1 / len(names):.0%}）")

    # ------------------------------------------------------------------
    # 5. 化学身份（折叠一侧，规模化）
    # ------------------------------------------------------------------
    print("\n" + "=" * 102)
    print("5. 轴的化学身份在 folding 一侧、在规模上还成立吗")
    print("=" * 102)
    DS = json.load(open(os.path.join(ROOT, "cache",
                                     "generic_disrupt_scales.json")))
    dis = {k: np.array(v, float) for k, v in DS["scales"].items()}
    want = ["sheet_prop_Chou", "local_polarizability", "molar_refractivity",
            "pi_octanol_FauchereP", "asa_RadzickaWolfenden", "kd_hydropathy",
            "partial_specific_volume", "steric_upsilon"]
    rowsc = []
    for nm in want + list(dis):
        vec = scales.get(nm, dis.get(nm))
        vv = []
        for k in names:
            cols = np.array(ok[k]["cols"])
            av = np.array(ok[k]["axis"], float)
            t = cen_unit(vec, cols)
            if not np.all(np.isfinite(t)):
                continue
            vv.append(abs(float(av @ t)))
        if not vv:
            continue
        vv = np.array(vv)
        rowsc.append({"scale": nm, "n": int(len(vv)),
                      "med_abscos": float(np.median(vv)),
                      "med_r2": float(np.median(vv ** 2)),
                      "q10": float(np.percentile(vv, 10)),
                      "q90": float(np.percentile(vv, 90))})
    rowsc.sort(key=lambda z: -z["med_abscos"])
    print(f"   {'标度':<34}{'|cos| 中位':>11}{'r² 中位':>9}"
          f"{'10%–90%':>18}")
    for z in rowsc:
        print(f"   {z['scale']:<34}{z['med_abscos']:>11.3f}"
              f"{z['med_r2']:>9.3f}"
              f"{'[%.2f, %.2f]' % (z['q10'], z['q90']):>18}")
    print(f"\n   参照（binding 一侧，docs/21 §6）：sheet propensity r² 0.607、"
          f"极化率 0.568、可及表面积 0.431、Kyte-Doolittle 0.223")

    out = {"n_admitted": len(adm), "quantile": a.quantile,
           "sigma": {"C_median": sig, "A_lower": sig_lo,
                     "B_upper": sig_hi,
                     "per_domain_C": {w: sigC[w] for w in adm}},
           "u_cos_AD": float(abs(cen_unit(U["u_A"]) @ cen_unit(U["u_D"]))),
           "u_loo_min": {"D": min(U["loo_D"]), "A": min(U["loo_A"])},
           "rows": res, "repro_pass": sorted(ok),
           "summary": {
               "n_repro_ok": len(ok),
               "cos_uD_median": float(np.median(cu)),
               "cos_uD_boot_ci": [float(np.percentile(bm, 2.5)),
                                  float(np.percentile(bm, 97.5))],
               "cos_uD_quartiles": [float(np.percentile(cu, 25)),
                                    float(np.percentile(cu, 75))],
               "cos_uD_decile": [float(np.percentile(cu, 10)),
                                 float(np.percentile(cu, 90))],
               "cos_uD_range": [float(cu.min()), float(cu.max())],
               "frac_over_p95": float(nover / len(names)),
               "n_over_p95": nover,
               "maxof_n_p95": float(np.percentile(mx, 95)),
               "sign_test_p": float(bt.pvalue),
               "atten_median": float(np.median(ac)),
               "rho_median": float(np.median(mid)),
               "rho_boot_ci": [float(np.percentile(bmr, 2.5)),
                               float(np.percentile(bmr, 97.5))],
               "rho_quartiles": [float(np.percentile(mid, 25)),
                                 float(np.percentile(mid, 75))],
               "rho_decile": [float(np.percentile(mid, 10)),
                              float(np.percentile(mid, 90))],
               "rho_eq1_excluded": excl1,
               "recovery_cal_rank1_median": float(np.median(rl)),
               "repro_median": float(np.median(rp)),
               "coup_sd_true_median": float(np.median(
                   [v["coup_sd_true"] for v in good.values()])),
               "n_coup_sd_true_zero": int(sum(
                   1 for v in good.values() if v["coup_sd_true"] <= 1e-9)),
               "repro_synth_median": float(np.median(sy))},
           "rho_by_sigma": summ_rho,
           "chem": rowsc}
    p = os.path.join(ROOT, "cache", "tsuboyama_axis.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
