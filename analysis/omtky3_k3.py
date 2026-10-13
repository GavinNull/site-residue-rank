#!/usr/bin/env python3
"""OMTKY3 第二轴对照在 k = 3 上重算（剔除 3SGB），零分布全部在 k = 3 上重生成。

为什么要重算（不是改字）：
  docs/24 §7.1 第 2 条把 `OMTKY3/SGPB/3SGB` 在秩 2 上判为**不可读**
  （stab cos2 = 0.657/0.659 对它自己 10 位点几何的秩 2 纯噪声 p95 = 0.689，
  边距 −0.029；上限余量 0.070，四者里唯一 < 0.10），人类裁定 D-046 据此
  把它剔出证据集。但 docs/13 §6.3 的「四个矩阵各自第二维可分辨」仍在原处，
  docs/30 §4 继承了它 —— 于是 docs/30 §4 说「四个」、§7(a) 说「存活的三个」。
  docs/31 §4.2 抓到这一处，并明确指出：PR = 2.71 / p = 0.0045 是 **k = 4**
  的量，改成「三个」就必须连零分布一起在 k = 3 上重算。

纪律（CLAUDE.md）：
  §10  工具先自验：本脚本硬性要求 cache/gate_harness_check.json 的 ok = True，
       并在算任何 k = 3 的数之前先逐位复现 docs/13 §6.3 的 k = 4 表
       （已知事实复现，不过就中止）。
  §10.3 任何「测不出结构」的结论之前，必须过理想条件下的还原性诊断 ——
       第 5 节在 k = 3 上做它（合成「共享第二轴」与「独立第二轴」各一组，
       既测能命中也测不误伤）。
  §5   每个零分布在**本次比较自己的** k 与环境维数 d 上生成；记录搜索量。
  §8   量级校准：各向同性基线不是 0，必须报出来并与观测对照。

数值层全部沿用，不重写：
  subspace_lib          extract_subspace / align / align_deflated / haar_*_null
                        / load_skempi / load_independent / load_all_dms
                        / dedup_by_content / load_scales
  subspace_resid_dim    centered_orth / spectrum（PR 与残差谱，原样 import）
  subspace_cond_null    eigs / cond_null / uncond_null
  omtky3_pr_power       rand_perp / draw_dirs / null_pr（功效诊断的合成器）
  rank_test_power       als_guarded（交叉核验用；**绝不**用 als_complete）
  cache/gate_rank2_calibration.json  逐几何的秩 2 纯噪声 floor p95（D-046 门槛）

产物：cache/omtky3_k3.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import (ROOT, align, align_deflated,  # noqa: E402
                          dedup_by_content, extract_subspace, haar_1d_null,
                          haar_pair_null, load_all_dms, load_independent,
                          load_scales, load_skempi)
from subspace_resid_dim import centered_orth, spectrum          # noqa: E402
from subspace_cond_null import cond_null, eigs, uncond_null     # noqa: E402
from omtky3_pr_power import draw_dirs, null_pr, rand_perp       # noqa: E402

ONE = np.ones(20) / np.sqrt(20)
NNULL = 4000          # 零分布次数（PR、λ、成对）
NBOOT_SYN = 400       # 合成矩阵级诊断的重复数
REF = "starr2020/WH1/huACE2"          # docs/13 §6.3 的共同参照
OMT4 = ["OMTKY3/chymotrypsin/1CHO", "OMTKY3/HLE-elastase/1PPF",
        "OMTKY3/subtilisin-Carlsberg/1R0R", "OMTKY3/SGPB/3SGB"]
DROP = "OMTKY3/SGPB/3SGB"             # D-046：秩 2 不可读
POOLED = "SKEMPI/OMTKY3/pooled-4proteases"
# docs/13 §6.3 的 k = 4 表（复现靶标；只用来核对工具，不参与任何判决）
DOCS13_K4 = {"OMTKY3/HLE-elastase/1PPF": (0.905, 0.761),
             "OMTKY3/SGPB/3SGB": (0.770, 0.290),
             "OMTKY3/chymotrypsin/1CHO": (0.921, 0.055),
             "OMTKY3/subtilisin-Carlsberg/1R0R": (0.870, 0.026)}

SEARCH_LOG = []        # §5：记录一共做了多少次比较 / 多少个零分布格


def log(kind, n, what):
    SEARCH_LOG.append({"kind": kind, "n": n, "what": what})


# ----------------------------------------------------------------------
# 0. 硬前置：工具自检必须已通过（CLAUDE.md §10）
# ----------------------------------------------------------------------
def require_harness():
    p = os.path.join(ROOT, "cache", "gate_harness_check.json")
    if not os.path.exists(p):
        raise RuntimeError("cache/gate_harness_check.json 不存在 —— "
                           "先跑 analysis/gate_harness_check.py")
    h = json.load(open(p))
    if not h.get("ok"):
        raise RuntimeError("gate_harness_check 未通过（ok != True）—— 中止")
    print("[前置] cache/gate_harness_check.json  ok = True")
    return h


# ----------------------------------------------------------------------
# 1. 载入 + 逐几何的可读性门（D-046：秩 2 纯噪声 floor p95）
# ----------------------------------------------------------------------
def load_pool():
    mats, _ = dedup_by_content(load_independent() + load_skempi()
                               + load_all_dms())
    by = {}
    for d in mats:
        B, c, _ = extract_subspace(d["M"], d["mask"])
        if B is not None and len(c) == 20:
            by[d["name"]] = {"B": B, "cols": np.asarray(c),
                             "censor": d.get("censor_frac"),
                             "shape": tuple(np.shape(d["M"]))}
    return by


def gate_table():
    """逐几何、逐秩的可读性门：stab cos2 > 本矩阵几何的秩 2 纯噪声 p95。

    数字**不在本脚本重算**，直接取 docs/24 落盘的标定结果
    （cache/gate_rank2_calibration.json，n_real = 120 / nboot = 200）。
    这是人类裁定 D-046 用的同一张表、同一个分位（p95）。
    """
    p = os.path.join(ROOT, "cache", "gate_rank2_calibration.json")
    g = json.load(open(p))
    rows = {r["name"]: r for r in g["rows"]}
    print("\n" + "=" * 86)
    print("[门] 逐几何秩 2 可读性（D-046 口径：stab cos2 > 本几何纯噪声 floor p95）")
    print(f"{'矩阵':<40}{'stab cos2':>11}{'floor p95':>11}{'边距':>9}{'判定':>10}")
    print("-" * 86)
    keep = []
    for n in OMT4 + [REF]:
        r = rows[n]
        m = r["stab_obs"] - r["floor_p95"]
        ok = m > 0
        print(f"{n:<40}{r['stab_obs']:>11.3f}{r['floor_p95']:>11.3f}"
              f"{m:>+9.3f}{('可读' if ok else '**不可读**'):>10}")
        if ok and n != REF:
            keep.append(n)
    print("-" * 86)
    print(f"   OMTKY3 面板里可读的 = {len(keep)} / 4  ->  k = {len(keep)}")
    print(f"   被剔除：{DROP}（与 D-046 一致）")
    assert keep == [n for n in OMT4 if n != DROP], "门的结果与 D-046 不一致"
    return keep, rows


# ----------------------------------------------------------------------
# 2. 已知事实复现：docs/13 §6.3 的 k = 4 表必须逐位回来（不过就中止）
# ----------------------------------------------------------------------
def reproduce_docs13(by):
    print("\n" + "=" * 86)
    print("[自检 R] 先复现 docs/13 §6.3 的 k = 4 表（已知事实；不过就中止）")
    print(f"   共同参照 = {REF}")
    ref = by[REF]
    worst = 0.0
    for n in OMT4:
        r = align(by[n]["B"], by[n]["cols"], ref["B"], ref["cols"])
        t1, t2 = DOCS13_K4[n]
        d1, d2 = abs(r["cos"][0] - t1), abs(r["cos"][1] - t2)
        worst = max(worst, d1, d2)
        print(f"   {n:<40} cosθ1 {r['cos'][0]:.3f} (靶 {t1:.3f}, Δ{d1:.3f})"
              f"   cosθ2 {r['cos'][1]:.3f} (靶 {t2:.3f}, Δ{d2:.3f})")
    log("comparison", len(OMT4), "docs/13 §6.3 复现（4 个矩阵 vs 共同参照）")
    print(f"   最大偏差 {worst:.4f}  {'OK' if worst < 0.01 else 'FAIL'}")
    if worst >= 0.01:
        raise RuntimeError("未复现 docs/13 §6.3 —— 后面的数不可用")
    return worst


# ----------------------------------------------------------------------
# 3. k = 3 的成对主角度
# ----------------------------------------------------------------------
def pairwise(by, keep, kd):
    n1 = haar_pair_null(20, n=NNULL, seed=7)
    n1d = haar_1d_null(20, n=NNULL, seed=7)
    p95 = (float(np.percentile(n1["cos1"], 95)),
           float(np.percentile(n1["cos2"], 95)))
    p95d = float(np.percentile(n1d, 95))
    log("null", 2 * NNULL, "成对 Haar 零分布（d = 20，2 维面 + 1 维）")
    print("\n" + "=" * 86)
    print("[成对 A] 存活的 3 个 OMTKY3 矩阵 vs 共同参照 " + REF)
    print(f"   N1 随机零分布 p95（d = 20）: cosθ1 {p95[0]:.3f}  "
          f"cosθ2 {p95[1]:.3f}  一维残差 {p95d:.3f}")
    print(f"{'矩阵':<40}{'cosθ1':>8}{'cosθ2':>8}{'投掉疏水后':>12}")
    print("-" * 86)
    ref = by[REF]
    vs_ref = []
    for n in keep:
        r = align(by[n]["B"], by[n]["cols"], ref["B"], ref["cols"])
        dfl = align_deflated(by[n]["B"], by[n]["cols"], ref["B"], ref["cols"],
                             kd, np.arange(20))
        vs_ref.append({"name": n, "cos1": r["cos"][0], "cos2": r["cos"][1],
                       "defl": dfl["cos_resid"] if dfl else None})
        print(f"{n:<40}{r['cos'][0]:>8.3f}{r['cos'][1]:>8.3f}"
              f"{(dfl['cos_resid'] if dfl else float('nan')):>12.3f}")
    c2 = np.array([x["cos2"] for x in vs_ref])
    c1 = np.array([x["cos1"] for x in vs_ref])
    print("-" * 86)
    print(f"   cosθ2 = " + " / ".join(f"{x:.3f}" for x in c2)
          + f"   中位 {np.median(c2):.3f}  全距 {c2.max()-c2.min():.3f}"
          f"  超 p95 {int((c2 > p95[1]).sum())}/{len(c2)}")
    print(f"   cosθ1 = " + " / ".join(f"{x:.3f}" for x in c1)
          + f"   中位 {np.median(c1):.3f}"
          f"  超 p95 {int((c1 > p95[0]).sum())}/{len(c1)}")
    log("comparison", len(keep), "3 个存活矩阵 vs 共同参照")

    print("\n[成对 B] 存活 3 个矩阵彼此（面板内 3 个配对；k = 4 时是 6 个）")
    print(f"{'配对':<44}{'cosθ1':>8}{'cosθ2':>8}{'投掉疏水后':>12}")
    print("-" * 86)
    within = []
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            a, b = by[keep[i]], by[keep[j]]
            r = align(a["B"], a["cols"], b["B"], b["cols"])
            dfl = align_deflated(a["B"], a["cols"], b["B"], b["cols"],
                                 kd, np.arange(20))
            lab = (keep[i].split("/")[1] + " | " + keep[j].split("/")[1])
            within.append({"a": keep[i], "b": keep[j], "cos1": r["cos"][0],
                           "cos2": r["cos"][1],
                           "defl": dfl["cos_resid"] if dfl else None})
            print(f"{lab:<44}{r['cos'][0]:>8.3f}{r['cos'][1]:>8.3f}"
                  f"{(dfl['cos_resid'] if dfl else float('nan')):>12.3f}")
    w1 = np.array([x["cos1"] for x in within])
    w2 = np.array([x["cos2"] for x in within])
    print("-" * 86)
    print(f"   cosθ1 中位 {np.median(w1):.3f} 范围 {w1.min():.3f}–{w1.max():.3f}"
          f"  超 p95 {int((w1 > p95[0]).sum())}/{len(w1)}"
          f"   （k = 4 时是 6/6，0.745–0.919）")
    print(f"   cosθ2 中位 {np.median(w2):.3f} 范围 {w2.min():.3f}–{w2.max():.3f}"
          f"  超 p95 {int((w2 > p95[1]).sum())}/{len(w2)}"
          f"   （k = 4 时是 3/6，0.079–0.586，中位 0.291）")
    log("comparison", len(within), "面板内两两（k = 3）")
    return {"vs_ref": vs_ref, "within": within,
            "null_p95": {"cos1": p95[0], "cos2": p95[1], "d1": p95d}}


# ----------------------------------------------------------------------
# 4. k = 3 的 PR（零分布在 k = 3 上重生成）+ 共识 λ
# ----------------------------------------------------------------------
def pr_at_k(by, keep, rng, label):
    k = len(keep)
    Qs = [centered_orth(by[n]["B"]) for n in keep]
    P = np.mean([Q @ Q.T for Q in Qs], axis=0)
    uu = np.linalg.eigh(P)[1][:, ::-1][:, 0]
    dirs = []
    for Q in Qs:
        c = Q.T @ uu
        c = c / np.linalg.norm(c)
        dirs.append(Q @ np.array([-c[1], c[0]]))
    ws, pr, n90 = spectrum(dirs)
    prn, n90n = [], []
    rng2 = np.random.default_rng(31337 + k)      # 零分布独立种子
    for _ in range(NNULL):
        ds = [rand_perp(uu, rng2) for _ in range(k)]
        _, p_, n_ = spectrum(ds)
        prn.append(p_); n90n.append(n_)
    prn = np.array(prn)
    p_one = float(np.mean(prn <= pr))
    log("null", NNULL, f"PR 零分布（k = {k}，d = 18 = u 的正交补）")
    print(f"\n[PR {label}]  k = {k}（零分布在 k = {k} 上重新生成，{NNULL} 次）")
    print(f"   残差谱（前 {min(k,4)}）= " + " ".join(f"{x:.3f}"
                                                   for x in ws[:min(k, 4)]))
    print(f"   PR = {pr:.2f}   （完全相同 = 1；完全各向同性 ≈ k = {k}）")
    print(f"   PR 零分布中位 {np.median(prn):.2f} "
          f"[{np.percentile(prn, 2.5):.2f}, {np.percentile(prn, 97.5):.2f}]"
          f"；单尾 5% 临界 PR < {np.percentile(prn, 5):.2f}")
    print(f"   **单尾 p（PR ≤ 观测）= {p_one:.4f}**")
    print(f"   覆盖 90% 需要 {n90} 维；零分布中位 {int(np.median(n90n))} 维")
    # §8 量级校准：各向同性基线不是 0
    pw = [abs(float(dirs[i] @ dirs[j])) for i in range(k)
          for j in range(i + 1, k)]
    share = float(np.mean([abs(d @ np.linalg.eigh(
        np.mean([np.outer(x, x) for x in dirs], axis=0))[1][:, -1])
        for d in dirs]))
    print(f"   量级校准：残差方向两两 |cos| = "
          + " ".join(f"{x:.2f}" for x in pw)
          + f"（中位 {np.median(pw):.2f}）；与主轴平均 |cos| = {share:.2f}")
    return {"k": k, "resid_eigs": ws[:min(k, 4)].tolist(), "pr": float(pr),
            "null_med": float(np.median(prn)),
            "null_ci": [float(np.percentile(prn, 2.5)),
                        float(np.percentile(prn, 97.5))],
            "null_crit05": float(np.percentile(prn, 5)),
            "p_one_sided": p_one, "n90": n90,
            "n90_null_med": float(np.median(n90n)),
            "pairwise_abscos": pw, "share_cos": share,
            "u_resid": uu.tolist()}


def consensus(by, keep, rng, label):
    k = len(keep)
    Bs = [by[n]["B"] for n in keep]
    w, V = eigs(Bs)
    cn, alpha = cond_null(k, V[:, 0], rng, nsim=NNULL, target_l1=float(w[0]))
    un = uncond_null(k, rng, nsim=NNULL)
    log("null", 2 * NNULL, f"λ 条件/无条件零分布（k = {k}）")
    p_l2 = float(np.mean(cn[:, 1] >= w[1]))
    print(f"\n[共识 {label}]  k = {k}")
    print(f"   观测 λ = {np.round(w[:3], 3).tolist()}")
    print(f"   N1 无条件零分布 p95 = "
          f"{np.round(np.percentile(un, 95, axis=0), 3).tolist()}")
    print(f"   条件零分布抖动角 α = {np.degrees(alpha):.1f}°；标定残差 "
          f"|模拟 λ1 中位 − 观测 λ1| = "
          f"{abs(float(np.median(cn[:, 0])) - float(w[0])):.4f}")
    print(f"   条件零分布中位 = {np.round(np.median(cn, axis=0), 3).tolist()}"
          f"；p95 = {np.round(np.percentile(cn, 95, axis=0), 3).tolist()}")
    print(f"   λ2 的条件 p = {p_l2:.4f}")
    return {"k": k, "lambda": w[:3].tolist(),
            "uncond_p95": np.percentile(un, 95, axis=0).tolist(),
            "cond_med": np.median(cn, axis=0).tolist(),
            "cond_p95": np.percentile(cn, 95, axis=0).tolist(),
            "alpha_deg": float(np.degrees(alpha)),
            "calib_resid": abs(float(np.median(cn[:, 0])) - float(w[0])),
            "p_lambda2": p_l2}


# ----------------------------------------------------------------------
# 5. k = 3 的还原性诊断（CLAUDE.md §10.3）—— 在解读 k = 3 的 PR 之前
# ----------------------------------------------------------------------
def power_k3(rng, ks=(3, 4)):
    """两层诊断，都在**各自的 k** 上生成零分布：

    (a) 方向级：直接合成 k 条残差方向（投影正态，kappa = 0 严格各向同性，
        kappa = None 完全共享）。既测命中（kappa 大必须显著）也测不误伤
        （kappa = 0 的拒绝率必须 ≈ 0.05）。沿用 omtky3_pr_power.draw_dirs，
        那个函数的第一版参数化是错的、被阴性对照抓到过（见其 docstring）。
    (b) 矩阵级：合成 k 个 10 位点 × 20 残基、σ 匹配真实水平的秩 2 矩阵，
        走**同一条** extract_subspace -> centered_orth -> spectrum 管线。
        这一层包含了抽取噪声，是真正对应「三个真实矩阵」的那个问题。
    """
    print("\n" + "=" * 86)
    print("[功效 §10.3] PR 统计量在 k = 3 上还能不能分开「共享」与「独立」")
    KAPPAS = [0.0, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0, None]
    out = {}
    u = rng.normal(size=20); u -= ONE * (ONE @ u); u /= np.linalg.norm(u)
    for k in ks:
        nl = null_pr(k, u, rng, n=NNULL)
        crit = float(np.percentile(nl, 5))
        log("null", NNULL, f"功效诊断 PR 零分布（方向级，k = {k}）")
        rows = []
        for kp in KAPPAS:
            prs, cosb = [], []
            for _ in range(NBOOT_SYN):
                v0 = rand_perp(u, rng)
                ds = draw_dirs(k, u, v0, kp, rng)
                cosb += [abs(float(d @ v0)) for d in ds]
                prs.append(spectrum(ds)[1])
            prs = np.array(prs)
            rows.append({"kappa": kp, "share_cos": float(np.mean(cosb)),
                         "pr_med": float(np.median(prs)),
                         "power": float((prs < crit).mean())})
        log("sim", len(KAPPAS) * NBOOT_SYN, f"功效诊断格（方向级，k = {k}）")
        xs = [r["share_cos"] for r in rows]; ps = [r["power"] for r in rows]
        s80 = None
        for i in range(len(xs) - 1):
            if ps[i] < 0.8 <= ps[i + 1]:
                t = (0.8 - ps[i]) / (ps[i + 1] - ps[i])
                s80 = xs[i] + t * (xs[i + 1] - xs[i]); break
        print(f"\n  [方向级 k = {k}] 零分布 PR 中位 {np.median(nl):.2f} "
              f"[{np.percentile(nl, 2.5):.2f}, {np.percentile(nl, 97.5):.2f}]"
              f"；单尾 5% 临界 PR < {crit:.2f}")
        print("     共享度 mean|cos| " + "".join(f"{r['share_cos']:>7.2f}"
                                                 for r in rows))
        print("     PR 中位          " + "".join(f"{r['pr_med']:>7.2f}"
                                                 for r in rows))
        print("     功效             " + "".join(f"{r['power']:>7.2f}"
                                                 for r in rows))
        print(f"     阴性对照（kappa = 0，构造上严格各向同性）拒绝率 "
              f"{rows[0]['power']:.3f}（应 ≈ 0.05）"
              f"  {'OK' if rows[0]['power'] < 0.09 else 'FAIL'}")
        print(f"     完全共享（kappa = None）的功效 = {rows[-1]['power']:.3f}"
              f"  {'能命中' if rows[-1]['power'] > 0.8 else '**连完全共享都测不出**'}")
        print(f"     80% 功效所需共享度 ≈ "
              + (f"{s80:.2f}" if s80 is not None else "（全区间内达不到）"))
        if rows[0]["power"] >= 0.09:
            raise RuntimeError(f"k = {k} 的阴性对照未通过 —— 数字不可用")
        out[k] = {"null_med": float(np.median(nl)),
                  "null_ci": [float(np.percentile(nl, 2.5)),
                              float(np.percentile(nl, 97.5))],
                  "crit05": crit, "rows": rows, "share80": s80,
                  "fp_rate": rows[0]["power"],
                  "power_full_share": rows[-1]["power"]}
    return out


def power_k3_matrix(rng, k=3, sigma=0.30, nrep=NBOOT_SYN // 2):
    """矩阵级还原性诊断：合成 k 个 10×20 秩 2 矩阵，走真实抽取管线。

    σ = 0.30 来自 cache/gate_rank2_calibration.json 对这些 10 位点几何
    匹配出的 sigma_matched（四个 OMTKY3 矩阵全部 0.3）。
    两种构造：
      shared  —— k 个矩阵共享第一轴 u **和**第二轴 v（PR 应 ≈ 1）
      indep   —— 共享 u，第二轴各自独立各向同性（PR 应落在 k 的零分布里）
    """
    print(f"\n  [矩阵级 k = {k}]  10 位点 × 20 残基，σ = {sigma}，"
          f"每种构造 {nrep} 次（同一条 extract_subspace 管线）")
    u = rng.normal(size=20); u -= ONE * (ONE @ u); u /= np.linalg.norm(u)
    nl = null_pr(k, u, rng, n=NNULL)
    crit = float(np.percentile(nl, 5))
    log("null", NNULL, f"矩阵级诊断用的 PR 零分布（k = {k}）")

    def one(kind):
        v = rand_perp(u, rng)
        Bs = []
        for _ in range(k):
            vv = v if kind == "shared" else rand_perp(u, rng)
            S = np.stack([u, vv], 1)                  # 20 × 2 残基侧载荷
            L = rng.normal(size=(10, 2)) * np.array([1.0, 0.6])
            M = L @ S.T + rng.normal(scale=sigma, size=(10, 20))
            B, c, _ = extract_subspace(M, np.ones((10, 20), bool))
            if B is None or len(c) != 20:
                return None
            Bs.append(B)
        Qs = [centered_orth(B) for B in Bs]
        P = np.mean([Q @ Q.T for Q in Qs], axis=0)
        uu = np.linalg.eigh(P)[1][:, ::-1][:, 0]
        dirs = []
        for Q in Qs:
            cc = Q.T @ uu; cc = cc / np.linalg.norm(cc)
            dirs.append(Q @ np.array([-cc[1], cc[0]]))
        return spectrum(dirs)[1]

    res = {}
    for kind in ("shared", "indep"):
        v = [x for x in (one(kind) for _ in range(nrep)) if x is not None]
        v = np.array(v)
        res[kind] = {"pr_med": float(np.median(v)),
                     "pr_ci": [float(np.percentile(v, 2.5)),
                               float(np.percentile(v, 97.5))],
                     "reject": float((v < crit).mean()), "n": int(len(v))}
        print(f"     {kind:<8} PR 中位 {np.median(v):.2f} "
              f"[{np.percentile(v, 2.5):.2f}, {np.percentile(v, 97.5):.2f}]"
              f"   单尾 5% 拒绝率 {res[kind]['reject']:.3f}")
        log("sim", len(v), f"矩阵级诊断 {kind}（k = {k}）")
    sep = res["shared"]["reject"] > 0.80 and res["indep"]["reject"] < 0.09
    print(f"     判定：{'可分辨（命中 > 0.80 且不误伤 < 0.09）' if sep else '**不可分辨**'}"
          f"（共享 {res['shared']['reject']:.2f} / 独立 {res['indep']['reject']:.2f}）")
    res["separable"] = bool(sep)
    res["crit05"] = crit
    res["sigma"] = sigma
    return res


# ----------------------------------------------------------------------
# 6. 各向同性基线（§8；docs/31 抓到 docs/30 把它写成「各向异性基线」）
# ----------------------------------------------------------------------
def isotropy_baseline(rng, d=18, n=200000):
    a = rng.normal(size=(n, d)); a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = rng.normal(size=(n, d)); b /= np.linalg.norm(b, axis=1, keepdims=True)
    c = np.abs(np.einsum("ij,ij->i", a, b))
    med, mean = float(np.median(c)), float(np.mean(c))
    ana = float(np.sqrt(2 / (np.pi * d)))
    log("null", n, f"各向同性基线模拟（d = {d}，两条随机一维方向的 |cos|）")
    print("\n" + "=" * 86)
    print(f"[基线 §8] **各向同性**基线（isotropy baseline），d = {d}，{n} 次")
    print(f"   两条随机一维方向的 |cos|：中位 {med:.3f}  均值 {mean:.3f}"
          f"   解析 E|cos| = sqrt(2/(πd)) = {ana:.3f}")
    print("   口径更正（docs/31 §4.2）：docs/30 把这个量写成「各向异性基线」，"
          "标签反了。")
    print("   它是**各向同性**下的期望值，是**离散度要超过的下限**，不是 0。")
    return {"d": d, "median": med, "mean": mean, "analytic_mean": ana,
            "label": "isotropy baseline（各向同性基线）"}


# ----------------------------------------------------------------------
# 7. 合并矩阵（pooled）：能不能当替代？
# ----------------------------------------------------------------------
def pooled_check(by, kd):
    print("\n" + "=" * 86)
    print("[合并矩阵] pooled OMTKY3 vs 共同参照（它过门槛，但它不是替代）")
    if POOLED not in by:
        print("   pooled 矩阵未抽出 —— 跳过")
        return None
    ref, p = by[REF], by[POOLED]
    r = align(p["B"], p["cols"], ref["B"], ref["cols"])
    dfl = align_deflated(p["B"], p["cols"], ref["B"], ref["cols"],
                         kd, np.arange(20))
    print(f"   {POOLED}:  cosθ1 {r['cos'][0]:.3f}  cosθ2 {r['cos'][1]:.3f}"
          f"  投掉疏水后 {(dfl['cos_resid'] if dfl else float('nan')):.3f}")
    print("   它只贡献 **1** 个子空间，所以 k = 1：既没有成对 cosθ2 的离散度，")
    print("   也不存在 PR（PR 在 k = 1 上恒等于 1，与共享无关）。")
    log("comparison", 1, "pooled 矩阵 vs 共同参照")
    return {"name": POOLED, "cos1": r["cos"][0], "cos2": r["cos"][1],
            "defl": dfl["cos_resid"] if dfl else None, "k_contributed": 1}


# ----------------------------------------------------------------------
def main():
    harness = require_harness()
    scales = load_scales(verbose=False)
    kd = scales["kd_hydropathy"]
    by = load_pool()
    repro = reproduce_docs13(by)
    keep, grows = gate_table()

    pw = pairwise(by, keep, kd)

    print("\n" + "=" * 86)
    print("[PR] 残差方向参与率 —— k = 3 与 k = 4 并列（各自的零分布）")
    rng = np.random.default_rng(20261013)
    pr3 = pr_at_k(by, keep, rng, "k=3 存活集")
    pr4 = pr_at_k(by, OMT4, rng, "k=4 原口径（仅作对照，含不可读的 3SGB）")

    cs3 = consensus(by, keep, np.random.default_rng(20261013 + 1), "k=3")
    cs4 = consensus(by, OMT4, np.random.default_rng(20261013 + 2), "k=4 对照")

    powr = power_k3(np.random.default_rng(424242))
    pmat = power_k3_matrix(np.random.default_rng(777000))
    base = isotropy_baseline(np.random.default_rng(1))
    pool = pooled_check(by, kd)

    # ---- 判决 ----
    print("\n" + "=" * 86)
    print("判决")
    print("=" * 86)
    usable = (powr[3]["fp_rate"] < 0.09
              and powr[3]["power_full_share"] > 0.80
              and pmat["separable"])
    print(f"  PR 统计量在 k = 3 上可用？  "
          f"{'是（理想条件下能把共享与独立分开）' if usable else '**否**'}")
    print(f"    方向级：阴性 {powr[3]['fp_rate']:.3f}、完全共享功效 "
          f"{powr[3]['power_full_share']:.3f}、80% 功效需共享度 "
          + (f"{powr[3]['share80']:.2f}" if powr[3]["share80"] else "达不到"))
    print(f"    矩阵级：共享构造拒绝率 {pmat['shared']['reject']:.3f}、"
          f"独立构造拒绝率 {pmat['indep']['reject']:.3f}")
    print(f"  k = 3 的 PR = {pr3['pr']:.2f}，零分布中位 {pr3['null_med']:.2f} "
          f"[{pr3['null_ci'][0]:.2f}, {pr3['null_ci'][1]:.2f}]，"
          f"单尾 p = {pr3['p_one_sided']:.4f}")
    print(f"  k = 4 的 PR = {pr4['pr']:.2f}，零分布中位 {pr4['null_med']:.2f} "
          f"[{pr4['null_ci'][0]:.2f}, {pr4['null_ci'][1]:.2f}]，"
          f"单尾 p = {pr4['p_one_sided']:.4f}")
    ov = (pr3["pr"] >= pr4["null_ci"][0] and pr3["pr"] <= pr4["null_ci"][1])
    print(f"  §6 检查：k = 3 的 PR 是否落在 k = 4 零分布区间内？{'是' if ov else '否'}"
          "（两个 k 的 PR 不可互比，这里只用来判断有没有增加分辨率）")
    c2 = sorted((x["cos2"] for x in pw["vs_ref"]), reverse=True)
    print(f"  存活 3 个的 cosθ2 = " + " / ".join(f"{x:.3f}" for x in c2)
          + f"；各向同性基线（d = 18）中位 {base['median']:.3f}")

    print("\n[§5 搜索量]")
    tot = {}
    for e in SEARCH_LOG:
        tot[e["kind"]] = tot.get(e["kind"], 0) + e["n"]
    for kk, v in sorted(tot.items()):
        print(f"   {kk:<12} {v:>9}")
    print(f"   零分布格数 = {len([e for e in SEARCH_LOG if e['kind']=='null'])}"
          f"；每一个都在自己的 k 与 d 上生成。"
          f"  候选函数搜索 = 0（本次不调参、不换统计量）")

    out = {"generated": "analysis/omtky3_k3.py",
           "harness_ok": bool(harness.get("ok")),
           "repro_docs13_max_dev": repro,
           "ref": REF, "dropped": DROP, "keep": keep,
           "gate": {n: {"stab_obs": grows[n]["stab_obs"],
                        "floor_p95": grows[n]["floor_p95"],
                        "margin": grows[n]["stab_obs"] - grows[n]["floor_p95"],
                        "ceiling_med": grows[n]["ceiling_med"]}
                    for n in OMT4 + [REF]},
           "pairwise": pw, "pr_k3": pr3, "pr_k4": pr4,
           "consensus_k3": cs3, "consensus_k4": cs4,
           "power_dir": {str(k): v for k, v in powr.items()},
           "power_matrix": pmat, "isotropy_baseline": base,
           "pooled": pool, "pr_usable_at_k3": bool(usable),
           "search_log": SEARCH_LOG, "nnull": NNULL}
    p = os.path.join(ROOT, "cache", "omtky3_k3.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
