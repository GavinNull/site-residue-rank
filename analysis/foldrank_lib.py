#!/usr/bin/env python3
"""docs/42：折叠一侧交互项的秩 —— 共享层（预先登记锁 + 判别集 + 合成 + 定秩程序 + 自检）。

预先登记：`cache/docs42_prereg.md`，sha256 锁在 `cache/docs42_prereg_lock.json`。
任何依赖本模块做结论的脚本必须先调 `verify_lock()`，锁不上就中止（CLAUDE.md §4）。

本模块**不**计算任何真实数据的秩。真实数据的定秩在 `foldrank_ranks.py`，
它在功效结果落盘之前拒绝运行。

数值层复用，不重写：
  rank_core.cv_rank_noleak / center_fit / selfcheck   （无泄露、带欠定行列保护 ALS）
  tsuboyama_load.load_cached                          （逐域矩阵，仓库约定：正 = 去稳定）
"""
import hashlib
import json
import math
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rank_core as RC                                      # noqa: E402
import tsuboyama_load as TL                                 # noqa: E402
from subspace_lib import ROOT                               # noqa: E402

PREREG = os.path.join(ROOT, "cache", "docs42_prereg.md")
LOCK = os.path.join(ROOT, "cache", "docs42_prereg_lock.json")

# ---- 预先登记 §2：定秩程序（与 within_system_rank / rank_core 的验证过的流程一致）
PROC = dict(max_rank=3, K=5, restarts=3, seed=3, iters=250, ridge=0.1)
# ---- 预先登记 §3：合成谱（元素级 RMS 的相对权重）
SPEC = {0: (), 1: (1.0,), 2: (1.0, 0.5), 3: (1.0, 0.5, 0.25)}
SIG_A, SIG_B = 0.112, 0.70


# ----------------------------------------------------------------------
def verify_lock(verbose=True):
    lk = json.load(open(LOCK))
    h = hashlib.sha256(open(os.path.join(ROOT, lk["file"]), "rb").read()
                       ).hexdigest()
    assert h == lk["sha256"], (
        "预先登记文件已被改动！CLAUDE.md §4 禁止事后修改论证。\n"
        f"  锁定 {lk['sha256']}\n  现在 {h}")
    gh = json.load(open(os.path.join(ROOT, "cache",
                                     "gate_harness_check.json")))
    assert gh.get("ok") is True, "cache/gate_harness_check.json 的 ok 不为 true"
    if verbose:
        print(f"  预先登记锁 {lk['file']} sha256 {h[:16]}… 一致"
              f"（锁定于 {lk['locked_at']}）  OK")
        print("  cache/gate_harness_check.json ok = true  OK")
    return {"prereg_sha256": h, "locked_at": lk["locked_at"]}


# ----------------------------------------------------------------------
def universe():
    """预先登记 §1 的判别集：docs/27 的 144 个域（p95 门槛 ∩ σ 可得 ∩ D-036 过闸）。

    -> (names, mats, cluster{name: WT_cluster}, sigC{name}, fout{name})
    """
    info, mats = TL.load_cached()
    gate = json.load(open(os.path.join(ROOT, "cache", "tsuboyama_gate.json")))
    SG = json.load(open(os.path.join(ROOT, "cache", "tsuboyama_sigma.json")))
    AX = json.load(open(os.path.join(ROOT, "cache", "tsuboyama_axis.json")))
    CE = json.load(open(os.path.join(ROOT, "cache", "tsuboyama_censor.json")))
    names = sorted(set(gate["pass_p95"]) & set(SG["rows"])
                   & set(AX["repro_pass"]))
    import csv
    cl = {r["name"]: r["WT_cluster"] for r in csv.DictReader(open(
        os.path.join(ROOT, "cache", "dms", "tsuboyama2023",
                     "Single_DMS_list.csv")))}
    sigC = {w: SG["rows"][w]["sigma_C"] for w in names}
    fout = {w: CE["rows"][w]["f_out"] for w in names}
    return names, mats, {w: cl[w] for w in names}, sigC, fout


# ----------------------------------------------------------------------
def detect(M, mask, seed=None):
    """定秩程序 P*：返回 (检出秩, trim95 曲线[0..max_rank], 全 RMSE 曲线)。

    检出秩 = trim95 RMSE 的 argmin（并列取小，np.argmin 语义）。
    """
    p = dict(PROC)
    if seed is not None:
        p["seed"] = seed
    e = RC.cv_rank_noleak(M, mask, **p)
    trim = np.array([e[r][1] for r in range(p["max_rank"] + 1)])
    full = np.array([e[r][0] for r in range(p["max_rank"] + 1)])
    return int(np.argmin(trim)), trim, full


def _orthonormal_centered(n, k, rng):
    X = rng.normal(size=(n, k))
    X -= X.mean(axis=0, keepdims=True)
    Q, _ = np.linalg.qr(X)
    return Q


def interaction(nr, nc, rank, s, rng, weights=None):
    """已知秩 rank 的交互项，元素级总 RMS = s（谱权重 weights，默认 SPEC）。"""
    if rank == 0 or s <= 0:
        return np.zeros((nr, nc))
    w = np.array(weights if weights is not None else SPEC[rank][:rank], float)
    lam = s * w / math.sqrt((w ** 2).sum()) * math.sqrt(nr * nc)
    Phi = _orthonormal_centered(nr, rank, rng)
    Psi = _orthonormal_centered(nc, rank, rng)
    return (Phi * lam) @ Psi.T


def geometry(M, mask):
    """本域自己的几何：主效应 (r, c)、观测残差 SD、κ（iid 单位噪声经同一
    中心化后留在观测格子上的方差比）。"""
    r, c = RC.center_fit(M, mask)
    R = M - r[:, None] - c[None, :]
    cso = float(R[mask].std())
    rng = np.random.default_rng(zlib.crc32(b"kappa") % (2 ** 31))
    ks = []
    for _ in range(20):
        Z = rng.normal(size=mask.shape)
        rr, cc = RC.center_fit(np.where(mask, Z, 0.0), mask)
        Rz = Z - rr[:, None] - cc[None, :]
        ks.append(float((Rz[mask] ** 2).mean()))
    return {"r": r, "c": c, "cso": cso, "kappa": float(np.mean(ks)),
            "row_sd": float(r.std()), "col_sd": float(c.std())}


def synth(geo, mask, rank, s, sigma, rng, weights=None):
    """主效应（本域实测）+ 已知秩交互（强度 s）+ 各向同性噪声（σ）。
    未观测格子填 0（下游从不读取）。"""
    nr, nc = mask.shape
    M = (geo["r"][:, None] + geo["c"][None, :]
         + interaction(nr, nc, rank, s, rng, weights)
         + rng.normal(0.0, sigma, (nr, nc)))
    return np.where(mask, M, 0.0)


def scenario(geo, sigma_C):
    """预先登记 §3：三个锚点的情景 (σ, s, 类别)。

    s_a = sqrt(max(cso² − κσ_a², 0))；若 s_a < 0.25·cso，情景与实测残差不相容，
    改用上限情景 s = cso（类别 'ceiling'）。秩 0 世界用方差匹配的零假设：
    σ0² = σ_a² + s²/κ。
    """
    out = {}
    for tag, sg in (("A", SIG_A), ("C", float(sigma_C)), ("B", SIG_B)):
        s2 = geo["cso"] ** 2 - geo["kappa"] * sg ** 2
        s = math.sqrt(s2) if s2 > 0 else 0.0
        if s >= 0.25 * geo["cso"]:
            kind = "consistent"
        else:
            s, kind = geo["cso"], "ceiling"
        sig0 = math.sqrt(sg ** 2 + s ** 2 / geo["kappa"])
        out[tag] = {"sigma": sg, "s": s, "kind": kind, "sigma_null": sig0,
                    "s_over_sigma": s / sg}
    return out


def rng_for(name, *tag):
    key = (name + "|" + "|".join(map(str, tag))).encode()
    return np.random.default_rng(zlib.crc32(key) + 977 * len(key))


# ----------------------------------------------------------------------
def selftest(names, mats, verbose=True):
    """预先登记 §9 的工具自检（用已知事实；不过即 AssertionError）。"""
    P = print if verbose else (lambda *a, **k: None)
    log = {}
    # H1 底层数值层自检
    RC.selfcheck(verbose=False)
    log["H1_rank_core_selfcheck"] = "pass"
    P("  [H1] rank_core.selfcheck()  OK")

    # H2 合成工具本身：谱、中心化、总 RMS（已知事实，不依赖定秩）
    rng = np.random.default_rng(1)
    for k in (1, 2, 3):
        I = interaction(60, 20, k, 0.8, rng)
        rms = float(np.sqrt((I ** 2).mean()))
        sv = np.linalg.svd(I, compute_uv=False)
        assert abs(rms - 0.8) < 1e-9, f"总 RMS {rms}"
        assert np.linalg.matrix_rank(I, tol=1e-8) == k
        assert abs(I.mean(axis=0)).max() < 1e-10 and \
            abs(I.mean(axis=1)).max() < 1e-10, "交互项未双向中心化"
        if k >= 2:
            assert abs(sv[1] / sv[0] - SPEC[k][1]) < 1e-9, sv
    log["H2_interaction_tool"] = "pass"
    P("  [H2] 合成交互：总 RMS、秩、双向中心化、谱比 = 预设  OK")

    # H3 κ：完备无缺失矩阵上 κ = (nr−1)(nc−1)/(nr nc)（已知解析值）
    Mfull = np.zeros((40, 20))
    g = geometry(Mfull + np.random.default_rng(0).normal(size=(40, 20)),
                 np.ones((40, 20), bool))
    ka = 39 * 19 / (40 * 20)
    assert abs(g["kappa"] - ka) < 0.03, (g["kappa"], ka)
    log["H3_kappa_full"] = {"kappa": g["kappa"], "analytic": ka}
    P(f"  [H3] κ(完备 40×20) = {g['kappa']:.3f} vs 解析 {ka:.3f}  OK")

    # H4/H5/H6：定秩程序在**真实几何**上两个方向（命中 / 不误伤）+ 量纲哨兵
    pick = [names[i] for i in range(0, len(names), max(1, len(names) // 12))
            ][:12]
    hit = []
    nul = []
    sent = []
    for w in pick:
        m = mats[w]
        geo = geometry(m["M"], m["mask"])
        for k in range(10):
            # 命中：C5 的设计点（lam1=1.5, lam2=0.75, σ=0.64 → s=1.677）
            Mz = synth(geo, m["mask"], 2, math.sqrt(1.5 ** 2 + 0.75 ** 2),
                       0.64, rng_for(w, "H4", k))
            hit.append(detect(Mz, m["mask"])[0])
            # 不误伤：无结构 σ=0.45
            M0 = synth(geo, m["mask"], 0, 0.0, 0.45, rng_for(w, "H5", k))
            r0, tr0, fu0 = detect(M0, m["mask"])
            nul.append(r0)
            sent.append(fu0[0] / 0.45)
    hit = np.array(hit)
    nul = np.array(nul)
    ph = float((hit == 2).mean())
    pn = float((nul == 0).mean())
    ms = float(np.median(sent))
    assert ph >= 0.80, f"[H4] 命中失败：已知秩 2 @C5 设计点只在 {ph:.2f} 检出 2"
    assert pn >= 0.70, f"[H5] 误伤：无结构只在 {pn:.2f} 检出 0"
    assert 0.8 < ms < 1.5, f"[H6] 量纲哨兵：秩 0 留出 RMSE / σ = {ms:.2f}"
    log["H4_hit"] = {"geoms": pick, "P_det2_given_true2": ph,
                     "hist": np.bincount(hit, minlength=4).tolist()}
    log["H5_nonharm"] = {"P_det0_given_true0": pn,
                         "hist": np.bincount(nul, minlength=4).tolist()}
    log["H6_sentinel"] = {"rank0_holdout_rmse_over_sigma": ms}
    P(f"  [H4] 命中：已知秩 2（C5 设计点）→ 检出 2 的比例 {ph:.2f}"
      f"（{len(hit)} 次，分布 {np.bincount(hit, minlength=4).tolist()}）  OK")
    P(f"  [H5] 不误伤：无结构 → 检出 0 的比例 {pn:.2f}"
      f"（分布 {np.bincount(nul, minlength=4).tolist()}）  OK")
    P(f"  [H6] 量纲哨兵：秩 0 留出 RMSE/σ 中位 {ms:.2f}（应≈1）  OK")
    return log
