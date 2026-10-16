#!/usr/bin/env python3
"""docs/43：binding 矩阵的秩 2 是否被定秩程序的「高 SNR 过冲」抬高 —— 共享层。

预先登记：`cache/docs43_prereg.md`，sha256 锁在 `cache/docs43_prereg_lock.json`。
任何依赖本模块做结论的脚本必须先调 `verify_lock()`，锁不上就中止（CLAUDE.md §4）。

本模块包含：
  * 预先登记锁、定秩程序（与头条同一口径：3 种子均值曲线的 argmin）
  * 重复噪声估计器（σ_meas）+ 对真实文件的对齐核对
  * 植入世界（真实几何 / 真实主效应 / 真实或随机载荷 / 高斯或 wild 噪声）与 cso 正向匹配
  * 三个候选校正 K1 / K2 / K3（预先登记 §9）
  * 自检（CLAUDE.md §10：既测命中，也测不误伤）

数值层复用，不重写：rank_core.cv_rank_noleak / center_fit / residual / fit_loadings /
als（= als_guarded 的向量化等价版）。**禁用 als_complete。**
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import csv
import hashlib
import json
import math
import os
import re
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

import rank_core as RC                                       # noqa: E402

PREREG = os.path.join(ROOT, "cache", "docs43_prereg.md")
LOCK = os.path.join(ROOT, "cache", "docs43_prereg_lock.json")
DMS = os.path.join(ROOT, "cache", "dms")
AA20 = list("ACDEFGHIKLMNPQRSTVWY")

# ---- 预先登记 §3：定秩程序
CV = dict(K=5, restarts=2, iters=150, ridge=0.1)
SEEDS = (1, 2, 3)
MAXR = 3
# ---- 预先登记 §4/§7
R_REP = 40
NPATH = 7
COMPAT = (0.8, 1.25)
SIGMA_LIT_SAME_SOURCE = 0.35          # rank_test_power.SIGMA["same_source"]（evidence/assay-noise.md）


# ======================================================================
# 预先登记锁
# ======================================================================
def verify_lock(verbose=True):
    lk = json.load(open(LOCK))
    h = hashlib.sha256(open(os.path.join(ROOT, lk["file"]), "rb").read()).hexdigest()
    assert h == lk["sha256"], (
        "预先登记文件与锁不一致 —— 按 CLAUDE.md §4 中止\n"
        f"  锁定 {lk['sha256']}\n  现在 {h}")
    if verbose:
        print(f"  预先登记锁 {lk['file']} sha256 {h[:16]}… 一致（锁定于 {lk['locked_at']}）  OK",
              flush=True)
    return {"prereg_sha256": h, "locked_at": lk["locked_at"]}


# ======================================================================
# 定秩程序（与头条同口径）
# ======================================================================
def curves_for(M, mask, seeds=SEEDS, max_rank=MAXR):
    """每个种子一条 trim95 曲线 -> list[dict rank->trim95]。M 在未观测处任意（这里填 0）。"""
    Mf = np.where(mask, M, 0.0)
    out = []
    for sd in seeds:
        e = RC.cv_rank_noleak(Mf, mask, max_rank=max_rank, seed=sd, **CV)
        out.append({r: e[r][1] for r in e})
    return out


def mean_curve(curves):
    ranks = sorted(curves[0])
    return {r: float(np.mean([c[r] for c in curves])) for r in ranks}


def argmin_rank(curve):
    m = min(curve.values())
    return min(r for r in curve if curve[r] <= m + 1e-15)


def detect(M, mask):
    """头条口径：3 种子均值曲线的 argmin（并列取小）。返回 (rank, mean_curve)。"""
    mc = mean_curve(curves_for(M, mask))
    return argmin_rank(mc), mc


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, cen - half), min(1.0, cen + half))


# ======================================================================
# 数据加载（复用 independence_audit.load_all：与 docs/41 同一矩阵）
# ======================================================================
_ALL = None


def load_all_matrices():
    global _ALL
    if _ALL is None:
        import independence_audit as IA
        _ALL = IA.load_all()
    return _ALL


def pick(name):
    for d in load_all_matrices():
        if d["name"] == name:
            return d
    raise KeyError(name)


S1_BACKBONES = ("Wuhan-Hu-1", "Eta", "Omicron_BA2")
S1_REP_LIGS = ("deer", "hamster", "bat", "cat")
S1_NOREP_LIG = "human_monomer"
OTHER = ("PDCoV/PDCoV/gAPN", "CR9114/WT_H1", "SKEMPI/OMTKY3/pooled-4proteases")


def scope_names():
    nm = []
    for b in S1_BACKBONES:
        for l in S1_REP_LIGS + (S1_NOREP_LIG,):
            nm.append(f"animalACE2/{b}/{l}")
    return nm + list(OTHER)


def _f(x):
    return np.nan if x in ("", "NA", "NaN", "nan") else float(x)


def build_reps(d):
    """返回 (Dm, Nrep, kind)。Dm[i,j] = rep1 − rep2（仅两重复格子，其余 NaN）；
    Nrep[i,j] ∈ {0,1,2}（0 = 未观测）；kind ∈ {'measured','none'}。

    **对齐核对（已知事实）**：矩阵每个观测格子的值必须等于 (rep1+rep2)/2（两重复）或该重复（一重复），
    偏差 > tol 即中止 —— 这同时核对了格子身份与「格子 = 重复均值」这个 √2 因子的前提。
    """
    M, mask, sites = d["M"], d["mask"], d["sites"]
    nr, nc = M.shape
    name = d["name"]
    cells = {}
    tol = 1e-4
    if name.startswith("animalACE2/") or name.startswith("PDCoV/"):
        if name.startswith("animalACE2/"):
            if d["ligand"] not in S1_REP_LIGS:
                return None, None, "none"
            fn, skey, tgt = "starr_animalACE2_final_variant_scores.csv", "position", d["backbone"]
            X = d["ligand"]
            cb, c1, c2 = f"bind_{X}", f"{X}_rep1", f"{X}_rep2"
        else:
            fn, skey, tgt = "starr_PDCoV_RBD_final_variant_scores.csv", "position", "PDCoV"
            cb, c1, c2 = "bind_gAPN", "gAPN_rep1", "gAPN_rep2"
        for r in csv.DictReader(open(os.path.join(DMS, fn))):
            if r.get("target") != tgt:
                continue
            cells[(r[skey], r["mutant"])] = (_f(r[cb]), _f(r[c1]), _f(r[c2]))
    elif name == "CR9114/WT_H1":
        pth = os.path.join(DMS, "cr9114", "cr9114_result_HC_WT_H1_KD_table_summary_kabat.csv")
        rx = re.compile(r"^([A-Z])(\d+)([A-Z_])$")
        for r in csv.DictReader(open(pth)):
            m = rx.match(r["mut"])
            if not m or r["mut_class"] not in ("missense", "silent"):
                continue
            cells[(int(m.group(2)), m.group(3))] = (
                _f(r["minus_delta_log_Kd"]), _f(r["minus_delta_log_Kd_rep1"]),
                _f(r["minus_delta_log_Kd_rep2"]))
        tol = 2e-3
    else:
        return None, None, "none"
    aas = d.get("aas")
    if aas is None:
        if name == "CR9114/WT_H1":
            import cr9114_parse as P
            import subspace_lib as L
            cr = P.load_all("dmlk", run_selftest=False)["WT_H1"]
            kc = cr["mask"].sum(0) >= L.MIN_PER_COL
            aas = [AA20[j] for j in range(20) if kc[j]]
        else:
            aas = AA20
    assert len(aas) == nc, f"{name}: 列数 {nc} 与残基标签 {len(aas)} 不符"
    Dm = np.full((nr, nc), np.nan)
    Nrep = np.zeros((nr, nc), int)
    maxdev = 0.0
    for i, s in enumerate(sites):
        for j, a in enumerate(aas):
            if not mask[i, j]:
                continue
            v, a1, a2 = cells[(s, a)]
            have = [x for x in (a1, a2) if np.isfinite(x)]
            assert len(have) >= 1, f"{name} ({s},{a}) 无重复值"
            dev = abs(M[i, j] - float(np.mean(have)))
            maxdev = max(maxdev, dev)
            assert dev < tol, f"{name} ({s},{a}) 矩阵值 {M[i, j]} ≠ 重复均值 {np.mean(have)}"
            Nrep[i, j] = len(have)
            if len(have) == 2:
                Dm[i, j] = a1 - a2
    return Dm, Nrep, "measured"


def sigma_meas_from(Dm, Nrep, mask):
    """预先登记 §2.2。σ_单 = SD(d)/√2；格子噪声² = σ_单²·(f2/2 + f1)。"""
    d = Dm[mask & np.isfinite(Dm)]
    n2 = int((Nrep[mask] == 2).sum())
    n1 = int((Nrep[mask] == 1).sum())
    tot = n1 + n2
    f2, f1 = n2 / tot, n1 / tot
    sd = float(np.std(d, ddof=1))
    mad = float(1.4826 * np.median(np.abs(d - np.median(d))))
    s1 = sd / math.sqrt(2)
    s1r = mad / math.sqrt(2)
    return {"sigma_single": s1, "sigma_cell": s1 * math.sqrt(f2 / 2 + f1),
            "sigma_single_robust": s1r,
            "sigma_cell_robust": s1r * math.sqrt(f2 / 2 + f1),
            "f2": f2, "f1": f1, "n_pairs": int(len(d))}


# ======================================================================
# 植入世界
# ======================================================================
class Geometry:
    """一个真实矩阵的几何：mask、主效应 r/c、cso、真实载荷拟合、σ_fit、值域。"""

    def __init__(self, d):
        self.name = d["name"]
        self.M = np.where(d["mask"], d["M"], 0.0)
        self.mask = d["mask"].copy()
        self.nr, self.nc = self.M.shape
        self.n = int(self.mask.sum())
        self.r, self.c = RC.center_fit(self.M, self.mask)
        self.lo = float(self.M[self.mask].min())
        self.hi = float(self.M[self.mask].max())
        R = RC.residual(self.M, self.mask)
        self.cso = float(R[self.mask].std())
        self.fit = {}
        for k in (1, 2):
            U, V, _ = RC.fit_loadings(self.M, self.mask, rank=k, restarts=4, seed=11)
            P = U @ V.T
            sse = float(((R[self.mask] - P[self.mask]) ** 2).sum())
            dof = (self.nr + self.nc - 1) + k * (self.nr + self.nc - k)
            self.fit[k] = {"P": P, "sigma": math.sqrt(sse / max(self.n - dof, 1))}
        self.Dm, self.Nrep, self.rep_kind = build_reps(d)
        self.sig = None
        if self.rep_kind == "measured":
            self.sig = sigma_meas_from(self.Dm, self.Nrep, self.mask)
        self._ab = {}

    # ---- 交互项
    def interaction(self, k, mode, rng):
        if k == 0:
            return np.zeros((self.nr, self.nc))
        if mode == "real":
            return self.fit[k]["P"].copy()
        # rand：位点/残基载荷随机、中心化、正交，谱 (1, 0.5)
        w = (1.0, 0.5)[:k]
        phis, psis = [], []
        for _ in range(k):
            for lst, n in ((phis, self.nr), (psis, self.nc)):
                v = rng.normal(0, 1, n)
                v -= v.mean()
                for q in lst:
                    v -= (v @ q) * q
                v /= np.linalg.norm(v)
                lst.append(v)
        I = np.zeros((self.nr, self.nc))
        for k_ in range(k):
            I += w[k_] * math.sqrt(self.nr * self.nc) * np.outer(phis[k_], psis[k_])
        return I

    def _scale_to(self, I, s):
        rms = math.sqrt(float((I[self.mask] ** 2).mean()))
        return I * (s / rms) if rms > 0 else I

    def wild_noise(self, rng):
        # 工具自检（G2）暴露：两个重复之间有系统性偏移（d 的均值 != 0，库间批次差），
        # 它对每个格子是同一常数，被主效应吸收，不是噪声；翻转符号前必须先去均值，
        # 否则合成噪声的 RMS 比真实重复差的 SD 大 ~4%（预先登记 4.1 的实现细节，不改变判决规则）。
        e = np.zeros((self.nr, self.nc))
        two = self.mask & np.isfinite(self.Dm)
        dc = self.Dm - float(np.mean(self.Dm[two]))
        sg = rng.choice([-1.0, 1.0], size=(self.nr, self.nc))
        e[two] = (sg[two] * dc[two] / 2.0)
        one = self.mask & (self.Nrep == 1)
        if one.any():
            pool = dc[two]
            e[one] = sg[one] * rng.choice(pool, size=int(one.sum())) / math.sqrt(2.0)
        return e

    def make(self, k, mode, s, sigma, noise, rng, clip=True):
        """Y = r + c + I(k, s) + ε，仅 mask 内有效（其余填 0）。返回 (Y, 被截断比例)。"""
        I = self._scale_to(self.interaction(k, mode, rng), s) if k > 0 else 0.0
        if noise == "gauss":
            eps = rng.normal(0, sigma, (self.nr, self.nc))
        elif noise == "wild":
            assert self.rep_kind == "measured"
            eps = self.wild_noise(rng)
        else:
            raise ValueError(noise)
        Y = self.r[:, None] + self.c[None, :] + I + eps
        cf = 0.0
        if clip:
            Yc = np.clip(Y, self.lo, self.hi)
            cf = float((Yc[self.mask] != Y[self.mask]).mean())
            Y = Yc
        return np.where(self.mask, Y, 0.0), cf

    # ---- cso 正向匹配：cso² = a s² + b σ²
    def ab(self, k, mode, ndraw=8, seed=12345):
        key = (k, mode)
        if key in self._ab:
            return self._ab[key]
        rng = np.random.default_rng(seed + 17 * k + (3 if mode == "rand" else 0))
        a = []
        for _ in range(ndraw if mode == "rand" else 1):
            Y, _ = self.make(k, mode, 1.0, 0.0, "gauss", rng, clip=False)
            a.append(float(RC.residual(Y, self.mask)[self.mask].std()) ** 2)
        b = []
        for _ in range(ndraw):
            Y, _ = self.make(0, mode, 0.0, 1.0, "gauss", rng, clip=False)
            b.append(float(RC.residual(Y, self.mask)[self.mask].std()) ** 2)
        self._ab[key] = (float(np.mean(a)), float(np.mean(b)))
        return self._ab[key]

    def s_for(self, k, mode, sigma):
        a, b = self.ab(k, mode)
        v = self.cso ** 2 - b * sigma ** 2
        return None if v <= 0 else math.sqrt(v / a)

    def sigma0(self):
        """k=0 对照：与真实总残差同量级的纯噪声。"""
        _, b = self.ab(0, "rand")
        return self.cso / math.sqrt(b)

    def path(self, k, sigma_lo):
        """预先登记 §2.5：σ 在 [σ_lo, σ_hi] 上等比 7 点；σ_hi = σ_fit(k)。截断记录。"""
        hi = self.fit[k]["sigma"]
        if sigma_lo >= hi:
            return []
        out = []
        for sg in np.geomspace(sigma_lo, hi, NPATH):
            s = self.s_for(k, "real", float(sg))
            out.append({"sigma": float(sg), "s": s,
                        "snr": None if s is None else s / float(sg)})
        return out


# ======================================================================
# 工具自检（预先登记 §4.3）
# ======================================================================
def selfcheck_noise_estimator(verbose=True):
    log = []
    rng = np.random.default_rng(2026)
    # (a) 命中：重尾噪声 + 15% 一重复，恢复格子噪声（与解析值差 < 5%）
    nr, nc = 150, 20
    mask = rng.random((nr, nc)) < 0.9
    truth = rng.normal(0, 2.0, (nr, nc))
    sg = 0.4
    t = rng.standard_t(6, size=(nr, nc, 2)) * sg / math.sqrt(6 / 4)   # t6 的 SD = sqrt(6/4)
    r1, r2 = truth + t[..., 0], truth + t[..., 1]
    one = rng.random((nr, nc)) < 0.15
    r2 = np.where(one, np.nan, r2)
    cell = np.where(one, r1, (r1 + r2) / 2)
    Dm = np.where(mask & ~one, r1 - r2, np.nan)
    Nrep = np.where(mask, np.where(one, 1, 2), 0)
    est = sigma_meas_from(Dm, Nrep, mask)
    f1 = float(one[mask].mean())
    true_cell = sg * math.sqrt((1 - f1) / 2 + f1)
    rel = abs(est["sigma_cell"] - true_cell) / true_cell
    # 直接对照：格子 − 真值 的经验 SD
    emp = float((cell - truth)[mask].std())
    assert rel < 0.05, f"噪声估计器偏差 {rel:.3f}"
    assert abs(est["sigma_cell"] - emp) / emp < 0.06, (est["sigma_cell"], emp)
    log.append(f"N1 命中：重尾(t6)+15%一重复 -> σ_cell 估计 {est['sigma_cell']:.4f} vs 解析 {true_cell:.4f} "
               f"vs 经验 {emp:.4f}（相对差 {rel:.1%}）  OK")
    # (b) 不误伤：两重复完全相同 -> σ=0；真值差异巨大（主效应）不进入估计
    Dm0 = np.where(mask, 0.0, np.nan)
    est0 = sigma_meas_from(Dm0, np.where(mask, 2, 0), mask)
    assert est0["sigma_cell"] == 0.0
    log.append("N2 不误伤：重复完全一致 -> σ=0  OK")
    # (c) 已知限度（写进文档的下界性质）：两重复共享的系统误差不可见
    shared = rng.normal(0, 1.0, (nr, nc))
    r1s, r2s = truth + shared + t[..., 0], truth + shared + t[..., 1]
    Dms = np.where(mask, r1s - r2s, np.nan)
    ests = sigma_meas_from(Dms, np.where(mask, 2, 0), mask)
    tot = float((((r1s + r2s) / 2) - truth)[mask].std())
    assert ests["sigma_cell"] < 0.6 * tot, "共享误差应使 σ_meas 明显小于总误差"
    log.append(f"N3 已知下界性质：共享误差 SD=1.0 时 σ_meas={ests['sigma_cell']:.3f} < 总误差 {tot:.3f}  OK")
    if verbose:
        for s in log:
            print("  " + s, flush=True)
    return log


def selfcheck_geometry(g, verbose=True, nrep=2):
    """单个几何的自检：cso 正向匹配线性（独立 (s,σ) 上相对差 < 4%）、wild 噪声保真。"""
    log = []
    sg = 0.5 * g.fit[1]["sigma"]
    for mode in ("real", "rand"):
        s = 0.6 * g.cso
        a, b = g.ab(1, mode)
        pred = math.sqrt(a * s * s + b * sg * sg)
        obs = []
        rng = np.random.default_rng(777)
        for _ in range(4):
            Y, _ = g.make(1, mode, s, sg, "gauss", rng, clip=False)
            obs.append(float(RC.residual(Y, g.mask)[g.mask].std()) ** 2)
        obs = math.sqrt(float(np.mean(obs)))
        rel = abs(pred - obs) / obs
        assert rel < 0.04, f"{g.name} {mode}: cso 正向匹配偏差 {rel:.3f}"
        log.append(f"G1 {g.name} [{mode}] cso 预测 {pred:.3f} vs 合成 {obs:.3f}（{rel:.1%}）  OK")
    if g.rep_kind == "measured":
        rng = np.random.default_rng(31)
        sds = []
        for _ in range(6):
            e = g.wild_noise(rng)
            two = g.mask & np.isfinite(g.Dm)
            # 合成「重复差」= 2*ε（两重复格子）：SD 应等于真实 d 的 SD
            sds.append(float(np.std(2 * e[two], ddof=1)))
        real_sd = float(np.std(g.Dm[g.mask & np.isfinite(g.Dm)], ddof=1))
        rel = abs(np.mean(sds) - real_sd) / real_sd
        assert rel < 0.02, f"wild 噪声重复差 SD 偏离真实 {rel:.3f}"
        log.append(f"G2 {g.name} wild 噪声保真：合成重复差 SD {np.mean(sds):.4f} vs 真实 {real_sd:.4f}  OK")
    if verbose:
        for s in log:
            print("  " + s, flush=True)
    return log


# ======================================================================
# 校正候选（预先登记 §9）
# ======================================================================
def als_joint(Y, mask, rank, rng, iters=150, ridge=0.1, r0=None, c0=None):
    """K1：联合拟合 Y ≈ r_i + c_j + U Vᵀ。r、c 是不受岭惩罚的偏置列；岭只作用于 U、V。
    欠定行/列保护同 als_guarded：观测数 < rank+2 的行/列，因子置 0、偏置取均值。
    返回完整预测 P = r + c + U Vᵀ。"""
    nr, nc = Y.shape
    k = rank
    W = mask.astype(float)
    r = np.zeros(nr) if r0 is None else r0.copy()
    c = np.zeros(nc) if c0 is None else c0.copy()
    U = rng.normal(0, 0.1, (nr, k))
    V = rng.normal(0, 0.1, (nc, k))
    Ym = np.where(mask, Y, 0.0)
    Ireg = np.diag([max(ridge, 1e-8)] * k + [1e-8]) if k > 0 else np.array([[1e-8]])
    cnt_r = mask.sum(1)
    cnt_c = mask.sum(0)
    ok_r = cnt_r >= k + 2
    ok_c = cnt_c >= k + 2
    for _ in range(iters):
        # 行步：θ_i=[u_i, r_i]，回归量 [v_j, 1]，目标 y−c_j
        Z = np.hstack([V, np.ones((nc, 1))])
        ZZ = (Z[:, :, None] * Z[:, None, :]).reshape(nc, -1)
        A = (W @ ZZ).reshape(nr, k + 1, k + 1) + Ireg
        b = (W * (Ym - c[None, :])) @ Z
        th = np.linalg.solve(A, b[:, :, None])[:, :, 0]
        U = th[:, :k]
        r = th[:, k]
        if (~ok_r).any():
            U[~ok_r] = 0.0
            tot = (W * (Ym - c[None, :])).sum(1)
            r[~ok_r] = np.where(cnt_r[~ok_r] > 0, tot[~ok_r] / np.maximum(cnt_r[~ok_r], 1), 0.0)
        # 列步：θ_j=[v_j, c_j]，回归量 [u_i, 1]，目标 y−r_i
        Z2 = np.hstack([U, np.ones((nr, 1))])
        ZZ2 = (Z2[:, :, None] * Z2[:, None, :]).reshape(nr, -1)
        A2 = (W.T @ ZZ2).reshape(nc, k + 1, k + 1) + Ireg
        b2 = (W * (Ym - r[:, None])).T @ Z2
        th2 = np.linalg.solve(A2, b2[:, :, None])[:, :, 0]
        V = th2[:, :k]
        c = th2[:, k]
        if (~ok_c).any():
            V[~ok_c] = 0.0
            tot = (W * (Ym - r[:, None])).sum(0)
            c[~ok_c] = np.where(cnt_c[~ok_c] > 0, tot[~ok_c] / np.maximum(cnt_c[~ok_c], 1), 0.0)
    return r[:, None] + c[None, :] + (U @ V.T if k > 0 else 0.0)


def cv_curve(M, mask, ranks, K=5, restarts=2, seed=3, iters=150, ridge=0.1, joint=False):
    """与 rank_core.cv_rank_noleak 同一折划分/重启选择；joint=True 时用 K1 的联合拟合。
    返回 {rank: trim95}。ranks 可为子集（K3 用）。"""
    rng = np.random.default_rng(seed)
    idx = np.argwhere(mask)
    rng.shuffle(idx)
    folds = np.array_split(idx, K)
    out = {}
    for rank in ranks:
        errs = []
        for f in folds:
            trn = mask.copy()
            for i, j in f:
                trn[i, j] = False
            if joint:
                r0, c0 = RC.center_fit(M, trn)
                best, bestP = np.inf, None
                for _ in range(restarts):
                    P = als_joint(M, trn, rank, rng, iters=iters, ridge=ridge, r0=r0, c0=c0)
                    tr = np.mean((M[trn] - P[trn]) ** 2)
                    if tr < best:
                        best, bestP = tr, P
                for i, j in f:
                    errs.append((M[i, j] - bestP[i, j]) ** 2)
            else:
                r, c = RC.center_fit(M, trn)
                R = M - r[:, None] - c[None, :]
                Yt = np.where(trn, R, 0.0)
                best, bestP = np.inf, None
                for _ in range(restarts):
                    P = RC.als(Yt, trn, rank, rng, iters=iters, ridge=ridge)
                    tr = np.mean((R[trn] - P[trn]) ** 2)
                    if tr < best:
                        best, bestP = tr, P
                for i, j in f:
                    errs.append((R[i, j] - bestP[i, j]) ** 2)
        out[rank] = float(math.sqrt(np.quantile(np.array(errs), 0.95)))
    return out


def detect_k1(M, mask):
    Mf = np.where(mask, M, 0.0)
    cs = [cv_curve(Mf, mask, range(MAXR + 1), seed=sd, joint=True, **{k: v for k, v in CV.items() if k != "K"})
          for sd in SEEDS]
    mc = mean_curve(cs)
    return argmin_rank(mc), mc


def tolerance_rank(curve, delta=0.02):
    m = min(curve.values())
    return min(r for r in curve if curve[r] <= (1 + delta) * m)


def detect_k2(M, mask, base_curve=None):
    if base_curve is None:
        _, base_curve = detect(M, mask)
    return tolerance_rank(base_curve), base_curve


def _perm_within_rows(E, mask, rng):
    out = np.zeros_like(E)
    for i in range(E.shape[0]):
        j = np.where(mask[i])[0]
        if len(j) > 1:
            out[i, j] = E[i, rng.permutation(j)]
        elif len(j) == 1:
            out[i, j] = E[i, j]
    return out


def detect_k3(M, mask, B=20, seed=0):
    """K3：顺序 bootstrap（预先登记 §9）。观测与代理用同一 g（单种子 3，ranks=[r-1,r]）。"""
    Mf = np.where(mask, M, 0.0)
    rng = np.random.default_rng(seed)
    r_, c_ = RC.center_fit(Mf, mask)
    R = RC.residual(Mf, mask)
    accepted = 0
    info = []
    for r in (1, 2, 3):
        cu = cv_curve(Mf, mask, [r - 1, r], seed=3, **{k: v for k, v in CV.items() if k != "K"})
        g_obs = (cu[r - 1] - cu[r]) / cu[r - 1]
        if r - 1 == 0:
            S = np.zeros_like(Mf)
        else:
            U, V, _ = RC.fit_loadings(Mf, mask, rank=r - 1, restarts=4, seed=11)
            S = U @ V.T
        E = np.where(mask, R - S, 0.0)
        gs = []
        for _b in range(B):
            Yb = r_[:, None] + c_[None, :] + S + _perm_within_rows(E, mask, rng)
            Yb = np.where(mask, Yb, 0.0)
            cb = cv_curve(Yb, mask, [r - 1, r], seed=3, **{k: v for k, v in CV.items() if k != "K"})
            gs.append((cb[r - 1] - cb[r]) / cb[r - 1])
        thr = float(np.max(gs))
        ok = g_obs > thr
        info.append({"r": r, "g_obs": float(g_obs), "g_null_max": thr})
        if ok:
            accepted = r
        else:
            break
    return accepted, info


# ======================================================================
# 世界执行（供 multiprocessing）
# ======================================================================
def seed_of(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


GEOMS = {}


def set_geoms(dct):
    GEOMS.clear()
    GEOMS.update(dct)


def run_task(t):
    """t: dict(mat, k, mode, s, sigma, noise, rep, tag, method)。返回 dict(t + det + curve + clipfrac)。"""
    g = GEOMS[t["mat"]]
    rng = np.random.default_rng(seed_of(t["mat"], t["tag"], t["rep"]))
    Y, cf = g.make(t["k"], t["mode"], t["s"], t["sigma"], t["noise"], rng, clip=t.get("clip", True))
    method = t.get("method", "base")
    out = dict(t)
    out["clip"] = cf
    if method == "base":
        det, mc = detect(Y, g.mask)
        out["det"], out["curve"] = det, mc
    elif method == "all":                # 基线 + K1 + K2（K3 另走）
        det, mc = detect(Y, g.mask)
        out["det"], out["curve"] = det, mc
        out["det_k2"] = tolerance_rank(mc)
        d1, mc1 = detect_k1(Y, g.mask)
        out["det_k1"], out["curve_k1"] = d1, mc1
    elif method == "k3":
        det, mc = detect(Y, g.mask)
        out["det"], out["curve"] = det, mc
        d3, info = detect_k3(Y, g.mask, B=20, seed=seed_of(t["mat"], t["tag"], t["rep"], "k3"))
        out["det_k3"], out["k3info"] = d3, info
    return out


# ======================================================================
# 校正的自检（已知事实）
# ======================================================================
def selfcheck_corrections(verbose=True):
    log = []
    rng = np.random.default_rng(5)
    # J1 无噪声完备：联合拟合应精确还原 主效应 + 秩 k
    for k in (0, 1, 2):
        a = rng.normal(0, 1.5, 25)
        b = rng.normal(0, 1.5, 18)
        T = a[:, None] + b[None, :]
        if k:
            T = T + rng.normal(0, 1, (25, k)) @ rng.normal(0, 1, (18, k)).T
        mk = np.ones((25, 18), bool)
        P = als_joint(T, mk, k, np.random.default_rng(1), iters=800, ridge=1e-8)
        rel = np.linalg.norm(T - P) / np.linalg.norm(T)
        assert rel < 1e-4, f"als_joint 无噪声秩{k}未还原 rel={rel:.2e}"
        log.append(f"J1 无噪声：联合拟合 秩{k} 相对还原误差 {rel:.1e}  OK")
    # J2 缺失 + 欠定行：不得爆（预测有限且量级同数据）
    T = rng.normal(0, 1, (30, 12)) + 3.0
    mk = rng.random((30, 12)) > 0.4
    mk[4] = False
    mk[:, 5] = False
    mk[7, :2] = True
    mk[7, 2:] = False
    P = als_joint(np.where(mk, T, 0.0), mk, 2, np.random.default_rng(2), iters=60, ridge=0.1)
    assert np.isfinite(P).all() and abs(P).max() < 50
    log.append("J2 缺失/欠定行列下联合拟合有限且不爆  OK")
    # J3 K2 规则：已知曲线 -> 已知结果（并测不误伤：曲线有明确大跌时不被 δ 吞掉）
    assert tolerance_rank({0: 1.00, 1: 0.90, 2: 0.895, 3: 0.95}) == 1       # 2 仅比 1 好 0.55% < δ -> 取 1
    assert tolerance_rank({0: 1.00, 1: 0.90, 2: 0.80, 3: 0.85}) == 2        # 明确优 -> 保留 2
    log.append("J3 K2 规则：微弱差异折回较小秩、明确差异保留  OK")
    # J4 cv_curve 非联合路径 == rank_core.cv_rank_noleak（逐位；防 D-023 式静默偏差）
    M0 = rng.normal(0, 1, (40, 14)) + rng.normal(0, 1.5, 40)[:, None]
    mk0 = rng.random((40, 14)) > 0.25
    e = RC.cv_rank_noleak(np.where(mk0, M0, 0.0), mk0, max_rank=2, seed=3, **CV)
    cc = cv_curve(np.where(mk0, M0, 0.0), mk0, [0, 1, 2], seed=3, **{k: v for k, v in CV.items() if k != "K"})
    for r in (0, 1, 2):
        assert abs(cc[r] - e[r][1]) < 1e-12, (r, cc[r], e[r][1])
    log.append("J4 cv_curve == rank_core.cv_rank_noleak（逐位, 秩0–2）  OK")
    if verbose:
        for s in log:
            print("  " + s, flush=True)
    return log


if __name__ == "__main__":
    verify_lock()
    selfcheck_noise_estimator()
    selfcheck_corrections()
    print("overshoot_lib 自检通过（几何相关的自检在 overshoot_binding.py 里随真实矩阵运行）")


# ======================================================================
# 逐格留出误差（供位点聚类 bootstrap；CLAUDE.md §5）
# ======================================================================
def cv_cell_errs(M, mask, ranks, seed, joint=False, K=5, restarts=2, iters=150, ridge=0.1):
    """与 cv_curve 同一折划分/重启选择；返回 (cells, {rank: 与 np.argwhere(mask) 同序的平方误差})。"""
    rng = np.random.default_rng(seed)
    idx = np.argwhere(mask)
    rng.shuffle(idx)
    folds = np.array_split(idx, K)
    key = {(int(i), int(j)): n for n, (i, j) in enumerate(np.argwhere(mask))}
    order = np.array([key[(int(i), int(j))] for i, j in idx])
    out = {}
    for rank in ranks:
        errs = []
        for f in folds:
            trn = mask.copy()
            for i, j in f:
                trn[i, j] = False
            best, bestP = np.inf, None
            if joint:
                r0, c0 = RC.center_fit(M, trn)
                for _ in range(restarts):
                    P = als_joint(M, trn, rank, rng, iters=iters, ridge=ridge, r0=r0, c0=c0)
                    tr = np.mean((M[trn] - P[trn]) ** 2)
                    if tr < best:
                        best, bestP = tr, P
                tgt = M
            else:
                r, c = RC.center_fit(M, trn)
                R = M - r[:, None] - c[None, :]
                Yt = np.where(trn, R, 0.0)
                for _ in range(restarts):
                    P = RC.als(Yt, trn, rank, rng, iters=iters, ridge=ridge)
                    tr = np.mean((R[trn] - P[trn]) ** 2)
                    if tr < best:
                        best, bestP = tr, P
                tgt = R
            for i, j in f:
                errs.append((tgt[i, j] - bestP[i, j]) ** 2)
        e = np.empty(len(idx))
        e[order] = np.array(errs)
        out[rank] = e
    return np.argwhere(mask), out


def cell_errs_mean(M, mask, joint, seeds=SEEDS, max_rank=MAXR):
    Mf = np.where(mask, M, 0.0)
    acc = None
    for sd in seeds:
        cells, E = cv_cell_errs(Mf, mask, range(max_rank + 1), sd, joint=joint)
        if acc is None:
            acc = {r: E[r].copy() for r in E}
        else:
            for r in E:
                acc[r] += E[r]
    return cells, {r: acc[r] / len(seeds) for r in acc}
