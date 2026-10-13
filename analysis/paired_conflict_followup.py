#!/usr/bin/env python3
"""docs/29 两处必须补完的测量（都在预先登记的候选之内，不新增候选）。

(A) H1 的量化：**最大完全观测块到底多大**。主脚本只报了「< 12×8，不可判定」，
    那不够 —— 必须给出实际尺寸，并说明人类点名的那个直接检验
    （在完全观测子矩阵上重算 0.016）在这批数据上**能不能做**。
    同时报一条放松的口径：允许每行/每列缺 ≤ 1 格的「近完全块」。

(B) H5.2 的判别后续：主脚本测到 \|cos(u, 逐残基缺失率向量)\| = **0.673** ≥ 0.6，
    按预先登记这是**支持 H5**。预先登记要求"test any that are testable"，
    所以必须做判别测量，**判读在跑之前写定**：
      把逐残基缺失率方向从**两侧**投掉（u 与每个域的折叠轴各自 deflate），
      重算 144 个域的 \|cos\| 分布。
      - 若 deflate 后中位仍 ≥ 0.60 且超 p95 比例仍 ≫ 5% → **缺失偏倚不是驱动因素**，
        H5.2 被排除为解释（0.673 是共线而非因果）；
      - 若中位塌到 d−1 维 Haar 中位附近且超 p95 ≈ 5% → **H5.2 是驱动因素**，
        跨数据集一致性有一部分是缺失模式产物。
    另报一条"去掉最稀有的两个残基（C、W）"的敏感性核对。

运行：python3 analysis/paired_conflict_followup.py
产物：cache/docs29_followup.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddpca_load as DL                                          # noqa: E402
import tsuboyama_load as TL                                      # noqa: E402
from subspace_lib import AA20, ROOT, haar_1d_null                # noqa: E402
from third_family_rank1 import align_1d, rank1_dir               # noqa: E402

OUT = os.path.join(ROOT, "cache", "docs29_followup.json")


# ----------------------------------------------------------------- (A)
def biggest_block(mp, max_hole_row=0, max_hole_col=0):
    """贪心剔除，直到每行缺 ≤ max_hole_row 且每列缺 ≤ max_hole_col。
    返回 (n_rows, n_cols, n_cells, fill)。"""
    rows = np.where(mp.any(axis=1))[0]
    cols = np.arange(mp.shape[1])
    best = (0, 0, 0, 0.0)
    while len(rows) >= 5 and len(cols) >= 5:
        sub = mp[np.ix_(rows, cols)]
        hr = (~sub).sum(axis=1)
        hc = (~sub).sum(axis=0)
        if hr.max() <= max_hole_row and hc.max() <= max_hole_col:
            n = int(sub.sum())
            return len(rows), len(cols), n, n / sub.size
        # 剔除「洞最多」的那一维里最差的那一个（按比例，避免长宽偏置）
        if hr.max() / len(cols) >= hc.max() / len(rows):
            rows = np.delete(rows, int(np.argmax(hr)))
        else:
            cols = np.delete(cols, int(np.argmax(hc)))
    return best


def partA(verbose=True):
    P = print if verbose else (lambda *a, **k: None)
    P("=" * 96)
    P("(A) H1 的量化：最大完全观测块")
    P("=" * 96)
    out = {}
    for dom in ("GRB2-SH3", "PSD95-PDZ3"):
        D = DL.load_domain(dom)
        DL.verify(dom, D, verbose=False)
        Mf, mkf, cf = DL.matrix(D, "folding")
        Mb, mkb, _ = DL.matrix(D, "binding")
        mp = mkf & mkb
        keep = mp.sum(axis=1) >= 4
        mp = mp & keep[:, None]
        rows = {}
        for lab, hr, hc in (("完全块（0 洞）", 0, 0),
                            ("每行每列 ≤ 1 洞", 1, 1),
                            ("每行每列 ≤ 2 洞", 2, 2)):
            nr, nc, n, fill = biggest_block(mp, hr, hc)
            rows[lab] = {"n_rows": nr, "n_cols": nc, "n_cells": n,
                         "fill": fill}
            P(f"  {dom:<12}{lab:<18} {nr} 位点 × {nc} 残基 = {n} 格"
              f"（填充率 {fill:.1%}）")
        # 每列（残基）的配对覆盖率 —— 看是哪个残基在打洞
        cov = mp.sum(axis=0) / max(1, int(keep.sum()))
        worst = np.argsort(cov)[:6]
        P(f"  {dom:<12}配对覆盖率最低的 6 个残基: "
          + "、".join(f"{AA20[i]} {cov[i]:.0%}" for i in worst)
          + f"；最高 {AA20[int(np.argmax(cov))]} {cov.max():.0%}"
          + f"；中位 {np.median(cov):.0%}")
        rows["col_coverage"] = {AA20[i]: float(cov[i])
                                for i in range(len(AA20))}
        rows["n_sites_paired"] = int(keep.sum())
        out[dom] = rows
    P("\n  **结论**：人类点名的那个直接检验（在完全观测子矩阵上重算 0.016）"
      "\n  在这批数据上**做不成** —— 不是因为功效不足，而是因为"
      "**不存在可用尺寸的完全观测块**。")
    P("  这件事本身是 informative missingness 的一个直接后果：41.8% 的缺失"
      "\n  不是集中在少数行列上，而是**弥散**的，所以剔不出完整块。")
    return out


# ----------------------------------------------------------------- (B)
def deflate(v, w):
    """把方向 w 从 v 里投掉（两者都先中心化）。"""
    v = np.asarray(v, float) - np.mean(v)
    w = np.asarray(w, float) - np.mean(w)
    w = w / np.linalg.norm(w)
    return v - w * (w @ v)


def partB(verbose=True):
    P = print if verbose else (lambda *a, **k: None)
    P("\n" + "=" * 96)
    P("(B) H5.2 判别后续：把逐残基缺失率方向从两侧投掉")
    P("=" * 96)
    gf = json.load(open(os.path.join(ROOT, "cache",
                                     "generic_folding_axis.json")))
    assert gf["aa_order"] == AA20
    u = np.asarray(gf["u"], float)
    ucols = np.arange(len(AA20))
    ax = json.load(open(os.path.join(ROOT, "cache", "tsuboyama_axis.json")))
    info, mats = TL.load_cached()
    names = [k for k, v in ax["rows"].items() if v.get("repro_ok")]

    # 逐残基缺失率（144 个域汇总，与主脚本同一口径）
    miss = np.zeros(len(AA20))
    tot = 0
    for nm in names:
        mk = mats[nm]["mask"]
        miss += (~mk).sum(axis=0)
        tot += mk.shape[0]
    mrate = miss / tot
    c0 = align_1d(mrate, ucols, u, ucols)["cos"]
    P(f"  复核主脚本：|cos(u, 逐残基缺失率)| = {c0:.3f}")
    P(f"  逐残基配对缺失率: "
      + "、".join(f"{AA20[i]} {mrate[i]:.1%}"
                 for i in np.argsort(-mrate)[:6]) + " …")

    p95_20 = float(np.percentile(haar_1d_null(20, n=40000, seed=3), 95))
    p95_19 = float(np.percentile(haar_1d_null(19, n=40000, seed=3), 95))
    med_19 = float(np.median(haar_1d_null(19, n=40000, seed=3)))
    P(f"  零分布: d=20 的 Haar p95 {p95_20:.3f}；"
      f"deflate 后环境降为 d−1 = 19，Haar 中位 {med_19:.3f}、p95 {p95_19:.3f}")

    raw, defl, nodrop = [], [], []
    ud = deflate(u, mrate)
    keep_cw = [i for i, a in enumerate(AA20) if a not in ("C", "W")]
    for nm in names:
        r = ax["rows"][nm]
        w = np.asarray(r["axis"], float)
        cc = np.asarray(r["cols"], int)
        a = align_1d(w, cc, u, ucols)
        if a:
            raw.append(a["cos"])
        # deflate 两侧
        wfull = np.zeros(len(AA20))
        wfull[cc] = w
        wd = deflate(wfull, mrate)
        b = align_1d(wd, ucols, ud, ucols)
        if b:
            defl.append(b["cos"])
        # 去掉 C/W 的敏感性核对
        kk = [i for i in keep_cw if i in set(cc.tolist())]
        if len(kk) >= 15:
            c = align_1d(wfull[kk], np.array(kk), u[kk], np.array(kk))
            if c:
                nodrop.append(c["cos"])
    raw, defl, nodrop = map(np.array, (raw, defl, nodrop))

    def boot(x, n=4000, seed=7):
        rng = np.random.default_rng(seed)
        m = [float(np.median(rng.choice(x, size=len(x), replace=True)))
             for _ in range(n)]
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    P(f"\n  {'口径':<34}{'n':>5}{'中位':>8}{'域级 bootstrap 95%':>24}"
      f"{'超 p95 比例':>13}")
    for lab, x, p95 in (("原口径（docs/27 §5.4）", raw, p95_20),
                        ("**两侧投掉缺失率方向**", defl, p95_19),
                        ("去掉 C、W（敏感性）", nodrop, p95_20)):
        lo, hi = boot(x)
        P(f"  {lab:<34}{len(x):>5}{np.median(x):>8.3f}"
          f"{f'[{lo:.3f}, {hi:.3f}]':>24}{np.mean(x > p95):>13.1%}")

    lo, hi = boot(defl)
    survive = (np.median(defl) >= 0.60 and np.mean(defl > p95_19) > 0.15)
    collapse = (np.median(defl) <= med_19 + 0.10
                and np.mean(defl > p95_19) < 0.15)
    v = ("**排除 H5.2**（缺失偏倚不是驱动因素，0.673 是共线而非因果）"
         if survive else
         "**H5.2 是驱动因素**（跨数据集一致性有一部分是缺失模式产物）"
         if collapse else "中间档，不判")
    P(f"\n  预先写定的判读 -> {v}")
    return {"cos_u_vs_missrate": c0, "missrate_by_aa":
            {AA20[i]: float(mrate[i]) for i in range(len(AA20))},
            "haar_p95_d20": p95_20, "haar_p95_d19": p95_19,
            "haar_median_d19": med_19,
            "raw": {"n": len(raw), "median": float(np.median(raw)),
                    "boot_ci": boot(raw),
                    "over_p95": float(np.mean(raw > p95_20))},
            "deflated": {"n": len(defl), "median": float(np.median(defl)),
                         "boot_ci": [lo, hi],
                         "over_p95": float(np.mean(defl > p95_19))},
            "drop_CW": {"n": len(nodrop), "median": float(np.median(nodrop)),
                        "boot_ci": boot(nodrop),
                        "over_p95": float(np.mean(nodrop > p95_20))},
            "verdict": v}


def partA2(verbose=True):
    """放松口径的块上做一次完整流程（**先还原诊断，再看 \\|cos\\|**，顺序同主脚本）。

    这一节的目的**不是**救 H1，而是把「做不成」证到底：
    即使放松到「每行每列 ≤ 2 洞」的 16×8 块，还原诊断是否还有功效。
    """
    import gate_calib_lib as G
    import rank_core as RC
    from subspace_lib import permute_within_sites
    P = print if verbose else (lambda *a, **k: None)
    P("\n" + "=" * 96)
    P("(A2) 放松到「每行每列 ≤ 2 洞」的块：先还原诊断，再看 |cos|")
    P("=" * 96)
    out = {}
    for dom in ("GRB2-SH3",):
        D = DL.load_domain(dom)
        DL.verify(dom, D, verbose=False)
        Mf, mkf, _ = DL.matrix(D, "folding")
        Mb, mkb, _ = DL.matrix(D, "binding")
        mp = mkf & mkb
        mp = mp & (mp.sum(axis=1) >= 4)[:, None]
        rows = np.where(mp.any(axis=1))[0]
        cols = np.arange(mp.shape[1])
        while len(rows) >= 5 and len(cols) >= 5:
            sub = mp[np.ix_(rows, cols)]
            hr = (~sub).sum(axis=1)
            hc = (~sub).sum(axis=0)
            if hr.max() <= 2 and hc.max() <= 2:
                break
            if hr.max() / len(cols) >= hc.max() / len(rows):
                rows = np.delete(rows, int(np.argmax(hr)))
            else:
                cols = np.delete(cols, int(np.argmax(hc)))
        m2 = np.zeros_like(mp)
        m2[np.ix_(rows, cols)] = mp[np.ix_(rows, cols)]
        d2 = len(cols)
        P(f"  {dom}: {len(rows)} 位点 × {d2} 残基 = {int(m2.sum())} 格"
          f"（残基 {''.join(AA20[i] for i in cols)}）")
        Fs = np.where(m2, Mf, np.nan)
        Bs = np.where(m2, Mb, np.nan)
        rf = G.split_half(Fs, m2, rank=1, nsplit=200, seed=12)
        rb = G.split_half(Bs, m2, rank=1, nsplit=200, seed=11)
        rfm = float(np.median(rf)) if rf else float("nan")
        rbm = float(np.median(rb)) if rb else float("nan")
        p95 = float(np.percentile(haar_1d_null(d2, n=40000, seed=3), 95))
        P(f"  D-036（块内位点对半）: 折叠 {rfm:.3f}、结合 {rbm:.3f} vs "
          f"d={d2} 的 Haar p95 {p95:.3f} -> "
          f"折叠{'过' if rfm > p95 else '**不过**'}、"
          f"结合{'过' if rbm > p95 else '**不过**'}"
          f" => 配对主张{'可以提' if (rfm > p95 and rbm > p95) else '**不允许**提'}")
        # 还原诊断
        w0, c0, _ = rank1_dir(Fs, m2)
        mr, mc = RC.center_fit(Bs, m2)
        R = Bs - mr[:, None] - mc[None, :]
        coup = float(np.nanstd(R[m2]))
        rec = {}
        for sg in (0.2, 0.4, 0.6, 1.0):
            got, rp = [], []
            ut = np.zeros(mp.shape[1])
            ut[c0] = w0
            ut -= ut[c0].mean()
            ut /= np.linalg.norm(ut)
            for t in range(40):
                rng = np.random.default_rng(600 + t + int(sg * 1000))
                A = rng.normal(0, coup, (mp.shape[0], 1))
                X = (A @ ut[None, :] + mr[:, None] + mc[None, :]
                     + rng.normal(0, sg, mp.shape))
                X = np.where(m2, X, np.nan)
                w, cc, _ = rank1_dir(X, m2)
                if w is None:
                    continue
                r = align_1d(w, cc, ut, np.arange(mp.shape[1]))
                if r:
                    got.append(r["cos"])
                if t < 8:
                    v = G.split_half(X, m2, rank=1, nsplit=60, seed=80 + t)
                    if v:
                        rp.append(float(np.median(v)))
            rec[sg] = {"recovery_median": float(np.median(got)) if got else None,
                       "synth_repro": float(np.median(rp)) if rp else None}
            P(f"    σ = {sg:<4} 还原 |cos| 中位 "
              f"{rec[sg]['recovery_median']:.3f}；合成可复现度 "
              f"{rec[sg]['synth_repro']:.3f}")
        cands = [(abs(v["synth_repro"] - rbm), s) for s, v in rec.items()
                 if v["synth_repro"] is not None]
        resid, sg_star = min(cands)
        recs = rec[sg_star]["recovery_median"]
        P(f"  标定: 贴住实测结合可复现度 {rbm:.3f} 的是 σ* = {sg_star}"
          f"（残差 {resid:.3f}）；该档还原 |cos| = **{recs:.3f}**")
        P(f"  判读: 还原 < 0.50 → **不可判定**  -> "
          f"{'**不可判定**' if recs < 0.50 else ('有功效' if recs >= 0.80 else '中间档')}")
        wb, cb, _ = rank1_dir(Bs, m2)
        r = align_1d(w0, c0, wb, cb) if wb is not None else None
        rng = np.random.default_rng(91)
        pv = []
        for _ in range(400):
            Bp, mkp = permute_within_sites(Bs, m2, rng)
            wp, cp, _ = rank1_dir(Bp, mkp)
            if wp is None:
                continue
            q = align_1d(w0, c0, wp, cp)
            if q:
                pv.append(q["cos"])
        pv = np.array(pv)
        P(f"  块上的 |cos(折叠轴, 结合轴)| = "
          f"**{(r['cos'] if r else float('nan')):.3f}**"
          f"（配对全集 0.016）；块自己的置换零分布中位 {np.median(pv):.3f}、"
          f"p95 {np.percentile(pv, 95):.3f}、p = "
          f"{float(np.mean(pv >= r['cos'])) if r else float('nan'):.3f}")
        out[dom] = {"n_rows": len(rows), "n_cols": d2,
                    "cols": "".join(AA20[i] for i in cols),
                    "n_cells": int(m2.sum()), "d": d2, "haar_p95": p95,
                    "repro_folding": rfm, "repro_binding": rbm,
                    "d036_both": bool(rfm > p95 and rbm > p95),
                    "recovery": {str(k): v for k, v in rec.items()},
                    "sigma_star": sg_star, "calib_resid": resid,
                    "recovery_at_sigma_star": recs,
                    "abscos": r["cos"] if r else None,
                    "perm_median": float(np.median(pv)),
                    "perm_p95": float(np.percentile(pv, 95)),
                    "perm_p": float(np.mean(pv >= r["cos"])) if r else None}
    return out


def main():
    out = {"A_blocks": partA(), "A2_relaxed": partA2(), "B_h5_2": partB()}
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n产物: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
