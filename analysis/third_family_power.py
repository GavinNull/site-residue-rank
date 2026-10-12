#!/usr/bin/env python3
"""可达上限诊断：候选的「定不出自己的轴」是现象还是我的方法不够？
（CLAUDE.md §10.3 强制；docs/23 §5）

§10.3：「任何『测不出结构』的结论，必须先过一次理想条件下的还原性诊断，
否则无法区分『现象无结构』与『我的方法不够』。」

`third_family_rank1.py` 测到 titeseq/4-4-20 的秩 1 对半分自对齐
只有 0.285（一维零分布 p95 = 0.445）。但 4-4-20 只有 **20 个位点**，
对半后每半 10 个位点。所以必须先问：**在 20 个位点这个几何上，
即使信号是完美的秩 1，对半分诊断最好能给多少？**
如果上限本身就很低，0.285 什么也不说明。

做法：**沿用候选自己的 mask 几何**（位点数、每行观测模式都不变），
只把数值换成「已知的一条真方向 + 可调噪声」，扫噪声水平，
在每个水平上报 (a) 秩 1 对交互方差的解释率、(b) 对半分自对齐中位。
然后看候选自己的解释率落在哪一格，读出那一格的上限。

同时给一个**不误伤**对照：把同一诊断套在既有保留集的
OMTKY3/1CHO（10 位点，已判定「第二维可分辨」）上 ——
如果 10 位点几何上的上限也很低，那说明本仓库对小矩阵一律该用
bootstrap 稳定性而不是对半分（docs/13 §4.3 已经这么做了），
这个结论必须和候选的判决一起报。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import (ROOT, ci, dc_fast, extract_subspace,  # noqa: E402
                          haar_1d_null, load_all_dms, load_independent,
                          load_skempi, dedup_by_content, pick_primary)
from third_family_rank1 import align_1d, rank1_dir  # noqa: E402
from third_family_screen import cand_magma, cand_titeseq  # noqa: E402

N_SPLIT = 400
N_REP = 12


def rank1_expl(M, mask):
    """秩 1 对**交互**（双向中心化后的残差）方差的样本内解释率。"""
    B, cols, err = extract_subspace(M, mask, rank=1)
    if B is None:
        return None, err
    keep_r = mask.sum(axis=1) >= 4
    Ms, mk = M[np.ix_(keep_r, cols)], mask[np.ix_(keep_r, cols)]
    R = dc_fast(np.where(mk, Ms, 0.0), mk)
    X = np.where(mk, R, 0.0)
    for _ in range(40):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        L = (U[:, :1] * s[:1]) @ Vt[:1]
        X = np.where(mk, R, L)
        X = X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    L = (U[:, :1] * s[:1]) @ Vt[:1]
    expl = 1.0 - ((R[mk] - L[mk]) ** 2).sum() / (R[mk] ** 2).sum()
    return float(expl), None


def split_half_1d(M, mask, nsplit=N_SPLIT, seed=42):
    rng = np.random.default_rng(seed)
    nr = M.shape[0]
    out = []
    for _ in range(nsplit):
        p = rng.permutation(nr)
        h1, h2 = p[: nr // 2], p[nr // 2:]
        w1, k1, _ = rank1_dir(M[h1], mask[h1])
        w2, k2, _ = rank1_dir(M[h2], mask[h2])
        if w1 is None or w2 is None:
            continue
        r = align_1d(w1, k1, w2, k2)
        if r:
            out.append(r["cos"])
    return out


def boot_stab_1d(M, mask, nboot=600, seed=7):
    """bootstrap 子空间（秩 1）与全数据方向的 |cos| —— docs/13 §4.3 的
    那个「算不出对半分时用的」度量，在秩 1 上的同口径版本。"""
    w0, c0, _ = rank1_dir(M, mask)
    if w0 is None:
        return []
    rng = np.random.default_rng(seed)
    nr = M.shape[0]
    out = []
    for _ in range(nboot):
        idx = rng.integers(0, nr, nr)
        wb, kb, _ = rank1_dir(M[idx], mask[idx], seed=int(rng.integers(1e9)))
        if wb is None:
            continue
        r = align_1d(w0, c0, wb, kb)
        if r:
            out.append(r["cos"])
    return out


def ceiling(mask, sigmas, nrep=N_REP, nsplit=N_SPLIT, seed=0):
    """在给定 mask 几何上，用**已知的一条真方向**合成，扫噪声。
    返回 [(sigma, expl 中位, 对半分中位, 对半分 [2.5,97.5], 成功率)]。"""
    nr, nc = mask.shape
    rows = []
    for sg in sigmas:
        es, cs, nfail = [], [], 0
        for t in range(nrep):
            rng = np.random.default_rng(seed + 1000 * t + int(sg * 1e4))
            v = rng.normal(size=nc); v -= v.mean(); v /= np.linalg.norm(v)
            a = rng.normal(0, 1.0, nr)
            X = np.outer(a, v) + rng.normal(0, sg, (nr, nc))
            e, err = rank1_expl(X, mask)
            if e is None:
                nfail += 1
                continue
            es.append(e)
            cs += split_half_1d(X, mask, nsplit=max(40, nsplit // nrep),
                                seed=7 + t)
        rows.append({"sigma": float(sg),
                     "expl": float(np.median(es)) if es else float("nan"),
                     "split": float(np.median(cs)) if cs else float("nan"),
                     "split_ci": ci(cs) if cs else (float("nan"),) * 2,
                     "n_split": len(cs), "n_fail": nfail})
    return rows


def main():
    nl = haar_1d_null(20, n=100000, seed=1)
    p95 = float(np.percentile(nl, 95))
    print("=" * 96)
    print("可达上限诊断（CLAUDE.md §10.3）：秩 1 对半分自对齐的几何上限")
    print("=" * 96)
    print(f"一维随机零分布 d=20：中位 {np.median(nl):.3f}、p95 {p95:.3f}\n")

    targets = []
    for d in cand_titeseq():
        targets.append(("titeseq/4-4-20/fluorescein", d))
    mg, _ = cand_magma()
    for d in mg:
        if d["name"] == "magma/topbinHA_rep2/319-345":
            targets.append((d["name"], d))
    # 不误伤对照：既有保留集里已判「可分辨」的两个（一个 10 位点小矩阵、
    # 一个大矩阵）—— 诊断必须能把它们和候选分开
    mats, _ = dedup_by_content(load_independent() + load_skempi()
                               + load_all_dms())
    mats = pick_primary(mats)
    for d in mats:
        if d["name"] in ("OMTKY3/chymotrypsin/1CHO", "starr2020/WH1/huACE2"):
            targets.append((d["name"] + "  [对照：已判可分辨]", d))

    out = {"p95_1d": p95, "null_med": float(np.median(nl)), "rows": []}
    sigmas = [0.15, 0.3, 0.5, 0.8, 1.2, 2.0]
    for name, d in targets:
        e, err = rank1_expl(d["M"], d["mask"])
        obs = split_half_1d(d["M"], d["mask"])
        om = float(np.median(obs)) if obs else float("nan")
        oc = ci(obs) if obs else (float("nan"),) * 2
        print("-" * 96)
        print(f"{name}")
        print(f"  几何：{d['M'].shape[0]} 位点 × {d['M'].shape[1]} 残基，"
              f"{int(d['mask'].sum())} 观测")
        print(f"  实测：秩 1 对交互方差的解释率 = "
              f"{('%.1f%%' % (100 * e)) if e is not None else 'N/A ' + str(err)}"
              f"；对半分自对齐中位 = "
              f"{('%.3f [%.2f, %.2f]' % (om, oc[0], oc[1])) if obs else 'nan（算不出）'}"
              f"（{len(obs)}/{N_SPLIT} 次成功）")
        print(f"  同几何的理想上限（已知真方向 + 可调噪声）：")
        print(f"     {'sigma':>7}{'秩1 解释交互':>13}"
              f"{'对半分自对齐中位 [2.5,97.5]':>30}{'成功次数':>9}")
        rows = ceiling(d["mask"], sigmas)
        for r in rows:
            print(f"     {r['sigma']:>7.2f}{r['expl']:>12.1%}"
                  f"{r['split']:>16.3f} [{r['split_ci'][0]:.2f},"
                  f"{r['split_ci'][1]:.2f}]{r['n_split']:>9}")
        # 候选的解释率落在哪一格
        verdict = ""
        if e is not None:
            cand = [r for r in rows if np.isfinite(r["expl"])]
            if cand:
                near = min(cand, key=lambda r: abs(r["expl"] - e))
                verdict = (f"实测解释率 {e:.1%} 最接近 sigma = {near['sigma']}"
                           f"（{near['expl']:.1%}），该格的对半分上限 = "
                           f"{near['split']:.3f}")
                print(f"  -> {verdict}")
                if np.isfinite(om):
                    if near["split"] < p95:
                        print(f"     **上限本身不超零分布 p95 "
                              f"({near['split']:.3f} < {p95:.3f})"
                              f" —— 实测的 {om:.3f} 什么也不说明："
                              f"这是方法不够，不是现象无结构**")
                    else:
                        print(f"     上限 {near['split']:.3f} 超 p95，"
                              f"而实测 {om:.3f} "
                              f"{'也超 —— 可分辨' if om > p95 else '不超 —— 这一次是现象，不是功效'}")
                else:
                    print("     实测对半分**算不出来**（对半后过不了列保护），"
                          "所以这个几何上对半分口径不可用")
        out["rows"].append({"name": name, "shape": list(d["M"].shape),
                            "n_obs": int(d["mask"].sum()),
                            "expl": e, "split_obs": om,
                            "split_obs_ci": list(oc), "n_split_obs": len(obs),
                            "ceiling": rows, "verdict": verdict})

    # ------------------------------------------------------------------
    # 第二部分：bootstrap 稳定性这个**实际在用的闸**的上限与下限
    # ------------------------------------------------------------------
    # 上面证明：对半分口径在 ≤43 位点的几何上要么无功效（上限 0.230 <
    # 零分布 p95），要么**根本算不出**（连已在证据集里的 OMTKY3/1CHO 也算不出）。
    # docs/13 §4.3 因此改用 bootstrap 稳定性、门槛 0.60。
    # 那就必须把这个闸也标定：它的**下限**（纯噪声给多少）和
    # **上限**（真方向给多少）。只给上限的护栏会被绕过（D-007）。
    print("\n" + "=" * 96)
    print("[第二部分] 实际在用的闸 —— bootstrap 稳定性的上下限标定")
    print("=" * 96)
    print("  下限 = 同几何、**没有任何真结构**（纯噪声）时的 stab；")
    print("  上限 = 同几何、一条已知真方向 + 匹配 SNR 时的 stab。")
    print("  实测值若贴住下限，这个闸在该几何上就什么也没说。\n")
    print(f"  {'体系':<42}{'实测 stab':>10}{"纯噪声下限 中位 [2.5,97.5]":>26}"
          f"{'真方向上限':>11}")
    print("  " + "-" * 92)
    boot_rows = []
    for name, d in targets:
        e, _ = rank1_expl(d["M"], d["mask"])
        obsv = boot_stab_1d(d["M"], d["mask"], nboot=600)
        om = float(np.median(obsv)) if obsv else float("nan")
        # 下限：纯噪声，同 mask。**闸用的是每个矩阵的中位数**，
        # 所以零分布必须是「每次噪声实现的中位数」的分布，不是把
        # 所有 bootstrap 抽样混在一起（混在一起会给出 [0.04, 0.97]
        # 这种无意义的宽区间，看起来像「实测落在区间内」）。
        flo = []
        for t in range(24):
            rng = np.random.default_rng(300 + t)
            X = rng.normal(0, 1.0, d["M"].shape)
            v = boot_stab_1d(X, d["mask"], nboot=200, seed=50 + t)
            if v:
                flo.append(float(np.median(v)))
        fm = float(np.median(flo)) if flo else float("nan")
        fc = ci(flo) if flo else (float("nan"),) * 2
        # 上限：真方向，噪声调到匹配实测解释率
        sg_best, hi = None, float("nan")
        best = None
        for sg in sigmas:
            ee = []
            for t in range(4):
                rng = np.random.default_rng(700 + t + int(sg * 1e4))
                v = rng.normal(size=d["M"].shape[1]); v -= v.mean()
                v /= np.linalg.norm(v)
                a = rng.normal(0, 1.0, d["M"].shape[0])
                X = np.outer(a, v) + rng.normal(0, sg, d["M"].shape)
                x, _ = rank1_expl(X, d["mask"])
                if x is not None:
                    ee.append(x)
            if not ee:
                continue
            gap = abs(float(np.median(ee)) - (e if e is not None else 0.3))
            if best is None or gap < best:
                best, sg_best = gap, sg
        if sg_best is not None:
            hv = []
            for t in range(24):
                rng = np.random.default_rng(900 + t)
                v = rng.normal(size=d["M"].shape[1]); v -= v.mean()
                v /= np.linalg.norm(v)
                a = rng.normal(0, 1.0, d["M"].shape[0])
                X = np.outer(a, v) + rng.normal(0, sg_best, d["M"].shape)
                q = boot_stab_1d(X, d["mask"], nboot=200, seed=80 + t)
                if q:
                    hv.append(float(np.median(q)))
            hi = float(np.median(hv)) if hv else float("nan")
            hc = ci(hv) if hv else (float("nan"),) * 2
        else:
            hc = (float("nan"),) * 2
        print(f"  {name[:41]:<42}{om:>10.3f}"
              f"{fm:>12.3f} [{fc[0]:.2f},{fc[1]:.2f}]"
              f"{hi:>11.3f} [{hc[0]:.2f},{hc[1]:.2f}]")
        boot_rows.append({"name": name, "stab_obs": om, "floor": fm,
                          "floor_ci": list(fc), "ceiling": hi,
                          "ceiling_ci": list(hc),
                          "sigma_matched": sg_best, "expl": e})
    print("\n  判读（门槛 0.60）。先判这个闸在该几何上**有没有判别力**：")
    print("  （真方向上限中位 − 纯噪声下限中位 < 0.10 => 没有判别力，"
          "实测值不可读）")
    for r in boot_rows:
        if not np.isfinite(r["stab_obs"]):
            continue
        sep = ((r["ceiling"] - r["floor"])
               if np.isfinite(r["ceiling"]) and np.isfinite(r["floor"])
               else float("nan"))
        if not np.isfinite(sep) or sep < 0.10:
            v = (f"**闸无判别力（上限−下限 = {sep:+.3f}）—— "
                 f"实测 {r['stab_obs']:.3f} 不可读，可分辨性未定**")
        elif r["stab_obs"] >= 0.60 and r["stab_obs"] > r["floor_ci"][1]:
            v = (f"过闸，且超出纯噪声下限的 97.5% 上界 "
                 f"({r['floor_ci'][1]:.3f})")
        elif r["stab_obs"] >= 0.60:
            v = (f"过 0.60，但**没有超出纯噪声下限的 97.5% 上界 "
                 f"({r['floor_ci'][1]:.3f})** —— 不算证据")
        else:
            v = "不过闸"
        print(f"     {r['name'][:44]:<46}{v}")
        r["verdict_boot"] = v
    out["boot_calibration"] = boot_rows

    p = os.path.join(ROOT, "cache", "third_family_power.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print("\n产物:", p)


if __name__ == "__main__":
    main()
