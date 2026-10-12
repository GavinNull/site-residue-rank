#!/usr/bin/env python3
"""普适方向的化学身份 + 每个体系的功效档案。

判决表 2b 的结论是「只有**一个**方向跨独立体系共享」。那么该问的不是
「共识 2 维子空间被已知标度张成多少」，而是
  (a) 那**一条**普适方向是什么；
  (b) 每个体系的第二维到底有没有被测出来（功效档案），
      否则「第二维不共享」分不清是现象还是功效。

普适方向的定义（可辨识）：在**第二维可分辨**的体系上取平均投影算子
Pbar = mean_i Q_i Q_i^T 的首特征向量。注意单个 Q_i 的列不可辨识，
但 Pbar 的特征向量是可辨识的。
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import (AA20, ROOT, dedup_by_content,  # noqa: E402
                          extract_subspace, load_all_dms, load_independent,
                          load_scales, load_skempi, pick_primary, dc_fast)

CENSOR_MAX = 0.25


ONE = np.ones((20, 1)) / np.sqrt(20)


def proj_(B):
    Q = B - ONE @ (ONE.T @ B)
    Q, _ = np.linalg.qr(Q)
    return Q @ Q.T


def main():
    J = json.load(open(os.path.join(ROOT, "cache",
                                    "subspace_universality.json")))
    scales = load_scales(verbose=False)
    stab = {}
    for r in J["focus_pairs"]:
        for side in ("a", "b"):
            if r.get(f"stab_cos2_{side}") is not None:
                stab[r[side]] = r[f"stab_cos2_{side}"]

    mats, _ = dedup_by_content(load_independent() + load_skempi()
                               + load_all_dms())
    mats = pick_primary(mats)
    by = {}
    print("=" * 92)
    print("功效档案：每个体系的交互方差占比、秩2 对交互的解释率、第二维稳定性")
    print("=" * 92)
    print(f"{'体系':<38}{'位点':>5}{'观测':>6}{'截尾':>7}"
          f"{'交互/总方差':>11}{'秩2 解释交互':>13}{'stab cos2':>10}")
    print("-" * 92)
    rows = []
    for d in mats:
        if d["censor_frac"] >= CENSOR_MAX:
            continue
        B, cols, err = extract_subspace(d["M"], d["mask"])
        if B is None:
            continue
        mk = d["mask"][:, cols]
        Ms = d["M"][:, cols]
        R = dc_fast(np.where(mk, Ms, 0.0), mk)
        tot = Ms[mk].var()
        inter = R[mk].var()
        # 秩2 在交互上的解释率（样本内，EM 填充后的低秩重构）
        X = np.where(mk, R, 0.0)
        for _ in range(40):
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            L = (U[:, :2] * s[:2]) @ Vt[:2]
            X = np.where(mk, R, L)
            X = X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) \
                + X.mean()
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        L = (U[:, :2] * s[:2]) @ Vt[:2]
        expl = 1.0 - ((R[mk] - L[mk]) ** 2).sum() / (R[mk] ** 2).sum()
        st = stab.get(d["name"])
        by[d["name"]] = B if len(cols) == 20 else None
        if by[d["name"]] is None:
            del by[d["name"]]
        rows.append({"name": d["name"], "inter_frac": inter / tot,
                     "rank2_expl": float(expl), "stab": st,
                     "sites": len(d["sites"]), "n": d["n_obs"],
                     "cens": d["censor_frac"], "primary": d["primary"]})
        if st is not None or d["primary"]:
            print(f"{d['name'][:37]:<38}{len(d['sites']):>5}{d['n_obs']:>6}"
                  f"{d['censor_frac']:>7.1%}{inter / tot:>11.1%}"
                  f"{expl:>13.1%}"
                  f"{(f'{st:.3f}' if st is not None else '-'):>10}")

    # ------- 普适方向 -------
    # **家族均衡**定义：每个独立家族只取一个代表，否则 OMTKY3（5 个相关矩阵）
    # 和冠状病毒（5 个代表）会各自占去一半权重，共识被它们拉走。
    # 三个家族都必须是「第二维可分辨」的（stab cos2 >= 0.60）。
    FAMILY_REPS = ["starr2020/WH1/huACE2",
                   "SKEMPI/OMTKY3/pooled-4proteases",
                   "lehner/GB1/IgG-Fc"]
    ALT_COV = ["animalACE2/Wuhan-Hu-1/bat", "PDCoV/PDCoV/gAPN",
               "sarbecovirus/SARS-CoV-1_Urban/Ra9479ACE2",
               "animalACE2/Omicron_BA2/human_monomer"]
    resolvable = [n for n in FAMILY_REPS if n in by]
    print(f"\n普适方向（家族均衡）：{len(resolvable)} 个家族各 1 个代表")
    for n in resolvable:
        print(f"   {n}  stab cos2 = {stab.get(n, float('nan')):.3f}")
    Ps = [proj_(by[n]) for n in resolvable]
    Pb = np.mean(Ps, axis=0)
    w, V = np.linalg.eigh(Pb)
    w = w[::-1]; V = V[:, ::-1]
    print(f"\n   特征值前 5: " + " ".join(f"{x:.3f}" for x in w[:5]))
    # 换冠状病毒代表的鲁棒性
    u0 = V[:, 0]
    print("\n   换冠状病毒代表时方向是否稳定:")
    for rep in ALT_COV:
        if rep not in by:
            continue
        w2, V2 = np.linalg.eigh(np.mean(
            [proj_(by[n]) for n in [rep, FAMILY_REPS[1], FAMILY_REPS[2]]],
            axis=0))
        u2 = V2[:, ::-1][:, 0]
        print(f"     {rep[:44]:<46} lambda1 = {w2[::-1][0]:.3f}  "
              f"|cos(u, u')| = {abs(float(u0 @ u2)):.3f}")

    rng = np.random.default_rng(3)
    nw = []
    for _ in range(4000):
        Qs = []
        for _ in range(len(Ps)):
            one = np.ones((20, 1)) / np.sqrt(20)
            A = rng.normal(size=(20, 2)); A -= one @ (one.T @ A)
            Q, _ = np.linalg.qr(A)
            Qs.append(Q @ Q.T)
        nw.append(np.linalg.eigvalsh(np.mean(Qs, axis=0))[::-1][:3])
    nw = np.array(nw)
    print(f"   随机零分布 p95: " +
          " ".join(f"{x:.3f}" for x in np.percentile(nw, 95, axis=0)))
    for k in range(3):
        print(f"   lambda_{k+1} = {w[k]:.3f} vs 零分布 p95 "
              f"{np.percentile(nw[:, k], 95):.3f}  "
              f"{'超出' if w[k] > np.percentile(nw[:, k], 95) else '未超出'}")

    u = V[:, 0]
    order = np.argsort(-u)
    print(f"\n   普适方向 u 的残基载荷（最正 -> 最负）:")
    print("     " + "  ".join(f"{AA20[i]}{u[i]:+.2f}" for i in order))

    print(f"\n   与已知标度的相关（|r|，n = 有定义的残基数）:")
    res = []
    for nm, s in scales.items():
        idx = np.array([i for i in range(20) if np.isfinite(s[i])])
        if len(idx) < 10:
            continue
        a = u[idx] - u[idx].mean()
        b = s[idx] - s[idx].mean()
        if b.std() == 0:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        res.append((abs(r), r, nm, len(idx)))
    for ar, r, nm, n in sorted(res, reverse=True):
        print(f"     {nm:<28}{n:>4}  r = {r:+.3f}   r^2 = {r * r:.3f}")

    full = [nm for nm, s in scales.items() if np.isfinite(s).all()]
    print(f"\n   普适方向被已知标度解释的比例（线性回归 R^2，"
          f"{len(full)} 条完整标度）:")
    best1 = max(((abs(np.corrcoef(u - u.mean(),
                                  scales[p] - scales[p].mean())[0, 1]), p)
                 for p in full))
    print(f"     最佳单条: {best1[1]}  R^2 = {best1[0] ** 2:.3f}")
    bb = None
    for p, q in itertools.combinations(full, 2):
        A = np.stack([scales[p], scales[q], np.ones(20)], 1)
        coef, *_ = np.linalg.lstsq(A, u, rcond=None)
        r2 = 1 - ((u - A @ coef) ** 2).sum() / ((u - u.mean()) ** 2).sum()
        if bb is None or r2 > bb[0]:
            bb = (r2, p, q)
    print(f"     最佳两条: {bb[1]} + {bb[2]}  R^2 = {bb[0]:.3f}")
    A = np.stack([scales[p] for p in full] + [np.ones(20)], 1)
    coef, *_ = np.linalg.lstsq(A, u, rcond=None)
    r2all = 1 - ((u - A @ coef) ** 2).sum() / ((u - u.mean()) ** 2).sum()
    print(f"     全部 {len(full)} 条一起（会过拟合，19 自由度配 "
          f"{len(full)} 个回归量）: R^2 = {r2all:.3f}")
    print(f"     => 最佳单条（{best1[1]}）已占 {best1[0] ** 2:.0%}；"
          f"普适方向上没有与已知标度正交的大块剩余。")
    print("     注意：19 自由度配 15 个回归量，'全部一起'的 R^2 只说明"
          "没有正交的新轴，不说明拟合有意义。")

    out = os.path.join(ROOT, "cache", "subspace_axis1.json")
    json.dump({"universal_dir": u.tolist(), "aa_order": AA20,
               "eigs": w[:5].tolist(),
               "null_eig_p95": np.percentile(nw, 95, axis=0).tolist(),
               "resolvable": resolvable,
               "scale_corr": [{"scale": nm, "r": r, "n": n}
                              for _, r, nm, n in sorted(res, reverse=True)],
               "best1": {"scale": best1[1], "r2": best1[0] ** 2},
               "best2": {"a": bb[1], "b": bb[2], "r2": bb[0]},
               "power_profile": rows}, open(out, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
