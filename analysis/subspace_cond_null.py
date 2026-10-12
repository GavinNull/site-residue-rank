#!/usr/bin/env python3
"""N3（成分1 混淆）在共识层面的正确做法：**条件零分布**。

问题：平均投影算子 Pbar 的 lambda_2 与「3 个完全随机 2 维面」比较是不公平的。
一旦这些子空间都包含同一条方向 u（去溶剂化），lambda_2 就会被抬高，
因为它们已经不再互相一般位。拿它去对比无条件随机零分布，会把
「只共享一条方向」误读成「共享两条」。

条件零分布 H0_cond：每个体系的子空间 = span{u, 该体系自己的随机第二方向}，
其中 u 是观测到的共享方向，随机第二方向在 u 的正交补（中心化 19 维内）里 Haar。
即：**第一维共享、第二维体系特异**。
  若观测 lambda_2 落在 H0_cond 内 => 只有一条方向普适。
  若观测 lambda_2 超出 H0_cond 的 p95 => 第二维也普适。

对照（既测能命中也测不误伤，§10.2）：同一检验在冠状病毒家族内部必须
给出「超出」（已知它们共享两维），在跨家族必须能区分。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import (AA20, ROOT, dedup_by_content,  # noqa: E402
                          extract_subspace, load_all_dms, load_independent,
                          load_skempi)

NSIM = 4000
ONE = np.ones((20, 1)) / np.sqrt(20)


def proj(B):
    Q = B - ONE @ (ONE.T @ B)
    Q, _ = np.linalg.qr(Q)
    return Q @ Q.T, Q


def eigs(Bs):
    Pb = np.mean([proj(B)[0] for B in Bs], axis=0)
    w, V = np.linalg.eigh(Pb)
    return w[::-1], V[:, ::-1]


def rand_in_complement(u, rng):
    a = rng.normal(size=20)
    a -= ONE[:, 0] * (ONE[:, 0] @ a)
    a -= u * (u @ a)
    return a / np.linalg.norm(a)


def _draw(k, u, alpha, rng):
    """k 个子空间，各含一条「u 加 alpha 角度抖动」的第一方向 + 随机第二方向。"""
    Ps = []
    ca, sa = np.cos(alpha), np.sin(alpha)
    for _ in range(k):
        u1 = ca * u + sa * rand_in_complement(u, rng)
        u1 /= np.linalg.norm(u1)
        a2 = rand_in_complement(u, rng)
        a2 -= u1 * (u1 @ a2)
        a2 /= np.linalg.norm(a2)
        Q, _ = np.linalg.qr(np.stack([u1, a2], 1))
        Ps.append(Q @ Q.T)
    return np.linalg.eigvalsh(np.mean(Ps, axis=0))[::-1][:3]


def cond_null(k, u, rng, nsim=NSIM, target_l1=None):
    """H0_cond: 第一维按**与观测同等程度**共享，第二维体系特异。

    为什么要标定 alpha（第一版是错的）：trace(Pbar) 恒等于 2。
    若让每个子空间都**精确**含 u，则 lambda_1 = 1.000，剩下只有 1.000
    分给其余方向，lambda_2 被人为压低；而观测的 lambda_1 只有 0.79-0.88，
    剩下 1.12-1.21。拿观测 lambda_2 去比那个零分布是**结构性偏向阳性**的，
    等于用一个不可能的零假设。
    修正：用二分法调抖动角 alpha，使模拟的 lambda_1 中位数等于观测 lambda_1，
    然后在**同一个 lambda_1 水平**下读 lambda_2 的分布。
    """
    if target_l1 is None:
        alpha = 0.0
    else:
        lo, hi = 0.0, np.pi / 2
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            m = np.median([_draw(k, u, mid, rng)[0] for _ in range(240)])
            if m > target_l1:
                lo = mid
            else:
                hi = mid
        alpha = 0.5 * (lo + hi)
    out = np.array([_draw(k, u, alpha, rng) for _ in range(nsim)])
    return out, alpha


def uncond_null(k, rng, nsim=NSIM):
    out = []
    for _ in range(nsim):
        Ps = []
        for _ in range(k):
            A = rng.normal(size=(20, 2))
            A -= ONE @ (ONE.T @ A)
            Q, _ = np.linalg.qr(A)
            Ps.append(Q @ Q.T)
        out.append(np.linalg.eigvalsh(np.mean(Ps, axis=0))[::-1][:3])
    return np.array(out)


def report(label, names, by, rng, boot=None):
    Bs = [by[n] for n in names]
    w, V = eigs(Bs)
    u = V[:, 0]
    k = len(Bs)
    cn, alpha = cond_null(k, u, rng, target_l1=float(w[0]))
    un = uncond_null(k, rng)
    print(f"\n[{label}]  {k} 个子空间")
    for n in names:
        print(f"    {n}")
    print(f"    观测 lambda = {np.round(w[:3], 3).tolist()}")
    print(f"    无条件随机零分布 p95        = "
          f"{np.round(np.percentile(un, 95, axis=0), 3).tolist()}")
    print(f"    条件零分布已标定到观测 lambda_1（抖动角 alpha = "
          f"{np.degrees(alpha):.1f} 度）")
    print(f"    条件零分布中位 = "
          f"{np.round(np.median(cn, axis=0), 3).tolist()}")
    print(f"    条件零分布 p95              = "
          f"{np.round(np.percentile(cn, 95, axis=0), 3).tolist()}")
    p2 = float(np.mean(cn[:, 1] >= w[1]))
    verdict = ("第二维也共享（超出条件零分布）" if w[1] >
               np.percentile(cn[:, 1], 95)
               else "只有一条方向共享（lambda_2 落在条件零分布内）")
    print(f"    lambda_2 在条件零分布下的 p = {p2:.4f}  -> {verdict}")
    return {"label": label, "names": names, "eigs": w[:3].tolist(),
            "u": u.tolist(), "cond_p95": np.percentile(cn, 95,
                                                       axis=0).tolist(),
            "cond_med": np.median(cn, axis=0).tolist(),
            "alpha_deg": float(np.degrees(alpha)),
            "uncond_p95": np.percentile(un, 95, axis=0).tolist(),
            "p_lambda2": p2}


def main():
    rng = np.random.default_rng(20261011)
    mats, _ = dedup_by_content(load_independent() + load_skempi()
                               + load_all_dms())
    by = {}
    for d in mats:
        B, c, _ = extract_subspace(d["M"], d["mask"])
        if B is not None and len(c) == 20 and d["censor_frac"] < 0.25:
            by[d["name"]] = B

    print("=" * 88)
    print("条件零分布检验：第二维是否也普适")
    print("=" * 88)
    print(f"每格 {NSIM} 次模拟")

    COV = ["starr2020/WH1/huACE2", "animalACE2/Wuhan-Hu-1/bat",
           "PDCoV/PDCoV/gAPN",
           "sarbecovirus/SARS-CoV-1_Urban/Ra9479ACE2",
           "animalACE2/Omicron_BA2/human_monomer"]
    OMT = ["OMTKY3/chymotrypsin/1CHO", "OMTKY3/HLE-elastase/1PPF",
           "OMTKY3/subtilisin-Carlsberg/1R0R", "OMTKY3/SGPB/3SGB"]
    out = []
    # 阳性对照：已知共享两维的冠状病毒家族 —— 检验必须给「超出」
    out.append(report("阳性对照：冠状病毒家族内（已知共享两维）",
                      COV, by, rng))
    # 家族均衡的跨家族检验
    out.append(report("跨家族（冠状病毒 / OMTKY3 / GB1，各 1 代表）",
                      ["starr2020/WH1/huACE2",
                       "SKEMPI/OMTKY3/pooled-4proteases",
                       "lehner/GB1/IgG-Fc"], by, rng))
    # OMTKY3 的 4 个蛋白酶：同骨架、不同配体、完整数据、各自第二维可分辨
    out.append(report("OMTKY3 同骨架 x 4 种蛋白酶（非病毒的配体对照）",
                      OMT, by, rng))
    # 全部：4 个 OMTKY3 单酶 + GB1 + 5 个冠状病毒
    out.append(report("可分辨体系全集（4 OMTKY3 单酶 + GB1 + 5 冠状病毒）",
                      OMT + ["lehner/GB1/IgG-Fc"] + COV, by, rng))

    print("\n" + "=" * 88)
    print("量级校验：标定后条件零分布的 lambda_1 中位必须等于观测 lambda_1")
    for r in out:
        d = abs(r["cond_med"][0] - r["eigs"][0])
        print(f"   [{r['label'][:36]:<38}] 零分布 lambda_1 中位 "
              f"{r['cond_med'][0]:.3f} vs 观测 {r['eigs'][0]:.3f}  "
              f"差 {d:.3f}  {'OK' if d < 0.02 else '<< 标定失败'}")

    print("\n负对照（§10.2 不误伤）：合成「只共享一维」的子空间，"
          "在同一检验下必须**不**超出")
    rng2 = np.random.default_rng(99)
    u0 = np.array(out[1]["u"])
    for k, lab in ((3, "3 个"), (10, "10 个")):
        for tgt in (0.875, 0.788):
            al = cond_null(k, u0, rng2, nsim=10, target_l1=tgt)[1]
            obs = np.median([_draw(k, u0, al, rng2)[1] for _ in range(400)])
            cn2, _ = cond_null(k, u0, rng2, nsim=1500, target_l1=tgt)
            pp = float(np.mean(cn2[:, 1] >= obs))
            print(f"   {lab}子空间、lambda_1 标到 {tgt}: 合成 lambda_2 = "
                  f"{obs:.3f}，条件零分布 p = {pp:.3f}  "
                  f"{'OK（未误报）' if pp > 0.05 else '<< 误报'}")

    print("\n普适方向 u（跨家族那一格）的残基载荷:")
    u = np.array(out[1]["u"])
    o = np.argsort(-u)
    print("   " + "  ".join(f"{AA20[i]}{u[i]:+.2f}" for i in o))

    p = os.path.join(ROOT, "cache", "subspace_cond_null.json")
    json.dump({"nsim": NSIM, "aa_order": AA20, "groups": out},
              open(p, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
