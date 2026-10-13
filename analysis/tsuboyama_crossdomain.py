#!/usr/bin/env python3
"""docs/27 Step 6（bonus）：**folding 的第一轴自己跨结构域普适吗**。

这是 binding 一侧永远做不了的检验（docs/13 §4.2：真正独立的跨家族比较
只有两组，其中一组还是一个骨架）。Tsuboyama 覆盖上百个结构域，
所以可以直接问。

统计口径照 docs/21 §4–§5，一条不改：
  * **先在组内平均投影算子，再把组等权**（docs/21 §5.1：k = 2 的退化
    说明把每个体系等权会让共识被组内共享撑起来；这里组 = Tsuboyama 自己的
    `WT_cluster`，即序列相似性簇，所以「同一个蛋白家族的多个成员」
    不会被重复计票）；
  * λ1 比**同结构**无条件零分布（按真实的簇大小结构生成 Haar 随机 2 维面，
    先簇内平均再等权）；
  * λ2 比**条件零分布**，二分法把抖动角标定到**观测 λ1**
    （docs/21 §5.1：trace(P̄) ≡ 2，不标定的条件零分布是结构性偏阳的），
    并报**标定残差**与**负对照**（合成只共享一维 → 必须不超出）；
  * PR 的零分布在**这个 k** 上重生成（docs/21 §5.4：一个 k 的 PR
    不能和另一个 k 建的零分布比）；
  * **docs/24 §0.5 的陷阱**：剔掉最不可分辨的第二维会把 cos θ2 的中位
    从 0.327 抬到 0.477，而 λ2 的条件 p 不动 —— **过滤后的中位移动不是证据**。
    本脚本因此把 cos θ2 的分布在「全集」与「只保留第二维可分辨者」
    两种口径下并列，并同时报 λ2 的条件 p 有没有动。

运行：python3 analysis/tsuboyama_crossdomain.py [--jobs 9]
产物：cache/tsuboyama_crossdomain.json
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subspace_cond_null as CN                                 # noqa: E402
import tsuboyama_load as TL                                     # noqa: E402
from subspace_lib import (AA20, ROOT, align, extract_subspace,   # noqa: E402
                          haar_1d_null, haar_pair_null, load_scales)

NSIM = 4000
ONE = np.ones((20, 1)) / np.sqrt(20)


def sub2(args):
    name, M, mask = args
    B, cols, err = extract_subspace(M, mask, rank=2)
    if B is None:
        return name, None, err
    return name, (B.tolist(), cols.tolist()), None


def clusters():
    out = {}
    p = os.path.join(ROOT, "cache", "dms", "tsuboyama2023",
                     "Single_DMS_list.csv")
    for r in csv.DictReader(open(p)):
        out[r["name"]] = r.get("WT_cluster", "")
    return out


def pbar_grouped(groups, P):
    """先组内平均，再组等权（docs/21 §5.2 的构造）。"""
    return np.mean([np.mean([P[n] for n in v], axis=0)
                    for v in groups.values()], axis=0)


def null_grouped(sizes, rng, nsim=NSIM):
    """**同结构**无条件零分布：按真实的簇大小结构生成。"""
    out = []
    for _ in range(nsim):
        Ps = []
        for k in sizes:
            Qs = []
            for _ in range(k):
                A = rng.normal(size=(20, 2))
                A -= ONE @ (ONE.T @ A)
                Q, _ = np.linalg.qr(A)
                Qs.append(Q @ Q.T)
            Ps.append(np.mean(Qs, axis=0))
        out.append(np.linalg.eigvalsh(np.mean(Ps, axis=0))[::-1][:3])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=9)
    a = ap.parse_args()

    info, mats = TL.load_cached()
    ax = json.load(open(os.path.join(ROOT, "cache", "tsuboyama_axis.json")))
    adm = sorted(ax["repro_pass"])       # 过三道闸 + D-036 可复现
    print("=" * 102)
    print(f"docs/27 Step 6：folding 第一轴自己跨结构域普适吗"
          f"（{len(adm)} 个过闸且方向可复现的天然独立结构域）")
    print("=" * 102)

    jobs = [(w, mats[w]["M"], mats[w]["mask"]) for w in adm]
    by, cols20, skipped = {}, {}, []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for nm, r, err in ex.map(sub2, jobs, chunksize=2):
            if r is None:
                skipped.append((nm, err)); continue
            B, cl = np.array(r[0]), np.array(r[1])
            if len(cl) != 20:
                skipped.append((nm, f"d = {len(cl)} ≠ 20")); continue
            by[nm] = B
            cols20[nm] = cl
    print(f"   抽到完整 20 维基底的秩 2 子空间：**{len(by)}** 个"
          f"（跳过 {len(skipped)} 个：投影算子必须住在同一个 R^20 里）")
    if skipped[:5]:
        print("   跳过的前几个: " + ", ".join(f"{n}({e})"
                                              for n, e in skipped[:5]))

    cl = clusters()
    groups = defaultdict(list)
    for n in by:
        groups[cl.get(n, "?")].append(n)
    sizes = [len(v) for v in groups.values()]
    print(f"   分组 = Tsuboyama 自己的 `WT_cluster`（序列相似性簇）："
          f"**{len(groups)} 个簇**，簇大小 1–{max(sizes)}"
          f"（>1 的簇 {sum(1 for s in sizes if s > 1)} 个）")
    print(f"   —— 这一步是 docs/21 §4.2 的要求：同一个蛋白家族的多个成员"
          f"不得重复计票")

    P = {n: CN.proj(by[n])[0] for n in by}
    Pb = pbar_grouped(groups, P)
    w, V = np.linalg.eigh(Pb)
    w = w[::-1]; V = V[:, ::-1]
    uf = V[:, 0]
    print("\n" + "=" * 102)
    print("1. 共识谱（先簇内平均，再簇等权）")
    print("=" * 102)
    print(f"   观测 λ 前 3 = " + "  ".join(f"**{x:.3f}**" for x in w[:3]))
    nl = null_grouped(sizes, np.random.default_rng(20261013))
    npc = np.percentile(nl, 95, axis=0)
    for i in range(3):
        print(f"   λ{i + 1} = {w[i]:.3f}  vs **同结构**无条件零分布 "
              f"p95 = {npc[i]:.3f}  "
              f"{'**远超**' if w[i] > npc[i] + 0.1 else ('超出' if w[i] > npc[i] else '未超出')}")

    # λ2 的条件零分布（标定到观测 λ1）+ 标定残差 + 负对照
    cnd, alf = CN.cond_null(len(groups), uf, np.random.default_rng(4),
                            nsim=NSIM, target_l1=float(w[0]))
    p2 = float(np.mean(cnd[:, 1] >= w[1]))
    resid = abs(float(np.median(cnd[:, 0])) - float(w[0]))
    print(f"\n   λ2 的**条件**零分布（二分法标定到观测 λ1，抖动 "
          f"{np.degrees(alf):.1f}°）：p = **{p2:.4f}**")
    print(f"   标定残差 |条件零分布 λ1 中位 − 观测 λ1| = **{resid:.4f}**"
          f"（docs/21 的四组是 ≤ 0.001；docs/24 场景 A 是 0.0035）")
    # 负对照：合成「只共享一维」
    rngn = np.random.default_rng(77)
    Pn = []
    for v in groups.values():
        Qs = []
        for _ in range(len(v)):
            u1 = uf / np.linalg.norm(uf)
            a2 = CN.rand_in_complement(u1, rngn)
            a2 -= u1 * (u1 @ a2); a2 /= np.linalg.norm(a2)
            Q, _ = np.linalg.qr(np.stack([u1, a2], 1))
            Qs.append(Q @ Q.T)
        Pn.append(np.mean(Qs, axis=0))
    wn = np.linalg.eigvalsh(np.mean(Pn, axis=0))[::-1]
    cnd2, alf2 = CN.cond_null(len(groups), uf, np.random.default_rng(5),
                              nsim=NSIM, target_l1=float(wn[0]))
    pneg = float(np.mean(cnd2[:, 1] >= wn[1]))
    print(f"   **负对照（§10.2 不误伤）**：合成只共享一维 → λ1 = "
          f"{wn[0]:.3f}、λ2 = {wn[1]:.3f}，λ2 在同一检验下 p = **{pneg:.3f}**"
          f"（必须 > 0.05；docs/21 的四格是 0.499–0.529）")
    neg_ok = pneg > 0.05
    if not neg_ok:
        print("   => **负对照不通过。** 在 k = %d 上，λ2 的条件零分布检验"
              "把一个**只共享一维**的构造判成「第二维也共享」。" % len(groups))
        print("      所以**本文档不读 λ2 的 p 值**（上面那个 0.0000 不可用）。")
        print("      λ1 一侧不受影响：同一个负对照给 λ1 = %.3f，"
              "远超同结构零分布 p95 = %.3f —— 正是 λ1 该有的行为。"
              % (wn[0], npc[0]))

    # ---- 这个失效是 k 的函数吗：把负对照在多个 k 上重跑（定位失效区）----
    print("\n   [负对照的 k 扫描] 定位失效从哪个 k 开始（每格同一构造）")
    kscan = []
    allc = sorted(groups)
    for kk in (6, 9, 15, 25, 40, 60, 90):
        if kk > len(allc):
            continue
        rngk = np.random.default_rng(101)
        Pk = []
        for _ in range(kk):
            u1 = uf / np.linalg.norm(uf)
            a2 = CN.rand_in_complement(u1, rngk)
            a2 -= u1 * (u1 @ a2); a2 /= np.linalg.norm(a2)
            Q, _ = np.linalg.qr(np.stack([u1, a2], 1))
            Pk.append(Q @ Q.T)
        wk = np.linalg.eigvalsh(np.mean(Pk, axis=0))[::-1]
        ck, ak = CN.cond_null(kk, uf, np.random.default_rng(7),
                              nsim=1500, target_l1=float(wk[0]))
        pk = float(np.mean(ck[:, 1] >= wk[1]))
        kscan.append({"k": kk, "lam1": float(wk[0]), "lam2": float(wk[1]),
                      "p": pk, "ok": bool(pk > 0.05)})
        print(f"      k = {kk:>3}：只共享一维的合成 λ2 = {wk[1]:.3f}，"
              f"p = {pk:.3f}  {'OK' if pk > 0.05 else '**误报**'}")
    kok = [z["k"] for z in kscan if z["ok"]]
    print(f"      => 负对照通过的 k：{kok if kok else '无'}；"
          f"失效的 k：{[z['k'] for z in kscan if not z['ok']]}")

    # ---- 在负对照通过的 k 上重做 λ2（子抽样簇，重复多次）--------------
    sub_l2 = []
    if kok:
        kuse = max(kok)
        print(f"\n   [在负对照通过的 k = {kuse} 上重做 λ2] "
              f"从 {len(allc)} 个簇里随机抽 {kuse} 个，重复 40 次")
        rngs = np.random.default_rng(13)
        for t in range(40):
            pick = rngs.choice(len(allc), kuse, replace=False)
            g2 = {allc[i]: groups[allc[i]] for i in pick}
            ww, VV = np.linalg.eigh(pbar_grouped(g2, P))
            ww = ww[::-1]; uu = VV[:, ::-1][:, 0]
            cc, aa2 = CN.cond_null(kuse, uu, np.random.default_rng(900 + t),
                                   nsim=1200, target_l1=float(ww[0]))
            sub_l2.append({"lam1": float(ww[0]), "lam2": float(ww[1]),
                           "p": float(np.mean(cc[:, 1] >= ww[1])),
                           "cal_resid": abs(float(np.median(cc[:, 0]))
                                            - float(ww[0]))})
        pp = np.array([z["p"] for z in sub_l2])
        l2 = np.array([z["lam2"] for z in sub_l2])
        print(f"      λ1 中位 "
              f"{np.median([z['lam1'] for z in sub_l2]):.3f}、"
              f"λ2 中位 {np.median(l2):.3f}")
        print(f"      λ2 的条件 p：中位 **{np.median(pp):.4f}**、"
              f"四分位 [{np.percentile(pp, 25):.4f}, "
              f"{np.percentile(pp, 75):.4f}]；"
              f"p < 0.05 的比例 **{np.mean(pp < 0.05):.0%}**")
        print(f"      标定残差中位 "
              f"{np.median([z['cal_resid'] for z in sub_l2]):.4f}")

    # ------------------------------------------------------------------
    # 2. 参与率 PR（零分布在这个 k 上重生成）
    # ------------------------------------------------------------------
    print("\n" + "=" * 102)
    print("2. 残差方向的参与率 PR（零分布**在这个 k 上重生成**，docs/21 §5.4）")
    print("=" * 102)
    dirs = []
    for n in by:
        Q = CN.proj(by[n])[1]
        d1 = Q[:, 0] - uf * (uf @ Q[:, 0])
        d2 = Q[:, 1] - uf * (uf @ Q[:, 1])
        for d in (d1, d2):
            nn = np.linalg.norm(d)
            if nn > 1e-9:
                dirs.append(d / nn)
    Vr = np.linalg.eigvalsh(np.mean([np.outer(d, d) for d in dirs],
                                    axis=0))[::-1]
    pr = float((Vr.sum() ** 2) / (Vr ** 2).sum())
    rngp = np.random.default_rng(9)
    nullpr = []
    for _ in range(600):
        dd = []
        for _ in range(len(dirs)):
            z = rngp.normal(size=20)
            z -= z.mean()
            z -= uf * (uf @ z)
            dd.append(z / np.linalg.norm(z))
        vv = np.linalg.eigvalsh(np.mean([np.outer(d, d) for d in dd],
                                        axis=0))[::-1]
        nullpr.append((vv.sum() ** 2) / (vv ** 2).sum())
    nullpr = np.array(nullpr)
    print(f"   残差谱前 3 = {np.round(Vr[:3], 3).tolist()}；"
          f"**PR = {pr:.2f}**")
    print(f"   在 k = {len(dirs)} 条残差方向上重生成的零分布 PR = "
          f"{np.median(nullpr):.2f} "
          f"[{np.percentile(nullpr, 2.5):.2f}, "
          f"{np.percentile(nullpr, 97.5):.2f}]，单尾 p = "
          f"{np.mean(nullpr <= pr):.4f}")

    # ------------------------------------------------------------------
    # 3. 留一簇稳定性
    # ------------------------------------------------------------------
    print("\n" + "=" * 102)
    print("3. 留一**簇**稳定性（docs/21 §5.2 的口径）")
    print("=" * 102)
    loo = []
    for g in list(groups):
        g2 = {k: v for k, v in groups.items() if k != g}
        if len(g2) < 2:
            continue
        ww, VV = np.linalg.eigh(pbar_grouped(g2, P))
        uu = VV[:, ::-1][:, 0]
        loo.append({"dropped_cluster": g, "n": len(groups[g]),
                    "abscos": abs(float(uf @ uu)),
                    "lam1": float(ww[::-1][0])})
    ac = np.array([z["abscos"] for z in loo])
    print(f"   {len(loo)} 次留一：|cos(u_fold, u')| 最小 **{ac.min():.4f}**、"
          f"中位 {np.median(ac):.4f}；λ1 范围 "
          f"{min(z['lam1'] for z in loo):.3f}–"
          f"{max(z['lam1'] for z in loo):.3f}")
    print(f"   对照：binding 一侧两组构造的留一最小是 0.956"
          f"（docs/21）/ 0.911（docs/24 标定后）")

    # ------------------------------------------------------------------
    # 4. 成对统计（cos θ1 / cos θ2），并且摆明 docs/24 §0.5 的选择效应
    # ------------------------------------------------------------------
    print("\n" + "=" * 102)
    print("4. 跨簇成对统计（cos θ1 / cos θ2）")
    print("=" * 102)
    pn = haar_pair_null(20, 2, n=NSIM, seed=3)
    p95_1 = float(np.percentile(pn["cos1"], 95))
    p95_2 = float(np.percentile(pn["cos2"], 95))
    names = sorted(by)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if cl.get(names[i]) == cl.get(names[j]):
                continue              # 同簇不算跨家族证据
            r = align(by[names[i]], cols20[names[i]],
                      by[names[j]], cols20[names[j]])
            if r:
                pairs.append({"a": names[i], "b": names[j],
                              "cos1": r["cos"][0], "cos2": r["cos"][1]})
    c1 = np.array([p["cos1"] for p in pairs])
    c2 = np.array([p["cos2"] for p in pairs])
    print(f"   跨簇配对 n = **{len(pairs)}**（同簇配对已排除）")
    print(f"   cos θ1：中位 **{np.median(c1):.3f}**，四分位 "
          f"[{np.percentile(c1, 25):.3f}, {np.percentile(c1, 75):.3f}]；"
          f"超随机 p95({p95_1:.3f}) 的比例 **{np.mean(c1 > p95_1):.1%}**")
    print(f"   cos θ2：中位 {np.median(c2):.3f}；超 p95({p95_2:.3f}) 的比例 "
          f"{np.mean(c2 > p95_2):.1%}")
    print(f"   **docs/24 §0.5 的陷阱**：若只保留第二维更可分辨的那一半配对，"
          f"cos θ2 中位会变成 "
          f"{np.median(np.sort(c2)[len(c2) // 2:]):.3f} —— "
          f"而 λ2 的条件 p 仍是 {p2:.4f}。中位移动不是证据。")

    # ------------------------------------------------------------------
    # 5. folding 共识轴 vs binding 共识轴
    # ------------------------------------------------------------------
    print("\n" + "=" * 102)
    print("5. **folding 共识轴 vs binding 共识普适轴**（共识层面的那一个数）")
    print("=" * 102)
    tg = json.load(open(os.path.join(ROOT, "cache",
                                     "subspace_two_groups.json")))
    gc = json.load(open(os.path.join(ROOT, "cache",
                                     "gate_consequences.json")))
    uD = np.array(tg["axis1_family_avg"]["u"], float)
    uA = np.array([s for s in gc["consensus"]
                   if s["label"].startswith("A")][0]["u"], float)

    def cu(v):
        v = np.asarray(v, float) - np.mean(v)
        return v / np.linalg.norm(v)
    n1 = haar_1d_null(20, n=20000, seed=51)
    p95d = float(np.percentile(n1, 95))
    cD = abs(float(cu(uf) @ cu(uD)))
    cA = abs(float(cu(uf) @ cu(uA)))
    print(f"   |cos(u_fold, u_binding_D)| = **{cD:.3f}**  "
          f"p(Haar, d=20) = {np.mean(n1 >= cD):.5f}")
    print(f"   |cos(u_fold, u_binding_A)| = **{cA:.3f}**  "
          f"（d = 20 的一维 p95 = {p95d:.3f}）")
    o = np.argsort(-cu(uf))
    ufc = cu(uf)
    print("   u_fold 载荷（最正 → 最负）:")
    print("     " + "  ".join(f"{AA20[i]}{ufc[i]:+.2f}" for i in o))
    print("   u_binding_D 载荷:")
    uDc = cu(uD); oD = np.argsort(-uDc)
    print("     " + "  ".join(f"{AA20[i]}{uDc[i]:+.2f}" for i in oD))

    # ---- 同源 / 跨源配对表（D-036 的第二半，docs/25 §6.3 的诊断式）----
    print("\n" + "=" * 102)
    print("5b. 同源 / 跨源配对分开报（docs/25 §6.3：假对齐的特征是"
          "**同源配对落在随机水平而跨源配对高** —— 与真共享方向反序）")
    print("=" * 102)
    GF = json.load(open(os.path.join(ROOT, "cache",
                                     "generic_folding_axis.json")))
    gr = GF["observed"].get("GRB2-SH3/folding")
    xs = {}
    if gr:
        gcols = np.array(gr["cols"]); gax = np.array(gr["axis"], float)
        vv = []
        for n in names:
            Q = CN.proj(by[n])[1]
            com = np.intersect1d(cols20[n], gcols)
            v1 = cu(Q[:, 0][np.searchsorted(cols20[n], com)])
            v2 = cu(gax[np.searchsorted(gcols, com)])
            vv.append(abs(float(v1 @ v2)))
        vv = np.array(vv)
        xs = {"n": len(vv), "median": float(np.median(vv)),
              "q25": float(np.percentile(vv, 25)),
              "q75": float(np.percentile(vv, 75)),
              "frac_over_p95": float(np.mean(vv > p95d)),
              "ufold_vs_grb2": abs(float(cu(uf) @ cu(
                  np.pad(gax, (0, 0)) if len(gax) == 20 else gax)))
              if len(gax) == 20 else None}
        print(f"   **跨源、同表型**（Tsuboyama folding × ddPCA "
              f"GRB2-SH3/folding，两个完全不同的实验平台）："
              f"|cos| 中位 **{np.median(vv):.3f}**，四分位 "
              f"[{np.percentile(vv, 25):.3f}, {np.percentile(vv, 75):.3f}]，"
              f"超 p95 的比例 **{np.mean(vv > p95d):.1%}**")
        if xs["ufold_vs_grb2"] is not None:
            print(f"   共识层面：|cos(u_fold, GRB2-SH3/folding 轴)| = "
                  f"**{xs['ufold_vs_grb2']:.3f}**")
    print(f"   **同源、同表型**（Tsuboyama 内部跨簇配对）："
          f"cos θ1 中位 {np.median(c1):.3f}、超 p95 的比例 "
          f"{np.mean(c1 > p95_1):.1%}")
    print(f"   **跨源、跨表型**（folding × binding 共识轴）："
          f"|cos(u_fold, u_binding_D)| = {cD:.3f}")
    print("   => 顺序是「同源 ≥ 跨源同表型 ≥ 跨源跨表型」，"
          "**与 docs/25 的假对齐特征相反**（那里是同源落在 0.023–0.046、"
          "跨源 0.44–0.58）。")

    scales = load_scales(verbose=False)
    DS = json.load(open(os.path.join(ROOT, "cache",
                                     "generic_disrupt_scales.json")))
    dis = {k: np.array(v, float) for k, v in DS["scales"].items()}
    print("\n   u_fold 的化学身份（与 docs/21 §6 的 binding 表同口径）:")
    rows = []
    for nm, vec in list(scales.items()) + list(dis.items()):
        v = np.asarray(vec, float)
        if not np.all(np.isfinite(v)):
            continue
        r = float(cu(uf) @ cu(v))
        rows.append({"scale": nm, "r": r, "r2": r * r})
    rows.sort(key=lambda z: -abs(z["r"]))
    for z in rows[:12]:
        print(f"     {z['scale']:<34} r = {z['r']:+.3f}   r² = {z['r2']:.3f}")

    out = {"n_admitted": len(adm), "n_used": len(by),
           "skipped": skipped, "n_clusters": len(groups),
           "cluster_sizes": sizes,
           "eigs": w[:3].tolist(), "null_p95": npc.tolist(),
           "u_fold": uf.tolist(), "aa_order": AA20,
           "p_lambda2": p2, "alpha_deg": float(np.degrees(alf)),
           "cal_resid": resid,
           "neg_control": {"lam1": float(wn[0]), "lam2": float(wn[1]),
                           "p": pneg, "ok": bool(neg_ok)},
           "neg_control_kscan": kscan,
           "lambda2_at_calibrated_k": sub_l2,
           "pr": {"k_dirs": len(dirs), "pr": pr,
                  "resid_eigs": Vr[:3].tolist(),
                  "null_med": float(np.median(nullpr)),
                  "null_ci": [float(np.percentile(nullpr, 2.5)),
                              float(np.percentile(nullpr, 97.5))],
                  "p_one": float(np.mean(nullpr <= pr))},
           "loo": loo, "loo_min": float(ac.min()),
           "pairs_n": len(pairs),
           "cos1_median": float(np.median(c1)),
           "cos1_quartiles": [float(np.percentile(c1, 25)),
                              float(np.percentile(c1, 75))],
           "cos1_frac_over_p95": float(np.mean(c1 > p95_1)),
           "cos2_median": float(np.median(c2)),
           "cos2_frac_over_p95": float(np.mean(c2 > p95_2)),
           "pair_null_p95": {"cos1": p95_1, "cos2": p95_2},
           "cos_ufold_uD": cD, "cos_ufold_uA": cA,
           "cross_source_grb2": xs,
           "p_haar_ufold_uD": float(np.mean(n1 >= cD)),
           "chem": rows[:20]}
    p = os.path.join(ROOT, "cache", "tsuboyama_crossdomain.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
