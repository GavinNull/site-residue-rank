#!/usr/bin/env python3
"""Part 1：把 G4 闸（bootstrap `stab cos2 >= 0.60`）的**秩 2 纯噪声下限**
逐矩阵标定出来。docs/23 §5.3/§9.2 登记的那条待办，人类已授权执行。

闸的定义（docs/13 §4.3、`subspace_universality.py`）：
  stab cos2 = median_b cos θ2( span(V_boot_b), span(V_全数据) )，秩 2。
docs/23 只标定了**秩 1**的下限/上限，而 G4 实际用的是秩 2 的那个量。

对每个成员矩阵，在它**自己的几何**（位点数、残基数、逐行 mask 模式）上：
  下限 = 纯噪声（**无任何低秩结构**）的 stab cos2「每次实现的中位」的分布；
  上限 = 已知秩 2 真结构、噪声调到匹配该矩阵自己的秩 2 解释率时的同一量。
下限的 p95 就是**几何相关的门槛**，用来替换固定的 0.60。

口径声明（照抄 docs/23 §5.2，不许偷换）：
闸用的是每个矩阵的中位数，所以零分布必须是「每次噪声实现的中位数」的分布，
**不是**把全部 bootstrap 抽样混在一起 —— 混在一起会给出宽到无意义的区间，
看起来像「实测落在区间内」。

前置：`analysis/gate_harness_check.py` 必须先通过（本脚本会强制调用判定）。
"""
import json
import os
import sys
from multiprocessing import Pool, cpu_count

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_calib_lib as G  # noqa: E402
from subspace_lib import ci  # noqa: E402

N_REAL = 120        # 噪声/真结构的实现次数（决定 p95/p99 的分辨率）
N_BOOT = 200        # 每次实现内部的 bootstrap 次数（与 docs/23 §5.2 同）
N_BOOT_OBS = 2000   # 实测值用 docs/13 的口径（2000 次）
GAP_MIN = 0.10      # docs/23 §5.2 的判别力阈值，原样沿用
STAB_FLAT = 0.60    # 现行固定门槛


def harness_gate():
    p = os.path.join(G.ROOT, "cache", "gate_harness_check.json")
    if not os.path.exists(p):
        raise RuntimeError("先跑 analysis/gate_harness_check.py")
    J = json.load(open(p))
    if not J.get("ok"):
        raise RuntimeError("harness 自检未通过 —— 按 CLAUDE.md §10 中止")
    print("[前置] gate_harness_check.json: 通过"
          "（秩 1 下限/上限复现 docs/23 §5.2）")
    return J


_MATS = None


def _mats():
    """进程内缓存。multiprocessing 的每个 worker 各加载一次。"""
    global _MATS
    if _MATS is None:
        _MATS = G.load_evidence()
    return _MATS


def one_matrix(arg):
    """一个矩阵的全部标定（供 Pool.map 用；种子全固定，结果与串行一致）。"""
    nm, pubv = arg
    d = _mats()[nm]
    shape, mask = d["M"].shape, d["mask"]
    e2, _err = G.rankr_expl(d["M"], mask, rank=2)
    obs = G.stab_boot(d["M"], mask, rank=2, nboot=N_BOOT_OBS, seed=7)
    om = float(np.median(obs)) if obs else float("nan")
    flo = G.floor_dist(shape, mask, rank=2, n_real=N_REAL, nboot=N_BOOT)
    fp = G.pcts(flo)
    sg, sgtab = G.match_sigma(shape, mask, 2, e2)
    hi = G.ceiling_dist(shape, mask, rank=2, sigma=sg, n_real=N_REAL,
                        nboot=N_BOOT)
    hm = float(np.median(hi))
    return {"name": nm, "shape": list(shape), "n_obs": int(mask.sum()),
            "expl_rank2": e2, "stab_obs": om, "stab_pub": pubv,
            "floor_med": fp[50], "floor_p90": fp[90], "floor_p95": fp[95],
            "floor_p99": fp[99], "floor_ci": list(ci(flo)),
            "n_floor": len(flo), "ceiling_med": hm,
            "ceiling_ci": list(ci(hi)), "n_ceiling": len(hi),
            "sigma_matched": sg, "sigma_table": sgtab,
            "gap": hm - fp[50]}


def one_sens(arg):
    nm, nr_, nb_ = arg
    d = _mats()[nm]
    f = G.floor_dist(d["M"].shape, d["mask"], rank=2, n_real=nr_, nboot=nb_)
    return (nm, nr_, nb_, float(np.percentile(f, 95)))


def cached_stab():
    """docs/13 / docs/21 口径下的实测 stab cos2（2000 次 bootstrap）。"""
    U = json.load(open(os.path.join(G.ROOT, "cache",
                                    "subspace_universality.json")))
    stab = {}
    for r in U["focus_pairs"]:
        for side in ("a", "b"):
            if r.get(f"stab_cos2_{side}") is not None:
                stab[r[side]] = r[f"stab_cos2_{side}"]
    return stab


def main():
    harness_gate()
    mats = G.load_evidence()
    pub = cached_stab()

    n2 = G.null2(20)
    print(f"\n量级校准：秩 2 随机零分布（d = 20）cos2 中位 "
          f"{np.median(n2['cos2']):.3f}、p95 {np.percentile(n2['cos2'], 95):.3f}；"
          f"cos1 中位 {np.median(n2['cos1']):.3f}、p95 "
          f"{np.percentile(n2['cos1'], 95):.3f}")
    print("  注意：bootstrap stab 的纯噪声下限**与这个零分布无关**。")
    print("  stab 不是「两个独立随机面」的比较，而是「同一份噪声的重抽 vs 自己」，")
    print("  噪声里的偶然结构会被两边同时看到 —— 这正是固定 0.60 失效的机制。")

    print("\n" + "=" * 112)
    print(f"Part 1：秩 2 `stab cos2` 的逐矩阵下限 / 上限"
          f"（{N_REAL} 次实现 × {N_BOOT} 次 bootstrap；实测 {N_BOOT_OBS} 次）")
    print("=" * 112)
    hdr = (f"{'体系':<40}{'位点':>5}{'观测':>6}{'秩2解释':>8}"
           f"{'实测stab':>9}{'下限中位':>9}{'p90':>7}{'p95':>7}{'p99':>7}"
           f"{'上限中位':>9}{'gap':>8}")
    print(hdr)
    print("-" * 112)
    with Pool(min(cpu_count(), 9)) as po:
        rows = po.map(one_matrix, [(nm, pub.get(nm)) for nm in G.EVIDENCE9])
    for r in rows:
        print(f"{r['name'][:39]:<40}{r['shape'][0]:>5}{r['n_obs']:>6}"
              f"{r['expl_rank2']:>8.1%}{r['stab_obs']:>9.3f}"
              f"{r['floor_med']:>9.3f}{r['floor_p90']:>7.3f}"
              f"{r['floor_p95']:>7.3f}{r['floor_p99']:>7.3f}"
              f"{r['ceiling_med']:>9.3f}{r['gap']:>+8.3f}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 112)
    print("逐矩阵判读（两步：先问闸在该几何上有没有判别力，再问实测是否过闸）")
    print("=" * 112)
    print(f"  判别力阈值：上限中位 − 下限中位 >= {GAP_MIN}（docs/23 §5.2 原样沿用）")
    print(f"  几何相关门槛：下限 p95（替换固定的 {STAB_FLAT}）\n")
    print(f"  {'体系':<40}{'实测':>8}{'固定0.60':>9}{'下限p95':>9}"
          f"{'新门槛':>8}  判读")
    print("  " + "-" * 108)
    for r in rows:
        readable = np.isfinite(r["gap"]) and r["gap"] >= GAP_MIN
        pass_flat = r["stab_obs"] >= STAB_FLAT
        pass_geo = r["stab_obs"] > r["floor_p95"]
        if not readable:
            v = (f"**闸无判别力（gap = {r['gap']:+.3f}）—— 实测不可读**")
        elif pass_geo:
            v = "过新门槛（实测超纯噪声 p95）"
        else:
            v = "**不过新门槛 —— 与纯噪声不可区分**"
        r["readable"] = bool(readable)
        r["pass_flat"] = bool(pass_flat)
        r["pass_geo"] = bool(pass_geo)
        r["verdict"] = v
        print(f"  {r['name'][:39]:<40}{r['stab_obs']:>8.3f}"
              f"{'过' if pass_flat else '不过':>9}{r['floor_p95']:>9.3f}"
              f"{'过' if pass_geo else '**不过**':>8}  {v}")

    fails = [r for r in rows if not (r["readable"] and r["pass_geo"])]
    print(f"\n  固定 0.60 下通过: {sum(r['pass_flat'] for r in rows)}/9")
    print(f"  几何相关门槛（下限 p95）下通过: "
          f"{sum(r['pass_geo'] and r['readable'] for r in rows)}/9")
    if fails:
        print("  **失败的成员**：")
        for r in fails:
            print(f"     {r['name']}  实测 {r['stab_obs']:.3f} "
                  f"vs 下限 p95 {r['floor_p95']:.3f}  ({r['verdict']})")

    # ------------------------------------------------------------------
    # 参数敏感性（CLAUDE.md §5：记录搜索量并给不敏感性或敏感性）
    # ------------------------------------------------------------------
    print("\n" + "=" * 112)
    print("参数敏感性（§5）：下限 p95 对 (实现次数, bootstrap 次数) 的依赖")
    print("=" * 112)
    probe = ["OMTKY3/SGPB/3SGB", "OMTKY3/chymotrypsin/1CHO",
             "starr2020/WH1/huACE2"]
    variants = [(24, 200), (60, 200), (120, 200), (60, 100), (60, 400)]
    print(f"  {'体系':<40}" + "".join(f"{f'{a}x{b}':>11}"
                                      for a, b in variants))
    print("  " + "-" * 108)
    jobs = [(nm, a, b) for nm in probe for a, b in variants]
    with Pool(min(cpu_count(), len(jobs))) as po:
        res = po.map(one_sens, jobs)
    tab = {(nm, a, b): v for nm, a, b, v in res}
    sens = []
    for nm in probe:
        vals = [tab[(nm, a, b)] for a, b in variants]
        print(f"  {nm[:39]:<40}" + "".join(f"{v:>11.3f}" for v in vals))
        sens.append({"name": nm, "variants": variants, "floor_p95": vals,
                     "spread": float(max(vals) - min(vals))})
    print(f"\n  变体数 = {len(variants)}（实现次数 3 档 + bootstrap 次数 3 档，"
          f"共享 60x200 这一格）。")
    mx = max(s["spread"] for s in sens)
    print(f"  下限 p95 在全部变体上的最大跨度 = {mx:.3f}。")
    print(f"  {'结论对参数不敏感' if mx < 0.05 else '**结论对参数敏感，必须按最保守一格读**'}")

    out = {"n_real": N_REAL, "n_boot": N_BOOT, "n_boot_obs": N_BOOT_OBS,
           "gap_min": GAP_MIN, "stab_flat": STAB_FLAT,
           "null2_cos1_p95": float(np.percentile(n2["cos1"], 95)),
           "null2_cos2_p95": float(np.percentile(n2["cos2"], 95)),
           "rows": rows, "sensitivity": sens,
           "n_pass_flat": int(sum(r["pass_flat"] for r in rows)),
           "n_pass_geo": int(sum(r["pass_geo"] and r["readable"]
                                 for r in rows)),
           "fail_names": [r["name"] for r in fails]}
    p = os.path.join(G.ROOT, "cache", "gate_rank2_calibration.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print("\n产物:", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
