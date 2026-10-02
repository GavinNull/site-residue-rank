#!/usr/bin/env python3
"""G3 盲预测打分：方向准确率 + 因子分层 + 置信度校准，全部带 bootstrap 区间。

只在预测锁定后可运行。输出聚合统计，默认不打印单对标签。

用法:
  python3 analysis/score_blind.py
  python3 analysis/score_blind.py --per-pair     # 逐对对错（仅 postmortem 用）
"""
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labelstore  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "predictions")

# docs/04-blind-protocol.md 预注册门槛，不得事后调整
GATE = {
    "min_stratum_n": 15,
    "stratum_lower_bound": 0.60,
    "n_boot": 5000,
}


def load_yaml(path):
    """极简 yaml 读取：本仓库预测文件只用平坦 key: value 与简单列表。"""
    d = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                inner = v[1:-1].strip()
                d[k] = [x.strip().strip('"\'') for x in inner.split(",") if x.strip()]
            else:
                d[k] = v.strip('"\'')
    return d


def boot_ci(hits, n_boot=None, alpha=0.05):
    """hits: 0/1 列表。返回 (点估计, 下界, 上界)。"""
    n_boot = n_boot or GATE["n_boot"]
    n = len(hits)
    if n == 0:
        return (float("nan"),) * 3
    point = sum(hits) / n
    rng = random.Random(20261003)
    means = []
    for _ in range(n_boot):
        s = sum(hits[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return point, lo, hi


def fmt(t):
    p, lo, hi = t
    return f"{p:.3f}  [{lo:.3f}, {hi:.3f}]"


def main():
    per_pair = "--per-pair" in sys.argv
    locked = labelstore.locked_pairs()
    files = sorted(glob.glob(os.path.join(PRED, "*.yaml")))
    files = [p for p in files if not os.path.basename(p).startswith("_")]

    records = []
    skipped = []
    for p in files:
        pr = load_yaml(p)
        pid = pr.get("pair_id")
        if pid not in locked:
            skipped.append((pid, "未锁定"))
            continue
        lab = labelstore.get(pid)
        if lab is None:
            skipped.append((pid, "无标签"))
            continue
        if abs(lab["abs_ddg_kcal"]) < 0.5:
            skipped.append((pid, "indistinguishable (<0.5 kcal/mol)"))
            continue
        try:
            conf = float(pr.get("confidence", 0.5))
        except ValueError:
            conf = 0.5
        records.append({
            "pair_id": pid,
            "hit": 1 if pr.get("direction") == lab["stronger"] else 0,
            "factor": pr.get("dominant_factor", "UNSET"),
            "conf": conf,
            "pred_bin": pr.get("magnitude_bin", ""),
            "true_mag": abs(lab["abs_ddg_kcal"]),
        })

    if not records:
        print("没有可打分的记录。")
        if skipped:
            print("跳过：")
            for pid, why in skipped:
                print(f"  {pid}: {why}")
        return

    hits = [r["hit"] for r in records]
    print("=" * 62)
    print(f"G3 盲预测打分   n = {len(records)}   "
          f"(跳过 {len(skipped)})   labels fp={labelstore.fingerprint()}")
    print("=" * 62)
    print(f"\n[1] 总方向准确率    {fmt(boot_ci(hits))}")
    print(f"    随机基线        0.500")

    print("\n[2] 按 dominant_factor 分层")
    by = {}
    for r in records:
        by.setdefault(r["factor"], []).append(r["hit"])
    usable = []
    for fac, hh in sorted(by.items(), key=lambda kv: -len(kv[1])):
        t = boot_ci(hh)
        mark = ""
        if len(hh) >= GATE["min_stratum_n"] and t[1] > GATE["stratum_lower_bound"]:
            mark = "  <= 进 G4 候选"
            usable.append(fac)
        elif len(hh) < GATE["min_stratum_n"]:
            mark = "  (n 不足，不判)"
        print(f"    {fac:<22} n={len(hh):<4} {fmt(t)}{mark}")

    print("\n[3] 置信度校准")
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    curve = []
    for lo, hi in bins:
        hh = [r["hit"] for r in records if lo <= r["conf"] < hi]
        if not hh:
            print(f"    [{lo:.1f},{hi:.1f})  n=0")
            continue
        t = boot_ci(hh)
        curve.append((lo, t[0]))
        print(f"    [{lo:.1f},{hi:.1f})  n={len(hh):<4} {fmt(t)}")
    monotone = all(curve[i][1] <= curve[i + 1][1] + 1e-9
                   for i in range(len(curve) - 1)) if len(curve) > 1 else None
    print(f"    单调性: {monotone}")

    print("\n[4] 量级分箱命中（非主指标，测量级直觉）")
    def tb(x):
        return "small" if x < 1 else ("medium" if x < 2 else "large")
    mh = [1 if r["pred_bin"] == tb(r["true_mag"]) else 0
          for r in records if r["pred_bin"]]
    if mh:
        print(f"    bin accuracy    n={len(mh):<4} {fmt(boot_ci(mh))}  (随机 ≈0.333)")
    else:
        print("    未填 magnitude_bin")

    print("\n[5] 预注册门槛（docs/04-blind-protocol.md）")
    tot = boot_ci(hits)
    print(f"    (a) 总准确率下界 > 基线上界      下界={tot[1]:.3f}  "
          f"<- 需与 G2 基线墙区间比较，见 state/gates.md")
    print(f"    (b) ≥2 个因子层可用              当前 {len(usable)} 个: {usable}")
    print(f"    (c) 校准曲线单调且高置信优于均值  单调={monotone}")

    if per_pair:
        print("\n[postmortem] 逐对")
        for r in records:
            print(f"    {r['pair_id']:<12} hit={r['hit']} "
                  f"factor={r['factor']:<20} conf={r['conf']}")

    out = os.path.join(ROOT, "state", "g3_report.json")
    with open(out, "w") as f:
        json.dump({"n": len(records), "overall": tot,
                   "usable_factors": usable, "calibrated": monotone,
                   "labels_fingerprint": labelstore.fingerprint()}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n写入 {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
