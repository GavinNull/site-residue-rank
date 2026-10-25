#!/usr/bin/env python3
"""拿掉 OMTKY3 的受控计算：144 个折叠域对「只含 RBD 家族」共识轴的对齐。

README / docs/48 §2.4 的推断：8 成员共识轴与折叠的中位 0.807，低于 RBD 家族自己的 0.983，
「似乎」来自家族等权纳入的 OMTKY3。此前没有做把 OMTKY3 单独拿掉的受控计算。

### 事先写下的预测（跑之前落盘）
  预测：若推断成立，只含 5 个冠状病毒的共识轴 u_C（gate_consequences 的 C 构造）对 144 域的
  |cos| 中位应高于 8 成员 u_A 的 0.807，且 bootstrap 区间不与 [0.784, 0.828] 重叠。
  证伪：u_C 的中位 ≤ 0.807，或区间重叠。
  地位：探索层，单次对比，不改动任何预先登记层判决；不据它改写 docs/48 的结论句。

### 自检（不过则中止）
  用缓存里每域存的 `axis`/`cols` 与 u_A 重算 cos，必须逐域复现缓存的 `cos_uA`（容差 1e-6），
  并复现 8 成员中位 0.807。这证明本脚本读对了轴、对对了位置，之后的 u_C 才可信。
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "cache", "tsuboyama_axis.json"), encoding="utf-8"))
g = json.load(open(os.path.join(ROOT, "cache", "gate_consequences.json"), encoding="utf-8"))
AA = json.load(open(os.path.join(ROOT, "cache", "subspace_two_groups.json"), encoding="utf-8"))["aa_order"]
rows, ok = d["rows"], set(d["repro_pass"])
keys = [k for k in rows if k in ok]
assert len(keys) == 144


def cen_unit(v, cols=None):
    v = np.asarray(v, float)
    if cols is not None:
        v = v[np.asarray(cols, int)]
    v = v - v.mean()
    return v / np.linalg.norm(v)


def cons(prefix):
    return np.array([c for c in g["consensus"] if c["label"].startswith(prefix)][0]["u"], float)


def cosines(u):
    return np.array([abs(float(cen_unit(rows[k]["axis"]) @ cen_unit(u, rows[k]["cols"]))) for k in keys])


def boot(v, n=4000):
    rng = np.random.default_rng(0)
    b = np.array([np.median(rng.choice(v, len(v))) for _ in range(n)])
    return float(np.median(v)), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


# ---- 自检：复现缓存的 cos_uA 与 0.807 ----
uA = cons("A")
cA = cosines(uA)
cache_cA = np.array([abs(rows[k]["cos_uA"]) for k in keys])
if np.max(np.abs(cA - cache_cA)) > 1e-6:
    print("自检失败：重算的 cos_uA 与缓存不一致，最大差", float(np.max(np.abs(cA - cache_cA))))
    sys.exit(2)
mA, ciA = boot(cA)
if abs(mA - 0.807) > 0.0006:
    print(f"自检失败：8 成员中位 {mA:.4f} ≠ 0.807"); sys.exit(2)
print(f"自检通过：逐域复现 cos_uA（最大差 < 1e-6），8 成员中位 {mA:.3f} [{ciA[0]:.3f}, {ciA[1]:.3f}]")

# ---- 受控对比 ----
p95 = np.array([rows[k]["p95_d"] for k in keys])
out = {"prediction_written_before_run": True}
for lab, pre in (("A. 8 成员（5 冠状病毒 + 3 OMTKY3）", "A"),
                 ("C. 只含 5 个冠状病毒（拿掉 OMTKY3）", "C")):
    v = cosines(cons(pre))
    m, ci = boot(v)
    # 配对差（同一批域）：C − A 的中位与区间
    out[pre] = {"median": m, "ci": ci, "n_over_p95": int((v > p95).sum())}
    print(f"{lab}：中位 {m:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  超自身 p95 {int((v > p95).sum())}/144")
vC = cosines(cons("C"))
diff = vC - cA
rng = np.random.default_rng(1)
bd = np.array([np.median(rng.choice(diff, len(diff))) for _ in range(4000)])
out["paired_diff_C_minus_A"] = {"median": float(np.median(diff)),
                                "ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                                "frac_C_greater": float((diff > 0).mean())}
print(f"配对差 C−A（同一批域）：中位 {np.median(diff):+.3f} "
      f"[{np.percentile(bd,2.5):+.3f}, {np.percentile(bd,97.5):+.3f}]  C>A 的域占 {(diff>0).mean():.0%}")
print(f"u_C 与 u_A 夹角余弦：{abs(float(cen_unit(cons('C')) @ cen_unit(uA))):.4f}")
json.dump(out, open(os.path.join(ROOT, "cache", "fold_align_loo_omty3.json"), "w"), indent=1)
