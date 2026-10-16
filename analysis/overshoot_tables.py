#!/usr/bin/env python3
"""docs/43 的表：从 cache/overshoot_*.json 生成可直接贴进文档的 markdown，并可用 --check 核对文档里的表行。
用法：python3 analysis/overshoot_tables.py > cache/overshoot_tables.md
      python3 analysis/overshoot_tables.py --check docs/43-overshoot-binding.md
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
C = lambda f: json.load(open(os.path.join(ROOT, "cache", f)))
noise, harness, bind = C("overshoot_noise.json"), C("overshoot_harness.json"), C("overshoot_binding.json")
sys.path.insert(0, HERE)
import overshoot_lib as OL  # noqa: E402


def short(nm):
    return nm.replace("animalACE2/", "").replace("PDCoV/PDCoV/", "PD-CoV/").replace("SKEMPI/OMTKY3/pooled-4proteases", "OMTKY3/pooled")


def fr(e):
    return f"{e['FR2'][0]}/{e['FR2'][1]}"


def window_of(snr, h):
    """snr 相对该几何窗口：用 harness 扫描里最近的几何不适用，这里只按全局描述返回区间。"""
    return ""


def tables():
    L = []
    L.append("### T1 每个矩阵的噪声与 s/σ（两端）\n")
    L.append("| 矩阵 | 标 | 观测 | cso | σ_meas(格子) | σ_fit1 | σ_meas²/σ_fit1² | s/σ @σ_meas | s/σ @σ_fit1 | 真实检出秩(3种子均值曲线) |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for nm in OL.scope_names():
        r = noise["matrices"][nm]
        p = r["path_k1"]
        a = f"{p[0]['snr']:.2f}" if p else "—"
        b = f"{p[-1]['snr']:.2f}" if p else "—"
        L.append(f"| {short(nm)} | {r['tag'] or ''} | {r['n']} | {r['cso']:.3f} | {r['sigma_lo']:.3f} | {r['sigma_fit1']:.3f} | "
                 f"{(r['sigma_lo']/r['sigma_fit1'])**2:.3f} | {a} | {b} | {r['real_det']} |")
    L.append("")
    L.append("### T2 双向 harness（设计点 s/σ=2.62，σ=0.64，R=40）\n")
    L.append("| 几何 | P(检出=2\\|真2) | P(检出≥2\\|真1) | P(检出=0\\|真0) |")
    L.append("|---|---:|---:|---:|")
    for nm, h in harness["harness"].items():
        f = lambda t: f"{t[0]}/{t[1]}"
        L.append(f"| {short(nm)} | {f(h['hit_k2_is2'])} | {f(h['nofool_k1_ge2'])} | {f(h['null_k0_is0'])} |")
    L.append("")
    L.append("### T3 SNR 扫描（σ=σ_fit1；R=40；格内 = P(检出=2\\|真2) / P(检出≥2\\|真1)）\n")
    snrs = list(harness["sweep"][list(harness["sweep"])[0]].keys())
    snrs = [s for s in snrs if s != "k0_is0"]
    L.append("| 几何 | " + " | ".join(f"s/σ={s}" for s in snrs) + " |")
    L.append("|---|" + "---:|" * len(snrs))
    for nm, d in harness["sweep"].items():
        cells = []
        for s in snrs:
            c2, c1 = d[s]["k2"], d[s]["k1"]
            cells.append(f"{c2[2]/40:.2f} / {(c1[2]+c1[3])/40:.2f}")
        L.append(f"| {short(nm)} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### T4 k=1 路径上的 P(检出≥2\\|真1)（real 载荷, 高斯噪声；p1=σ_meas … p7=σ_fit1；括号 = s/σ）\n")
    L.append("| 矩阵 | 标 | 预先登记判决 | " + " | ".join(f"p{i}" for i in range(1, 8)) + " |")
    L.append("|---|---|---|" + "---:|" * 7)
    for nm in OL.scope_names():
        r = bind[nm]
        if not r.get("path"):
            L.append(f"| {short(nm)} | {r['tag'] or ''} | {r['verdict']} | " + " | ".join(["—"] * 7) + " |")
            continue
        cells = [f"{e['FR2'][0]}/{e['FR2'][1]} ({e['snr']:.1f})" for e in r["path"]]
        L.append(f"| {short(nm)} | {r['tag'] or ''} | {r['verdict']} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### T5 k=2 路径上的 P(检出=2\\|真2) 与 P(检出≥3\\|真2)\n")
    L.append("| 矩阵 | " + " | ".join(f"p{i}" for i in range(1, 8)) + " |")
    L.append("|---|" + "---:|" * 7)
    for nm in OL.scope_names():
        r = bind[nm]
        if not r.get("path2"):
            continue
        L.append(f"| {short(nm)} | " + " | ".join(f"{e['H'][0]}/{e['H'][1]}; ≥3: {e['k2_dist'][3]}/{e['H'][1]} ({e['snr']:.1f})" for e in r["path2"]) + " |")

    # ---------------- 变体（rand 载荷 / wild 噪声）
    L.append("")
    L.append("### T6 变体：随机载荷（rand, p1/p4/p7）与 wild 噪声（p1）下的 P(检出≥2\\|真1)；括号 = 同点 real/gauss 的值\n")
    L.append("| 矩阵 | rand p1 | rand p4 | rand p7 | wild p1 | real p1 | real p4 | real p7 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for nm in OL.scope_names():
        r = bind[nm]
        if not r.get("path"):
            continue
        v = r.get("variants", {})
        g = lambda key: (f"{v[key]['rate'][0]}/{v[key]['rate'][1]}" if key in v else "—")
        rp = {e["p"]: e for e in r["path"] if "FR2" in e}
        L.append(f"| {short(nm)} | {g('k1_rand_p1')} | {g('k1_rand_p4')} | {g('k1_rand_p7')} | {g('k1_wild_p1')} | "
                 f"{fr(rp[1])} | {fr(rp[4])} | {fr(rp[7])} |")
    # ---------------- 事后：相容度（预先登记 q 与事后 q1）
    parts = {}
    for nm in OL.scope_names():
        fn = os.path.join(ROOT, "cache", "overshoot_binding_parts", nm.replace("/", "__") + ".json")
        if os.path.exists(fn):
            parts[nm] = json.load(open(fn))
    L.append("")
    L.append("### T7（事后，未预先登记）预先登记的相容度 q（秩 0 的 CV 误差比）不能区分路径上的点；q1（秩 1 的 CV 误差比）可以\n")
    L.append("| 矩阵 | q（p1）| q（p7）| q1（p1）| q1（p4）| q1（p7）| FR2>0 的点里 q1 的最大值 | FR2=0 的点里 q1 的最小值 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    allmax = []
    for nm, P in parts.items():
        rc = {int(k): v for k, v in noise["matrices"][nm]["real_curve"].items()}
        by = {}
        for r in P["res"]:
            by.setdefault(r["tag"], []).append(r)
        pts = []
        for p in range(1, 8):
            rs = by.get(f"k1_real_p{p}")
            if not rs:
                continue
            q1 = float(np.median([r["curve"][1] for r in rs]) / rc[1])
            q0 = float(np.median([r["curve"][0] for r in rs]) / rc[0])
            f = sum(r["det"] >= 2 for r in rs)
            pts.append((p, q0, q1, f))
        if not pts:
            continue
        pos = [x[2] for x in pts if x[3] > 0]
        zer = [x[2] for x in pts if x[3] == 0]
        if pos:
            allmax.append(max(pos))
        d = {x[0]: x for x in pts}
        L.append(f"| {short(nm)} | {d[1][1]:.2f} | {d[7][1]:.2f} | {d[1][2]:.2f} | {d[4][2]:.2f} | {d[7][2]:.2f} | "
                 f"{(f'{max(pos):.2f}' if pos else '—')} | {(f'{min(zer):.2f}' if zer else '—')} |")
    # ---------------- GC
    gc = C("overshoot_gc.json")
    L.append("")
    L.append("### T8 校正候选在植入数据上的预先登记检验（G4 合并；K3 合并 gAPN+OMTKY3）\n")
    L.append("| 方法 | s/σ | P(检出≥2\\|真1) | P(检出=2\\|真2) | 通过 |")
    L.append("|---|---:|---:|---:|---|")
    for meth in ("baseline", "K1", "K2", "baseline_on_K3_geoms", "K3"):
        m = gc["methods"][meth]
        first = True
        for lv, v in m["levels"].items():
            L.append(f"| {meth if first else ''} | {lv} | {v['FR2'][0]}/{v['FR2'][1]} | {v['H'][0]}/{v['H'][1]} | "
                     f"{('null ' + str(m['null_is0'][0]) + '/' + str(m['null_is0'][1]) + ('; 通过' if m['pass'] else '; 未通过')) if first else ''} |")
            first = False
    # ---------------- 应用
    ap = C("overshoot_apply.json")
    kb = C("overshoot_k1_boot.json")
    L.append("")
    L.append("### T9 通过 GC 的 K1、K3 在 18 个真实矩阵上的检出秩（头条口径：3 种子均值曲线 argmin；K3 单种子顺序检验）\n")
    L.append("| 矩阵 | 现行 | K1 | K3 | 现行 g12 [区间] | K1 g12 [区间] | 现行 g23 | K1 g23 |")
    L.append("|---|---:|---:|---:|---|---|---|---|")
    ff = lambda x: f"{100*x[0]:+.1f}% [{100*x[1]:+.1f},{100*x[2]:+.1f}]"
    for o in ap["res"]:
        nm = o["name"]
        b, k = kb["base"]["per_matrix"][nm], kb["K1"]["per_matrix"][nm]
        L.append(f"| {short(nm)} | {o['base']} | {o['K1']} | {o['K3']} | {ff(b['g12'])} | {ff(k['g12'])} | {ff(b['g23'])} | {ff(k['g23'])} |")
    L.append("")
    L.append("### T10 S1（15 个矩阵）秩间增益，位点联合重抽 2000 次\n")
    L.append("| 程序 | g01 | g12 | g23 | g02 | g12>0 的矩阵 |")
    L.append("|---|---|---|---|---|---:|")
    for lab, nmn in (("base", "现行"), ("K1", "K1")):
        s = kb[lab]["S1"]
        L.append(f"| {nmn} | {ff(s['g01'])} | {ff(s['g12'])} | {ff(s['g23'])} | {ff(s['g02'])} | {s['g12_pos_matrices']}/15 |")
    # ---------------- 消融
    ab = C("overshoot_ablate.json")
    L.append("")
    L.append("### T11 消融（预先登记 §8，次要）：20 个 CV 切分种子下检出秩 0/1/2/3 的计数\n")
    L.append("| 矩阵 | s2/s1 | A0 真实 | A1 去第二成分 | A2 去两成分 |")
    L.append("|---|---:|---|---|---|")
    for nm, r in ab["res"].items():
        L.append(f"| {short(nm)} | {r['s2']/r['s1']:.2f} | {r['A0']} | {r['A1']} | {r['A2']} |")
    return "\n".join(L)


if __name__ == "__main__":
    t = tables()
    if "--check" in sys.argv:
        doc = open(sys.argv[sys.argv.index("--check") + 1]).read()
        bad = [ln for ln in t.splitlines() if ln.startswith("| ") and ln not in doc]
        print("核对：缺失/不一致的表行", len(bad))
        for b in bad[:10]:
            print("  ", b)
        sys.exit(1 if bad else 0)
    print(t)
