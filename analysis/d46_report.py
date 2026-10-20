#!/usr/bin/env python3
"""docs/46 Part A 报告：按预先登记 §5（逐字沿用 docs/45 §9）分类、判决；两层并列（锁定层 / 修订 A1 层）。
只读 cache/d46/*；入口断言三层锁中的前两层（Part A 不依赖 A2）。"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d46_lib as D                              # noqa: E402
import d46_run as RN                             # noqa: E402
import u2split_lib as U                          # noqa: E402
import u2split_run as RUN                        # noqa: E402

OUT = RN.OUT
EXTS = RN.EXTS
S1, PATHM, SC = RN.S1, RN.PATHM, RN.SCOPE
R = D.R_WORLD


def short(n):
    return n.replace("animalACE2/", "").replace("Wuhan-Hu-1", "WH1").replace("Omicron_", "").replace("PDCoV/PDCoV/", "")


def classify(x1, x2, ws, gate_sep, faithful):
    """ws[kind] = list of 7 个 (y 复现中位数组)。返回 (cls, pos1, pos2)。"""
    pos1 = max((1 + (y >= x2).sum()) / (R + 1) for y in ws["R1"])
    pos2 = max((1 + (y <= x2).sum()) / (R + 1) for y in ws["R2"])
    if not faithful:
        return "F", pos1, pos2
    if not gate_sep:
        return "U", pos1, pos2
    if pos1 <= 0.05:
        return "R" + ("(弱于R2)" if pos2 <= 0.05 else ""), pos1, pos2
    if pos2 <= 0.05:
        return "N", pos1, pos2
    return "I", pos1, pos2


def system_verdict(classes, S1_names):
    cl = {n: classes[n] for n in S1_names}
    assess = [n for n, c in cl.items() if c not in ("U", "F")]
    nR = sum(c.startswith("R") for n, c in cl.items() if n in assess)
    nN = sum(c == "N" for n, c in cl.items() if n in assess)
    nI = sum(c == "I" for n, c in cl.items() if n in assess)
    nU = sum(c == "U" for c in cl.values())
    nF = sum(c == "F" for c in cl.values())
    na = len(assess)
    if na < 2 / 3 * len(S1_names):
        v = "UNUSABLE_CONTROL"           # 忠实且可分辨的 S1 矩阵 < 2/3 × 14
    elif nR >= 2 / 3 * na:
        v = "REPRODUCIBLE"
    elif nN >= 2 / 3 * na:
        v = "NOT_REPRODUCIBLE"
    else:
        v = "UNDECIDED"
    return {"n_assessable": na, "R": nR, "N": nN, "I": nI, "U": nU, "F": nF, "verdict": v, "classes": cl}


def total_from_ext(vs):
    if all(v == "REPRODUCIBLE" for v in vs):
        return "REPRODUCIBLE"
    if all(v == "NOT_REPRODUCIBLE" for v in vs):
        return "NOT_REPRODUCIBLE"
    if any(v == "UNUSABLE_CONTROL" for v in vs):
        return "UNUSABLE_CONTROL"
    return "CANNOT_ASSESS (extraction-dependent/undecided)"


def main():
    D.verify_lock46()
    U.verify_lock()
    gate = json.load(open(os.path.join(OUT, "gate.json")))
    perm = pickle.load(open(os.path.join(OUT, "perm.pkl"), "rb"))
    real = pickle.load(open(os.path.join(OUT, "real.pkl"), "rb"))
    fid = json.load(open(os.path.join(OUT, "fidelity.json")))
    fm = RN.faithful_map()
    g, noise = RUN.load_geoms()
    old = json.load(open(os.path.join(ROOT_, "cache", "u2split", "results.json")))
    pm = json.load(open(os.path.join(ROOT_, "cache", "u2split", "postmortem_noclip.json")))
    res = {"gate": gate, "matrices": {}, "verdict": {}, "fid_maps": fm}

    for n in SC:
        res["matrices"][n] = {}
        for ie, e in enumerate(EXTS):
            sp = real[n][e]["splits"]
            c1, c2, dd = sp[:, 0], sp[:, 1], sp[:, 2]
            dmed = int(np.nanmedian(dd))
            hn = U.haar_null(dmed)
            pn = perm[n][e]
            m = {"d": dmed, "cos1": U.boot_median(c1, seed=1), "cos2": U.boot_median(c2, seed=2),
                 "contig": real[n][e]["contig"],
                 "haar_cos2_p95": float(np.quantile(hn["cos2"], 0.95)),
                 "perm_cos2_med_p95": [float(np.nanmedian(pn[:, 1])), float(np.nanquantile(pn[:, 1], 0.95))],
                 "perm_cos1_med_p95": [float(np.nanmedian(pn[:, 0])), float(np.nanquantile(pn[:, 0], 0.95))],
                 "n_fail": int(np.isnan(c2).sum())}
            if n in PATHM:
                x1, x2 = float(np.nanmedian(c1)), float(np.nanmedian(c2))
                m["by_world"] = {}
                for w in RN.WORLDS:
                    ws = {}
                    pts = {}
                    for kind in ("R1", "R2"):
                        arrs = [RN.load_world(w, n, kind, p)[:, :, ie, :] for p in range(7)]
                        ws[kind] = [np.nanmedian(a[:, :, 1], axis=1) for a in arrs]
                        pts[kind] = [{"cos2_med": float(np.median(y)), "q05": float(np.quantile(y, 0.05)),
                                      "q95": float(np.quantile(y, 0.95)),
                                      "cos1_med": float(np.median(np.nanmedian(a[:, :, 0], axis=1)))}
                                     for y, a in zip(ws[kind], arrs)]
                    a0 = RN.load_world(w, n, "R0", 0)[:, :, ie, :]
                    y0 = np.nanmedian(a0[:, :, 1], axis=1)
                    sep = gate[w]["H2"][e]["separable"][n]
                    faithful = fm[w][n]["all_pass"]
                    cls, pos1, pos2 = classify(x1, x2, ws, sep, faithful)
                    cls_locked, _, _ = classify(x1, x2, ws, sep, fm[w][n]["locked_all_pass"])
                    f1b = max(v["f1b"] for k, v in fid[n]["pts"].items() if k.startswith(w + "|"))
                    m["by_world"][w] = {"class": cls, "class_locked_criterion": cls_locked, "pos1": float(pos1), "pos2": float(pos2),
                                        "separable": bool(sep), "faithful_A1": bool(faithful),
                                        "faithful_locked": bool(fm[w][n]["locked_all_pass"]),
                                        "n_fail_pts_A1": fm[w][n]["n_fail"], "max_f1b": float(f1b),
                                        "paths": pts, "R0_med": float(np.median(y0)), "R0_q95": float(np.quantile(y0, 0.95))}
            res["matrices"][n][e] = m

    # ---- 体系层
    S1p = [n for n in S1 if n in PATHM]
    for w in RN.WORLDS:
        res["verdict"][w] = {}
        for layer in ("A1", "locked"):
            res["verdict"][w][layer] = {}
            for e in EXTS:
                key = "class" if layer == "A1" else "class_locked_criterion"
                classes = {n: res["matrices"][n][e]["by_world"][w][key] for n in S1p}
                res["verdict"][w][layer][e] = system_verdict(classes, S1p)
            res["verdict"][w][layer]["total"] = total_from_ext([res["verdict"][w][layer][e]["verdict"] for e in EXTS])
    # 世界裁决（预先登记 §5，A1 层）
    def world_adjudicate(layer):
        usable = {w: res["verdict"][w][layer]["total"] for w in RN.WORLDS
                  if res["verdict"][w][layer]["total"] != "UNUSABLE_CONTROL" and all(gate[w][h][e]["pass"] for h in ("H1", "H2") for e in EXTS)}
        if not usable:
            return "CANNOT_ASSESS (无忠实对照 / 门不过)", usable
        vs = set(usable.values())
        if len(usable) == 2:
            if len(vs) == 1:
                return usable[RN.WORLDS[0]], usable
            return "CANNOT_ASSESS (对照世界依赖)", usable
        return list(usable.values())[0] + "（仅一个对照世界）", usable
    res["verdict"]["final_A1"], res["verdict"]["usable_A1"] = world_adjudicate("A1")
    res["verdict"]["final_locked"], res["verdict"]["usable_locked"] = world_adjudicate("locked")

    # ---- 与 docs/45 并列
    res["vs_docs45"] = {}
    for n in S1p:
        row = {"d45_prereg": {e: old["matrices"][n][e]["class"] for e in EXTS},
               "d45_posthoc": {e: pm[e][n][0] for e in EXTS}}
        res["vs_docs45"][n] = row
    agree = {}
    for w in RN.WORLDS:
        for e in EXTS:
            a_n, a_tot = 0, 0
            a_pre = 0
            for n in S1p:
                c = res["matrices"][n][e]["by_world"][w]["class"]
                if c in ("F", "U"):
                    continue
                a_tot += 1
                ph = res["vs_docs45"][n]["d45_posthoc"][e]
                a_n += int(c.startswith("R") == (ph == "R"))
                pr = res["vs_docs45"][n]["d45_prereg"][e]
                a_pre += int(c.startswith("R") == pr.startswith("R"))
            agree[f"{w}|{e}"] = {"n_compared": a_tot, "agree_posthoc_RvsNot": a_n, "agree_prereg_RvsNot": a_pre}
    res["agree"] = agree

    # ---- 描述汇总：R1 / R2 / R0 世界 cos θ₂
    summ = {}
    for w in RN.WORLDS:
        for e in EXTS:
            for kind in ("R1", "R2"):
                for p in (0, 3, 6):
                    vals_all = [res["matrices"][n][e]["by_world"][w]["paths"][kind][p]["cos2_med"] for n in S1p]
                    vals_f = [res["matrices"][n][e]["by_world"][w]["paths"][kind][p]["cos2_med"] for n in S1p
                              if res["matrices"][n][e]["by_world"][w]["faithful_A1"]]
                    summ[f"{w}|{e}|{kind}|{p}"] = {"all": [float(np.min(vals_all)), float(np.median(vals_all)), float(np.max(vals_all))],
                                                    "faithful": ([float(np.min(vals_f)), float(np.median(vals_f)), float(np.max(vals_f))] if vals_f else None),
                                                    "n_faithful": len(vals_f)}
            vals = [res["matrices"][n][e]["by_world"][w]["R0_med"] for n in S1p]
            summ[f"{w}|{e}|R0"] = [float(np.min(vals)), float(np.median(vals)), float(np.max(vals))]
            # R1 在全部 7 点上、所有忠实矩阵的最大复现中位
            mx = [max(p["cos2_med"] for p in res["matrices"][n][e]["by_world"][w]["paths"]["R1"]) for n in S1p
                  if res["matrices"][n][e]["by_world"][w]["faithful_A1"]]
            summ[f"{w}|{e}|R1_max_over_points_faithful"] = [float(np.min(mx)), float(np.median(mx)), float(np.max(mx))] if mx else None
    res["summary"] = summ
    # S1 的真实中位
    res["S1_real"] = {e: U.boot_median([res["matrices"][n][e]["cos2"][0] for n in S1], seed=3) for e in EXTS}
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), ensure_ascii=False, indent=1,
              default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else bool(o) if isinstance(o, np.bool_) else str(o))
    write_tables(res)
    print("FINAL (锁定判据层):", res["verdict"]["final_locked"])
    print("FINAL (修订 A1 层):", res["verdict"]["final_A1"])


ROOT_ = os.path.dirname(HERE)


def write_tables(res):
    L_ = []
    L_.append("# docs/46 Part A 表（自动生成；cache/d46/results.json）\n")
    L_.append("## 逐矩阵：真实 cos θ₂（300 分半，新种子）与分类\n")
    L_.append("| 矩阵 | cos θ₁ EM | cos θ₂ EM [CI] | cos θ₂ K1 [CI] | 置换 p95 EM/K1 | 连续 EM/K1 | G 忠实(A1) | RR 忠实(A1) | G 类 EM/K1 | RR 类 EM/K1 | docs45 预登记 EM/K1 | docs45 事后 EM/K1 |")
    L_.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in SC:
        mE, mK = res["matrices"][n]["EM"], res["matrices"][n]["K1"]
        def ci(t): return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
        row = f"| {short(n)} | {mE['cos1'][0]:.3f} | {ci(mE['cos2'])} | {ci(mK['cos2'])} | {mE['perm_cos2_med_p95'][1]:.2f}/{mK['perm_cos2_med_p95'][1]:.2f} | {cg(mE['contig'])}/{cg(mK['contig'])} |"
        if n in res["vs_docs45"] or n in PATHM:
            bw = mE["by_world"]; bk = mK["by_world"]
            row += f" {'是' if bw['G']['faithful_A1'] else '否('+str(bw['G']['n_fail_pts_A1'])+')'} | {'是' if bw['RR']['faithful_A1'] else '否('+str(bw['RR']['n_fail_pts_A1'])+')'} |"
            row += f" {bw['G']['class']}/{bk['G']['class']} | {bw['RR']['class']}/{bk['RR']['class']} |"
            if n in res["vs_docs45"]:
                v = res["vs_docs45"][n]
                row += f" {v['d45_prereg']['EM']}/{v['d45_prereg']['K1']} | {v['d45_posthoc']['EM']}/{v['d45_posthoc']['K1']} |"
            else:
                row += " — | — |"
        else:
            row += " — | — | — | — | — | — |"
        L_.append(row)
    L_.append("\n## 体系层\n")
    L_.append("| 层 | 世界 | 抽取器 | 可评估 | R | N | I | U | F | 体系层 | 世界总判 |")
    L_.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for layer in ("locked", "A1"):
        for w in RN.WORLDS:
            for e in EXTS:
                v = res["verdict"][w][layer][e]
                L_.append(f"| {layer} | {w} | {e} | {v['n_assessable']} | {v['R']} | {v['N']} | {v['I']} | {v['U']} | {v['F']} | {v['verdict']} | {res['verdict'][w][layer]['total']} |")
    L_.append(f"\n最终（锁定判据层）：**{res['verdict']['final_locked']}**；最终（修订 A1 层）：**{res['verdict']['final_A1']}**\n")
    L_.append("## 门\n")
    for w in RN.WORLDS:
        for e in EXTS:
            L_.append(f"- {w}/{e}: H1 假警报 {gate_fmt(res['gate'][w]['H1'][e]['pooled'])}（{'过' if res['gate'][w]['H1'][e]['pass'] else '未过'}），H2 S1 可分辨 {res['gate'][w]['H2'][e]['frac_S1']:.2f}（{'过' if res['gate'][w]['H2'][e]['pass'] else '未过'}）")
    L_.append("\n## 没有第二成分时 cos θ₂ 多大（R1 世界；矩阵间 [最小, 中位, 最大]；取路径第 0/3/6 点）\n")
    L_.append("| 世界 | 抽取器 | 点 | 全部 14 | 仅忠实矩阵 (n) |")
    L_.append("|---|---|---|---|---|")
    for w in RN.WORLDS:
        for e in EXTS:
            for p in (0, 3, 6):
                s = res["summary"][f"{w}|{e}|R1|{p}"]
                fa = "—" if s["faithful"] is None else f"{s['faithful'][0]:.2f}, {s['faithful'][1]:.2f}, {s['faithful'][2]:.2f} ({s['n_faithful']})"
                L_.append(f"| {w} | {e} | {p} | {s['all'][0]:.2f}, {s['all'][1]:.2f}, {s['all'][2]:.2f} | {fa} |")
    L_.append("\n## R2 世界、R0 世界\n")
    for w in RN.WORLDS:
        for e in EXTS:
            r2 = [res["summary"][f"{w}|{e}|R2|{p}"]["all"] for p in (0, 3, 6)]
            L_.append(f"- {w}/{e}: R2 点0/3/6 中位 {r2[0][1]:.2f}/{r2[1][1]:.2f}/{r2[2][1]:.2f}（最小 {r2[0][0]:.2f}/{r2[1][0]:.2f}/{r2[2][0]:.2f}）；R0 {res['summary'][f'{w}|{e}|R0'][0]:.2f}, {res['summary'][f'{w}|{e}|R0'][1]:.2f}, {res['summary'][f'{w}|{e}|R0'][2]:.2f}")
    L_.append("\n## 与 docs/45 一致数（R 与非 R；部分循环，不当作确认）\n")
    for k, v in res["agree"].items():
        L_.append(f"- {k}: 可比较 {v['n_compared']}；与事后一致 {v['agree_posthoc_RvsNot']}；与预登记一致 {v['agree_prereg_RvsNot']}")
    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(L_))


def cg(c):
    return 'NA' if c is None else f'{c[1]:.2f}'


def gate_fmt(x):
    return f"{x:.3f}"


if __name__ == "__main__":
    main()
