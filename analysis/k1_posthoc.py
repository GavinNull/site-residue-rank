#!/usr/bin/env python3
"""docs/44 事后敏感性（**未预先登记**，只作旁证，不进任何判决）：S2 只取 gAPN / 只取 pAPN，现行与 K1 并列。
动机：预先登记的体系层口径是「有功效矩阵的增益中位」，S2 只有 gAPN、pAPN 两个有功效矩阵，中位 = 二者平均，
docs/41 §5.2 已经指出 S2 的「支持」靠 gAPN 一个。这里把它拆开。复用 cache/k1_cells.pkl（k1_recompute 的逐格误差）。
"""
import json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import k1_recompute as KR
ROOT = KR.ROOT
blob = pickle.load(open(KR.CELLS_PKL, "rb"))["res"]
scope = KR.load_scope()
out = {}
for nm in KR.S2_NAMES:
    for arm in KR.ARMS:
        cells, per_seed = blob[(nm, arm)]
        cs = KR.curve_stats(per_seed)
        u = {"sites": list(scope[nm]["sites"]), "rows": cells[:, 0], "E": cs["Ebar"]}
        o = KR.pooled([u], [nm], 700 + KR.S2_NAMES.index(nm))
        out[f"{nm}|{arm}"] = {k: [o[k]["point"]] + list(o[k]["ci"]) for k in ("g01", "g12", "g23", "g02")}
        print(nm, arm, {k: f"{100*v[0]:+.1f}% [{100*v[1]:+.1f},{100*v[2]:+.1f}]" for k, v in out[f'{nm}|{arm}'].items()})
json.dump(out, open(os.path.join(ROOT, "cache", "k1_posthoc.json"), "w"), indent=1)

# ---- 事后（未预先登记）：K1 检出秩 vs 第二代截尾 g2（缓存值），描述 D-029「相关对检测器代际稳健」在 K1 下是否仍成立
import numpy as np
R = json.load(open(os.path.join(ROOT, "cache", "k1_recompute.json")))
c34 = json.load(open(os.path.join(ROOT, "cache", "censor_gen3_final.json")))["units"]
g2 = np.array([c["g2"] for c in c34]); g1 = np.array([c["g1"] for c in c34])
rk = np.array([u["rank_k1"] for u in R["units34"]]); ro = np.array([u["rank_old"] for u in R["units34"]])
post = {}
for nm, g in (("gen1(=gen3 here)", g1), ("gen2", g2)):
    for pr, r in (("old", ro), ("k1", rk)):
        sp = KR.CR.boot_corr(g, r, 4000, np.random.default_rng(7), "spearman")
        pe = KR.CR.boot_corr(g, r, 4000, np.random.default_rng(7), "pearson")
        post[f"{nm}|{pr}"] = {"spearman": [sp["point"], sp["lo"], sp["hi"]], "pearson": [pe["point"], pe["lo"], pe["hi"]]}
        print(nm, pr, "spearman %.2f [%.2f, %.2f]  pearson %.2f [%.2f, %.2f]" % (sp["point"], sp["lo"], sp["hi"], pe["point"], pe["lo"], pe["hi"]))
out["censor_generations"] = post
json.dump(out, open(os.path.join(ROOT, "cache", "k1_posthoc.json"), "w"), indent=1)
