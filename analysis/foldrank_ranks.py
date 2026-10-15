#!/usr/bin/env python3
"""docs/42 Step 3–5：真实折叠矩阵的定秩 + 簇级统计 + 截尾协变。

**拒绝运行**，除非：预先登记锁一致、功效结果 cache/foldrank_power.json 存在、
且至少一个锚点被纳入的簇数 ≥ 20（预先登记 §5）。
只对「在放行锚点下被纳入」的域计算真实秩。

运行：python3 analysis/foldrank_ranks.py --jobs 9
产物：cache/foldrank_real.json
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse                                             # noqa: E402
import json                                                 # noqa: E402
import sys                                                  # noqa: E402
from concurrent.futures import ProcessPoolExecutor         # noqa: E402

import numpy as np                                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foldrank_lib as F                                    # noqa: E402
import foldrank_stats as ST                                 # noqa: E402
from subspace_lib import ROOT                               # noqa: E402

POWER = os.path.join(ROOT, "cache", "foldrank_power.json")
OUT = os.path.join(ROOT, "cache", "foldrank_real.json")
_G = {}


def _init():
    names, mats, cl, sigC, fout = F.universe()
    _G.update(mats=mats)


def real_one(name):
    m = _G["mats"][name]
    out = {}
    for sd in (3, 13, 23):
        d, trim, full = F.detect(m["M"], m["mask"], seed=sd)
        out[str(sd)] = {"det": d, "trim": trim.tolist(), "full": full.tolist()}
    return name, out


def gains(trim):
    t = np.asarray(trim, float)
    return {"g01": (t[0] - t[1]) / t[0], "g12": (t[1] - t[2]) / t[1],
            "g23": (t[2] - t[3]) / t[2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=9)
    a = ap.parse_args()
    print("[硬前置]")
    F.verify_lock()
    assert os.path.exists(POWER), "功效结果不存在：Step 2 必须先于任何真实定秩"
    pw = json.load(open(POWER))
    proceed = [t for t in ("A", "C", "B") if pw["anchors"][t]["proceed"]]
    if not proceed:
        print("预先登记 §5：没有任何锚点被纳入簇数 >= 20 -> 拒绝对真实数据定秩。")
        return 3
    print(f"  放行锚点: {proceed}  OK")
    names, mats, cl, sigC, fout = F.universe()
    todo = sorted({w for t in proceed for w in pw["anchors"][t]["admitted"]})
    print(f"  需要真实定秩的域: {len(todo)}（放行锚点下被纳入的并集）")
    real = {}
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=_init) as ex:
        for i, (nm, r) in enumerate(ex.map(real_one, todo, chunksize=1)):
            real[nm] = r
            if (i + 1) % 10 == 0:
                print(f"   ... {i + 1}/{len(todo)}", flush=True)
    res = {"proceed": proceed, "real": real, "anchors": {}}

    for t in proceed:
        A = pw["anchors"][t]
        reps = A["reps"]                       # {cluster: domain}
        cls = sorted(reps)
        dom = [reps[c] for c in cls]
        det = np.array([real[w]["3"]["det"] for w in dom])
        trim = np.array([real[w]["3"]["trim"] for w in dom])
        g = [gains(x) for x in trim]
        g01 = np.array([x["g01"] for x in g])
        g12 = np.array([x["g12"] for x in g])
        g23 = np.array([x["g23"] for x in g])
        P = np.stack([ST.smooth_rows(pw["domains"][w]["anchor"][t]["confusion"])
                      for w in dom])               # (n, 4, 4)
        print(f"\n[统计自检] 锚点 {t}")
        sl = ST.selftest(P)
        r = {"stats_selftest": sl,
             "n_clusters": len(cls), "clusters": cls, "reps": dom,
             "det_hist": np.bincount(det, minlength=4).tolist(),
             "sign_1v2": {"n1": int((det == 1).sum()),
                          "n2": int((det == 2).sum()),
                          "p_two_sided": ST.exact_sign(int((det == 2).sum()),
                                                       int((det == 1).sum()))},
             "gof": ST.world_gof(P, det),
             "lambda12": ST.lambda12(P, det),
             "mixture_pi": ST.mixture_em(P, det).tolist(),
             "g01": ST.boot_median(g01), "g12": ST.boot_median(g12),
             "g23": ST.boot_median(g23),
             "g23_neg_frac": float((g23 < 0).mean()),
             "g12_pos_frac": float((g12 > 0).mean()),
             "sens_median": float(np.median([
                 pw["domains"][w]["anchor"][t]["sens"] for w in dom]))}
        # 种子稳健性
        r["seed_agree"] = {
            sd: float(np.mean([real[w][sd]["det"] == real[w]["3"]["det"]
                               for w in dom])) for sd in ("13", "23")}
        # 截尾协变
        fo = np.array([fout[w] for w in dom])
        r["fout"] = {"range": [float(fo.min()), float(fo.max())],
                     "median": float(np.median(fo)),
                     "spearman_g12": ST.boot_spearman(fo, g12),
                     "spearman_det": ST.boot_spearman(fo, det.astype(float))}
        order = np.argsort(fo)
        terc = np.array_split(order, 3)
        r["fout"]["terciles"] = [
            {"fout_range": [float(fo[i].min()), float(fo[i].max())],
             "n": int(len(i)),
             "det_hist": np.bincount(det[i], minlength=4).tolist(),
             "g12_median": float(np.median(g12[i]))} for i in terc]
        # 量级对照：真实归一化曲线中位 vs 合成各世界的曲线中位
        real_curve = np.median(trim / trim[:, [0]], axis=0).tolist()
        synth_curve = {k: np.median([pw["domains"][w]["anchor"][t]
                                     ["curve_median"][str(k)] for w in dom],
                                    axis=0).tolist() for k in range(4)}
        r["magnitude"] = {"real_median_curve": real_curve,
                          "synth_median_curve_by_true_rank": synth_curve}
        # 域层（次要）
        adm = pw["anchors"][t]["admitted"]
        det_d = np.array([real[w]["3"]["det"] for w in adm])
        r["domain_level"] = {
            "n_domains": len(adm),
            "det_hist": np.bincount(det_d, minlength=4).tolist(),
            "g12": ST.boot_median([gains(real[w]["3"]["trim"])["g12"]
                                   for w in adm]),
            "g23": ST.boot_median([gains(real[w]["3"]["trim"])["g23"]
                                   for w in adm])}
        # 次要聚合：簇内被纳入域的归一化曲线均值 -> argmin
        mean_det = []
        for c in cls:
            mem = [w for w in adm if cl[w] == c]
            cv = np.mean([np.array(real[w]["3"]["trim"])
                          / real[w]["3"]["trim"][0] for w in mem], axis=0)
            mean_det.append(int(np.argmin(cv)))
        r["cluster_meancurve_det_hist"] = np.bincount(
            mean_det, minlength=4).tolist()
        res["anchors"][t] = r
        print(f"\n锚点 {t}: n_clusters={len(cls)} 检出秩分布 {r['det_hist']}")
    res["stats_selftest_note"] = "见 foldrank_stats.selftest（在脚本 --selftest 下运行）"
    json.dump(res, open(OUT, "w"), default=float)
    print(f"\n写入 {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
