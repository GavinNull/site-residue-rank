#!/usr/bin/env python3
"""docs/46 Part B 主流程：结合第二方向 vs 表达量交互子空间。

预先登记：cache/docs46_prereg.md §6 + A1 + A2（三个锁都断言）。
阶段（顺序即纪律：数据与解析自检 -> 表达量自己可复现吗 -> 植入配对对照 -> 才做真实配对比较）：
  data      构建表达量/同源结合矩阵，解析自检（expr == 重复均值），文库一致性
  repro     E1 重复相关、E2 重复间子空间、E3 位点分半 + 保真 + 分类 -> 门 G-E
  planted   植入配对世界（SHARED / PARTIAL / INDEP）-> 门 G-B1、G-B2
  compare   真实配对比较、分类、CR9114 签名检查、次要描述
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
import d46_lib as D                                  # noqa: E402
import d46_partb_lib as B                            # noqa: E402
import u2split_lib as U                              # noqa: E402
import u2split_run as RUN                            # noqa: E402
import subspace_lib as L                             # noqa: E402
import rank_core as RC                               # noqa: E402

OUT = os.path.join(ROOT, "cache", "d46")
PB = os.path.join(OUT, "partb")
os.makedirs(PB, exist_ok=True)
EXTS = U.EXTRACTORS
BGS = {"Wuhan-Hu-1": "WH1", "Eta": "Eta", "Omicron_BA2": "BA2"}
LIGS = ("deer", "hamster", "bat", "cat", "human_monomer")
SOURCES = {  # 名 -> (文件, 目标, 对应的 S1 骨架)
    "WH1_v1": (B.OMI, "Wuhan-Hu-1_v1", "Wuhan-Hu-1"),
    "WH1_v2": (B.OMI, "Wuhan-Hu-1_v2", "Wuhan-Hu-1"),
    "Eta": (B.OMI, "Eta", "Eta"),
    "BA2": (B.OMI, "Omicron_BA2", "Omicron_BA2"),
    "PDCoV": (B.PDC, "PDCoV", None),
}
PRIMARY_SRC = {"Wuhan-Hu-1": "WH1_v2", "Eta": "Eta", "Omicron_BA2": "BA2"}


def P(*a):
    print(*a, flush=True)


def jdump(obj, path):
    json.dump(obj, open(path, "w"), ensure_ascii=False, indent=1,
              default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, np.integer)
              else bool(o) if isinstance(o, np.bool_) else str(o))


def need():
    B.verify_locks(False)
    assert os.path.exists(os.path.join(OUT, "selftest.json")), "先跑 d46_run.py selftest"
    assert json.load(open(os.path.join(OUT, "fidelity_teeth.json")))["ok"], "A1 保真判据的牙齿未过"


def sites_of(src):
    meta = json.load(open(os.path.join(ROOT, "cache", "u2split", "meta.json")))
    bg = SOURCES[src][2]
    key = f"animalACE2/{bg}/deer" if bg else "PDCoV/PDCoV/gAPN"
    return meta[key]["sites"]


# ======================================================================
def stage_data(a):
    need()
    B.verify_locks()
    data = {"expr": {}, "bindH": {}, "inventory": {}, "libid": {}}
    # ---- 列清单（事实）
    for fn in ("starr_animalACE2_final_variant_scores.csv", "starr2020_rbd_single_mut_effects.csv", B.OMI, B.PDC,
               "starr_SARSr_RBD_final_variant_scores_lib40_41.csv", "starr2022_variants_final_variant_scores.csv"):
        d = B._df(fn)
        ec = [c for c in d.columns if "expr" in c.lower() and not c.startswith("n_")]
        data["inventory"][fn] = {"cols": list(d.columns), "expr_cols": ec,
                                 "nonnull": {c: int(d[c].notna().sum()) for c in ec}}
    for src, (fn, tgt, bg) in SOURCES.items():
        sites = sites_of(src)
        M, k = B.build_mat(fn, tgt, "expr", sites)
        Dm, Nrep, maxdev = B.build_reps_arrays(fn, tgt, ["expr_rep1", "expr_rep2"], M, k, sites)
        M1, k1 = B.build_mat(fn, tgt, "expr_rep1", sites)
        M2, k2 = B.build_mat(fn, tgt, "expr_rep2", sites)
        g = B.make_geom(f"EXPR/{src}", M, k, Dm, Nrep)
        data["expr"][src] = {"M": M, "mask": k, "M1": M1, "k1": k1, "M2": M2, "k2": k2, "Dm": Dm, "Nrep": Nrep,
                             "sites": sites, "geom": g, "maxdev_mean_of_reps": maxdev,
                             "n": int(k.sum()), "nr_obs": int(k.any(1).sum()),
                             "n_two": int((Nrep == 2).sum()), "n_one": int((Nrep == 1).sum())}
        P(f"  表达量 {src}: n = {k.sum()}，两重复 {int((Nrep == 2).sum())}，单重复 {int((Nrep == 1).sum())}，"
          f"解析自检 expr == 重复均值 max|Δ| = {maxdev:.1e}  OK；σ_meas(cell) = {g.sig['sigma_cell']:.3f}，"
          f"σ_fit1 = {g.fit[1]['sigma']:.3f}，σ_fit2 = {g.fit[2]['sigma']:.3f}，cso = {g.cso:.3f}，值域 [{g.lo:.2f}, {g.hi:.2f}]")
        if src != "PDCoV":
            Mb, kb = B.build_mat(fn, tgt, "bind", sites)
            _, _, md = B.build_reps_arrays(fn, tgt, ["bind_rep1", "bind_rep2", "bind_rep3"], Mb, kb, sites)
            data["bindH"][src] = {"M": Mb, "mask": kb, "n": int(kb.sum()), "maxdev": md}
            P(f"  同源结合(hACE2) {src}: n = {kb.sum()}，解析自检 bind == 重复均值 max|Δ| = {md:.1e}  OK")
    # ---- 文库一致性（n_bc 的 Spearman）
    A = B._df("starr_animalACE2_final_variant_scores.csv")
    O = B._df(B.OMI)
    for src, (fn, tgt, bg) in SOURCES.items():
        if bg is None:
            continue
        a = A[A.target == bg].set_index(["position", "mutant"])
        o = O[O.target == tgt].set_index(["position", "mutant"])
        j = a.join(o, how="inner", lsuffix="_a", rsuffix="_o")
        x = j[[f"n_bc_{l}" for l in LIGS]].mean(axis=1)
        out = {}
        for c in ("n_bc_expr", "n_bc_bind"):
            m = x.notna() & j[c].notna()
            out[c] = float(pd.Series(x[m]).corr(j[c][m], method="spearman"))
        data["libid"][src] = out
    pickle.dump(data, open(os.path.join(PB, "data.pkl"), "wb"))
    jdump({"inventory": data["inventory"], "libid": data["libid"],
           "expr": {k: {kk: v[kk] for kk in ("maxdev_mean_of_reps", "n", "nr_obs", "n_two", "n_one")} for k, v in data["expr"].items()}},
          os.path.join(PB, "data_facts.json"))


def load_data():
    return pickle.load(open(os.path.join(PB, "data.pkl"), "rb"))


# ======================================================================
# repro
# ======================================================================
_D = {}
_EPATH = {}


def _init_b():
    global _D, _EPATH
    RUN._init_worker()
    _D = load_data()
    _EPATH = pickle.load(open(os.path.join(PB, "epaths.pkl"), "rb")) if os.path.exists(os.path.join(PB, "epaths.pkl")) else {}


def rep_corr(M1, k1, M2, k2):
    both = k1 & k2
    a, b = M1[both], M2[both]
    raw = float(np.corrcoef(a, b)[0, 1])
    R1 = L.dc_fast(np.where(both, M1, 0.0), both)[both]
    R2 = L.dc_fast(np.where(both, M2, 0.0), both)[both]
    return {"raw": raw, "dc": float(np.corrcoef(R1, R2)[0, 1]), "n": int(both.sum())}


def yardstick_binding():
    """结合列重复的同口径一致性（同骨架，animalACE2 的 deer/hamster/bat/cat 两个重复）。"""
    A = B._df("starr_animalACE2_final_variant_scores.csv")
    meta = json.load(open(os.path.join(ROOT, "cache", "u2split", "meta.json")))
    out = {}
    for bg in BGS:
        sites = meta[f"animalACE2/{bg}/deer"]["sites"]
        rows = []
        for lig in ("deer", "hamster", "bat", "cat"):
            M1, k1 = B.build_mat("starr_animalACE2_final_variant_scores.csv", bg, f"{lig}_rep1", sites)
            M2, k2 = B.build_mat("starr_animalACE2_final_variant_scores.csv", bg, f"{lig}_rep2", sites)
            rows.append(rep_corr(M1, k1, M2, k2))
        out[bg] = {"raw": [r["raw"] for r in rows], "dc": [r["dc"] for r in rows]}
    return out


def _epath(g):
    sig = g.sig["sigma_cell"]
    return {"k1": g.path(1, sig), "k2": g.path(2, sig)}


def _efid_task(src):
    g = _D["expr"][src]["geom"]
    ref = D.real_ref(g)
    ep = _EPATH[src]
    res = {"pts": {}, "ref": {"cso": ref["cso"]}}
    for world in ("G", "RR"):
        for kind, kk in (("R1", "k1"), ("R2", "k2")):
            for p, pt in enumerate(ep[kk]):
                res["pts"][f"{world}|{kind}|{p}"] = D.fid_point(g, ref, world, kind, pt, "e")
    for kind in ("R1", "R2"):
        res["pts"][f"REF|{kind}|0"] = D.fid_point(g, ref, "REF", kind, None, "eref")
    return src, res


def _ecell_task(c):
    w, src, kind, p = c
    path = os.path.join(PB, "eworlds", f"{w}__{src}__{kind}__{p}.npz")
    if os.path.exists(path):
        return c
    g = _D["expr"][src]["geom"]
    pt = None
    if kind in ("R1", "R2"):
        pt = _EPATH[src]["k1" if kind == "R1" else "k2"][p]
    R, S = D.R_WORLD, D.S_WORLD
    out = np.full((R, S, len(EXTS), 3), np.nan)
    for rep in range(R):
        rng = np.random.default_rng(D.seed46("bw", w, src, kind, p, rep))
        Y = D.make_world46(g, w, kind, rng, point=pt)
        for ie, ext in enumerate(EXTS):
            out[rep, :, ie, :] = U.many_splits(Y, g.mask, ext, S, D.seed46("bws", w, src, kind, p, rep, ext))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, cos=out)
    return c


def _ereal_task(args):
    src, ext = args
    e = _D["expr"][src]
    g = e["geom"]
    arr = U.many_splits(g.M, g.mask, ext, B.S_REAL_B, D.seed46("breal", src, ext))
    pn = U.perm_null(g.M, g.mask, ext, D.N_PERM_MAT, D.S_PERM, D.seed46("bperm", src, ext))
    sites = np.array(e["sites"])
    import u2split_run as R_
    cont = R_.contiguous_split(g.M, g.mask, sites, ext, D.seed46("bcont", src, ext))
    # E2：重复间（同位点，独立文库）
    r1 = U.extract(ext, e["M1"], e["k1"], 5)
    r2 = U.extract(ext, e["M2"], e["k2"], 6)
    e2 = U.compare(r1, r2)
    rng = np.random.default_rng(D.seed46("be2perm", src, ext))
    e2n = []
    for _ in range(30):
        Mp, mp = L.permute_within_sites(np.where(e["k2"], e["M2"], np.nan), e["k2"], rng)
        rr = U.extract(ext, np.where(mp, Mp, 0.0), mp, int(rng.integers(1 << 30)))
        c = U.compare(r1, rr)
        if c is not None:
            e2n.append(c)
    return src, ext, arr, pn, cont, e2, np.array(e2n)


def stage_repro(a):
    need()
    B.verify_locks()
    data = load_data()
    t0 = time.time()
    # ---- E1
    E1 = {"expr": {}, "yardstick_binding": yardstick_binding()}
    for src, e in data["expr"].items():
        E1["expr"][src] = rep_corr(e["M1"], e["k1"], e["M2"], e["k2"])
    for src, e in data["bindH"].items():
        pass
    jdump(E1, os.path.join(PB, "E1.json"))
    P("  E1 重复相关（逐变体，原始 / 双向去均值残差）：")
    for src, r in E1["expr"].items():
        P(f"    表达量 {src}: 原始 r = {r['raw']:.3f}，去均值残差 r = {r['dc']:.3f}（n = {r['n']}）")
    for bg, r in E1["yardstick_binding"].items():
        P(f"    结合标尺 {bg} (4 配体): 原始 r = {', '.join(f'{x:.3f}' for x in r['raw'])}；去均值残差 r = {', '.join(f'{x:.3f}' for x in r['dc'])}")
    # ---- 路径与保真
    ep = {src: _epath(e["geom"]) for src, e in data["expr"].items()}
    pickle.dump(ep, open(os.path.join(PB, "epaths.pkl"), "wb"))
    for src, p in ep.items():
        P(f"    {src}: 噪声路径点数 k1 = {len(p['k1'])}，k2 = {len(p['k2'])}（空 = σ_meas ≥ σ_fit，噪声支配）")
    srcs = [s for s in data["expr"] if len(ep[s]["k1"]) == 7 and len(ep[s]["k2"]) == 7]
    with mp.Pool(a.jobs, initializer=_init_b) as pool:
        fid = dict(pool.map(_efid_task, srcs, chunksize=1))
    jdump(fid, os.path.join(PB, "efid.json"))
    faithful = {}
    for src, f in fid.items():
        faithful[src] = {}
        for w in ("G", "RR"):
            pts = {k: v for k, v in f["pts"].items() if k.startswith(w + "|")}
            faithful[src][w] = {"A1": all(v["pass_am"] for v in pts.values()), "n_fail": sum(not v["pass_am"] for v in pts.values()),
                                "max_f1b": max(v["f1b"] for v in pts.values())}
        refp = [v["pass_am"] for k, v in f["pts"].items() if k.startswith("REF|")]
        faithful[src]["REF_pass"] = all(refp)
        P(f"    保真(A1) {src}: G {'忠实' if faithful[src]['G']['A1'] else '不忠实(' + str(faithful[src]['G']['n_fail']) + '点)'}，"
          f"RR {'忠实' if faithful[src]['RR']['A1'] else '不忠实(' + str(faithful[src]['RR']['n_fail']) + '点)'}，REF {'过' if faithful[src]['REF_pass'] else '未过'}")
        # REF 未过 = 判据在该几何上过严（工具的错，不是世界的错）：记录并在报告里标出；按 A2 的字面，无忠实世界 => 判决层门不过；
        # 另开「探索层」（不门控于保真度，明确标注，不进任何判决）供人类参考。
    jdump(faithful, os.path.join(PB, "efaithful.json"))
    # ---- 世界（仅忠实的）
    cells = []
    for src in srcs:
        for w in ("G", "RR"):
            for p in range(7):
                cells.append((w, src, "R1", p)); cells.append((w, src, "R2", p))
            cells.append((w, src, "R0", 0))
    P(f"  表达量植入世界单元（全部算出；判决层只用忠实者，探索层用全部）：{len(cells)}")
    with mp.Pool(a.jobs, initializer=_init_b) as pool:
        for i, _ in enumerate(pool.imap_unordered(_ecell_task, cells, chunksize=1), 1):
            if i % 10 == 0 or i == len(cells):
                P(f"    {i}/{len(cells)}  {time.time() - t0:.0f}s")
    # ---- 真实 + E2
    tasks = [(s, e) for s in data["expr"] for e in EXTS]
    with mp.Pool(a.jobs, initializer=_init_b) as pool:
        res = pool.map(_ereal_task, tasks, chunksize=1)
    real = {}
    for src, ext, arr, pn, cont, e2, e2n in res:
        real.setdefault(src, {})[ext] = {"splits": arr, "perm": pn, "contig": cont, "e2": e2, "e2_null": e2n}
    pickle.dump(real, open(os.path.join(PB, "ereal.pkl"), "wb"))
    P(f"  表达量真实分半 + E2 完成 {time.time() - t0:.0f}s")


def eload(w, src, kind, p):
    return np.load(os.path.join(PB, "eworlds", f"{w}__{src}__{kind}__{p}.npz"))["cos"]


def classify_pos(x2, ys1, ys2, sep, faithful):
    R = D.R_WORLD
    pos1 = max((1 + (y >= x2).sum()) / (R + 1) for y in ys1)
    pos2 = max((1 + (y <= x2).sum()) / (R + 1) for y in ys2)
    if not faithful:
        return "F", pos1, pos2
    if not sep:
        return "U", pos1, pos2
    if pos1 <= 0.05:
        return "R" + ("(弱于R2)" if pos2 <= 0.05 else ""), pos1, pos2
    if pos2 <= 0.05:
        return "N", pos1, pos2
    return "I", pos1, pos2


def stage_gateE(a):
    """门 G-E（二维）与 G-E1（一维，描述）。"""
    need()
    data = load_data()
    real = pickle.load(open(os.path.join(PB, "ereal.pkl"), "rb"))
    faithful = json.load(open(os.path.join(PB, "efaithful.json")))
    ep = pickle.load(open(os.path.join(PB, "epaths.pkl"), "rb"))
    out = {}
    for src in data["expr"]:
        o = {"sigma_cell": data["expr"][src]["geom"].sig["sigma_cell"],
             "sigma_fit1": data["expr"][src]["geom"].fit[1]["sigma"], "sigma_fit2": data["expr"][src]["geom"].fit[2]["sigma"],
             "path_len": [len(ep[src]["k1"]), len(ep[src]["k2"])], "by_ext": {}, "E2": {}}
        if src not in faithful:
            o["gate_E"] = False
            o["reason"] = "噪声支配（σ_meas ≥ σ_fit），无路径，无功效"
            out[src] = o
            continue
        perm_all = {}
        for ie, ext in enumerate(EXTS):
            r = real[src][ext]
            c1, c2, dd = r["splits"][:, 0], r["splits"][:, 1], r["splits"][:, 2]
            dmed = int(np.nanmedian(dd))
            pn = r["perm"]
            bw = {}
            for w in ("G", "RR"):
                ys1 = [np.nanmedian(eload(w, src, "R1", p)[:, :, ie, 1], axis=1) for p in range(7)]
                ys2 = [np.nanmedian(eload(w, src, "R2", p)[:, :, ie, 1], axis=1) for p in range(7)]
                p95 = float(np.nanquantile(pn[:, 1], 0.95))
                y0 = eload(w, src, "R0", 0)[:, :, ie, 1]
                ok = any(np.median(y2) > np.quantile(y1, 0.95) for y1, y2 in zip(ys1, ys2))
                cls, pos1, pos2 = classify_pos(float(np.nanmedian(c2)), ys1, ys2, ok, True)
                if not faithful[src][w]["A1"]:
                    cls_f = "F"
                # 一维（描述）：cos θ₁ 对 R0？这里只报真实 cos θ₁ 对置换 p95
                bw[w] = {"class": cls, "faithful_A1": bool(faithful[src][w]["A1"]), "pos1": float(pos1), "pos2": float(pos2), "separable": bool(ok),
                         "R0_exceed_perm_p95": float(np.nanmean(y0 > p95)),
                         "R1_med_pts": [float(np.median(y)) for y in ys1], "R2_med_pts": [float(np.median(y)) for y in ys2]}
            e2 = r["e2"]
            e2n = r["e2_null"]
            o["by_ext"][ext] = {"cos1": U.boot_median(c1, seed=11), "cos2": U.boot_median(c2, seed=12), "d": dmed,
                                "perm_p95_cos2": float(np.nanquantile(pn[:, 1], 0.95)), "perm_p95_cos1": float(np.nanquantile(pn[:, 0], 0.95)),
                                "contig": r["contig"], "worlds": bw}
            o["E2"][ext] = {"cos1": e2[0] if e2 else None, "cos2": e2[1] if e2 else None, "d": e2[2] if e2 else None,
                            "null_p95_cos2": float(np.quantile(e2n[:, 1], 0.95)) if len(e2n) else None,
                            "null_p95_cos1": float(np.quantile(e2n[:, 0], 0.95)) if len(e2n) else None}
        # 门
        usable_worlds = [w for w in ("G", "RR") if faithful[src][w]["A1"]]
        o["usable_worlds"] = usable_worlds
        cond = []
        for ext in EXTS:
            for w in usable_worlds:
                cond.append(o["by_ext"][ext]["worlds"][w]["class"].startswith("R"))
            cond.append(o["E2"][ext]["cos2"] is not None and o["E2"][ext]["cos2"] > o["E2"][ext]["null_p95_cos2"])
        o["gate_E"] = bool(len(usable_worlds) > 0 and all(cond))        # 判决层（A1：只用忠实世界）
        cond_x = []
        for ext in EXTS:
            for w in ("G", "RR"):
                cond_x.append(o["by_ext"][ext]["worlds"][w]["class"].startswith("R"))
            cond_x.append(o["E2"][ext]["cos2"] is not None and o["E2"][ext]["cos2"] > o["E2"][ext]["null_p95_cos2"])
        o["gate_E_expl"] = bool(all(cond_x))                            # 探索层：G 与 RR 都 R（不门控于保真度）
        o["gate_E1"] = bool(all(o["E2"][ext]["cos1"] is not None and o["E2"][ext]["cos1"] > o["E2"][ext]["null_p95_cos1"]
                                and o["by_ext"][ext]["cos1"][0] > o["by_ext"][ext]["perm_p95_cos1"] for ext in EXTS))
        out[src] = o
    # H1 汇总（表达量矩阵 pooled，每个世界）
    h1 = {}
    for w in ("G", "RR"):
        v = [o["by_ext"][e]["worlds"][w]["R0_exceed_perm_p95"] for o in out.values() if "by_ext" in o
             for e in EXTS if w in o["by_ext"][e]["worlds"] and "R0_exceed_perm_p95" in o["by_ext"][e]["worlds"][w]]
        h1[w] = float(np.mean(v)) if v else None
    out["_H1_pooled"] = h1
    jdump(out, os.path.join(PB, "gateE.json"))
    for src, o in out.items():
        if src.startswith("_"):
            continue
        P(f"  {src}: 路径点数 {o['path_len']}；G-E 二维(判决层) {'通过' if o.get('gate_E') else '未通过'}；G-E 探索层 {'通过' if o.get('gate_E_expl') else '未通过'}；G-E1 一维 {'通过' if o.get('gate_E1') else '未通过'}")
        for ext in EXTS:
            if "by_ext" not in o:
                continue
            b = o["by_ext"][ext]
            P(f"     [{ext}] 分半 cos θ₁ {b['cos1'][0]:.3f}，cos θ₂ {b['cos2'][0]:.3f} [{b['cos2'][1]:.3f}, {b['cos2'][2]:.3f}]，置换 p95 {b['perm_p95_cos2']:.3f}；"
              f"类 { {w: v['class'] for w, v in b['worlds'].items()} }；E2 重复间 cos θ₁ {o['E2'][ext]['cos1']:.3f} cos θ₂ {o['E2'][ext]['cos2']:.3f}（零 p95 {o['E2'][ext]['null_p95_cos2']:.3f}）")
    P("  H1 pooled(表达量):", h1)


# ======================================================================
# 配对
# ======================================================================
def pair_list(data):
    prs = []
    for bg in BGS:
        for lig in LIGS:
            for src, (fn, tgt, b2) in SOURCES.items():
                if b2 == bg:
                    prs.append({"id": f"{BGS[bg]}/{lig}|{src}", "kind": "cross", "bind": f"animalACE2/{bg}/{lig}", "src": src,
                                "primary": PRIMARY_SRC[bg] == src, "bg": bg})
    prs.append({"id": "gAPN|PDCoV", "kind": "same_S2", "bind": "PDCoV/PDCoV/gAPN", "src": "PDCoV", "primary": True, "bg": "PDCoV"})
    for src in ("WH1_v1", "WH1_v2", "Eta", "BA2"):
        prs.append({"id": f"hACE2({src})|{src}", "kind": "same_omicron", "bind": f"bindH:{src}", "src": src, "primary": True,
                    "bg": SOURCES[src][2]})
    return prs


_PG = {}


def _init_pair():
    global _PG
    _init_b()
    _PG = {"geoms": RUN._G, "noise": RUN._NOISE, "data": _D,
           "ep": pickle.load(open(os.path.join(PB, "epaths.pkl"), "rb"))}


def bind_arrays(name):
    if name.startswith("bindH:"):
        b = _D["bindH"][name.split(":")[1]]
        return b["M"], b["mask"]
    g = RUN._G[name]
    return g.M, g.mask


def _planted_task(args):
    pi, pr, mode, noise, pt = args
    path = os.path.join(PB, "planted", f"{pi}__{mode}__{noise}__{pt}.npz")
    if os.path.exists(path):
        return args
    gb = RUN._G[pr["bind"]]
    nb = RUN._NOISE[pr["bind"]]["path_k2"][B.PT_IDX[pt]]
    ge = _D["expr"][pr["src"]]["geom"]
    pe = _EPATH[pr["src"]]["k2"][B.PT_IDX[pt]]
    out = np.full((B.R_B, len(EXTS), 4), np.nan)
    for rep in range(B.R_B):
        rng = np.random.default_rng(D.seed46("bp", pi, mode, noise, pt, rep))
        Yb, Ye = B.planted_pair(gb, ge, mode, rng, nb, pe, noise)
        for ie, ext in enumerate(EXTS):
            c = B.pair_many(Yb, gb.mask, Ye, ge.mask, ext, B.S_B, D.seed46("bps", pi, mode, noise, pt, rep, ext))
            f = B.pair_full(Yb, gb.mask, Ye, ge.mask, ext, D.seed46("bpf", pi, mode, noise, pt, rep, ext))
            out[rep, ie] = [np.nanmedian(c[:, 0]), np.nanmedian(c[:, 1]), f[0], f[1]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, y=out)
    return args


def stage_planted(a):
    need()
    B.verify_locks()
    data = load_data()
    gateE = json.load(open(os.path.join(PB, "gateE.json")))
    efa = json.load(open(os.path.join(PB, "efaithful.json")))
    fm = None
    import d46_run as RN
    fm = RN.faithful_map()
    prs = pair_list(data)
    tasks, elig = [], {}
    for pi, pr in enumerate(prs):
        if pr["kind"] == "same_omicron":
            elig[pr["id"]] = "descriptive_only"
            continue
        if pr["bind"] not in fm["RR"]:
            elig[pr["id"]] = "binding 无噪声路径"
            continue
        ge_ = gateE.get(pr["src"], {})
        if not ge_.get("gate_E_expl"):
            elig[pr["id"]] = "G-E 未过（探索层也未过：表达量第二方向不可复现 / 无路径）"
            continue
        tier = []
        if not ge_.get("gate_E"):
            tier.append("表达量侧无忠实世界/门未过")
        if not fm["RR"][pr["bind"]]["all_pass"]:
            tier.append("结合侧 RR 不忠实(A1)")
        if not efa.get(pr["src"], {}).get("RR", {}).get("A1"):
            tier.append("表达量侧 RR 不忠实(A1)")
        elig[pr["id"]] = "ok" if not tier else "exploratory: " + "; ".join(tier)
        for mode in ("SHARED", "PARTIAL", "INDEP"):
            for noise in ("RR", "G"):
                for pt in range(3):
                    tasks.append((pi, pr, mode, noise, pt))
    jdump({"elig": elig, "pairs": prs}, os.path.join(PB, "pair_elig.json"))
    for k, v in elig.items():
        P(f"    {k}: {v}")
    P(f"  植入配对任务 {len(tasks)}（每任务 {B.R_B} 复现）")
    t0 = time.time()
    with mp.Pool(a.jobs, initializer=_init_pair) as pool:
        for i, _ in enumerate(pool.imap_unordered(_planted_task, tasks, chunksize=1), 1):
            if i % 20 == 0 or i == len(tasks):
                P(f"    {i}/{len(tasks)} {time.time() - t0:.0f}s")


def _preal_task(args):
    pi, pr = args
    Mb, kb = bind_arrays(pr["bind"])
    e = _D["expr"][pr["src"]]
    out = {}
    for ie, ext in enumerate(EXTS):
        r = {}
        r["avg"] = B.pair_many(Mb, kb, e["M"], e["mask"], ext, B.S_REAL_B, D.seed46("bpr", pi, ext, "avg"))
        r["full"] = B.pair_full(Mb, kb, e["M"], e["mask"], ext, D.seed46("bprf", pi, ext))
        r["rep1"] = B.pair_many(Mb, kb, e["M1"], e["k1"], ext, B.S_REAL_B, D.seed46("bpr", pi, ext, "r1"))
        r["rep2"] = B.pair_many(Mb, kb, e["M2"], e["k2"], ext, B.S_REAL_B, D.seed46("bpr", pi, ext, "r2"))
        r["perm"] = B.pair_perm_null(Mb, kb, e["M"], e["mask"], ext, B.NPERM_B, B.S_B, D.seed46("bpp", pi, ext))
        out[ext] = r
    # 描述：结合自身分半（同源结合矩阵的可复现性）
    # 3 维表达量敏感性（仅 EM）、残基主效应投影
    rng = np.random.default_rng(D.seed46("bp3", pi))
    c3 = []
    for _ in range(100):
        ia, ib = U.split_sites(Mb.shape[0], rng)
        bA = U.extract("EM", Mb[ia], kb[ia], 1)
        eB3 = L.extract_subspace(e["M"][ib], e["mask"][ib], rank=3, method="emsvd", seed=2)
        if bA is None or eB3[0] is None:
            continue
        a_ = L.align(bA[0], bA[1], eB3[0], eB3[1])
        if a_ is not None:
            c3.append((a_["cos"][0], a_["cos"][1], a_["n_res"]))
    out["c3"] = np.array(c3)
    # 残基主效应（列主效应）投影
    r_b, c_b = RC.center_fit(Mb, kb)
    r_e, c_e = RC.center_fit(e["M"], e["mask"])
    rb = U.extract("EM", Mb, kb, 1)
    proj = {}
    if rb is not None:
        cols = rb[1]
        for nm, cvec in (("expr_colmain", c_e), ("bind_colmain", c_b)):
            v = cvec[cols] - cvec[cols].mean()
            v = v / np.linalg.norm(v)
            Q = rb[0] - rb[0].mean(axis=0, keepdims=True)
            Qo, _ = np.linalg.qr(Q)
            proj[nm] = float(np.linalg.norm(Qo.T @ v))
        proj["d"] = int(len(cols))
    out["proj"] = proj
    return pi, out


def stage_compare(a):
    need()
    B.verify_locks()
    data = load_data()
    prs = pair_list(data)
    t0 = time.time()
    with mp.Pool(a.jobs, initializer=_init_pair) as pool:
        res = dict(pool.map(_preal_task, list(enumerate(prs)), chunksize=1))
    pickle.dump({"pairs": prs, "res": res}, open(os.path.join(PB, "preal.pkl"), "wb"))
    P(f"  真实配对比较完成 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["data", "repro", "gateE", "planted", "compare"])
    ap.add_argument("--jobs", type=int, default=9)
    a = ap.parse_args()
    {"data": stage_data, "repro": stage_repro, "gateE": stage_gateE, "planted": stage_planted, "compare": stage_compare}[a.stage](a)
