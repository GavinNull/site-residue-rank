#!/usr/bin/env python3
"""docs/47 真实数据：D-036 可复现门 -> Q1 -> Q2/Q3。阶段 repro / q1 / q23。
入口断言：预先登记锁、selftest.json、gate.json（q23 还要求主校正 ia 的门通过，否则中止）。"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse, json, multiprocessing as mp, os, pickle, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import exprconf_lib as E
import subspace_lib as L
import rank_k1 as K1
import independence_audit as IA
import independence_stats as IS
OUT = E.OUT
ROOT = E.ROOT


def P(*a):
    print(*a, flush=True)


def jdump(o, path):
    json.dump(o, open(path, "w"), ensure_ascii=False, indent=1,
              default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else bool(x) if isinstance(x, np.bool_) else str(x))


def need(gate_ok=False):
    E.verify_lock47(False)
    assert os.path.exists(os.path.join(OUT, "selftest.json")), "先跑 selftest"
    assert os.path.exists(os.path.join(OUT, "gate.json")), "先过双向对照世界的门（gate.json 不存在）—— §4 禁止在门之前读真实结论"
    if gate_ok:
        g = json.load(open(os.path.join(OUT, "gate.json")))
        assert g["gates"]["ia"]["ALL"], "主校正 (i-a) 的门未通过 —— 预先登记：判决 C，不推进校正后的结论"


# ---- 折叠参照 --------------------------------------------------------
def load_fold():
    ta = json.load(open(os.path.join(ROOT, "cache", "tsuboyama_axis.json")))
    cd = json.load(open(os.path.join(ROOT, "cache", "tsuboyama_crossdomain.json")))
    tg = json.load(open(os.path.join(ROOT, "cache", "subspace_two_groups.json")))
    assert cd["aa_order"] == E.AA20
    ok = set(ta["repro_pass"])
    doms = [(np.array(r["axis"], float), np.array(r["cols"], int), r["repro"], r["cos_uD"]) for k, r in ta["rows"].items() if k in ok]
    assert len(doms) == 144
    uf = np.array(cd["u_fold"], float)
    uD = np.array(tg["axis1_family_avg"]["u"], float)
    # 已知事实自检：u_fold 与 u_D 的 |cos| 复现 docs/27 的 0.845；144 域对 u_D 的中位复现 0.755
    c = abs(float(E.cenu(uf) @ E.cenu(uD)))
    assert abs(c - 0.8452) < 5e-4, c
    med = float(np.median([E.axis_cos((a, c_), E.tr(uD))[0] for a, c_, _, _ in doms]))
    assert abs(med - 0.755) < 1e-3, med
    return {"doms": doms, "u_fold": uf, "u_D": uD, "repro_med": float(np.median([d[2] for d in doms]))}


def fold144(r1, F):
    """r1 = (B或向量, cols) -> (中位 |cos|, 超各自 d 的 p95 的比例)。"""
    v, ex = [], 0
    for a, c_, _, _ in F["doms"]:
        x = E.axis_cos(r1, (a, c_))
        if x is None:
            continue
        v.append(x[0]); ex += x[0] > E.haar1_p95(x[1])
    return float(np.median(v)), float(ex / len(v)), v


# ---- repro -------------------------------------------------------------
def split_axis(M, mask, ext, S, seed):
    rng = np.random.default_rng(seed)
    out = np.full(S, np.nan)
    ds = []
    for s in range(S):
        ia, ib = E.U.split_sites(M.shape[0], rng)
        sd = int(rng.integers(1 << 30))
        a = E.X(ext, M[ia], mask[ia], 1, sd); b = E.X(ext, M[ib], mask[ib], 1, sd + 1)
        x = E.axis_cos(a, b)
        if x:
            out[s] = x[0]; ds.append(x[1])
    return out, (int(np.median(ds)) if ds else 20)


def _repro_task(t):
    kind, key = t
    pairs = E.load_all()
    Ed = E._GEOMS["Ed"]
    if kind == "bind":
        p = [q for q in pairs if q["id"] == key][0]
        M, m = p["Mb"], p["kb"]
    else:
        M, m = Ed["expr"][key]["M"], Ed["expr"][key]["mask"]
    res = {}
    for ext in E.EXTS:
        v, d = split_axis(M, m, ext, E.S_REAL, E.seed47("repro", kind, key, ext))
        res[ext] = {"med": float(np.nanmedian(v)), "q05": float(np.nanquantile(v, .05)), "q95": float(np.nanquantile(v, .95)), "d": d,
                    "p95": E.haar1_p95(d), "pass": bool(np.nanmedian(v) > E.haar1_p95(d))}
    return t, res


def stage_repro(a):
    need()
    pairs = E.load_all()
    tasks = [("bind", p["id"]) for p in pairs] + [("expr", s) for s in ("WH1_v1", "WH1_v2", "Eta", "BA2", "PDCoV")]
    with mp.Pool(a.jobs) as pool:
        res = dict(pool.map(_repro_task, tasks, chunksize=1))
    F = load_fold()
    out = {f"{k[0]}|{k[1]}": v for k, v in res.items()}
    out["fold"] = {"repro_med_split_half_first_axis": F["repro_med"], "n": 144}
    jdump(out, os.path.join(OUT, "repro.json"))
    for k, v in res.items():
        P(f"  {k[0]:4s} {k[1]:16s} EM {v['EM']['med']:.3f} [{v['EM']['q05']:.2f},{v['EM']['q95']:.2f}] (p95 {v['EM']['p95']:.3f}) {'过' if v['EM']['pass'] else '不过'}   K1 {v['K1']['med']:.3f} {'过' if v['K1']['pass'] else '不过'}")
    P(f"  折叠 144 域 位点分半第一轴中位 {F['repro_med']:.3f}")


# ---- Q1 ----------------------------------------------------------------
def _axis(M, m, ext):
    return E.X(ext, M, m, 1, 3)


def _boot_task(t):
    pid = t
    pairs = E.load_all()
    p = [q for q in pairs if q["id"] == pid][0]
    F = load_fold()
    K = p["kb"] & p["ke"]
    rng = np.random.default_rng(E.seed47("q1boot", pid))
    nr = K.shape[0]
    v = []
    for _ in range(E.NBOOT):
        idx = rng.integers(0, nr, nr)
        b = E.X("EM", p["Mb"][idx], K[idx], 1, 5)
        e = E.X("EM", p["Me"][idx], K[idx], 1, 6)
        x = E.axis_cos(b, e)
        bf = E.axis_cos(b, E.tr(F["u_fold"])); ef = E.axis_cos(e, E.tr(F["u_fold"]))
        if x and bf and ef:
            v.append((x[0], bf[0], ef[0]))
    v = np.array(v)
    q = lambda c: [float(np.quantile(v[:, c], .025)), float(np.median(v[:, c])), float(np.quantile(v[:, c], .975))]
    return pid, {"be": q(0), "bf": q(1), "ef": q(2)}


def stage_q1(a):
    need()
    pairs = E.load_all()
    Ed = E._GEOMS["Ed"]
    F = load_fold()
    rep = json.load(open(os.path.join(OUT, "repro.json")))
    srcs = ("WH1_v1", "WH1_v2", "Eta", "BA2", "PDCoV")
    res = {"bind": {}, "expr": {}, "ref": {}}
    uf, uD = E.tr(F["u_fold"]), E.tr(F["u_D"])
    res["ref"]["uD_uF"] = E.axis_cos(uD, uf)[0]
    for ext in E.EXTS:
        for s in srcs:
            e = _axis(Ed["expr"][s]["M"], Ed["expr"][s]["mask"], ext)
            m144, f144, _ = fold144(e, F)
            res["expr"][f"{s}|{ext}"] = {"vs_ufold": E.axis_cos(e, uf)[0], "vs_uD": E.axis_cos(e, uD)[0], "fold144_med": m144, "fold144_frac": f144, "repro": rep[f"expr|{s}"][ext]["med"]}
        for p in pairs:
            b = _axis(p["Mb"], p["kb"], ext)
            m144, f144, _ = fold144(b, F)
            res["bind"][f"{p['id']}|{ext}"] = {"vs_ufold": E.axis_cos(b, uf)[0], "vs_uD": E.axis_cos(b, uD)[0], "fold144_med": m144, "fold144_frac": f144,
                                               "repro": rep[f"bind|{p['id']}"][ext]["med"], "cls": p["cls"], "unit": p["unit"], "src": p["src"]}
    # 全部 (结合 × 表达量) 配对
    allp = {}
    for ext in E.EXTS:
        for p in pairs:
            b = _axis(p["Mb"], p["kb"], ext)
            for s in srcs:
                e = _axis(Ed["expr"][s]["M"], Ed["expr"][s]["mask"], ext)
                x = E.axis_cos(b, e)
                if p["src"] == s:
                    cls = p["cls"]
                elif p["src"] in ("WH1_v1", "WH1_v2") and s in ("WH1_v1", "WH1_v2"):
                    cls = "SAME-LIB-OTHERVER"
                elif "PDCoV" in (p["src"], s):
                    cls = "CROSS-FAMILY"
                else:
                    cls = "CROSS-BACKBONE"
                allp[f"{p['id']}|{s}|{ext}"] = {"cos": x[0], "d": x[1], "cls": cls, "unit": p["unit"]}
    res["pairs"] = allp
    # u_D × 表达量源（u_D 本身含冠状病毒结合矩阵）
    # 偏相关（轴层面，符号对齐后）
    def partial(cbf, cbe, cef):
        return (cbf - cbe * cef) / np.sqrt(max((1 - cbe ** 2) * (1 - cef ** 2), 1e-12))
    pc = {}
    for p in pairs:
        for ext in E.EXTS:
            b = _axis(p["Mb"], p["kb"], ext)
            e = _axis(Ed["expr"][p["src"]]["M"], Ed["expr"][p["src"]]["mask"], ext)
            com = np.intersect1d(b[1], e[1])
            vb, ve = E.cenu(b[0][:, 0][np.searchsorted(b[1], com)]), E.cenu(e[0][:, 0][np.searchsorted(e[1], com)])
            vf = E.cenu(F["u_fold"][com])
            if vb @ ve < 0: ve = -ve
            sb = 1.0 if vb @ vf >= 0 else -1.0
            cbf, cef, cbe = abs(vb @ vf), float(ve @ vf) * (1 if vb @ vf >= 0 else -1), float(vb @ ve)
            pc[f"{p['id']}|{ext}"] = {"c_bf": float(cbf), "c_be": cbe, "c_ef_signed": float(cef), "partial": float(partial(cbf, cbe, cef))}
    res["partial"] = pc
    jdump(res, os.path.join(OUT, "q1.json"))
    P("  Q1 已写 q1.json")


def stage_q1boot(a):
    need()
    pairs = E.load_all()
    ids = [p["id"] for p in pairs]
    with mp.Pool(a.jobs) as pool:
        boot = dict(pool.map(_boot_task, ids, chunksize=1))
    jdump(boot, os.path.join(OUT, "q1boot.json"))
    P("  Q1 bootstrap 已写 q1boot.json")


# ---- Q2/Q3 ------------------------------------------------------------
def _boot_gain(Mc, Kc, seeds=E.CV_SEEDS):
    cells = None
    Es = []
    for sd in seeds:
        cells, Ed_ = K1.cv_cell_errors_k1(Mc, Kc, max_rank=3, seed=sd, **{k: v for k, v in K1.CV_HEADLINE.items() if k != "K"})
        Es.append(Ed_)
    Ebar = {r: np.mean([e[r] for e in Es], axis=0) for r in range(4)}
    t = {r: K1.trim95(Ebar[r]) for r in range(4)}
    pt = {"g01": (t[0] - t[1]) / t[0], "g12": (t[1] - t[2]) / t[1], "g23": (t[2] - t[3]) / t[2]}
    unit = {"sites": list(range(Mc.shape[0])), "rows": cells[:, 0], "E": Ebar}
    sb = IA.site_boot_gains([unit], E.NB_GAIN, seed=E.seed47("gboot", int(Kc.sum())), pairs=((0, 1), (1, 2), (2, 3)))
    ci = {k: IS.pct_ci(sb[pr][:, 0]) for k, pr in (("g01", (0, 1)), ("g12", (1, 2)), ("g23", (2, 3)))}
    return pt, ci


def _q23_task(pid):
    pairs = E.load_all()
    p = [q for q in pairs if q["id"] == pid][0]
    F = load_fold()
    uf, uD = E.tr(F["u_fold"]), E.tr(F["u_D"])
    oa = E.obs_expr_axis(p)
    out = {}
    for corr in E.CORRS + ("ic",):
        Vexp = E._vexp_for(corr, p["Me"], p["ke"], p["src"], 17)
        Mc, Kc, info = E.correct(p["Mb"], p["kb"], p["Me"], p["ke"], corr, Vexp)
        for ext in E.EXTS:
            r1 = E.X(ext, Mc, Kc, 1, 3); r2 = E.X(ext, Mc, Kc, 2, 4)
            m144, f144, v144 = fold144(r1, F)
            rec = {"n": int(Kc.sum()), "share": info["share"], "beta": info.get("beta", 0.0),
                   "vs_ufold": E.axis_cos(r1, uf)[0], "vs_uD": E.axis_cos(r1, uD)[0], "fold144_med": m144, "fold144_frac": f144,
                   "vs_exprobs": E.axis_cos(r1, oa[ext])[0] if oa[ext] else None}
            sp = E.split_stats(p["Mb"], p["kb"], p["Me"], p["ke"], corr, ext, p["src"], E.S_REAL, E.seed47("q23split", pid, corr, ext))
            ax = sp[:, 0]
            rec["split_ax"] = [float(np.nanmedian(ax)), float(np.nanquantile(ax, .05)), float(np.nanquantile(ax, .95))]
            rec["split_c2"] = [float(np.nanmedian(sp[:, 1])), float(np.nanquantile(sp[:, 1], .05)), float(np.nanquantile(sp[:, 1], .95))]
            rec["split_d"] = int(np.nanmedian(sp[:, 2]))
            if ext == "K1":
                pt, ci = _boot_gain(Mc, Kc)
                rec["gain"] = pt; rec["gain_ci"] = ci
            out[f"{corr}|{ext}"] = rec
    return pid, out


def need_a():
    E.verify_lock47(False); E.verify_lock47a(False)
    ga = json.load(open(os.path.join(OUT, "gate_a.json")))
    assert ga["gates"]["ic"]["ALL"], "修订层主校正 (i-c) 的门未通过 —— 不推进校正后的结论（判决 C）"


def stage_q23a(a):
    """修订 A1 层：入口断言 (i-c) 的门通过。输出含 none/ia/ib/ii/ic 五种，ia/ib/ii 在各自的门上的状态见 gate.json / gate_a.json。"""
    need_a()
    stage_q23_core(a)


def stage_q23k2(a):
    """修订 A2：仅 K2 层（表达量源二维可复现：WH1_v1/v2、Eta）的配对；入口断言 K2 层 (i-c) 门通过。探索性，不进判决。"""
    E.verify_lock47(False); E.verify_lock47a(False); E.verify_lock47b(False)
    g = json.load(open(os.path.join(OUT, "gate_a_k2.json")))
    assert g["gates"]["ic"]["ALL"], "K2 层 (i-c) 的门未通过"
    pairs = E.load_all()
    elig = set(json.load(open(os.path.join(OUT, "eligible.json")))["eligible"])
    ids = [p["id"] for p in pairs if p["src"] in ("WH1_v1", "WH1_v2", "Eta") and p["id"] in elig]
    P("  K2 层配对:", ids)
    with mp.Pool(a.jobs) as pool:
        res = dict(pool.map(_q23_task, ids, chunksize=1))
    jdump(res, os.path.join(OUT, "q23_k2.json"))
    P("  Q2/Q3(K2) 已写 q23_k2.json")


def stage_q23(a):
    need(gate_ok=True)
    stage_q23_core(a)


def stage_q23_core(a):
    pairs = E.load_all()
    ids = [p["id"] for p in pairs]
    t0 = time.time()
    with mp.Pool(a.jobs) as pool:
        res = dict(pool.map(_q23_task, ids, chunksize=1))
    jdump(res, os.path.join(OUT, "q23.json"))
    P(f"  Q2/Q3 已写 q23.json  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--jobs", type=int, default=9)
    a = ap.parse_args()
    {"repro": stage_repro, "q1": stage_q1, "q1boot": stage_q1boot, "q23": stage_q23, "q23a": stage_q23a, "q23k2": stage_q23k2}[a.stage](a)
