#!/usr/bin/env python3
"""docs/47 Q1 的 D-036 跨源补充：非酵母展示的结合体系（OMTKY3 × 蛋白酶，SKEMPI 体外）与其它体系的第一轴，对表达量轴与折叠轴。
这些体系的读出不经过 RBD 展示量，所以「展示量/表达量读出混杂」对它们不适用。"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import exprconf_lib as E
import exprconf_real as R
import subspace_lib as L


def main():
    R.need()
    F = R.load_fold()
    pairs = E.load_all(); Ed = E._GEOMS["Ed"]
    mats = {d["name"]: d for d in L.load_independent() + L.load_skempi()}
    srcs = ("WH1_v1", "WH1_v2", "Eta", "BA2", "PDCoV")
    ea = {s: E.X("EM", Ed["expr"][s]["M"], Ed["expr"][s]["mask"], 1, 3) for s in srcs}
    out = {}
    ax = {}
    for n, d in mats.items():
        M, m = np.asarray(d["M"], float), np.asarray(d["mask"], bool)
        r = E.X("EM", np.where(m, M, 0.0), m, 1, 3)
        if r is None:
            continue
        ax[n] = r
        m144, f144, _ = R.fold144(r, F)
        out[n] = {"n_sites": int(M.shape[0]), "vs_ufold": E.axis_cos(r, E.tr(F["u_fold"]))[0], "vs_uD": E.axis_cos(r, E.tr(F["u_D"]))[0],
                  "fold144_med": m144, "fold144_frac": f144, "d": E.axis_cos(r, E.tr(F["u_fold"]))[1],
                  "vs_expr": {s: E.axis_cos(r, ea[s])[0] for s in srcs}}
    # RBD 15 个矩阵的平均投影算子的第一特征向量（RBD 共识轴）与各量的比较
    vs = []
    for p in pairs:
        if p["cls"] == "SAME-LIB":
            r = E.X("EM", p["Mb"], p["kb"], 1, 3)
            v = np.zeros(20); v[r[1]] = E.cenu(r[0][:, 0]) if len(r[1]) == 20 else 0
            if len(r[1]) == 20:
                vs.append(v)
    P_ = sum(np.outer(v, v) for v in vs) / len(vs)
    w, U = np.linalg.eigh(P_)
    u_rbd = U[:, -1]
    out["_RBD_consensus"] = {"lam1": float(w[-1]), "n": len(vs), "vs_ufold": E.axis_cos(E.tr(u_rbd), E.tr(F["u_fold"]))[0], "vs_uD": E.axis_cos(E.tr(u_rbd), E.tr(F["u_D"]))[0],
                             "vs_expr": {s: E.axis_cos(E.tr(u_rbd), ea[s])[0] for s in srcs}, "fold144_med": R.fold144(E.tr(u_rbd), F)[0]}
    # OMTKY3 四个蛋白酶两两 |cos|（跨配体一致性，D-036 在 10 位点体系里能做的替代）
    om = [n for n in ax if n.startswith("OMTKY3/") ]
    pw = [E.axis_cos(ax[a], ax[b])[0] for i, a in enumerate(om) for b in om[i + 1:]]
    out["_OMTKY3_pairwise"] = {"n": len(pw), "median": float(np.median(pw)), "range": [float(min(pw)), float(max(pw))], "d": 20}
    # 表达量轴 vs u_D 的化学：载荷
    out["_loadings"] = {"u_fold": dict(zip(E.AA20, map(float, E.cenu(F["u_fold"])))), "u_D": dict(zip(E.AA20, map(float, E.cenu(F["u_D"])))),
                        "u_rbd": dict(zip(E.AA20, map(float, E.cenu(u_rbd)))), **{f"expr_{s}": dict(zip(E.AA20, map(float, E.cenu(ea[s][0][:, 0])))) for s in ("WH1_v2",)}}
    json.dump(out, open(os.path.join(E.OUT, "nondisplay.json"), "w"), ensure_ascii=False, indent=1)
    for n, v in out.items():
        if n.startswith("_"):
            if n != "_loadings": print(n, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})
            continue
        print(f"{n:42s} sites {v['n_sites']:3d} vs_ufold {v['vs_ufold']:.3f} vs_uD {v['vs_uD']:.3f} fold144 {v['fold144_med']:.3f} expr(WH1_v2/Eta/BA2/PDCoV) "
              f"{v['vs_expr']['WH1_v2']:.2f}/{v['vs_expr']['Eta']:.2f}/{v['vs_expr']['BA2']:.2f}/{v['vs_expr']['PDCoV']:.2f}")
    ld = out["_loadings"]
    for k, v in ld.items():
        print(k, " ".join(f"{a}{v[a]:+.2f}" for a in sorted(v, key=lambda a: -v[a])))


if __name__ == "__main__":
    main()
