#!/usr/bin/env python3
"""docs/47 Step 7：表达量第一轴的化学身份 —— 与结合第一轴（docs/35 §3.3）同一批标度、同一 r² 口径。
自检（已知事实）：u_fold 对 CHOP780202 的 r² 复现 docs/27 §6.5 的 0.786（±0.005）；标度来自 L.load_scales()（内建 aaindex 解析自检）。"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import exprconf_lib as E
import exprconf_real as R
import subspace_lib as L
OUT, ROOT = E.OUT, E.ROOT


def r2(axis20, sc):
    ok = np.isfinite(sc)
    a = axis20[ok] - axis20[ok].mean(); s = sc[ok] - sc[ok].mean()
    return float((a @ s) ** 2 / ((a @ a) * (s @ s))), float((a @ s) / np.sqrt((a @ a) * (s @ s)))


def vec20(r):
    """(B, cols) -> 20 维（缺失残基 nan）。"""
    v = np.full(20, np.nan)
    v[r[1]] = r[0][:, 0]
    return v


def main():
    R.need()
    sc = {k: np.asarray(v, float) for k, v in L.load_scales(verbose=False).items()}
    gj = json.load(open(os.path.join(ROOT, "cache", "generic_disrupt_scales.json")))
    for k, v in gj["scales"].items():
        sc["GEN:" + k] = np.asarray(v, float)
    F = R.load_fold()
    uf = F["u_fold"]
    k0 = "sheet_prop_Chou"
    chk = r2(uf, sc[k0])[0]
    assert abs(chk - 0.786) < 0.006, f"自检失败：u_fold 对 {k0} r² = {chk:.3f}，docs/27 为 0.786"
    E.P = print
    print(f"  自检 u_fold × {k0}: r² = {chk:.3f}（docs/27 §6.5: 0.786） OK")
    rng = np.random.default_rng(E.seed47("step7null"))
    names = list(sc)
    nullmax, null1 = [], []
    for _ in range(20000):
        v = rng.normal(size=20); v -= v.mean()
        rr = [r2(v, sc[n])[0] for n in names]
        nullmax.append(max(rr)); null1.append(rr[0])
    nm95, n195 = float(np.quantile(nullmax, .95)), float(np.quantile(null1, .95))
    out = {"null_single_p95": n195, "null_max_over_scales_p95": nm95, "n_scales": len(names), "axes": {}}
    Ed = E._GEOMS.get("Ed") or (E.load_all(), E._GEOMS["Ed"])[1]
    axes = {"u_fold": uf, "u_D(binding)": F["u_D"]}
    for s in ("WH1_v1", "WH1_v2", "Eta", "BA2", "PDCoV"):
        for ext in E.EXTS:
            axes[f"expr:{s}|{ext}"] = vec20(E.X(ext, Ed["expr"][s]["M"], Ed["expr"][s]["mask"], 1, 3))
    for k, a in axes.items():
        ok = np.isfinite(a)
        rows = sorted(((n, *r2(np.where(ok, a, np.nan)[ok] if False else a, sc[n])) for n in names), key=lambda x: -x[1]) if ok.all() else \
            sorted(((n, *r2(a[ok], sc[n][ok])) for n in names), key=lambda x: -x[1])
        out["axes"][k] = [{"scale": n, "r2": r, "r": rr} for n, r, rr in rows]
    json.dump(out, open(os.path.join(OUT, "step7.json"), "w"), ensure_ascii=False, indent=1)
    print(f"  零分布 p95：单条标度 {n195:.3f}；{len(names)} 条标度取最大 {nm95:.3f}")
    for k in axes:
        top = out["axes"][k][:5]
        print(f"  {k:22s}", "; ".join(f"{t['scale']} r²={t['r2']:.2f}({t['r']:+.2f})" for t in top))


if __name__ == "__main__":
    main()
