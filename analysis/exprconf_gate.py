#!/usr/bin/env python3
"""docs/47 双向对照世界的门（预先登记 §3）。阶段：selftest / fid / worlds / report。先于任何真实结论。"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse, json, multiprocessing as mp, os, pickle, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import exprconf_lib as E
import d46_lib as D
OUT = E.OUT
WD = os.path.join(OUT, "worlds"); os.makedirs(WD, exist_ok=True)
CORRS_G = ("none", "ia", "ib", "ii", "dst")
GAIN_SCEN = ("BS", "GAM2", "PURE2")
PWD = os.path.join(OUT, "worlds_pw"); os.makedirs(PWD, exist_ok=True)
GAIN_SCEN_A = ("BS", "GAM2", "PURE2", "GAMF2", "PUREF2")
WDA = os.path.join(OUT, "worlds_a"); os.makedirs(WDA, exist_ok=True)


def P(*a):
    print(*a, flush=True)


def jdump(o, path):
    json.dump(o, open(path, "w"), ensure_ascii=False, indent=1,
              default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else bool(x) if isinstance(x, np.bool_) else str(x))


def stage_selftest(a):
    E.verify_lock47()
    import rank_core as RC, rank_k1 as K1
    RC.selfcheck(verbose=False); P("  rank_core.selfcheck 全过  OK")
    K1.selfcheck(verbose=False); P("  rank_k1.selfcheck 全过  OK")
    gh = json.load(open(os.path.join(E.ROOT, "cache", "gate_harness_check.json")))
    assert gh.get("ok") is True; P("  cache/gate_harness_check.json ok = true  OK")
    import subprocess
    for sc in ("subspace_selftest.py", "censor_density_check.py", "gate_harness_check.py"):
        r = subprocess.run([sys.executable, os.path.join(HERE, sc)], capture_output=True, text=True, cwd=E.ROOT)
        assert r.returncode == 0, f"{sc} 未通过:\n{r.stdout[-1500:]}\n{r.stderr[-800:]}"
        P(f"  {sc} 通过  OK")
    gh = json.load(open(os.path.join(E.ROOT, "cache", "gate_harness_check.json"))); assert gh.get("ok") is True
    log = E.selfcheck47()
    # 世界生成器的已知事实（T8）：γ 部分的 rms = γ·s；ρ 实现；Vb ⟂ Ve
    pairs = E.load_all()
    p = [q for q in pairs if q["id"] == "WH1/deer"][0]
    rng = np.random.default_rng(1)
    Y, tr = E.build_world(p, "GAM2", 3, rng)
    s = p["path"][3]["s"]
    m = p["gb"].mask
    rms = lambda A: float(np.sqrt((A[m] ** 2).mean()))
    assert abs(rms(tr["Ie"]) - 2.0 * s) / (2 * s) < 1e-9 and abs(rms(tr["Ib"]) - s) / s < 1e-9
    assert np.abs(tr["Vb"].T @ tr["Ve"]).max() < 1e-8
    # ρ：校正后噪声与表达量噪声的实现相关
    rs = []
    for k in range(4):
        Yc, trc = E.build_world(p, "CORR0.30", 3, np.random.default_rng(k))
        Y0, _ = E.build_world(p, "BS", 3, np.random.default_rng(k))
        # 同一 rng 种子 => 同一 Ib，差即噪声差；与表达量噪声的相关
        K = m & p["ke"]
        De = (E.RC.residual(p["ge"].M, p["ke"]) - p["ge"].fit[2]["P"])
        rs.append(float(np.corrcoef((Yc - Y0)[K], De[K])[0, 1]))
    # (Yc−Y0) 含 √(1−ρ²)·噪声_b 与 噪声_b 之差，不是干净的相关；改用直接检验：噪声部分 = Yc − (r+c+Ib)
    rr = []
    for k in range(6):
        rng_ = np.random.default_rng(100 + k)
        Yc, trc = E.build_world(p, "CORR0.30", 3, rng_)
        noise = (Yc - p["gb"].r[:, None] - p["gb"].c[None, :] - trc["Ib"] - trc["Ie"])
        K = m & p["ke"]
        De = (E.RC.residual(p["ge"].M, p["ke"]) - p["ge"].fit[2]["P"])
        rr.append(float(np.corrcoef(noise[K], De[K])[0, 1]))
    assert abs(np.mean(rr) - 0.30) < 0.03, f"T8 实现的噪声相关 {np.mean(rr):.3f} ≠ 0.30"
    log.append(f"T8 世界生成器：γ 部分 rms = γ·s、Vb ⟂ Ve（精确）；CORR0.30 实现的噪声相关 {np.mean(rr):.3f}（目标 0.30）  OK")
    P("  " + log[-1])
    jdump({"log": log, "lock": E.verify_lock47(False)}, os.path.join(OUT, "selftest.json"))
    P("先决自检全部通过")


def need():
    E.verify_lock47(False)
    assert os.path.exists(os.path.join(OUT, "selftest.json")), "先跑 selftest（CLAUDE.md §10）"


def stage_fid(a):
    """SAME-EXP 结合矩阵（omicron hACE2）的 RR 世界保真（A1 判据，k1 与 k2 的 7 个路径点）。"""
    need()
    pairs = E.load_all()
    res = {}
    for p in pairs:
        if p["cls"] != "SAME-EXP":
            continue
        g = p["gb"]
        ref = D.real_ref(g)
        sig = g.sig["sigma_cell"]
        ok = True
        info = {}
        for kind, k in (("R1", 1), ("R2", 2)):
            path = g.path(k, sig)
            info[kind] = len(path)
            for pi, pt in enumerate(path):
                r = D.fid_point(g, ref, "RR", kind, pt, "ec47fid")
                ok &= r["pass_am"]
                info[f"{kind}{pi}"] = {k_: r[k_] for k_ in ("f1a", "f2", "f3", "pass_am")}
        ref_ok = all(D.fid_point(g, ref, "REF", kd, None, "ec47ref")["pass_am"] for kd in ("R1", "R2"))
        res[p["id"]] = {"RR_faithful_A1": bool(ok and info["R1"] == 7 and info["R2"] == 7), "REF_pass": bool(ref_ok), "path_len": [info["R1"], info["R2"]], "pts": info}
        P(f"  {p['id']}: 路径点 {info['R1']}/{info['R2']}；RR 忠实(A1) {res[p['id']]['RR_faithful_A1']}；REF {ref_ok}")
    jdump(res, os.path.join(OUT, "fid_bindH.json"))


def set_faithful():
    pairs = E.load_all()
    fb = json.load(open(os.path.join(OUT, "fid_bindH.json"))) if os.path.exists(os.path.join(OUT, "fid_bindH.json")) else {}
    for p in pairs:
        if p["cls"] == "SAME-EXP":
            p["faithful_bindH"] = bool(fb.get(p["id"], {}).get("RR_faithful_A1", False))
    return pairs


_PAIRS = None


def _init():
    global _PAIRS
    _PAIRS = set_faithful()


def _task(t):
    pi, scen, pt = t
    path = os.path.join(WD, f"{pi}__{scen}__{pt}.pkl")
    if os.path.exists(path):
        return t
    p = _PAIRS[pi]
    gain_reps = 12
    recs = []
    for rep in range(E.R_W):
        rng = np.random.default_rng(E.seed47("world", p["id"], scen, pt, rep))
        Y, tr = E.build_world(p, scen, pt, rng)
        wg = scen in GAIN_SCEN and pt == 3 and rep < gain_reps
        recs.append(E.eval_instance(p, Y, tr, E.seed47("eval", p["id"], scen, pt, rep), corrs=CORRS_G, with_gain=wg))
    pickle.dump(recs, open(path, "wb"))
    return t


def _task_a(t):
    pi, scen, pt = t
    path = os.path.join(WDA, f"{pi}__{scen}__{pt}.pkl")
    if os.path.exists(path):
        return t
    p = _PAIRS[pi]
    recs = []
    for rep in range(E.R_W):
        rng = np.random.default_rng(E.seed47("world", p["id"], scen, pt, rep))
        Y, tr = E.build_world(p, scen, pt, rng)
        wg = scen in GAIN_SCEN_A and pt == 3 and rep < 12
        recs.append(E.eval_instance(p, Y, tr, E.seed47("eval", p["id"], scen, pt, rep), corrs=E.CORRS_A, with_gain=wg))
    pickle.dump(recs, open(path, "wb"))
    return t


def _task_pw(t):
    pi, scen, pt = t
    path = os.path.join(PWD, f"{pi}__{scen}__{pt}.pkl")
    if os.path.exists(path):
        return t
    p = _PAIRS[pi]
    recs = []
    for rep in range(E.R_W):
        rng = np.random.default_rng(E.seed47("world", p["id"], scen, pt, rep))
        Y, tr = E.build_world(p, scen, pt, rng)
        recs.append(E.eval_instance(p, Y, tr, E.seed47("eval", p["id"], scen, pt, rep), corrs=("none", "ic"), exts=("EM", "K1"), split_ext=("EM",), with_gain=True))
    pickle.dump(recs, open(path, "wb"))
    return t


def stage_power(a):
    """§10.3 还原性诊断（修订 A2 层）：结合特异振幅 f·s，f ∈ {0, .1, .2, .3, .5}；K2 层配对；校正 none/ic；K1 g01 + 第一轴自复现。"""
    need()
    E.verify_lock47a(); E.verify_lock47b()
    pairs = set_faithful()
    el = [i for i, p in enumerate(pairs) if p["src"] in ("WH1_v1", "WH1_v2", "Eta") and len(p["path"]) == 7 and (p["faithful"] is True or (p["cls"] == "SAME-EXP" and p.get("faithful_bindH")))]
    tasks = [(i, sc, 3) for i in el for sc in E.POWER]
    t0 = time.time()
    with mp.Pool(a.jobs, initializer=_init) as pool:
        for k, _ in enumerate(pool.imap_unordered(_task_pw, tasks, chunksize=1), 1):
            if k % 10 == 0 or k == len(tasks):
                P(f"    {k}/{len(tasks)}  {time.time() - t0:.0f}s")
    P("ALL DONE PW")


def stage_worlds_a(a):
    """修订 A1 层：corr ∈ none/ia/ic，13 个世界（含 GAMF/PUREF2）。种子与锁定层逐世界相同（同一 build_world 随机数）。"""
    need()
    E.verify_lock47a()
    pairs = set_faithful()
    el = [i for i, p in enumerate(pairs) if len(p["path"]) == 7 and (p["faithful"] is True or (p["cls"] == "SAME-EXP" and p.get("faithful_bindH")))]
    tasks = [(i, sc, pt) for i in el for sc in E.SCEN for pt in E.PT_IDX]
    t0 = time.time()
    with mp.Pool(a.jobs, initializer=_init) as pool:
        for k, _ in enumerate(pool.imap_unordered(_task_a, tasks, chunksize=1), 1):
            if k % 20 == 0 or k == len(tasks):
                P(f"    {k}/{len(tasks)}  {time.time() - t0:.0f}s")
    P("ALL DONE A")


def stage_worlds(a):
    need()
    pairs = set_faithful()
    el = [i for i, p in enumerate(pairs) if len(p["path"]) == 7 and (p["faithful"] is True or (p["cls"] == "SAME-EXP" and p.get("faithful_bindH")))]
    P("  合格配对:", [pairs[i]["id"] for i in el])
    P("  不合格（排除）:", [(p["id"], p["faithful"], len(p["path"])) for i, p in enumerate(pairs) if i not in el])
    tasks = [(i, sc, pt) for i in el for sc in E.SCEN if sc not in ('GAMF1', 'GAMF2', 'PUREF2') for pt in E.PT_IDX]
    t0 = time.time()
    with mp.Pool(a.jobs, initializer=_init) as pool:
        for k, _ in enumerate(pool.imap_unordered(_task, tasks, chunksize=1), 1):
            if k % 20 == 0 or k == len(tasks):
                P(f"    {k}/{len(tasks)}  {time.time() - t0:.0f}s")
    json.dump({"eligible": [pairs[i]["id"] for i in el]}, open(os.path.join(OUT, "eligible.json"), "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--jobs", type=int, default=9)
    a = ap.parse_args()
    {"selftest": stage_selftest, "fid": stage_fid, "worlds": stage_worlds, "worlds_a": stage_worlds_a, "power": stage_power}[a.stage](a)
