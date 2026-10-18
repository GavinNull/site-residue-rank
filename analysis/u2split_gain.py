#!/usr/bin/env python3
"""docs/45 Step 6 的量级校准：把第二成分按幅度 λ 植入后，K1 留出增益 g12 = (trim95_r1 − trim95_r2)/trim95_r1 是多少。
（植入世界同 u2split_lib.make_world 的 R2L，中点噪声。目的：说明「可复现但太小」是否成立。事后/描述性，不进判决。）"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import json, multiprocessing as mp, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import u2split_lib as U, u2split_run as RUN, rank_k1 as K1

NAMES = ["animalACE2/Wuhan-Hu-1/hamster", "animalACE2/Wuhan-Hu-1/cat", "animalACE2/Eta/deer",
         "animalACE2/Eta/cat", "animalACE2/Omicron_BA2/human_monomer", "PDCoV/PDCoV/gAPN"]
LAMS = (0.0, 0.25, 0.5, 1.0)
REPS = 4


def task(a):
    n, lam, rep = a
    g, nz = RUN._G[n], RUN._NOISE[n]
    pt = nz["path_k2"][RUN.DOSE_POINT]
    rng = np.random.default_rng(U.seed_of("gain", n, lam, rep))
    Y = U.make_world(g, "R2L", rng, point=pt, lam=lam)
    h = K1.headline(Y, g.mask, "k1", max_rank=2)
    c = h["mean_curve"]
    return n, lam, rep, (c[1] - c[2]) / c[1], (c[0] - c[1]) / c[0], U.planted_share2(g, pt, lam)


if __name__ == "__main__":
    U.verify_lock()
    tasks = [(n, l, r) for n in NAMES for l in LAMS for r in range(REPS)]
    with mp.Pool(9, initializer=RUN._init_worker) as pool:
        res = pool.map(task, tasks, chunksize=1)
    out = {}
    for n, lam, rep, g12, g01, sh in res:
        out.setdefault(f"{lam}", {"g12": [], "g01": [], "share2": []})
        out[f"{lam}"]["g12"].append(g12); out[f"{lam}"]["g01"].append(g01); out[f"{lam}"]["share2"].append(sh)
    for k, v in out.items():
        print(k, "planted share2 median %.3f | K1 g12 median %.3f [min %.3f max %.3f] | g01 median %.3f"
              % (np.median(v["share2"]), np.median(v["g12"]), min(v["g12"]), max(v["g12"]), np.median(v["g01"])))
    json.dump(out, open(os.path.join(RUN.OUT, "gain.json"), "w"), indent=1)
