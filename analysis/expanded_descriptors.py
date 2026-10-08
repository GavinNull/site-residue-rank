#!/usr/bin/env python3
"""为扩充后的 SKEMPI 数据（20 体系 / 7 个矩阵体系）算位点描述子。

此前的跨体系验证只有 3 个体系（OMTKY3 40 位点、4-4-20 13、BPTI 2），
而 OMTKY3 一家占主导，样本外测试近乎无意义。
把取代数门槛从 6 降到 4 后独立体系 7 -> 20，可建矩阵的体系 1 -> 7。
"""
import collections
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_orientation import parse_pdb, sub, norm, unit, dot, ROOT
from rotamer_build import CB_SPEC, nerf

SHELLS = ((1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 6.0), (6.0, 8.0))


def site_descriptors(pid, chain, resi, nsamp=800, excl=2.7, bur_r=8.0):
    """返回 (burial, 各壳自由比例)。伙伴 = 该 PDB 里除本链外的全部非水原子。"""
    path = os.path.join(ROOT, "cache", "pdb", f"{pid}.pdb")
    if not os.path.exists(path):
        return None
    atoms = parse_pdb(path)
    own = {a["name"]: a["xyz"] for a in atoms
           if a["chain"] == chain and a["resi"] == str(resi) and not a["het"]}
    if not all(x in own for x in ("N", "CA", "C")):
        return None
    nm, a, b, c, bd, an, dh = CB_SPEC
    cb = nerf(own[a], own[b], own[c], bd, an, dh[1])
    u = unit(sub(cb, own["CA"]))
    env = [x["xyz"] for x in atoms if x["res"] != "HOH"
           and not (x["chain"] == chain and x["resi"] == str(resi)
                    and not x["het"])
           and norm(sub(x["xyz"], cb)) < 12.0]
    bur = sum(1 for p in env if norm(sub(p, cb)) <= bur_r)
    E = np.array(env) if env else np.zeros((0, 3))
    rng = np.random.default_rng(42)
    out = []
    for lo, hi in SHELLS:
        free = tot = 0
        for _ in range(nsamp):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            if dot(u, tuple(v)) < 0.0:
                continue
            r = (lo ** 3 + (hi ** 3 - lo ** 3) * rng.random()) ** (1 / 3)
            p = np.array(cb) + r * v
            tot += 1
            if len(E) == 0 or np.min(np.linalg.norm(E - p, axis=1)) >= excl:
                free += 1
        out.append(free / tot if tot else float("nan"))
    return float(bur), out


def main():
    rows = list(csv.DictReader(open(os.path.join(
        ROOT, "cache", "skempi_thr4.csv"))))
    sites = sorted({(r["system"], r["pdb"], r["chain"], r["position"])
                    for r in rows})
    print(f"skempi_thr4: {len(rows)} 记录, {len(sites)} 个位点")
    out = {}
    ok = bad = 0
    for s in sites:
        d = site_descriptors(s[1], s[2], s[3])
        if d is None:
            bad += 1
            continue
        out["|".join(s)] = {"burial": d[0], "shells": d[1]}
        ok += 1
    print(f"算出描述子 {ok} 个位点，失败 {bad}（多为 Gly 无 CB 或链/编号不匹配）")
    json.dump({"shells": SHELLS, "sites": out},
              open(os.path.join(ROOT, "cache", "expanded_site_desc.json"), "w"))
    # 按体系报告覆盖
    bysys = collections.Counter(k.split("|")[0] for k in out)
    print("\n每个体系算出的位点数:")
    for s, n in bysys.most_common():
        print(f"   {s[:46]:<46} {n:>3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
