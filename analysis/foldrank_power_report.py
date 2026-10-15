#!/usr/bin/env python3
"""docs/42 功效结果的补充汇总（次要量，预先登记 §3 的 (a)(b) 与量级核对）。读 cache/foldrank_power.json。"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import ROOT
R = json.load(open(os.path.join(ROOT, "cache", "foldrank_power.json")))
D = R["domains"]; names = sorted(D)
print("== 量级核对：合成秩2情景实现的残差SD / 实测cso（应≈1）")
for t in "ACB":
    v = [D[w]["anchor"][t]["cso_sim_over_obs"] for w in names]
    print(f" {t}: 中位 {np.median(v):.3f}  范围 [{min(v):.3f},{max(v):.3f}]")
print("\n== 情景强度 s/σ 分布")
for t in "ACB":
    v = np.array([D[w]["anchor"][t]["s_over_sigma"] for w in names])
    k = np.mean([D[w]["anchor"][t]["kind"] == "consistent" for w in names])
    print(f" {t}: s/σ 中位 {np.median(v):.2f} 四分位 [{np.percentile(v,25):.2f},{np.percentile(v,75):.2f}] 范围[{v.min():.2f},{v.max():.2f}]；相容比例 {k:.2f}")
print("\n== sens 与 s/σ 的关系（锚点A、C 的原始情景；按 s/σ 分箱，域数/平均 P(检出2|真2)/平均 P(检出2|真1)/P(检出3|真2)）")
for t in "AC":
    x = np.array([D[w]["anchor"][t]["s_over_sigma"] for w in names])
    se = np.array([D[w]["anchor"][t]["sens"] for w in names])
    fp = np.array([D[w]["anchor"][t]["fp1"] for w in names])
    d3 = np.array([D[w]["anchor"][t]["confusion"][2][3] for w in names])
    d1 = np.array([D[w]["anchor"][t]["confusion"][2][1] for w in names])
    bins = [0, .5, .7, .9, 1.2, 2, 3, 4, 5, 6, 99]
    print(f" 锚点 {t}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum():
            print(f"   s/σ∈[{lo},{hi}): n={m.sum():3d}  sens={se[m].mean():.2f}  fp1={fp[m].mean():.2f}  真2检出1={d1[m].mean():.2f} 检出3={d3[m].mean():.2f}")
print("\n== 补充(a)：σ_C 上 s 网格，P(检出2|真2) 与 P(检出2|真1)（域均值；s 绝对值 kcal/mol）")
G = {}
for w in names:
    for k, v in D[w]["grid"].items():
        s, r = k.split("|"); G.setdefault((float(s), int(r)), []).append(v)
for (s, r) in sorted(G):
    a = np.array(G[(s, r)]); print(f"  s={s:.2f} 真{r}: 检出分布均值 {np.round(a.mean(0),2).tolist()}  P(检出2) 达 0.8 的域 {int((a[:,2]>=.8).sum())}/{len(a)}")
print("\n== 上界：σ_C 下，把实测残差全当秩2 信号的上限(s=cso)时 P(检出2|真2) ≥ 0.8 的域（用 s 网格就近：s 网格最大0.70）")
cso = np.array([D[w]["cso"] for w in names]); print(f" cso 中位 {np.median(cso):.3f} 范围[{cso.min():.3f},{cso.max():.3f}]；cso>0.70 的域 {(cso>0.70).sum()}")
print("\n== 补充(b)：谱等权秩2 的 P(检出2|真2)")
for t in "ACB":
    a = np.array([D[w]["equal_spec"][t] for w in names]); cons = np.array([D[w]["anchor"][t]["kind"]=="consistent" for w in names])
    print(f" {t}: 均值 {a[:,2].mean():.2f}  ≥0.8 的域 {(a[:,2]>=.8).sum()}（相容情景内 {((a[:,2]>=.8)&cons).sum()}）；检出分布均值 {np.round(a.mean(0),2).tolist()}")
print("\n== 被纳入域与其 s/σ")
for t in "ACB":
    print(t, [(w, round(D[w]['anchor'][t]['s_over_sigma'],2), D[w]['cluster']) for w in names if D[w]['anchor'][t]['admitted']])
print("\n== 上限情景(ceiling)满足判据的域:", [(w, D[w]['cluster']) for w in names if D[w]['anchor']['B']['ceiling_admitted']])
print("\n== 锚点A：秩1真值被检出为>=2 的域比例（C6b 偏高偏差）")
f = np.array([sum(D[w]["anchor"]["A"]["confusion"][1][2:]) for w in names]); print(f" 均值 {f.mean():.2f}; 中位 {np.median(f):.2f}")
print("\n== 簇数检查:", len({D[w]['cluster'] for w in names}), " 最大可能被纳入簇数(锚点A, 即便放宽 sens>=0.5 且 fp1<=0.5):",
      len({D[w]['cluster'] for w in names if D[w]['anchor']['A']['sens']>=.5 and D[w]['anchor']['A']['fp1']<=.5 and D[w]['anchor']['A']['fp0']<=.2}))
