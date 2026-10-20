#!/usr/bin/env python3
"""docs/47 汇总表（自动生成 cache/exprconf/tables.md）。只读已有 json/pkl，不做新的统计推断。"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import exprconf_lib as E
OUT = E.OUT
J = lambda n: json.load(open(os.path.join(OUT, n)))
f3 = lambda x: f"{x:.3f}"
rng_ = lambda v: f"{np.median(v):.3f} [{min(v):.3f}, {max(v):.3f}]"


def main():
    pairs = E.load_all()
    rep, q1, q1b, nd, gate, ga, gk2, gk1, q23, s7 = (J("repro.json"), J("q1.json"), J("q1boot.json"), J("nondisplay.json"), J("gate.json"),
                                                     J("gate_a.json"), J("gate_a_k2.json"), J("gate_a_k1.json"), J("q23_k2.json"), J("step7.json"))
    L = []
    L.append("## T1 可复现性（位点不相交分半，第一轴 |cos|，250 个分半，EM；K1 相差 ≤ 0.02）\n")
    L.append("| 方向 | 中位 [5%, 95%] | d=20 随机 p95 | 过闸 |\n|---|---|---|---|")
    for p in pairs:
        r = rep[f"bind|{p['id']}"]["EM"]
        L.append(f"| 结合 {p['id']} ({p['cls']}) | {r['med']:.3f} [{r['q05']:.2f}, {r['q95']:.2f}] | {r['p95']:.3f} | {'过' if r['pass'] else '不过'} |")
    for s in ("WH1_v1", "WH1_v2", "Eta", "BA2", "PDCoV"):
        r = rep[f"expr|{s}"]["EM"]
        L.append(f"| 表达量 {s} | {r['med']:.3f} [{r['q05']:.2f}, {r['q95']:.2f}] | {r['p95']:.3f} | {'过' if r['pass'] else '不过'} |")
    L.append(f"| 折叠（144 域，docs/27） | {rep['fold']['repro_med_split_half_first_axis']:.3f}（逐域位点分半中位） | 0.443 | 过 |")
    # Q1
    L.append("\n## T2 Q1：第一轴两两 |cos|（EM；20 维；d=20 随机 p95 = 0.442）\n")
    cls = {}
    for k, v in q1["pairs"].items():
        if k.endswith("|EM"):
            cls.setdefault(v["cls"], []).append(v["cos"])
    L.append("| 配对类别（结合 × 表达量） | n | 中位 [min, max] |\n|---|---|---|")
    for c in ("SAME-EXP", "SAME-LIB", "SAME-LIB-OTHERVER", "CROSS-BACKBONE", "CROSS-FAMILY", "CROSS-FAMILY-S2"):
        if c in cls:
            L.append(f"| {c} | {len(cls[c])} | {rng_(cls[c])} |")
    b_uf = [v["vs_ufold"] for k, v in q1["bind"].items() if k.endswith("|EM")]
    b_144 = [v["fold144_med"] for k, v in q1["bind"].items() if k.endswith("|EM")]
    e_uf = [v["vs_ufold"] for k, v in q1["expr"].items() if k.endswith("|EM")]
    e_144 = [v["fold144_med"] for k, v in q1["expr"].items() if k.endswith("|EM")]
    L.append("\n| 比较 | 对象 | 中位 [min, max] |\n|---|---|---|")
    L.append(f"| 结合 × 折叠（共识 u_fold） | 20 个结合矩阵的各自第一轴 | {rng_(b_uf)} |")
    L.append(f"| 结合 × 折叠（144 域逐域中位） | 同上 | {rng_(b_144)} |")
    L.append(f"| 表达量 × 折叠（共识 u_fold） | 5 个表达量源 | {rng_(e_uf)} |")
    L.append(f"| 表达量 × 折叠（144 域逐域中位） | 同上 | {rng_(e_144)} |")
    ub = [v["vs_uD"] for k, v in q1["bind"].items() if k.endswith("|EM")]
    ue = [v["vs_uD"] for k, v in q1["expr"].items() if k.endswith("|EM")]
    L.append(f"| 结合普适轴 u_D × 各结合矩阵第一轴 | 20 | {rng_(ub)} |")
    L.append(f"| 结合普适轴 u_D × 表达量第一轴 | 5 | {rng_(ue)} |")
    L.append(f"| u_D × u_fold（docs/27 的共识值） | — | {q1['ref']['uD_uF']:.3f} |")
    r = nd["_RBD_consensus"]
    L.append(f"| RBD 共识轴（15 个 S1 矩阵的平均投影算子首特征向量，λ1 = {r['lam1']:.3f}）× u_fold / u_D / 144 域 | — | {r['vs_ufold']:.3f} / {r['vs_uD']:.3f} / {r['fold144_med']:.3f} |")
    L.append("\n偏相关（轴层面，(c_bf − c_be·c_ef)/√((1−c_be²)(1−c_ef²))，符号对齐）：\n")
    pc = [v["partial"] for k, v in q1["partial"].items() if k.endswith("|EM")]
    L.append(f"20 个结合矩阵的中位 {np.median(pc):.3f}，范围 [{min(pc):.3f}, {max(pc):.3f}]（三个轴两两 ≥ 0.95 时该量对测量噪声极敏感，不单独解读）")
    L.append("\n位点 bootstrap（3000 次；对子空间对齐向下偏，只读相对差）结合×表达量 |cos| 的中位 [2.5%, 97.5%]：")
    L.append("\n| 配对 | be | bf | ef |\n|---|---|---|---|")
    for k, v in q1b.items():
        L.append(f"| {k} | {v['be'][1]:.3f} [{v['be'][0]:.3f}, {v['be'][2]:.3f}] | {v['bf'][1]:.3f} [{v['bf'][0]:.3f}, {v['bf'][2]:.3f}] | {v['ef'][1]:.3f} [{v['ef'][0]:.3f}, {v['ef'][2]:.3f}] |")
    L.append("\n## T3 非酵母展示体系的第一轴（D-036 跨源）\n")
    L.append("| 体系 | 位点 | × u_fold | × u_D | × 折叠144 | × 表达量 WH1_v2 / Eta / BA2 / PDCoV |\n|---|---|---|---|---|---|")
    for n, v in nd.items():
        if n.startswith("_"):
            continue
        L.append(f"| {n} | {v['n_sites']} | {v['vs_ufold']:.3f} | {v['vs_uD']:.3f} | {v['fold144_med']:.3f} | {v['vs_expr']['WH1_v2']:.2f} / {v['vs_expr']['Eta']:.2f} / {v['vs_expr']['BA2']:.2f} / {v['vs_expr']['PDCoV']:.2f} |")
    pw = nd["_OMTKY3_pairwise"]
    L.append(f"\nOMTKY3 四个蛋白酶两两第一轴 |cos|：中位 {pw['median']:.3f}，范围 [{pw['range'][0]:.3f}, {pw['range'][1]:.3f}]（d=20 p95 0.442）")
    # gates
    L.append("\n## T4 门\n")
    L.append("| 层 | 校正 | HIT-1 | HIT-2 | HIT-3 | FA-1/2 | FA-3 | 全过 |\n|---|---|---|---|---|---|---|---|")
    yn = lambda b: "过" if b else "**不过**"
    for nm, g in (("锁定·池化", gate["gates"]), ("A1·池化", ga["gates"]), ("A2·K2 层", gk2["gates"]), ("A2·K1 层", gk1["gates"])):
        for c, v in g.items():
            L.append(f"| {nm} | {c} | {yn(v['HIT1'])} | {yn(v['HIT2'])} | {yn(v['HIT3'])} | {yn(v['FA12'])} | {yn(v['FA3'])} | {yn(v['ALL'])} |")
    L.append(f"\nδ0（A1·池化，(i-c) 干净世界下降 p95）= {ga['delta0']['p95_drop_ic']:.4f}，取 {ga['delta0']['delta0']:.2f}；牙齿：NOOP 在 HIT-1 失败 = {ga['teeth']['NOOP_fails_HIT1']}；DESTROY（锁定层）失败 = {gate['teeth']['DESTROY_fails_HIT2_or_FA2']}")
    L.append("\n梯度（A1·池化，EM；中位）：\n\n| 世界 | 未校正 cb1 / ce1 | ia cb1 / ce1 | ic cb1 / ce1 |\n|---|---|---|---|")
    for sc in ("BS", "GAM0.25", "GAM0.5", "GAM1", "GAM2", "GAMF1", "GAMF2"):
        a, b, c = (ga["criteria"][x]["ladder"][sc] for x in ("none", "ia", "ic"))
        L.append(f"| {sc} | {a['cb1']:.2f} / {a['ce1']:.2f} | {b['cb1']:.2f} / {b['ce1']:.2f} | {c['cb1']:.2f} / {c['ce1']:.2f} |")
    L.append("\n只报告、不入门（A1·池化，EM 中位）：\n\n| 世界 | 校正 | 校正前 cb1 / ce1obs | 校正后 cb1 / ce1obs |\n|---|---|---|---|")
    for sc in ("CORR0.30", "PARRES", "PARCELL"):
        for c in ("ia", "ic"):
            r = ga["criteria"][c]["report"][sc]
            L.append(f"| {sc} | {c} | {r['cb1_unc']:.2f} / {r['ce1obs_unc']:.2f} | {r['cb1_cor']:.2f} / {r['ce1obs_cor']:.2f} |")
    # K2 stratum
    L.append("\n## T5 K2 层（探索性，不进判决）：未校正 vs 校正后（ic；K1 增益的位点 bootstrap 2000 区间）\n")
    L.append("| 配对 | n 格 | 去掉的 rms 份额 | 第一轴×u_fold 前→后 | ×144域中位 前→后 | ×表达量轴(观测) 前→后 | 第一轴分半自复现 前→后 | g01 前 | g01 后 | g12 后 | g23 后 |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for pid, v in q23.items():
        a, c = v["none|EM"], v["ic|EM"]
        ga_, gc = v["none|K1"], v["ic|K1"]
        L.append(f"| {pid} | {c['n']} | {c['share']:.2f} | {a['vs_ufold']:.3f}→{c['vs_ufold']:.3f} | {a['fold144_med']:.3f}→{c['fold144_med']:.3f} | {a['vs_exprobs']:.3f}→{c['vs_exprobs']:.3f} | {a['split_ax'][0]:.2f}→{c['split_ax'][0]:.2f} | "
                 f"{ga_['gain']['g01']:+.3f} [{ga_['gain_ci']['g01'][0]:+.3f},{ga_['gain_ci']['g01'][1]:+.3f}] | {gc['gain']['g01']:+.3f} [{gc['gain_ci']['g01'][0]:+.3f},{gc['gain_ci']['g01'][1]:+.3f}] | "
                 f"{gc['gain']['g12']:+.3f} [{gc['gain_ci']['g12'][0]:+.3f},{gc['gain_ci']['g12'][1]:+.3f}] | {gc['gain']['g23']:+.3f} [{gc['gain_ci']['g23'][0]:+.3f},{gc['gain_ci']['g23'][1]:+.3f}] |")
    # unit aggregation
    L.append("\n单位层（K2 层，ic，EM 第一轴；SAME-LIB 单位 = 骨架内合格 animal 配体的中位；只报每单位的值与范围）：\n")
    L.append("| 单位 | 配对 | ×u_fold 前 | ×u_fold 后 | 降幅 | ×144 前 | ×144 后 | 分半自复现 前 | 后 | 去掉份额 | 分类(δ0=0.05) |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for lab, sel in (("WH1 animal", lambda p: p.startswith("WH1/")), ("Eta animal", lambda p: p.startswith("Eta/")), ("SAME-EXP hACE2", lambda p: p.startswith("hACE2"))):
        ids_ = [p for p in q23 if sel(p)]
        g = lambda c, key, sub=None: np.array([q23[i][c][key] if sub is None else q23[i][c][key][sub] for i in ids_], float)
        cu, cc = g("none|EM", "vs_ufold"), g("ic|EM", "vs_ufold")
        u1, c1 = g("none|EM", "fold144_med"), g("ic|EM", "fold144_med")
        su, sc_ = g("none|EM", "split_ax", 0), g("ic|EM", "split_ax", 0)
        sh = g("ic|EM", "share")
        mu, mc = float(np.median(cu)), float(np.median(cc))
        cl = "塌缩" if (mc <= 0.443 and mu - mc > 0.05) else ("完整存活" if mu - mc <= 0.05 else "部分存活")
        rg = lambda a: f"{np.median(a):.3f} [{a.min():.3f}, {a.max():.3f}]"
        L.append(f"| {lab} | {len(ids_)} | {rg(cu)} | {rg(cc)} | {mu - mc:.3f} | {rg(u1)} | {rg(c1)} | {rg(su)} | {rg(sc_)} | {rg(sh)} | {cl} |")
    # sensitivities
    L.append("\n敏感性（K2 层；各校正的中位；注意各自的门状态：ia 不过 HIT-1/2/3；ib 不过 FA-2/3；ii 不过 HIT-3、FA-3；ic 过）：\n")
    L.append("| 校正 | ×u_fold 中位 [min,max] | 分半自复现 中位 | g01 后 中位 | g01 区间下界>0 的矩阵数 | 去掉份额 中位 |\n|---|---|---|---|---|---|")
    for c in ("none", "ia", "ib", "ii", "ic"):
        vf = [q23[i][f"{c}|EM"]["vs_ufold"] for i in q23]
        sp = [q23[i][f"{c}|EM"]["split_ax"][0] for i in q23]
        g01 = [q23[i][f"{c}|K1"]["gain"]["g01"] for i in q23]
        lo = sum(q23[i][f"{c}|K1"]["gain_ci"]["g01"][0] > 0 for i in q23)
        sh = [q23[i][f"{c}|EM"]["share"] for i in q23]
        L.append(f"| {c} | {rng_(vf)} | {np.median(sp):.3f} | {np.median(g01):+.3f} | {lo}/{len(q23)} | {np.median(sh):.2f} |")
    # step 7
    L.append("\n## T6 Step 7：第一轴与标度的 r²（EM；零分布 单条 p95 = %.3f；%d 条标度取最大 p95 = %.3f）\n" % (s7["null_single_p95"], s7["n_scales"], s7["null_max_over_scales_p95"]))
    want = ["sheet_prop_Chou", "pi_octanol_FauchereP", "kd_hydropathy", "partial_specific_volume", "molar_refractivity", "local_polarizability", "steric_upsilon", "GEN:D1_folding_ddG_ddPCA", "GEN:D2_blosum62_nonexchangeability", "GEN:D5_disrupt_PC1", "GEN:D3_starr_expression_loss"]
    axes = ["u_fold", "u_D(binding)", "expr:WH1_v1|EM", "expr:WH1_v2|EM", "expr:Eta|EM", "expr:BA2|EM", "expr:PDCoV|EM"]
    L.append("| 标度 | " + " | ".join(a.replace("expr:", "表达量 ").replace("|EM", "") for a in axes) + " |\n|---|" + "---|" * len(axes))
    for w in want:
        row = []
        for a in axes:
            d = {r["scale"]: r for r in s7["axes"][a]}
            row.append(f"{d[w]['r2']:.2f}")
        L.append(f"| {w.replace('GEN:', '')} | " + " | ".join(row) + " |")
    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
