#!/usr/bin/env python3
"""由 cache/independence_audit.json 生成 docs/41 用的 markdown 表（不重算任何东西）。

输出 cache/independence_tables.md。docs/41 里的表就是从这里贴的，
所以文档数字与 JSON 的一致性可以机械核对（python3 analysis/independence_tables.py --check docs/41-independence-audit.md）。
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "cache", "independence_audit.json")))


def pc(x, d=1, sign=True):
    if x is None:
        return "n/a"
    return f"{100 * x:+.{d}f}%" if sign else f"{100 * x:.{d}f}%"


def ci(c, d=1):
    return f"[{pc(c[0], d)}, {pc(c[1], d)}]"


def build():
    L = []
    pn = R["panel"]
    rp = R["reproduction"]

    # ---- 15 个子集
    L.append("### T1 十五个子集（秩 1→2、2→3 增益，位点联合 bootstrap 2000 次的区间）\n")
    L.append("| 骨架 | 配体 | 观测 | 检出秩(3 种子) | 秩1→2 | 区间 | 秩2→3 | 区间 | 秩0→2 | 区间 | 单独判 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in R["matrices"]:
        if m["system"] != "S1":
            continue
        _, b, lg = m["name"].split("/")
        c = m["ci"]
        L.append(f"| {b} | {lg} | {m['n_obs']} | {m['mode_rank']} ({m['stab']:.1f}) | "
                 f"{pc(c['g12']['point'])} | {ci(c['g12']['ci'])} | {pc(c['g23']['point'])} | {ci(c['g23']['ci'])} | "
                 f"{pc(c['g02']['point'])} | {ci(c['g02']['ci'])} | {m['status_matrix']} |")
    n_s2 = sum(1 for m in R["matrices"] if m["system"] == "S1" and m["status_matrix"] == "SUPPORTS_2")
    n_le1 = sum(1 for m in R["matrices"] if m["system"] == "S1" and m["status_matrix"] == "SUPPORTS_LE1")
    n_ind = sum(1 for m in R["matrices"] if m["system"] == "S1" and m["status_matrix"] == "INDETERMINATE")
    L.append(f"\n逐子集单独判：区间在 0 之上 **{n_s2}**，跨 0 **{n_ind}**，在 0 之下 **{n_le1}**"
             f"（点估计为正的有 {rp['n_pos']}）。\n")

    # ---- 单元阶梯
    a = pn["U3"]["g12"]["anova"]
    a23 = pn["U3"]["g23"]["anova"]
    u4 = pn["U4_site"]
    cb = pn["U3U4_combined"]
    L.append("### T2 面板（15 个子集）在各候选单元下的秩 1→2 检验\n")
    L.append("| 单元 | 独立单元数 / 有效 n | 统计量 | p 或区间 | 判读 |")
    L.append("|---|---|---|---|---|")
    L.append(f"| U0 子集（原文） | 15 名义 | {pn['U0']['k_pos']}/15 为正 | 单侧精确 p = {pn['U0']['p_one_sided']:.5f}；双侧 {rp['p_two_sided']:.5f} | 把 15 当独立 |")
    u1, u2 = pn["U1_backbone"], pn["U2_ligand"]
    L.append(f"| U1 骨架 | 3 | {u1['k_pos_median']}/3 | 形式精确 p = {u1['p_formal']:.4f}（最小可达 {u1['min_attainable_p']:.4f}）；块翻转 p = {u1['blockflip_p_count']:.4f} | **不可解释**：n=3，任何结果都 >0.05 |")
    L.append(f"| U2 配体 | 5 | {u2['k_pos_median']}/5 | 形式精确 p = {u2['p_formal']:.4f}（最小可达 {u2['min_attainable_p']:.4f}）；块翻转 p = {u2['blockflip_p_count']:.4f} | **形式值，非证据**；且自检显示单向聚类在交叉设计下虚警约 10% |")
    L.append(f"| U3 骨架×配体交叉 | n_eff = {a['n_eff']:.1f}（分量截到 0；ICC 骨架 {a['icc_row']:.2f}、配体 {a['icc_col']:.2f}），Satterthwaite df = {a['df']:.1f} | 均值增益 {pc(a['mean'],2)} | t 区间 {ci(a['ci'],1)}；单侧 t p = {pn['U3']['g12']['p_one_sided_t']:.3f} | 唯一在交叉设计下校准过的检验；df 只有 {a['df']:.1f} |")
    L.append(f"| U3 交叉 bootstrap（中位） | 3 + 5 水平 | 中位增益 | 骨架+配体 {ci(pn['U3']['g12']['boot_crossed_ci'])}；仅骨架 {ci(pn['U3']['g12']['boot_rows_ci'])}；仅配体 {ci(pn['U3']['g12']['boot_cols_ci'])} | 3 个骨架水平上欠覆盖，区间偏窄 |")
    L.append(f"| U4 位点（联合重抽） | 201 位点，1 个面板；计数的设计效应 {u4['design_effect_count']:.2f}（n_eff = {u4['n_eff_count']:.1f}） | 中位增益 | {ci(u4['median_g12_ci'])}；重抽下 ≥14/15 为正的比例 {u4['npos_g12_dist']['frac_eq_14_or_more']:.2f}，≥13/15 为 {u4['npos_g12_dist']['frac_ge_13']:.2f}，中位数 {u4['npos_g12_dist']['median']:.0f}/15 | 只含位点抽样 |")
    L.append(f"| **U3+U4 同时重抽骨架、配体、位点** | 3 + 5 水平 + 201 位点 | 中位增益 | **{ci(cb['g12']['ci'])}**；重抽中 ≤0 的比例 {cb['g12']['frac_le0']:.3f} | 最宽、最诚实的面板内区间；跨 0 |")
    pp = pn["perm_null"]
    L.append(f"| 位点内联合残基置换（保留跨矩阵共享格子噪声） | {pp['nperm']} 次 | 为正的子集数 | 观测(seed1) {pp['obs_count_seed1']}/15；零分布均值 {pp['null_count_mean']:.2f}，最大 {pp['null_count_max']} | p ≤ {pp['p_count']:.4f}（分辨率 1/{pp['nperm']+1}）。排除「共享噪声单独造出多数为正」 |")
    L.append("")

    # ---- 内部极小在面板内
    im = pn["interior_min"]
    L.append("### T3 内部极小在面板内（秩 2→3，负 = 秩 3 更差）\n")
    L.append("| 单元 | 结果 |")
    L.append("|---|---|")
    L.append(f"| U0 | 秩 3 劣于秩 2：{im['k_leg2_rank3_worse']}/15；两条腿（2 优于 1 且 3 劣于 2）同时成立：{im['k_both_legs']}/15 |")
    L.append(f"| U1/U2 块翻转（秩3更差的计数） | 骨架 p = {im['leg2_blockflip_rows_p']:.3f}（下限 0.125）；配体 p = {im['leg2_blockflip_cols_p']:.4f}（下限 0.031） |")
    L.append(f"| U3 交叉 t | 均值 {pc(a23['mean'],2)}，t 区间 {ci(a23['ci'])}，df = {a23['df']:.1f}；单侧 p = {pn['U3']['g23']['p_one_sided_t_worse']:.4f} |")
    L.append(f"| U4 位点 | 中位 {pc(rp['median_g23_same_definition'],2)}；区间 {ci(u4['median_g23_ci'])} |")
    L.append(f"| U3+U4 同时重抽 | 区间 {ci(cb['g23']['ci'])}；≥0 的比例 {cb['g23']['frac_ge0']:.3f} |")
    L.append(f"| 秩 0→2（相对可加模型） | U3+U4 区间 {ci(cb['g02']['ci'])}；U4 区间 {ci(u4['median_g02_ci'])} |")
    L.append("")

    # ---- 依赖结构
    d = pn["dependence"]
    L.append("### T4 15 个子集彼此多不独立（描述性证据）\n")
    L.append("| 配对类型 | 对数 | 交互残差相关（均值 [最小, 最大]） | 逐格损失差（秩1−秩2）相关均值 | 逐位点损失差相关均值 |")
    L.append("|---|---|---|---|---|")
    for k, nm in (("same_ligand_diff_backbone", "同配体、不同骨架"), ("same_backbone_diff_ligand", "同骨架、不同配体"), ("diff_both", "骨架与配体都不同")):
        r1 = d["interaction_residual_corr"][k]
        L.append(f"| {nm} | {r1['n_pairs']} | {r1['mean']:.2f} [{r1['min']:.2f}, {r1['max']:.2f}] | "
                 f"{d['cell_loss_diff_corr'][k]['mean']:.2f} | {d['site_loss_diff_corr'][k]['mean']:.2f} |")
    L.append("")

    # ---- 体系矩阵
    L.append("### T5 全部 31 个矩阵（Step 4 的明细）\n")
    L.append("| 体系 | 矩阵 | 位点 | 观测 | Gen-3 截尾 | 滴定窗外(docs/25) | 检出秩(稳定度) | 秩1→2 [区间] | 秩2→3 [区间] | 秩0→2 [区间] | 理想条件还原：命中2 / 保0 / 秩1植入误判2优1 | 有功效 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in R["matrices"]:
        if m["system"] == "S1":
            continue
        c = m["ci"]
        cen = "n/a (n<200)" if m["censor_gen3"] is None else f"{100*m['censor_gen3']:.1f}%"
        fo = m.get("f_outside_titration_docs25")
        fo = "—" if fo is None else f"{100*fo:.1f}%"
        rc = m["rec"]
        L.append(f"| {m['system']} | {m['name']}{'' if m['vote_matrix'] else '（不投票）'} | {m['sites']} | {m['n_obs']} | {cen} | {fo} | "
                 f"{m['mode_rank']} ({m['stab']:.1f}) | {pc(c['g12']['point'])} {ci(c['g12']['ci'],0)} | "
                 f"{pc(c['g23']['point'])} {ci(c['g23']['ci'],0)} | {pc(c['g02']['point'])} {ci(c['g02']['ci'],0)} | "
                 f"{rc['plant2']['p_detect2']:.2f} / {rc['plant0']['p_detect0']:.2f} / {rc['plant1']['p_r2_beats_r1']:.2f} | "
                 f"{'是' if m['powered'] else '**否**'} |")
    L.append("")

    # ---- 体系层
    L.append("### T6 独立体系 × 检出秩（体系层；多矩阵体系取有功效矩阵的增益中位，位点联合重抽区间）\n")
    L.append("| 体系 | 标签 | 有功效/投票矩阵 | 秩1→2 [区间] | 秩2→3 [区间] | 秩0→2 [区间] | 预登记状态 | 内部极小(两条腿) | 事后更严口径(2优于0) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for sid, s in R["systems"].items():
        if s["status"] == "UNINFORMATIVE":
            L.append(f"| {sid} | {s['name']}{'' if s['in_primary_vote'] else '（ddPCA，主投票外）'} | {s['n_powered']}/{s['n_vote_matrices']} | — | — | — | UNINFORMATIVE | — | — |")
            continue
        L.append(f"| {sid} | {s['name']}{'' if s['in_primary_vote'] else '（ddPCA，主投票外）'} | {s['n_powered']}/{s['n_vote_matrices']} | "
                 f"{pc(s['g12_point'])} {ci(s['g12_ci'])} | {pc(s['g23_point'])} {ci(s['g23_ci'])} | {pc(s['g02_point'])} {ci(s['g02_ci'])} | "
                 f"**{s['status']}** | {'是' if s['interior_min_supported'] else '否'} | {'是' if s['posthoc_strict_supports_2'] else '否'} |")
    L.append("")
    L.append("事后敏感性（未预先登记，只作旁证，不进主投票）：\n")
    for k, v in R["posthoc_sensitivity_systems"].items():
        L.append(f"- {k}：秩1→2 {pc(v['g12']['point'])} {ci(v['g12']['ci'])}，秩2→3 {pc(v['g23']['point'])} {ci(v['g23']['ci'])}，"
                 f"秩0→2 {pc(v['g02']['point'])} {ci(v['g02']['ci'])} → {v['status']}")
    L.append("")

    # ---- 投票
    L.append("### T7 体系层投票\n")
    L.append("| 集合 | 口径 | 支持秩2 | 支持≤1 | 投票体系数 n | 最小可达单侧 p | 形式精确 p | 可解释(最小可达≤0.01) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, v in R["system_votes"].items():
        sset, how = k.split("|")
        L.append(f"| {sset} | {'区间' if how=='ci' else '点估计符号'} | {v['k_support2']} | {v['k_support_le1']} | {v['n_voting']} | "
                 f"{v['min_attainable_p']:.4f} | {v['p_formal']:.3f} | {'是' if v['interpretable'] else '**否**'} |")
    L.append("")
    cu = R["coarse_units"]
    L.append("粗单元（只计数）：家族 3（含 ddPCA 为 6）、测定平台 3（含 ddPCA 为 4）、数据集 1；"
             f"家族 n=3 的最小可达 p = {cu['U7_family_primary']['min_attainable_p']:.3f}，n=1 无检验。\n")

    # ---- S1 扩展
    L.append("### T8 S1 扩展（34 单元，来自缓存；描述性，全部同一实验室/平台/家族）：Gen-3 截尾分层 × 检出秩\n")
    L.append("| Gen-3 截尾 | 单元数 | 检出秩分布 | 来源 |")
    L.append("|---|---|---|---|")
    for k, v in R["s1_extended_34_by_gen3"].items():
        L.append(f"| {k} | {v['n']} | {v['ranks']} | {v['by_dataset']} |")
    L.append("")
    sv = R["search_volume"]
    L.append(f"搜索量：候选单元 {sv['units_considered']} 个；聚类/置换/混合模型/bootstrap 类检验与区间共 {sv['tests_run']} 项；"
             f"定秩矩阵 {sv['matrices_run']} 个；还原性诊断 CV 运行 {sv['recovery_cv_runs']} 次；置换 CV 运行 {sv['perm_cv_runs']} 次。\n")
    return "\n".join(L) + "\n"


def main():
    out = build()
    p = os.path.join(ROOT, "cache", "independence_tables.md")
    open(p, "w").write(out)
    print(f"写入 {p} ({len(out)} 字符)")
    if "--check" in sys.argv:
        doc = open(sys.argv[sys.argv.index("--check") + 1]).read()
        # 每一行表行必须原样出现在文档里
        miss = [ln for ln in out.splitlines() if ln.startswith("| ") and ln not in doc]
        print(f"文档缺少 {len(miss)} 行表行")
        for ln in miss[:10]:
            print("  ", ln[:120])
        return 1 if miss else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
