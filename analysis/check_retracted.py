#!/usr/bin/env python3
"""撤回数字的机械检查器 —— 让 D-064 可执行，而不只是一条规矩。

### 为什么需要它

D-064 写的是「撤回一个成员之后，必须全仓库 grep 它曾支撑过的每一句话」。
这条规矩登记后**几小时内就在我自己的 docs/30 §4 里被违反了一次**：
已撤回的 k=4 口径 PR = 2.71 / p = 0.0045 与已作废的 λ2 = 0.627 / p = 0.2787
仍作为 `facts:` 活在正文里，且与它们上方的撤回段**直接打架**。
审计（docs/31）没抓到，因为审计查的是「数字对不对」，
而这类问题是「这个数字还该不该出现」—— 两回事。

所以把它做成检查器：**每个被撤回的数字登记在案，出现在允许区之外即失败。**

### 允许区

一个被撤回的数字出现在正文里**不一定**是错的：
论证「为什么撤回它」时必须引用它。所以判据不是「不许出现」，
而是「只许出现在**声明了撤回语境**的段落里」。
段落级判据：该出现点所在的段落（空行分隔）必须含 ALLOW_MARKERS 之一。

### 自检（§10，双向）

`--selftest` 造两份合成文本：一份把撤回数字放进无标记段落（**必须报错**），
一份放进带标记的段落（**必须放过**）。两向都过才算这个检查器可用。
这正是 §10.2 要求的「既测能命中，也测不误伤」——
一个只会报警的检查器最终会被 `# noqa` 绕过。
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 允许引用撤回数字的语境标记。出现在同一段落内即视为合法。
# 扩词要克制：过宽的标记会让这个检查器变成装饰（§10.2 的另一面 ——
# 过敏的护栏会被绕过，而过松的护栏等于没有）。
# 「事故」「砍掉」是叙述一次撤回时的必然用词，故纳入；
# 不纳入「真相」「其实」这类泛化词。
ALLOW_MARKERS = [
    "撤回", "作废", "不得再引用", "不得引用", "已更正", "已改",
    "原记", "此前", "旧口径", "postmortem", "更正", "不成立",
    "事故", "砍掉",
]

# (正则, 这个数字是什么, 出处/撤回依据)
RETRACTED = [
    (r"\bPR\s*=\s*2\.71\b", "k=4 口径的 OMTKY3 残差参与率", "D-066 / docs/33"),
    (r"\b0\.0045\b", "k=4 口径 PR 的单尾 p", "D-066 / docs/33"),
    (r"\b0\.2787\b", "两组 λ2 的条件 p（检验已整体作废）", "D-069 / docs/34"),
    (r"λ2\s*=\s*0\.627", "两组 λ2 观测值（同上）", "D-069 / docs/34"),
    (r"不相同，但也不各向同性", "k=4 下那条限定", "D-066"),
    (r"\b7/7\b", "「≥25% 截尾 → 全给秩 0」切点", "D-029 / docs/22"),
    (r"≥25%[^。]{0,24}8/9", "同上切点的 8/9 版本", "D-029 / docs/22"),
    (r"−0\.786|-0\.786", "截尾相关的三位小数点估计", "D-029（改报区间）"),
    (r"\b0\.891\b", "avidity/展示方向对齐（单次未重复分选）", "D-036 / docs/25 §6"),
    (r"\b0\.962\b", "GB1/folding 轴与 u 的对齐（自身不可复现）", "D-036 / docs/26"),
    (r"r²\s*=\s*0\.652", "GB1 folding 主效应相关（自身不可复现）", "D-036 / docs/26"),
    (r"\b0\.568\b.{0,40}(放宽|relaxed)", "docs/29 放宽块（三重取消资格）", "D-061"),
    (r"决定性对照", "OMTKY3 面板的旧定位（已降为辅证）", "人类裁定 2026-10-13"),
    # ---- docs/41 独立性审计（2026-10-15）新增 ----
    (r"\b0\.00049\b", "14/15 符号检验 p（把 15 个相关子集当独立样本）", "D-088 / docs/41"),
    (r"5\.54\s*%", "秩 2→3 的 −5.54%（无法从原始数据复现，同一定义给 −6.1%）", "D-088 / docs/41"),
    (r"7\s*个(蛋白)?骨架[、，,]\s*9\s*个配体", "「7 骨架 9 配体」（无任何脚本或缓存可追溯）", "D-088 / docs/41"),
    (r"14/15", "14/15 作为符号检验证据（子集不独立）", "D-088 / docs/41"),
]


# ---- docs/44（D-091：头条定秩程序换成 K1）新增：**现行程序下算出的数字**被取代，不得再作为当前事实出现 ----
# 只在 `--docs44` 下强制（默认运行只打印待处理条数，不影响退出码）：
# 在人类把 docs/44 §5 的替换文本应用到 docs/35 / docs/32 / README 之前，这些数字就活在稿子里，
# 若默认强制，这条门禁会立刻变红并长期红着——而长期红的门禁的下一步总是放宽。
# 已并入强制路径（见 __main__）；`--docs44` 开关不再需要，保留兼容。
DOCS44_MARKERS = ["去偏前", "现行程序", "旧程序", "docs/44 之前", "旧口径", "已被取代", "被取代"]
SUPERSEDED_DOCS44 = [
    (r"\+3\.5%\s*\[\+0\.7%", "S1 秩 1→2 增益（现行程序，逐格口径）", "D-091 / docs/44：K1 下为 +1.3% [−1.4%, +4.7%]（seed 41）"),
    (r"\+4\.5%[^。\n]{0,40}\[\+0\.8%, \+7\.3%\]", "S1 秩 1→2 中位 +4.5%（现行程序）", "D-091 / docs/44"),
    (r"\[[−-]1\.6%, \+9\.4%\]", "现行程序下三源联合重抽的 g12 区间", "D-091 / docs/44：K1 下为 [−4.7%, +6.9%]"),
    (r"\+11\.4%", "秩 0→2 中位 +11.4%（现行程序）", "D-091 / docs/44：K1 下为 +8.7%"),
    (r"43\.2%[^。\n]{0,40}10\.9%[^。\n]{0,40}0\.1%", "按检出秩分层的截尾中位 43.2/10.9/0.1（现行/原秩）", "D-091 / docs/44：K1 下为 43.2 / 10.9 / 0.2"),
    (r"\b15/18\b", "现行程序 15/18 为秩 2（口径含 CR9114 WT_H1）", "D-091 / docs/44：CR9114 移出后范围内为 14/17"),
    (r"\b12/18\b", "K1 12/18 为秩 2（口径含 CR9114 WT_H1）", "D-091 / docs/44：范围内为 12/17"),
    (r"16/21|76%（16/21", "截尾<10% 的 21 个组合里 16 个秩 2（现行程序）", "D-091 / docs/44：K1 下为 14/21"),
    (r"[−-]7\.0%\s*\[[−-]8\.[78]%", "S1 秩 2→3（现行程序）", "D-091 / docs/44：K1 下为 −11.6% [−15.3%, −8.8%]"),
]

# **作用域：只扫「对外声明主张」的文档。**
#
# 第一版扫全仓库，得到 127 处 —— 而其中绝大多数是正当的：
# docs/19、22、33、34 等**本身就是执行撤回的那些文档**，论证「为什么撤回」
# 必须大量引用被撤回的数字；docs/16 是 §4 冻结的预注册，不得修改；
# 其余分析文档是「当时相信什么」的记录，改掉它们会毁掉审计链。
# 所以「不许出现」这个判据只在**前向主张文档**上成立。
# 新的手稿加进 LIVE_CLAIM 即纳入检查。
LIVE_CLAIM = [
    "docs/30-manuscript.md",
    "docs/32-methods-paper.md",
    "docs/35-paper-draft.md",
    "README.md",     # 公开仓库的第一眼；docs/41 审计时 p=0.00049 就写在这里
]


def paragraphs(text):
    """返回 (起始行号, 段落文本)。段落按空行切分。"""
    out, buf, start = [], [], 1
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip():
            if not buf:
                start = n
            buf.append(line)
        elif buf:
            out.append((start, "\n".join(buf)))
            buf = []
    if buf:
        out.append((start, "\n".join(buf)))
    return out


def scan_text(text, label):
    bad = []
    for start, para in paragraphs(text):
        if any(m in para for m in ALLOW_MARKERS):
            continue
        for pat, what, why in RETRACTED:
            if re.search(pat, para):
                bad.append((label, start, what, why))
    return bad


def scan_text_docs44(text, label):
    bad = []
    for start, para in paragraphs(text):
        if any(m in para for m in ALLOW_MARKERS + DOCS44_MARKERS):
            continue
        for pat, what, why in SUPERSEDED_DOCS44:
            if re.search(pat, para):
                bad.append((label, start, what, why))
    return bad


def scan_repo_docs44():
    bad = []
    for rel in LIVE_CLAIM:
        full = os.path.join(ROOT, rel)
        if os.path.exists(full):
            with open(full, encoding="utf-8", errors="replace") as fh:
                bad += scan_text_docs44(fh.read(), rel)
    return bad


def selftest_docs44():
    hit = "S1 的秩 1→2 增益为 +3.5% [+0.7%, +7.2%]，是当前结论。"
    miss = "现行程序下 S1 的秩 1→2 增益为 +3.5% [+0.7%, +7.2%]，已被取代。"
    ok = bool(scan_text_docs44(hit, "h")) and not scan_text_docs44(miss, "m")
    two = scan_text_docs44(hit + "\n\n" + miss, "t")
    ok = ok and bool(two)
    print("docs44 条目自检：" + ("通过（命中/不误伤/段落级）" if ok else "未通过"))
    return ok


def scan_repo():
    bad, missing = [], []
    for rel in LIVE_CLAIM:
        full = os.path.join(ROOT, rel)
        if not os.path.exists(full):
            missing.append(rel)
            continue
        with open(full, encoding="utf-8", errors="replace") as fh:
            bad += scan_text(fh.read(), rel)
    if missing:
        print(f"警告：LIVE_CLAIM 里有 {len(missing)} 个文件不存在：{missing}")
    return bad


def selftest():
    """§10.2：既测能命中，也测不误伤。"""
    hit = "这个对照给出 PR = 2.71，单尾 p = 0.0045，清楚在零分布之外。"
    miss = "原记 PR = 2.71 / p = 0.0045 已按 D-066 撤回，不得再引用。"
    a = scan_text(hit, "<synthetic-hit>")
    b = scan_text(miss, "<synthetic-miss>")
    ok = True
    if not a:
        print("自检失败：无标记段落里的撤回数字**没有**被抓到（漏报）")
        ok = False
    if b:
        print(f"自检失败：带撤回标记的段落被误报 {len(b)} 次（误伤）")
        ok = False
    # 标记必须是段落级而非全文级：同一文件里别处有「撤回」不应豁免本段
    two = hit + "\n\n" + miss
    c = scan_text(two, "<synthetic-two-para>")
    if not c:
        print("自检失败：标记作用域是全文而不是段落 —— 一处「撤回」会豁免整个文件")
        ok = False
    print("自检通过：命中 / 不误伤 / 标记作用域为段落级" if ok else "自检未通过")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    if not selftest():
        print("检查器自身未通过自检，拒绝给出扫描结论（CLAUDE.md §10）")
        sys.exit(2)
    if not selftest_docs44():
        sys.exit(2)
    # docs/44 并入强制路径（2026-10-17）。此前 docs44 条目只在 `--docs44` 下才影响退出码，
    # 默认运行只打印条数——于是 docs/35 里有 9 处现行程序数字时，默认门禁仍然 exit 0。
    # 这与「make_figures --check 的 PASS 不覆盖 §3.3 表」同形：绿灯不能当作
    # 它没有覆盖的东西也被检查过。现在两条路径都强制，--docs44 仅保留作明细开关。
    pend = scan_repo_docs44()
    rows = scan_repo()
    print(f"登记的撤回条目 {len(RETRACTED)} 条 + 旧程序取代条目 {len(SUPERSEDED_DOCS44)} 条；"
          f"前向主张文档 {len(LIVE_CLAIM)} 个")
    print(f"扫描结果：撤回数字 {len(rows)} 处；旧程序数字未加标签/未替换 {len(pend)} 处\n")
    for rel, line, what, why in sorted(rows):
        print(f"  {rel}:{line}  {what}（{why}）")
    for rel, line, what, why in sorted(pend):
        print(f"  [旧程序] {rel}:{line}  {what}（{why}）")
    if rows or pend:
        print("\n处理方式：删掉，或把它放进一个声明了撤回/取代语境的段落。")
    sys.exit(1 if (rows or pend) else 0)
