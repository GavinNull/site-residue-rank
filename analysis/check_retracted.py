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
    rows = scan_repo()
    print(f"\n登记的撤回条目 {len(RETRACTED)} 条，前向主张文档 {len(LIVE_CLAIM)} 个；扫描结果：{len(rows)} 处需处理\n")
    for rel, line, what, why in sorted(rows):
        print(f"  {rel}:{line}  {what}（{why}）")
    if rows:
        print("\n处理方式：删掉，或把它放进一个声明了撤回语境的段落。")
    sys.exit(1 if rows else 0)
