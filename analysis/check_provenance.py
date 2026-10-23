#!/usr/bin/env python3
"""出处表正向核对（CLAUDE.md §3）：表中每个数字必须真的出现在它所引的文档里。

覆盖范围（也是局限）：
  - 只核对「数字是否出现在被引 docs/NN 全文」，不核对章节号，也不核对该数字是不是
    那个位置上的那个量（数字相撞不是出处，D-092 的教训）。
  - 来源列不含 docs/NN 的行（D-xxx、脚本、JOURNAL、附录 A 计数）**不核对**，
    单独计数并打印，不当作通过。
  - 反方向（正文数字在表里有对应）不在此脚本内。
先自检再扫描：自检不过则中止。
"""
import glob, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = ["docs/48-paper-v2.md", "docs/32-methods-paper.md"]
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def norm(s):
    return s.replace("−", "-").replace("–", "-")


def doc_text(n, cache={}):
    if n not in cache:
        fs = glob.glob(os.path.join(ROOT, "docs", f"{n}-*.md"))
        cache[n] = norm(open(fs[0], encoding="utf-8").read()) if fs else None
    return cache[n]


def extra_text(rel, cache={}):
    if rel not in cache:
        cache[rel] = norm(open(os.path.join(ROOT, rel), encoding="utf-8").read())
    return cache[rel]


def numbers(cell):
    out = []
    for tok in re.findall(r"`([^`]*)`", cell):
        out += [m.group(0).rstrip(",") for m in NUM.finditer(norm(tok))]
    return out


def check_rows(text, texts=None):
    """返回 (已核对行数, 未核对行数, 缺失[(行号, 数字, 引用)])"""
    checked = skipped = 0
    missing = []
    for i, ln in enumerate(text.splitlines(), 1):
        if not ln.startswith("|") or ln.count("|") < 4:
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) < 3 or set(cells[1]) <= set("-: "):
            continue
        if "`" not in cells[1]:
            continue            # 表头或不含数字的行，不计入
        refs = re.findall(r"docs/(\d+)", cells[-1])
        if "analysis/" in cells[-1] or (not refs and "D-" not in cells[-1]
                                        and "JOURNAL" not in cells[-1]):
            skipped += 1        # 来源是脚本或附录计数：本脚本不核对，如实计入未核对
            continue
        srcs = [(texts or {}).get(r) if texts else doc_text(r) for r in refs]
        if re.search(r"D-\d+", cells[-1]):
            srcs.append(extra_text("state/decisions.md"))
        if "JOURNAL" in cells[-1]:
            srcs.append(extra_text("JOURNAL.md"))
        srcs = [s for s in srcs if s]
        if not srcs:
            skipped += 1
            continue
        checked += 1
        for n in numbers(cells[1]):
            core = n.replace(",", "")
            if not any(n in s or core in s.replace(",", "") for s in srcs):
                missing.append((i, n, "/".join(refs)))
    return checked, skipped, missing


def selfcheck():
    ok = True
    # 已知事实：从 docs/44 里取一个真实存在的小数，必须被放过
    t44 = doc_text("44")
    real = next(m.group(0) for m in re.finditer(r"\d\.\d{3}", t44))
    row = f"| a | b | c |\n|---|---|---|\n| x | `{real}` | docs/44 §0 |\n"
    if check_rows(row)[2]:
        print("自检失败：真实存在的数字被误报"); ok = False
    # 植入：不存在的数字必须被抓
    row = "| a | b | c |\n|---|---|---|\n| x | `0.123456789` | docs/44 §0 |\n"
    if len(check_rows(row)[2]) != 1:
        print("自检失败：不存在的数字没被抓"); ok = False
    # 引错文档必须被抓（数字只在别的文档里）
    only_other = next(m.group(0) for m in re.finditer(r"\d\.\d{4}", t44)
                      if m.group(0) not in (doc_text("41") or ""))
    row = f"| a | b | c |\n|---|---|---|\n| x | `{only_other}` | docs/41 §0 |\n"
    if len(check_rows(row)[2]) != 1:
        print("自检失败：引错文档没被抓"); ok = False
    # 负号归一：docs 里是 − 还是 - 都应命中
    row = "| a | b | c |\n|---|---|---|\n| x | `−0.60` | docs/44 §0 |\n"
    if check_rows(row)[2]:
        print("自检失败：负号归一失效（−0.60 应在 docs/44 找到）"); ok = False
    # 无 docs 引用的行必须记入未核对，而不是通过
    row = "| a | b | c |\n|---|---|---|\n| x | `0.123456789` | 附录 A 逐行计数 |\n"
    c, s, m = check_rows(row)
    if not (c == 0 and s == 1):
        print("自检失败：无 docs 引用的行应计入未核对"); ok = False
    return ok


def main():
    if not selfcheck():
        print("自检未通过，中止"); sys.exit(2)
    print("自检通过：放过真数 / 抓住假数 / 抓住引错文档 / 负号归一 / 无引用行计入未核对")
    bad = 0
    for t in TARGETS:
        c, s, m = check_rows(norm(open(os.path.join(ROOT, t), encoding="utf-8").read()))
        print(f"{t}: 已核对 {c} 行，未核对（非 docs 来源）{s} 行，缺失数字 {len(m)} 个")
        for i, n, r in m:
            print(f"   {t}:{i}  `{n}` 不在 docs/{r} 中")
        bad += len(m)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
