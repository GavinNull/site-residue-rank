#!/usr/bin/env python3
"""仓库自检：标签泄露扫描 + 结构完整性 + gate 状态。

泄露扫描是本脚本的主要目的。pairs/ 下出现任何亲和力数值或强弱措辞，
该对即被标记为污染——因为污染过的论证与干净论证事后无法区分。

用法: python3 analysis/check_repo.py
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数值型泄露：亲和力单位与量纲
NUM_PATTERNS = [
    (r"\b(?:Kd|Ki|KD|KI|IC50|EC50|Kdiss)\s*[=:≈~<>]\s*[\d.]", "亲和力数值"),
    (r"[\d.]+\s*(?:nM|pM|µM|uM|μM|mM)\b", "浓度量纲"),
    (r"(?:ΔΔG|ddG|dG|ΔG)\s*[=:≈~]\s*[-+]?[\d.]", "自由能数值"),
    (r"[-+]?\d+(?:\.\d+)?\s*kcal\s*/?\s*mol", "kcal/mol"),
    (r"[-+]?\d+(?:\.\d+)?\s*kJ\s*/?\s*mol", "kJ/mol"),
    (r"\bp(?:Kd|Ki|IC50)\s*[=:]\s*[\d.]", "p-标度"),
    (r"\bfold\s*(?:-|\s)?(?:change|loss|gain|increase|decrease)", "倍数变化"),
    (r"\d+\s*(?:-|\s)?fold\b", "倍数"),
]

# 措辞型泄露：强弱结论
WORD_PATTERNS = [
    (r"(?:more|less|most|least|higher|lower|weaker|stronger|tighter|"
     r"reduced|enhanced|improved|abolish\w*|impair\w*|disrupt\w*)\s+"
     r"(?:affinity|binding|potency|activity)", "英文强弱措辞"),
    (r"(?:affinity|binding|potency)\s+(?:was|is|were|are)\s+"
     r"(?:higher|lower|reduced|increased|enhanced|weaker|stronger)", "英文强弱措辞"),
    (r"(?:亲和力|结合力|活性|效力)\s*(?:更|较|明显|显著)?\s*"
     r"(?:强|弱|高|低|提升|下降|增强|减弱|丧失)", "中文强弱措辞"),
    (r"(?:更强|更弱|更高|更低|最强|最弱)(?:的)?\s*(?:结合|亲和)", "中文强弱措辞"),
    (r"\bA_stronger\b|\bB_stronger\b", "方向标签"),
    (r"loss[- ]of[- ]function|gain[- ]of[- ]function", "功能得失结论"),
]

ALLOW_FILES = {"schema/pair.template.yaml"}  # 模板里的字段名不算泄露


def scan_pairs():
    hits = []
    for p in sorted(glob.glob(os.path.join(ROOT, "pairs", "*.yaml"))):
        rel = os.path.relpath(p, ROOT)
        if rel in ALLOW_FILES:
            continue
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if line.lstrip().startswith("#"):
                    continue
                for pat, label in NUM_PATTERNS + WORD_PATTERNS:
                    if re.search(pat, line, re.IGNORECASE):
                        hits.append((rel, i, label, line.strip()[:90]))
    return hits


def scan_distance_fields():
    """距离/体积这类合法数值不该被上面的正则误伤，这里只做提示性统计。"""
    n = len(glob.glob(os.path.join(ROOT, "pairs", "*.yaml")))
    return n


def check_structure():
    required = [
        "README.md", "CLAUDE.md", "JOURNAL.md",
        "docs/01-design.md", "docs/02-pair-criteria.md",
        "docs/03-validation.md", "docs/04-blind-protocol.md",
        "schema/pair.template.yaml", "schema/factor-vocabulary.md",
        "analysis/labelstore.py", "analysis/lock.py", "analysis/score_blind.py",
        "state/gates.md", "state/decisions.md",
    ]
    return [r for r in required if not os.path.exists(os.path.join(ROOT, r))]


def main():
    print("=" * 62)
    print("仓库自检")
    print("=" * 62)

    missing = check_structure()
    print(f"\n[结构] 必需文件 {'全部就位' if not missing else '缺失:'}")
    for m in missing:
        print("   缺", m)

    npairs = scan_distance_fields()
    nlab = 0
    try:
        sys.path.insert(0, os.path.join(ROOT, "analysis"))
        import labelstore
        nlab = len(labelstore.inventory())
        nlock = len(labelstore.locked_pairs())
    except Exception as e:
        nlock = 0
        print(f"\n[标签] labelstore 不可用: {e}")
    print(f"\n[计数] pairs={npairs}  labels={nlab}  locked_predictions={nlock}")

    hits = scan_pairs()
    print(f"\n[泄露扫描] pairs/ 下 {len(hits)} 处可疑")
    if hits:
        print("  !! 违反 CLAUDE.md §1。涉及的对应作废或改写后重新编号。")
        for rel, i, label, text in hits[:40]:
            print(f"   {rel}:{i}  [{label}]  {text}")
        if len(hits) > 40:
            print(f"   ... 另有 {len(hits) - 40} 处")

    print("\n[下一步] 见 state/gates.md")
    sys.exit(1 if (hits or missing) else 0)


if __name__ == "__main__":
    main()
