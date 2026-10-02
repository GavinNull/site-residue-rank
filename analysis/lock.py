#!/usr/bin/env python3
"""锁定盲预测：记录内容 SHA256，之后任何改动都会被 verify 检出。

用法:
  python3 analysis/lock.py            # 锁定所有未锁定的预测文件
  python3 analysis/lock.py --verify   # 校验已锁定文件未被改动
"""
import datetime
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "predictions")
LOCKFILE = os.path.join(PRED, "LOCK.jsonl")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_locks():
    out = {}
    if os.path.exists(LOCKFILE):
        with open(LOCKFILE) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r["file"]] = r
    return out


def pred_files():
    return sorted(p for p in glob.glob(os.path.join(PRED, "*.yaml"))
                  if not os.path.basename(p).startswith("_"))


def main():
    verify = "--verify" in sys.argv
    locks = load_locks()
    new, bad, ok = [], [], 0
    for p in pred_files():
        rel = os.path.relpath(p, ROOT)
        h = sha(p)
        if rel in locks:
            if locks[rel]["sha256"] != h:
                bad.append(rel)
            else:
                ok += 1
        elif not verify:
            pid = os.path.basename(p).split(".")[0]
            new.append({
                "file": rel,
                "pair_id": pid,
                "sha256": h,
                "locked_at": datetime.datetime.now().isoformat(timespec="seconds"),
            })

    if bad:
        print("!! 已锁定的预测文件被改动，违反 CLAUDE.md §4：")
        for b in bad:
            print("   ", b)
        sys.exit(2)

    if new:
        with open(LOCKFILE, "a") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"锁定 {len(new)} 个新预测。")
    print(f"校验通过 {ok} 个。总锁定数 {len(locks) + len(new)}。")


if __name__ == "__main__":
    main()
