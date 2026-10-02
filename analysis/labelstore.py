#!/usr/bin/env python3
"""标签存储：机制性隔离，而不是靠自律。

标签以 base64 存放于 labels/labels.b64，使得
  - cat / grep / 随手 Read 都读不出数值
  - 任何读取都必须经过本模块，从而被 guard 拦住

guard 规则：某个 pair 的标签只在该 pair 的盲预测已锁定后才可解出。
"""
import base64
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.path.join(ROOT, "labels", "labels.b64")
LOCKFILE = os.path.join(ROOT, "predictions", "LOCK.jsonl")


def _read_raw():
    if not os.path.exists(STORE):
        return {}
    with open(STORE, "rb") as f:
        blob = f.read().strip()
    if not blob:
        return {}
    return json.loads(base64.b64decode(blob).decode("utf-8"))


def _write_raw(d):
    os.makedirs(os.path.dirname(STORE), exist_ok=True)
    blob = base64.b64encode(json.dumps(d, ensure_ascii=False, indent=0,
                                       sort_keys=True).encode("utf-8"))
    with open(STORE, "wb") as f:
        f.write(blob)


def locked_pairs():
    """已锁定盲预测的 pair_id 集合。"""
    out = set()
    if not os.path.exists(LOCKFILE):
        return out
    with open(LOCKFILE) as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(json.loads(line)["pair_id"])
    return out


def put(pair_id, ddg_kcal, sign, source_doi, note=""):
    """写入标签。ddg_kcal 为 |ΔΔG|，sign 为 'A_stronger' / 'B_stronger'。"""
    assert sign in ("A_stronger", "B_stronger"), sign
    d = _read_raw()
    d[pair_id] = {
        "abs_ddg_kcal": float(ddg_kcal),
        "stronger": sign,
        "source_doi": source_doi,
        "note": note,
    }
    _write_raw(d)
    return len(d)


def get(pair_id, *, allow_unlocked=False):
    """解出单个标签。未锁定盲预测则拒绝。"""
    if not allow_unlocked and pair_id not in locked_pairs():
        raise PermissionError(
            f"{pair_id} 的盲预测尚未锁定，拒绝揭晓标签。"
            f" 先写 predictions/{pair_id}.<predictor>.yaml 并运行 analysis/lock.py"
        )
    return _read_raw().get(pair_id)


def inventory():
    """不泄露数值的清单：只给 pair_id、是否可分辨、来源。"""
    d = _read_raw()
    rows = []
    for pid, v in sorted(d.items()):
        rows.append({
            "pair_id": pid,
            "distinguishable": abs(v["abs_ddg_kcal"]) >= 0.5,
            "magnitude_bin": _bin(abs(v["abs_ddg_kcal"])),
            "source_doi": v["source_doi"],
            "locked": pid in locked_pairs(),
        })
    return rows


def _bin(x):
    # 粗分箱也是信息泄露的一种，故 inventory 只在 --force 时打印
    if x < 0.5:
        return "indistinguishable"
    return "binned"


def fingerprint():
    d = _read_raw()
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stat"
    if cmd == "stat":
        d = _read_raw()
        lk = locked_pairs()
        print(f"labels: {len(d)}   locked predictions: {len(lk)}   "
              f"revealable: {len(set(d) & lk)}")
        print(f"fingerprint: {fingerprint()}")
    elif cmd == "inventory":
        for r in inventory():
            print(r)
    else:
        print(__doc__)
