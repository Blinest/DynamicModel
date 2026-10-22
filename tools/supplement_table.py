# -*- coding: utf-8 -*-
"""supplement_table.py — 给位姿表**定向补采样**并合并

背景：`main.py build 40 --n 1200 --force` 用固定 LHS 种子，重跑只会得到
**同样的** 1200 个 τ（40N 下 776 入库）。要真正加点，必须
① 换种子生成**新的** τ；② 与已有表**合并**，而不是覆盖。

本脚本做三件事：
  1. 读旧表 → 把 κ 曲线抽到 `--nodes`（默认 61）保持与新增样本同网格
  2. 用 `--seed` 生成新 τ（LHS），剔除与旧表过近的点（`--min-dist` 相对距离）
  3. `build_table(taus=…)` 只解这些新点 → 合并 → 存盘

    python tools/supplement_table.py --tau 40 --add 655 --seed 1

⚠ 采样点数 655 是"尝试次数"；40N 接受率约 65% ⇒ 预计 +425 个入库点。
⚠ 合并后表是 `--nodes` 节点的精简网格（tau/dL/p_tip/R_tip/res/x 不受影响）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TC_NO_JIT", "1")

from ik_table import (load_table, save_table, build_table, lhs,   # noqa: E402
                      default_params)


def decimate(recs, nodes):
    """把每条记录的 κ 曲线抽到 `nodes` 个节点（保持 kS/kK/kV 一致）。"""
    out = []
    for r in recs:
        kc = r.get("kcoef") or {}
        S = np.asarray(kc.get("S", []), float)
        K = np.asarray(kc.get("K", []), float)
        V = kc.get("V", None)
        if S.size != nodes:
            idx = np.linspace(0, S.size - 1, nodes).round().astype(int)
            nk = dict(S=S[idx], K=K[idx])
            if V is not None:
                nk["V"] = np.asarray(V, float)[idx]
            r = dict(r); r["kcoef"] = nk
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=40.0)
    ap.add_argument("--add", type=int, default=655, help="新增尝试数")
    ap.add_argument("--seed", type=int, default=1, help="新采样的 LHS 种子")
    ap.add_argument("--nodes", type=int, default=61)
    ap.add_argument("--min-dist", type=float, default=0.02,
                    help="与旧点的最小相对距离（按 τmax 归一化）")
    ap.add_argument("--batch", type=int, default=200)
    a = ap.parse_args()

    path = f"vc_table_{a.tau:g}.npz"
    if not os.path.exists(path):
        sys.exit(f"找不到 {path}")
    old = load_table(path)
    P = old.params or default_params()
    n_old = len(old.recs)
    print(f"[1] 旧表 {path}: {n_old} 点  EI={P.EI:.4g} k_rod={P.k_rod:.4g}", flush=True)

    # ── 新 τ：换种子 LHS，剔除与旧点过近的 ──
    u = lhs(a.add * 2, 6, seed=a.seed)                # 多抽一倍，筛完再用
    tau_old = np.array([r["tau"] for r in old.recs], float) / a.tau
    keep = []
    for row in u:
        if len(keep) >= a.add:
            break
        if tau_old.size and np.min(np.linalg.norm(tau_old - row, axis=1)) < a.min_dist:
            continue
        keep.append(row)
    new_taus = np.array(keep, float) * a.tau
    print(f"[2] 新 τ: 生成 {len(new_taus)} 个（已剔除与旧点 <{a.min_dist} 的点）", flush=True)

    t0 = time.time()
    recs_new = build_table(N=len(new_taus), tau_max=a.tau, params=P,
                           taus=new_taus, batch_size=a.batch,
                           continuation=tuple(np.linspace(0.1, 1.0, 16)),
                           verbose=True)
    print(f"[3] 新增入库 {len(recs_new)}/{len(new_taus)}  "
          f"用时 {(time.time()-t0)/60:.1f} min", flush=True)

    # ── 合并（旧表先抽到同一网格）──
    merged = decimate(old.recs, a.nodes) + decimate(recs_new, a.nodes)
    # 按 τ 去重（保留先出现的）
    seen, uniq = set(), []
    for r in merged:
        k = tuple(np.round(np.asarray(r["tau"], float), 6))
        if k in seen:
            continue
        seen.add(k); uniq.append(r)
    print(f"[4] 合并: {n_old} + {len(recs_new)} -> {len(uniq)} 点（去重后）", flush=True)

    if os.path.exists(path):
        bak = path + ".prev2"
        os.replace(path, bak)
        print(f"[5] 旧表 -> {bak}")
    save_table(path, uniq, params=P)
    print(f"[6] 已存 {path}  {os.path.getsize(path)/1e6:.1f} MB  "
          f"节点数={a.nodes}  总用时 {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
