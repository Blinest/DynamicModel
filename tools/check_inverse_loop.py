# -*- coding: utf-8 -*-
"""check_inverse_loop.py — 逆解闭环验收（用表内的点当目标）

流程: 从表里随机取 N 个点 (p_tip, R_tip) 当目标 → `solve_pose` 求 τ →
      再用**正解**从 τ 算回位姿 → 与目标比位置误差 / 角度误差 / 查表耗时。

跑: TC_NO_JIT=1 python tools/check_inverse_loop.py [--table 60] [--n 6]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TC_NO_JIT", "1")

from ik_table import load_table, default_params          # noqa: E402
from inverse_solver import LookupInverseSolver           # noqa: E402
from tendon_coupling import shooting                     # noqa: E402


def rot_deg(Ra, Rb):
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="60", help="40 或 60 或 npz 路径")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    path = a.table if a.table.endswith(".npz") else f"vc_table_{a.table}.npz"

    T = load_table(path)
    P = T.params if getattr(T, "params", None) is not None else default_params()
    print(f"表 {path}: {len(T.recs)} 点  EI={P.EI:.4g}  k_rod={P.k_rod:.4g}")
    S = LookupInverseSolver(table=T)

    rng = np.random.default_rng(a.seed)
    idx = rng.choice(len(T.recs), size=min(a.n, len(T.recs)), replace=False)
    rows, ok = [], 0
    for i in idx:
        r = T.recs[int(i)]
        p = np.asarray(r["p_tip"], float)
        R = np.asarray(r["R_tip"], float)
        t0 = time.time()
        sol = S.solve_pose(p, R)
        dt = time.time() - t0
        tau = np.asarray(getattr(sol, "tau", sol.get("tau") if isinstance(sol, dict) else np.nan), float)
        conv = bool(getattr(sol, "converged", True))
        # 正解回代（F_tip 必须给零向量, 否则走 7 维残差会崩）
        o = shooting(tau, P, F_tip=np.zeros(3), g_world=(0, 0, -9.81), tol=1e-9, max_iter=60)
        dp = float(np.linalg.norm(o["p_curve"][-1] - p) * 1000.0)
        da = rot_deg(o["R_curve"][-1], R)
        ok += 1 if (conv and dp < 1.0) else 0
        rows.append((int(i), conv, np.linalg.norm(tau), dp, da, dt))
        print(f"  #{int(i):4d} conv={conv!s:5s} |τ|={np.linalg.norm(tau):6.1f} N  "
              f"Δp={dp:6.3f} mm  Δθ={da:6.3f}°  {dt*1000:7.0f} ms")

    if rows:
        dp = np.array([x[3] for x in rows]); da = np.array([x[4] for x in rows])
        dt = np.array([x[5] for x in rows])
        print("\n" + "=" * 66)
        print(f"通过 (<1 mm 且收敛): {ok}/{len(rows)}")
        print(f"Δp 中位={np.median(dp):.4f} mm  p90={np.percentile(dp,90):.4f}  max={dp.max():.4f}")
        print(f"Δθ 中位={np.median(da):.4f}°   max={da.max():.4f}")
        print(f"单次耗时 中位={np.median(dt)*1000:.0f} ms")
        return 0 if ok == len(rows) else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
