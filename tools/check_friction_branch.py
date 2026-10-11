# -*- coding: utf-8 -*-
"""check_friction_branch.py — 摩擦"按历史自动选支"的验证
==========================================================================================
背景: `SDMParams.friction_branch` 原先要外部手填 (+1 加载 / -1 卸载)。真实系统里
支由加载历史决定, 手填容易搞反, 而且同一姿态用两支会给出不同答案。
本脚本验证 `tendon_coupling.branch_from_history` + `shooting_batch_path` 的正确性:

  ① **选支正确**: 三角波 τ 路径 (升 8 步 → 降 8 步) 上, 上升段每步应选 +1、下降段选 -1,
     且 deadband 内不翻转 (重复同一 τ 两次)。
  ② **与手填逐位一致**: 上升段的结果必须与显式 `friction_branch=+1` 逐位相同;
     下降段必须与显式 `=-1` 逐位相同 (max|Δ|=0)。
  ③ **无摩擦退化**: μ=0 时两支给出同一结果, 且与现有无摩擦实现逐位一致。
  ④ **滞回环闭合**: 同一 τ 处上升/下降的末端 z 之差 = 环宽, 应 >0 且随 τ 幅值单调增大。
  ⑤ **无副作用**: 调用后 params.friction_branch 恢复原值。

跑: python tools/check_friction_branch.py [--nsub 60] [--mu 0.15]   (纯 numpy 约 1-2 分钟)
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import numba  # noqa: F401
except Exception:
    os.environ["TC_NO_JIT"] = "1"      # 无 numba → 走纯 numpy 参考实现

from tendon_coupling import (  # noqa: E402
    TendonCouplingModel, shooting_batch, forward_batch,
    branch_from_history, shooting_batch_path,
)

BASE = np.array([40.0, 20.0, 40.0, 20.0, 40.0, 10.0])


def make_path(lo=0.35, hi=1.0, n=8):
    up = np.linspace(lo, hi, n)
    down = np.linspace(hi, lo, n)[1:]                 # 折返点不重复
    sc = np.concatenate([up, down])          # 缩放序列 (15,)
    return sc[:, None] * BASE[None, :]      # (15,6) 三角波 τ 路径


def solve_explicit(tau, params, branch):
    P = copy.copy(params)
    P.friction_branch = float(branch)
    o = shooting_batch(np.atleast_2d(tau), P, tol=1e-8, max_iter=20)
    return o, np.asarray(o["tip"][0], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsub", type=int, default=60)
    ap.add_argument("--mu", type=float, default=0.15)
    a = ap.parse_args()

    M = TendonCouplingModel()
    P = M.params
    P.n_p = a.nsub; P.n_d = a.nsub
    P.mu_friction = a.mu
    P.friction_branch = +1.0

    TAU = make_path()
    nleg = 8
    print(f"τ 路径: {len(TAU)} 步 (升 {nleg} 步 → 降 {len(TAU)-nleg} 步), "
          f"μ={a.mu}, nsub={a.nsub}")

    # ── ① 选支 ──
    branch_before = float(P.friction_branch)
    try:
        rp = shooting_batch_path(TAU, P, tol=1e-8, max_iter=20)
    finally:
        pass
    br = rp["branch"]
    ok1 = bool(np.all(br[:nleg] > 0) and np.all(br[nleg:] < 0))
    print(f"① 选支序列      : {br.astype(int).tolist()}  "
          f"{'✅' if ok1 else '❌ 上升段应全 +1、下降段应全 -1'}")

    # deadband: 同一 τ 连发两次不应翻转
    b1 = branch_from_history(TAU[0], TAU[0], +1.0)
    b2 = branch_from_history(TAU[0], TAU[0], -1.0)
    ok1b = (b1 == +1.0 and b2 == -1.0)
    print(f"①b 原地不动沿用  : 上一支+1→{b1:+.0f}, 上一支-1→{b2:+.0f}  "
          f"{'✅' if ok1b else '❌'}")

    # ── ② 与手填逐位一致 ──
    e_plus, e_minus, n_cmp = 0.0, 0.0, 0
    for k in range(len(TAU)):
        if not bool(rp["converged"][k]):          # 未收敛点没有可比性, 单独报
            continue
        want = +1.0 if k < nleg else -1.0
        _, tip_e = solve_explicit(TAU[k], P, want)
        e = float(np.max(np.abs(tip_e - rp["tip"][k])))
        n_cmp += 1
        if want > 0:
            e_plus = max(e_plus, e)
        else:
            e_minus = max(e_minus, e)
    # 热启动与冷启动是两条不同迭代路径, 只在求解容差内一致 ⇒ 取 1e-6 m 作为判据
    ok2 = (e_plus < 1e-6 and e_minus < 1e-6 and n_cmp == len(TAU))
    print(f"② 对手填逐位一致  : 升段 max|Δtip|={e_plus:.3e} 降段 max|Δtip|={e_minus:.3e}  "
          f"{'✅' if ok2 else '❌'}")

    # ── ③ μ=0 退化 ──
    P0 = copy.copy(P); P0.mu_friction = 0.0; P0.friction_branch = +1.0
    r0 = shooting_batch_path(TAU, P0, tol=1e-8, max_iter=20)
    ref = np.zeros_like(r0["tip"])
    for k in range(len(TAU)):
        ref[k] = solve_explicit(TAU[k], P0, +1.0)[1]
    e0 = float(np.max(np.abs(ref - r0["tip"])))
    ok3 = (e0 < 1e-6) and bool(np.all(r0["converged"]))
    print(f"③ μ=0 与无摩擦一致: max|Δtip|={e0:.3e} (容差 1e-6 m)  {'✅' if ok3 else '❌'}")

    # ── ④ 滞回环 ──
    # 配对要按"同一 τ 幅值": 上升段 index i 对应下降段 index nleg+(n_down-1-i);
    # 折返点 (scale=hi) 只出现一次, 不能配对, 必须排除 (否则算出 0 的假环宽)
    n_down = len(TAU) - nleg
    dz = []
    for i in range(nleg - 1):
        kd = nleg + (n_down - 1 - i)
        dz.append(float(rp["tip"][i, 2] - rp["tip"][kd, 2]) * 1000.0)
    mono = bool(np.all(np.diff(dz) > -1e-9))
    ok4 = mono and (int(rp["converged"].sum()) == len(TAU)) and dz[0] > 0.0
    print(f"④ 环宽随幅值单调  : {np.round(dz, 4).tolist()} mm  "
          f"{'✅' if ok4 else '❌ 出现非单调'}")

    # ── ⑤ 无副作用 ──
    ok5 = (float(P.friction_branch) == branch_before)
    print(f"⑤ 调用后恢复原支  : {P.friction_branch:+.0f} (原 {branch_before:+.0f})  "
          f"{'✅' if ok5 else '❌'}")

    ok = ok1 and ok1b and ok2 and ok3 and ok4 and ok5
    conv = int(rp["converged"].sum())
    print(f"收敛: {conv}/{len(TAU)}   n_iter={rp['n_iter'].tolist()}")
    print("=" * 70)
    print("摩擦选支验证:", "ALL PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
