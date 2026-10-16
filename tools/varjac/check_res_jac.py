# -*- coding: utf-8 -*-
"""check_res_jac.py — ∂res/∂x₀ 解析（变分传播）与中心差分逐元素对拍【阶段 2】

三级：
  ① res 与 `forward_batch_ref` 一致（镜像积分没搬错）
  ② Jres = ∂res/∂x₀ 与中心差分一致（变分方程 + ∂R/∂x₀ + 锚点项雅可比）
  ③ Jtip = ∂tip/∂x₀ 与中心差分一致
  ④ 计时：解析 J vs 6 次/12 次差分正解

跑: TC_NO_JIT=1 python tools/varjac/check_res_jac.py [--n 3] [--sub 8] [--time]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
os.environ.setdefault("TC_NO_JIT", "1")

from res_jac import res_jac  # noqa: E402
from tendon_coupling import TendonCouplingModel, forward_batch_ref  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    if not ok:
        FAIL.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--sub", type=int, default=30,
                    help="每段子步数。⚠ 默认 30=真实默认工况；"
                         "nsub<=8 时 RK2 会让状态撞 ±3/±80 钳位，钳位处不可导 + 混沌放大，"
                         "对拍会给出 1e2 量级的**假**误差（值路径和对拍都无意义）")
    ap.add_argument("--h", type=float, default=1e-7)
    ap.add_argument("--time", action="store_true", help="额外做计时对比")
    a = ap.parse_args()

    M = TendonCouplingModel()
    P = M.params
    P.n_p = a.sub; P.n_d = a.sub
    print(f"n_p=n_d={P.n_p}  lat={getattr(P,'lattice',None)}  "
          f"rod={getattr(P,'rod_constraint',False)}  mu={getattr(P,'mu_friction',0.0)}")

    rng = np.random.default_rng(20261014)
    BASE_TAU = np.array([40.0, 20.0, 40.0, 20.0, 40.0, 10.0])
    e1 = e2 = e3 = 0.0
    for t in range(a.n):
        # ⚠ 工况必须落在**稳定区**：nsub 粗(<=8)时 RK2 会让状态撞 ±3/±80 钳位,
        #   钳位处不可导 + 混沌放大, 对拍毫无意义(实测相对误差可达 1e2 且是假的)。
        #   真实使用是 nsub=30(默认) / 60 / 100 / 300(建表), 全部稳定。
        x0 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) + rng.normal(0, 0.005, 6)
        if t == 1:
            x0 = np.array([1.4e-3, -2.4e-3, 0.9244, 0.3670, 0.2119, 0.0])  # 收敛解
        tau = BASE_TAU * (0.6 + 0.15 * t)

        res, tip, J, Jtip = res_jac(x0, tau, P)
        ref_res, ref_tip = forward_batch_ref(x0[None, :], tau[None, :], P)
        ref_res = ref_res[0]; ref_tip = ref_tip[0]
        d1 = float(np.max(np.abs(res - ref_res)))
        d1t = float(np.max(np.abs(tip - ref_tip)))
        sc = max(np.max(np.abs(ref_res)), 1e-30)
        e1 = max(e1, d1 / sc, d1t / max(np.max(np.abs(ref_tip)), 1e-30))

        # ── 中心差分 ──
        h = a.h
        Jfd = np.zeros(J.shape); Jtfd = np.zeros((3, 6))
        for k in range(6):
            xp = x0.copy(); xm = x0.copy()
            xp[k] += h; xm[k] -= h
            rp, tp_ = forward_batch_ref(xp[None, :], tau[None, :], P)
            rm, tm_ = forward_batch_ref(xm[None, :], tau[None, :], P)
            Jfd[:, k] = (rp[0] - rm[0]) / (2 * h)
            Jtfd[:, k] = (tp_[0] - tm_[0]) / (2 * h)
        e2 = max(e2, float(np.max(np.abs(J - Jfd))) / max(np.max(np.abs(Jfd)), 1e-30))
        e3 = max(e3, float(np.max(np.abs(Jtip - Jtfd)))
                 / max(np.max(np.abs(Jtfd)), 1e-30))

    print(f"[①] res/tip vs forward_batch_ref : 相对差 {e1:.3e}")
    check("① 镜像积分一致 (<1e-12)", e1 < 1e-12)
    print(f"[②] Jres vs 中心差分            : 相对差 {e2:.3e}")
    check("② ∂res/∂x₀ 解析正确 (<1e-5)", e2 < 1e-5)
    print(f"[③] Jtip vs 中心差分            : 相对差 {e3:.3e}")
    check("③ ∂tip/∂x₀ 解析正确 (<1e-5)", e3 < 1e-5)

    if a.time:
        x0 = np.array([0.0, 0.0, 1.0, 0.2, -0.1, 0.02]); tau = np.full(6, 30.0)
        t0 = time.perf_counter()
        for _ in range(3):
            res_jac(x0, tau, P)
        t_ana = (time.perf_counter() - t0) / 3
        xx = np.tile(x0, (1, 1)); tt = np.tile(tau, (1, 1))
        t0 = time.perf_counter()
        forward_batch_ref(xx, tt, P)
        t_fwd = time.perf_counter() - t0
        print(f"[④] 计时: 解析 {t_ana*1e3:.2f} ms/次 | 单次正解 {t_fwd*1e3:.2f} ms "
              f"| 前向差分(6) ≈ {t_fwd*6*1e3:.2f} ms | 中心差分(12) ≈ {t_fwd*12*1e3:.2f} ms")
        print(f"    → 相对前向差分 {t_fwd*6/t_ana:.2f}×, 相对中心差分 {t_fwd*12/t_ana:.2f}×")

    print("=" * 74)
    if FAIL:
        print(f"✗ {len(FAIL)} 项失败: {FAIL}")
        return 1
    print("✓ 通过 —— ∂res/∂x₀ 与 ∂tip/∂x₀ 已有解析式。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
