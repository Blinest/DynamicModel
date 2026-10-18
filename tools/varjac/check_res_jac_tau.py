# -*- coding: utf-8 -*-
"""check_res_jac_tau.py — ∂res/∂τ 的解析变分 vs 中心差分

两级：
  ① 值路径：`res_jac_tau` 的 res 必须与 `res_jac` 逐位一致（同一条积分镜像）
  ② 导数：Jtau = ∂res/∂τ 与 res 对 τ 的中心差分逐元素对拍

⚠ 必须用 nsub≥15：nsub 小时状态撞 ±3/±80 钳位，钳位处不可导 + 混沌放大，
  该工况下的对拍无意义（实测 nsub=5 时 res 值本身差 1.9e3）。

跑: TC_NO_JIT=1 python tools/varjac/check_res_jac_tau.py [--nsub 30] [--h 1e-4]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
os.environ.setdefault("TC_NO_JIT", "1")

from res_jac import res_jac, res_jac_tau  # noqa: E402
from tendon_coupling import TendonCouplingModel  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    if not ok:
        FAIL.append(name)


def rel(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = np.max(np.abs(a - b))
    s = max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-30)
    return d, d / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsub", type=int, default=30)
    ap.add_argument("--h", type=float, default=1e-4, help="τ 的差分步长 [N]")
    ap.add_argument("--n", type=int, default=3, help="用例数")
    a = ap.parse_args()

    M = TendonCouplingModel()
    P = M.params
    P.n_p = a.nsub; P.n_d = a.nsub
    P.mu_friction = 0.0
    print(f"nsub={a.nsub}  lat={getattr(P,'lattice',None)}  rod_on={getattr(P,'rod_constraint',None)}")

    rng = np.random.default_rng(20261018)
    e_val = e_tau = 0.0
    t_ana = t_fd = 0.0
    import time

    for trial in range(a.n):
        tau = rng.uniform(15.0, 55.0, size=6)
        # 偏离直杆的构型（含 v 偏离 E3 的分量，否则"只在直杆正确"的 bug 会藏住）
        x0 = np.array([0.004, -0.003, 0.996,
                       0.35 + 0.1 * rng.normal(), 0.18, 0.06])

        t0 = time.perf_counter()
        res_t, tip_t, Jtau = res_jac_tau(x0, tau, P)
        t_ana += time.perf_counter() - t0

        res_r, tip_r, _, _ = res_jac(x0, tau, P)
        d, r = rel(res_t, res_r)
        e_val = max(e_val, d)

        t0 = time.perf_counter()
        ncol = res_t.size
        Jfd = np.zeros((ncol, 6))
        for j in range(6):
            tp = tau.copy(); tm = tau.copy()
            tp[j] += a.h; tm[j] -= a.h
            rp = res_jac(x0, tp, P)[0]
            rm = res_jac(x0, tm, P)[0]
            Jfd[:, j] = (rp - rm) / (2 * a.h)
        t_fd += time.perf_counter() - t0

        _, r2 = rel(Jtau, Jfd)
        e_tau = max(e_tau, r2)
        print(f"  用例{trial}: |res|={np.linalg.norm(res_t):.3e}  "
              f"∂res/∂τ 相对差={r2:.3e}  max|J|={np.max(np.abs(Jtau)):.3e}")

    print(f"[①] res 值路径一致       : max 绝对差 {e_val:.3e}")
    check("① 值路径与 res_jac 逐位一致 (<1e-12)", e_val < 1e-12)
    print(f"[②] ∂res/∂τ vs 中心差分  : max 相对差 {e_tau:.3e}")
    check("② ∂res/∂τ 解析正确 (<1e-6)", e_tau < 1e-6)
    print(f"[③] 计时: 解析 {t_ana*1e3:.1f} ms / 差分 {t_fd*1e3:.1f} ms "
          f"→ {t_fd/max(t_ana,1e-9):.2f}× (差分 = 12 趟正解)")

    print("=" * 74)
    if FAIL:
        print(f"✗ {len(FAIL)} 项失败: {FAIL}")
        return 1
    print("✓ 通过 —— ∂res/∂τ 可用，进而 ∂tip/∂τ（位姿雅可比 J）可解析给出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
