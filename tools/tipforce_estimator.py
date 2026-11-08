# -*- coding: utf-8 -*-
"""tipforce_estimator.py — 实时尖端力估计（查表 + 一次矩阵乘）

为什么能实时
------------
反推 `F = J⁺·(τ_meas − τ₀)` **只是一次 3×6 矩阵乘**（微秒级）。
贵的部分（构 J、解 τ）只在**离线**做一次，存成表；运行期只查表。

   离线：按 (θ, φ) 网格构 J，存 J⁺（Moone-Penrose 伪逆）
   在线：F̂ = J⁺(θ,φ) · (τ − τ₀)          ← 约 10 µs
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tdcr_model_v2 as V2          # noqa: E402
import tipforce_from_tension as T   # noqa: E402


def build_table(thetas=(30, 60, 90, 120, 150), phis=(0, 60, 120, 180, 240, 300),
                dF=1e-3):
    """离线：每个工作点算 J，存 (theta, phi, tau0, dl0, Jinv)"""
    m = V2.TdcrModelV2(n_sub=3)
    rows = []
    t0 = time.time()
    for th in thetas:
        for ph in phis:
            psi = m.psi
            A0 = None
            lo, hi = 0.0, 30.0
            for _ in range(20):
                A = 0.5 * (lo + hi)
                tau = np.maximum(0.0, m.T0 + A * np.cos(psi - np.radians(ph)))
                if m.bend_angle(tau) < th:
                    lo = A
                else:
                    hi = A
            tau0 = np.maximum(0.0, m.T0 + A * np.cos(psi - np.radians(ph)))
            dl0 = m.delta_L(tau0)
            J = np.zeros((6, 3))
            for k in range(3):
                F = np.zeros(3); F[k] = dF
                tau_f = T.solve_tau(m, dl0, F, tau0)
                J[:, k] = (tau_f - tau0) / dF
            rows.append((th, ph, tau0, dl0, np.linalg.pinv(J)))
    return m, rows, time.time() - t0


def main():
    m, rows, dt = build_table()
    print(f"离线建表：{len(rows)} 个工作点（5 θ × 6 φ），用时 {dt:.1f} s")
    print(f"  平均 {dt/len(rows)*1000:.0f} ms/点（含 3 次牛顿解 τ）\n")

    # 在线：查表 + 一次矩阵乘
    th_q, ph_q = 90, 60
    idx = min(range(len(rows)), key=lambda i: abs(rows[i][0]-th_q)+abs(rows[i][1]-ph_q))
    th, ph, tau0, dl0, Jp = rows[idx]
    print(f"在线查询 (θ={th}, φ={ph}):")

    F_true = np.array([0.004, 0.006, 0.0])           # 7.2 mN
    tau_f = T.solve_tau(m, dl0, F_true, tau0)
    dtau = tau_f - tau0

    t0 = time.perf_counter()
    for _ in range(1000):
        F_hat = Jp @ dtau
    t1 = time.perf_counter()
    print(f"  1000 次估计用时 {(t1-t0)*1000:.2f} ms ⇒ 单次 {(t1-t0)*1e6/1000:.1f} µs")
    print(f"  F_true={np.round(F_true*1e3,2)} mN  F_hat={np.round(F_hat*1e3,2)} mN  "
          f"误差 {np.linalg.norm(F_hat-F_true)/np.linalg.norm(F_true)*100:.1f}%")

    print(f"  结论：在线部分 {(t1-t0)*1e6/1000:.0f} µs ⇒ "
          f"{(t1-t0)*1e6/1000:.0f} µs 级，1 kHz 循环完全够")


if __name__ == "__main__":
    main()
