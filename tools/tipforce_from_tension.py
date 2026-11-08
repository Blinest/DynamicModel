# -*- coding: utf-8 -*-
"""tipforce_from_tension.py — 丝长锁定时，能否从 6 路张力反推尖端外力？

场景
----
电机锁死 ⇒ 六根丝的位移 ΔL 固定。此时在尖端施加外力 F_tip，
臂形几乎不动，但各丝张力会变。问题：
    τ（6 个已知）⇒ F_tip（几个未知？）

做法
----
1. 固定 ΔL = ΔL₀（由初始 τ₀ 得到），作为"锁定"约束
2. 施加 F_tip，解 τ 使 delta_L(τ, F) = ΔL₀（6 未知 / 6 约束，牛顿迭代）
3. 求灵敏度矩阵 J = ∂τ/∂F（6×3），做 SVD 看秩与条件数
   · 秩 3 ⇒ 三分量都能反推
   · 秩 2 ⇒ 只能反推横向两分量（轴向与预紧不可分）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tdcr_model_v2 as V2   # noqa: E402

RNG = np.random.default_rng(0)


def solve_tau(m, dl_target, F, tau0, it=12, tol=1e-4):
    """给定固定 ΔL 与外载 F，解 τ（牛顿 + 数值雅可比）。"""
    tau = tau0.copy()
    for _ in range(it):
        r = m.delta_L(tau, F_tip=F) - dl_target
        if np.abs(r).max() < tol:
            break
        J = np.zeros((6, 6))
        for j in range(6):
            h = 1e-3 * max(1.0, abs(tau[j]))
            tp = tau.copy(); tp[j] += h
            J[:, j] = (m.delta_L(tp, F_tip=F) - m.delta_L(tau, F_tip=F)) / h
        try:
            d = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            d = -np.linalg.lstsq(J, r, rcond=None)[0]
        tau = np.maximum(0.0, tau + 0.7 * d)
    return tau


def main():
    m = V2.TdcrModelV2(n_sub=3)
    # 初始状态：90° 弯曲，ΔL 由它确定（= 电机锁在此位置）
    A, b, tau0 = m.solve_bend(90.0)
    dl0 = m.delta_L(tau0)
    print(f"锁定状态：θ={b:.1f}°  τ₀={np.round(tau0,2)}")
    print(f"  ΔL₀={np.round(dl0,2)} mm（电机锁死 = 保持这个长度）\n")

    # 灵敏度：每方向 1 mN
    dF = 1e-3
    J = np.zeros((6, 3))
    base = tau0
    for k, name in enumerate("xyz"):
        F = np.zeros(3); F[k] = dF
        tau_f = solve_tau(m, dl0, F, base)
        J[:, k] = (tau_f - base) / dF
        print(f"  F_{name}={dF*1e3:.0f} mN ⇒ Δτ={np.round((tau_f-base)*1e3,2)} mN")

    print("\n灵敏度矩阵 J = ∂τ/∂F  (mN/N)：")
    for i in range(6):
        print("  M%d: " % (i + 1) + "  ".join(f"{J[i,k]:8.4f}" for k in range(3)))

    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    print(f"\n奇异值: {np.round(S,4)}")
    print(f"  秩(>1% max) = {int((S > 0.01*S[0]).sum())} / 3")
    print(f"  条件数 = {S[0]/S[-1]:.1f}")
    print(f"  最小奇异值对应方向 (体坐标) = {np.round(Vt[-1],3)}")
    print("\n⇒ 若秩=2：只有**垂直于臂轴的两个横向分量**可反推；")
    print("   轴向力只让 6 根丝同向变化，与'预紧变动'无法区分。")


if __name__ == "__main__":
    main()
