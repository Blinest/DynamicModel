# -*- coding: utf-8 -*-
"""friction_tipforce_bias.py — 摩擦对尖端力辨识的影响（定量）

问题
----
辨识尖端力时假设 κ(s) = P(L−s)/EI(s)。实际若有摩擦，会给曲率叠加一项
**未知的摩擦偏置** κ_f(s)（来源于各腱张力不对称 / 盘处摩擦累积），
辨识器会把这部分错误地算到 P 上 ⇒ 产生偏差。

做法
----
真值生成：κ_true = κ_P + κ_f，其中
    κ_f(s) = A_f · exp(−s/λ)        （λ=0.12 m，根部集中，与摩擦衰减同形）
    κ_P(s) = P(L−s)/EI(s)
辨识时**只用 κ_P 模型**，扫 A_f 与 sign，看回收的 P 偏多少。

并把 A_f 换算成物理量：
    A_f = ΔM / EI ⇒ ΔM = A_f·EI = r_disk·ΔT ⇒ ΔT = ΔM/r_disk
    其中 ΔT 即"摩擦造成的等效张力不平衡"

依赖：casadi
"""
import os
import sys

import numpy as np
import casadi as ca

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import casadi_tipforce as T   # noqa: E402

R_DISK = 5e-3
LAM = 0.12


def kappa_true(P, A_f, ei, sign=+1):
    s_mid = (np.arange(T.NSEG) + 0.5) * T.SEG_LEN
    kP = P * (T.L_TOT - s_mid) / ei
    kf = sign * A_f * np.exp(-s_mid / LAM)
    return kP + kf


def fit_P(kv_obs, ei, noise_mm=1.0, seed=0):
    """用**只含 P 的模型**去拟合给定的 κ 场（换成形状 + 噪声）。"""
    rng = np.random.default_rng(seed)
    sh = T.roller(kv_obs)
    obs = sh + rng.normal(0, noise_mm * 1e-3, sh.shape)
    opti = ca.Opti()
    P = opti.variable()
    opti.subject_to(opti.bounded(0.0, P, 1.0))
    outs = T.rollout_casadi(P, ei)
    cost = 0
    for i in range(T.NSEG):
        px, pz, th = outs[i]
        cost += 1e6 * (px - obs[i, 0])**2 + 1e6 * (pz - obs[i, 1])**2
        cost += 20.0 * (th - sh[i, 2])**2
    opti.minimize(cost)
    opti.set_initial(P, 0.02)
    opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": False})
    return float(opti.solve().value(P))


def main():
    ei = T.seg_ei()
    P_true = 0.02                       # 20 mN
    print(f"真值尖端力 P = {P_true*1e3:.0f} mN，观测噪声 1 mm，λ={LAM*1e3:.0f} mm\n")
    print("  A_f(1/m) | 等效 ΔT(N) | 回收 P(mN) |   偏差    | 相对误差")
    print("  ---------+------------+------------+-----------+---------")
    base = None
    for A_f in (0.0, 0.02, 0.05, 0.10, 0.20, 0.40):
        kv = kappa_true(P_true, A_f, ei, +1)
        Pf = fit_P(kv, ei)
        if base is None:
            base = Pf
        dT = A_f * ei[0] / R_DISK
        bias = (Pf - P_true) * 1e3
        err = abs(Pf - P_true) / P_true * 100
        print(f"   {A_f:5.2f}   |   {dT:6.3f}   |   {Pf*1e3:7.3f}  | "
              f"{bias:+7.3f} mN | {err:5.1f}%")

    print("\n反向摩擦（符号相反）:")
    for A_f in (0.05, 0.10, 0.20):
        kv = kappa_true(P_true, A_f, ei, -1)
        Pf = fit_P(kv, ei)
        print(f"   A_f={A_f:4.2f}  ΔT={A_f*ei[0]/R_DISK:5.3f} N  "
              f"-> P={Pf*1e3:7.3f} mN (偏差 {(Pf-P_true)*1e3:+6.3f} mN)")

    print("\n等效换算（以 ΔM = A_f·EI, ΔT = ΔM/r_disk, r=5 mm）:")
    for A_f in (0.05, 0.10, 0.20):
        print(f"   A_f={A_f:4.2f} 1/m  <=>  ΔM={A_f*ei[0]*1e3:6.3f} mN·m  "
              f"<=>  ΔT={A_f*ei[0]/R_DISK:5.3f} N")
    print("\n  对照：T0=10 N、μ=0.15 的典型摩擦不平衡约 0.15–1.5 N"
          " => A_f 约 0.01–0.11 1/m")


if __name__ == "__main__":
    main()
