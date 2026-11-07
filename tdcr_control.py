# -*- coding: utf-8 -*-
"""tdcr_control.py — 基于本文模型的腱驱动弯曲控制

模型要点（2026-11-08 定案参数）
------------------------------
· 张力合成   τ_i = T0 + A·cos(α_i − φ)        α_i = 0/60/…/300°
· 分段曲率   段1（s∈[0,L1]）受 6 根腱 ⇒ κ1 = 3·A·r_disk/EI
             段2（s∈[L1,L]）只受 B 段 3 根 ⇒ κ2 = κ1/2
             总弯角 θ = κ1·L1 + κ2·L2 = 0.3·κ1（L1=L2=0.2）
· 摩擦          κ(s) = C/(1+μCs)，C=(e^{μθ}−1)/(μL)  ⇒ 根部曲率最大
· 方位          实测 φ_meas = 0.744·φ_cmd + 117°（标定层，纯旋转表达不了这个增益）
· 丝长          ΔL 由孔位轨迹积分（a 段丝只贴到 L1，位移约为 B 段的一半）

提供
----
    solve_tension(theta_deg, dir_deg)   目标弯角/方位 → 6 路张力
    predict(tau)                        张力 → (总弯角, 方位, 分段曲率, ΔL)
    control_step(target_deg, measured_deg)  单步闭环修正（P 型）
"""
import numpy as np

# ── 定案参数 ─────────────────────────────────────────────────────
EI = 0.01037            # N·m²
R_DISK = 5e-3           # m
L1 = L2 = 0.2           # m
T0 = 10.0               # N  基准预紧
TAU_MAX = 40.0          # N
MU = 0.15               # 腱-鞘摩擦
ALPHA_DEG = np.array([0, 60, 120, 180, 240, 300], float)
SEG_A = [0, 2, 4]       # 锚在 L1
SEG_B = [1, 3, 5]       # 锚在末端
DIR_GAIN, DIR_OFFSET = 0.744, 117.0     # 实测标定（dir_cmd → dir_meas）


def dir_to_cmd(dir_target):
    """目标方位(实测口径) → 下发方位(模型口径)。"""
    return (float(dir_target) - DIR_OFFSET) / DIR_GAIN


def dir_to_actual(dir_cmd):
    return float(dir_cmd) * DIR_GAIN + DIR_OFFSET


def kappa_seg(theta_deg):
    """总弯角 → (κ1, κ2)。θ = κ1·L1 + κ2·L2，κ2 = κ1/2。"""
    th = np.radians(theta_deg)
    k1 = th / (L1 + L2 / 2.0)
    return k1, k1 / 2.0


def solve_tension(theta_deg, dir_deg):
    """目标弯角/方位 → 6 路张力（解析反解）。"""
    k1, _ = kappa_seg(theta_deg)
    A = k1 * EI / (3 * R_DISK)              # 差动幅值
    phi_cmd = np.radians(dir_to_cmd(dir_deg))
    tau = T0 + A * np.cos(np.radians(ALPHA_DEG) - phi_cmd)
    return tau, A, phi_cmd


def predict(tau, mu=MU):
    """张力 → (θ, φ_meas, κ1, κ2, ΔL 极差, ΣT)。"""
    tau = np.asarray(tau, float)
    a = np.radians(ALPHA_DEG)
    M = tau * np.cos(a), tau * np.sin(a)
    mx, my = np.sum(tau * np.cos(a)), np.sum(tau * np.sin(a))
    # 差动幅值：对 τ 做一次拟合 A·cos(α−φ)
    # τ_i = T0 + A cos(α_i−φ) ⇒ 用 Στ·cosα、Στ·sinα 求
    A = 2.0 / 6.0 * np.hypot(mx, my) * 1.0
    phi = np.arctan2(my, mx)
    k1 = 3 * A * R_DISK / EI
    k2 = k1 / 2.0
    theta = np.degrees(k1 * (L1 + L2 / 2.0))
    # ΔL：B 段丝全行程，A 段丝一半（只贴到 L1）
    dL_B = R_DISK * k1 * (L1 + L2) * 1e3
    dL_A = R_DISK * k1 * L1 * 1e3
    dL = np.zeros(6)
    dL[SEG_A] = dL_A
    dL[SEG_B] = dL_B
    # dir_to_actual 返回的已是**度**，不要再套 np.degrees（会×57.3）
    return theta, dir_to_actual(np.degrees(phi)), k1, k2, dL, tau.sum()


def control_step(theta_target, theta_measured, A_prev, gain=0.8):
    """乘性 P 修正：θ ∝ A ⇒ A_new = A_prev·(1 + gain·(θ_t/θ_m − 1))。

    比加性修正更稳：转角与差动幅值近似成正比，乘性更新天然按比例收敛。
    """
    k1_t, _ = kappa_seg(theta_target)
    A_cmd = k1_t * EI / (3 * R_DISK)
    if theta_measured <= 1e-6 or A_prev <= 0:
        return A_cmd                      # 起步：直接给解析解
    ratio = theta_target / theta_measured
    return A_prev * (1.0 + gain * (ratio - 1.0))


def _selftest():
    print("=== 开环：目标 → 张力 → 模型预测 ===")
    print("%8s %8s %10s %10s %8s %8s" % ("目标θ", "目标φ", "预测θ", "预测φ", "ΣT", "A"))
    ok = True
    for th in (30, 60, 90, 120, 150, 180):
        for ph in (0, 90, 180, 270):
            tau, A, _ = solve_tension(th, ph)
            if tau.min() < -0.1 or tau.max() > TAU_MAX:
                ok = False
                continue
            pth, pph, k1, k2, dL, sT = predict(tau)
            if abs(ph - 0) == 0 or ph == 180:
                print("%8.1f %8.1f %10.2f %10.2f %8.1f %8.2f"
                      % (th, ph, pth, pph, sT, A))
    print("\n角度反解精度：", end="")
    errs = []
    for th in (30, 90, 150):
        tau, _, _ = solve_tension(th, 0)
        pth, *_ = predict(tau)
        errs.append(abs(pth - th))
    print("最大误差 %.2f°" % max(errs), "✓" if max(errs) < 0.5 else "✗")

    print("\n=== 分段曲率与 ΔL（目标 90°）===")
    tau, A, _ = solve_tension(90, 0)
    pth, pph, k1, k2, dL, sT = predict(tau)
    print("  κ1=%.2f κ2=%.2f 1/m（比值 %.2f）" % (k1, k2, k1 / k2))
    print("  ΔL: A 段丝 %.2f mm，B 段丝 %.2f mm（比值 %.2f，符合“A 段只贴 L1”）"
          % (dL[0], dL[1], dL[1] / dL[0]))
    print("  ΣT=%.1f N，τ 范围 %.1f–%.1f N" % (sT, tau.min(), tau.max()))

    print("\n=== 闭环：3 步迭代（假设实测=真值×0.8 的增益误差）===")
    for th in (60, 120):
        A = 0.0
        meas = 0.0
        print("  目标 %.0f°：" % th, end="")
        for it in range(4):
            A = control_step(th, meas, A) if it else kappa_seg(th)[0] * EI / (3 * R_DISK)
            tau = T0 + A * np.cos(np.radians(ALPHA_DEG))
            if tau.min() < 0:
                print(" [τ<0 截断]", end="")
                break
            pred, *_ = predict(tau)
            meas = 0.8 * pred                 # 模拟真机增益只有 80%
            print(" %.1f°" % meas, end="")
        print(" %s" % ("✓" if abs(meas - th) < 1.0 else "✗"))


if __name__ == "__main__":
    _selftest()
