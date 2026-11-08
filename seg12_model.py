# -*- coding: utf-8 -*-
"""seg12_model.py — 12 段曲率模型（把臂体分成 12 段，每段一个 κ）

为什么加这个参数
----------------
2 段模型只有 (κ1, φ) 两个参数 ⇒ 末端在弯曲平面内只有 **1 个自由度**（一条曲线），
任意 (ρ, z) 不可达（实测差 60–224 mm）。

12 段模型给 12 个 κ，但受物理约束（张力只能差动）实际用于控制在两个量：
    θ   总弯角
    p   弯曲集中指数：κ_i ∝ w_i^p，w_i 为基线权重（摩擦衰减 + 6丝/3丝）
⇒ 末端在平面内变成 **2 自由度（曲面）**，可以精确命中平面内的目标点。

基线权重
--------
    w_i ∝ 1/(1+μ·C·s_i)   （Capstan 衰减，根部大）
         × fact(段)        （s<200mm 的段受 6 根腱 ⇒ 力矩 2 倍于 s>200mm 的段）
"""
import numpy as np

EI = 0.01037
R_DISK = 5e-3
L = 0.4
L1 = 0.2
MU = 0.15
NSEG = 20          # 真实样机 20 个间隔盘 = 20 段（每段 20 mm）
SEG_LEN = L / NSEG
T0 = 10.0


def base_weights(mu=MU):
    s = (np.arange(NSEG) + 0.5) * SEG_LEN
    C = 1.0 / (1.0 + mu * 10.0 * s)          # 用 κ≈10 1/m 的量级做衰减参考
    fact = np.where(s <= L1, 1.0, 0.5)       # 段1 受 6 根腱、段2 只受 3 根
    w = C * fact
    return w / w.sum()


def kappa_vec(theta_deg, p=1.0, mu=MU):
    """总弯角 θ、集中指数 p ⇒ 12 段曲率 (1/m)。"""
    w = base_weights(mu) ** p
    w = w / w.sum()
    theta = np.radians(theta_deg)
    return theta * w / SEG_LEN          # Σ κ_i·SEG_LEN = θ


def shape(kvec, n_sub=8):
    """按 12 段曲率积分骨架，返回 p(m (N,2), 转角, 各段起点)。"""
    p = np.zeros(2)
    th = 0.0
    P = [p.copy()]
    for k in kvec:
        ds = SEG_LEN / n_sub
        for _ in range(n_sub):
            p = p + np.array([np.sin(th), np.cos(th)]) * ds
            th += k * ds
            P.append(p.copy())
    return np.array(P), th


def tip(theta_deg, p):
    P, th = shape(kappa_vec(theta_deg, p))
    return P[-1, 0] * 1e3, P[-1, 1] * 1e3, np.degrees(th)


def solve_target(rho_mm, z_mm, iters=60):
    """(θ, p) 命中平面目标：先粗网格搜最近点，再做局部细化（Newton 易发散）。"""
    ths = np.linspace(1.0, 200.0, 60)
    ps = np.linspace(0.15, 10.0, 40)
    best = (1e9, 90.0, 1.0)
    for th in ths:
        for p in ps:
            x, z, _ = tip(th, p)
            d = np.hypot(x - rho_mm, z - z_mm)
            if d < best[0]:
                best = (d, th, p)
    d0, th, p = best
    # 局部细化：小步长爬山
    step_th, step_p = 2.0, 0.15
    for _ in range(iters):
        improved = False
        for dth in (-step_th, 0, step_th):
            for dp in (-step_p, 0, step_p):
                t2 = float(np.clip(th + dth, 0.5, 200))
                p2 = float(np.clip(p + dp, 0.12, 12.0))
                x, z, _ = tip(t2, p2)
                d = np.hypot(x - rho_mm, z - z_mm)
                if d < d0 - 1e-6:
                    d0, th, p, improved = d, t2, p2, True
        if not improved:
            step_th *= 0.5
            step_p *= 0.5
            if step_th < 1e-3:
                break
    x, z, _ = tip(th, p)
    return th, p, d0, (x, z)


def dl_of(theta_deg, p=1.0):
    """6 路 ΔL：A 段丝只看前 6 段、B 段丝看全 12 段（按各自累积转角）。"""
    k = kappa_vec(theta_deg, p)
    th_A = np.sum(k[:6]) * SEG_LEN          # 前 200 mm 的转角
    th_B = np.sum(k) * SEG_LEN
    return R_DISK * th_A * 1e3, R_DISK * th_B * 1e3


if __name__ == "__main__":
    print("=== 基线与形状 ===")
    w = base_weights()
    print("  基线权重 (12 段)：", np.round(w, 3))
    print("  段长 %.3f mm" % (SEG_LEN * 1e3))
    for th in (30, 90, 180):
        x, z, d = tip(th, 1.0)
        a, b = dl_of(th, 1.0)
        print(f"  θ={th:3d}°  tip=({x:7.1f},{z:7.1f}) mm  ΔL_A={a:5.2f} ΔL_B={b:5.2f}")

    print("\n=== 之前 2 段模型打不到的目标，12 段能否命中 ===")
    for rho, z in ((60, 400), (150, 380), (250, 320), (300, 240), (200, 20)):
        th, p, res, tp = solve_target(rho, z)
        a, b = dl_of(th, p)
        flag = "✓" if res < 1.0 else "✗"
        print(f"  目标({rho:3.0f},{z:3.0f}) → θ={th:6.1f}° p={p:5.2f} "
              f"命中({tp[0]:6.1f},{tp[1]:6.1f}) 残差 {res:6.2f} mm "
              f"ΔL_A={a:5.2f} ΔL_B={b:5.2f}  {flag}")
