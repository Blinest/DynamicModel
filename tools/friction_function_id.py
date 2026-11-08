# -*- coding: utf-8 -*-
"""friction_function_id.py — 直接辨识**每根丝自己的摩擦力函数** f_j(T)

比单个 μ 更强的地方
------------------
Capstan 模型假设摩擦力与被拉张力成正比（f = μT）。真实结构里还有：
  · 盘孔处的**定值**卡滞（预压、贴壁）   ⇒ 与张力无关的常数项 c
  · 材料/几何非均匀                     ⇒ 逐丝不同
所以把摩擦力写成两参数函数：

    f_j(T) = μ_j·T + c_j          （与张力成正比的部分 + 常数部分）

沿丝积分（Θ = 累计转折角）：

    F_j(Θ) = ∫₀^Θ f_j(T(φ)) dφ ≈ (μ_j·τ + c_j)·Θ      （τ 近似恒定）

实测能得到的是**滞回半宽**（加载与卸载张力差的一半）：

    F_j(Θ) = [τ_up(Θ) − τ_dn(Θ)] / 2

于是两次实验即可分离两个参数：
    ① 固定预紧 τ，扫弯角 Θ  ⇒ F 对 Θ 的**斜率** = μ_j·τ + c_j
    ② 固定弯角 Θ，扫预紧 τ  ⇒ F 随 τ 的**斜率** = μ_j·Θ
       ⇒ μ_j = 斜率/(Θ)，c_j = (①的斜率) − μ_j·τ

本脚本：用已知 (μ_j, c_j) 造数据（含传感器噪声），跑上述辨识，报回收精度。
"""
import numpy as np

NWIRE = 6
RNG = np.random.default_rng(7)
SENSOR_NOISE = 0.05      # N，单次张力读数噪声
SEG_NOISE = 0.001        # N，Δτ 的重复性噪声


def truth():
    """6 根丝的真实摩擦参数（μ 有差异，c 全为正=定值卡滞）。"""
    mu = np.array([0.11, 0.14, 0.09, 0.16, 0.12, 0.13])
    c = np.array([0.10, 0.06, 0.13, 0.05, 0.08, 0.09])
    return mu, c


def measure(mu, c, tau, Theta, rng, n_rep=5):
    """返回实测滞回半宽 F = (tau_up - tau_dn)/2（N），含噪声。"""
    F = (mu * tau + c) * Theta
    return F + rng.normal(0, SENSOR_NOISE / np.sqrt(n_rep), F.shape)


def identify(rng):
    mu_t, c_t = truth()
    Thetas = np.radians(np.array([30, 45, 60, 90, 120, 150]))
    taus = np.array([6.0, 8.0, 10.0, 13.0, 16.0])

    # ① 固定 τ=10 N，扫 Θ → 斜率 k1 = μ·τ + c
    F1 = np.array([measure(mu_t, c_t, 10.0, th, rng) for th in Thetas])
    k1 = np.array([np.polyfit(Thetas, F1[:, j], 1)[0] for j in range(NWIRE)])

    # ② 固定 Θ=90°，扫 τ → 斜率 k2 = μ·Θ  ⇒ μ = k2/Θ
    th0 = np.radians(90.0)
    F2 = np.array([measure(mu_t, c_t, t, th0, rng) for t in taus])
    k2 = np.array([np.polyfit(taus, F2[:, j], 1)[0] for j in range(NWIRE)])

    mu_hat = k2 / th0
    c_hat = k1 - mu_hat * 10.0
    return mu_t, c_t, mu_hat, c_hat


def main():
    mu_t, c_t, mu_h, c_h = identify(RNG)
    print("每根丝的摩擦函数 f_j(T) = mu_j*T + c_j")
    print("   σ=0.05 N 传感器噪声, 每点重复 5 次\n")
    print("  丝  |  mu真值  mu辨识  误差  |  c真值  c辨识   误差")
    print("  ----+-----------------------+---------------------")
    for j in range(NWIRE):
        print(f"  M{j+1}  |  {mu_t[j]:.3f}  {mu_h[j]:.3f}  "
              f"{abs(mu_h[j]-mu_t[j])/mu_t[j]*100:5.1f}%  |  "
              f"{c_t[j]:.3f}  {c_h[j]:.3f}  "
              f"{abs(c_h[j]-c_t[j]):5.3f}")
    print(f"\n  mu 平均相对误差 {np.mean(np.abs(mu_h-mu_t)/mu_t)*100:.1f}%")
    print(f"  c  平均绝对误差 {np.mean(np.abs(c_h-c_t)):.3f} N")

    print("\n对应摩擦力（τ=10 N, 90° 弯曲，每根丝的总摩擦阻力）：")
    for j in range(NWIRE):
        F = (mu_t[j]*10.0 + c_t[j]) * np.radians(90)
        print(f"  M{j+1}: {F:5.2f} N   (占张力 {F/10*100:4.1f}%)")

    print("\n辨识所需实验：")
    exp = ["① 扫弯角（τ 固定）：6 个角度 × 6 根丝",
           "② 扫预紧（Θ 固定 90°）：5 个张力 × 6 根丝",
           "共 66 次加载/卸载往返，每次约 20 s ⇒ 约 22 分钟"]
    for e in exp:
        print("   " + e)


if __name__ == "__main__":
    main()
