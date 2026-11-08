# -*- coding: utf-8 -*-
"""friction_id_fast.py — 快速标定：连续三角扫描 + 6 丝并行（22 min → 2 min）

原方案慢在哪
------------
66 次"弯到角度 → 停 → 回直 → 停"，每次 20 s。慢在**分段停顿**和**逐丝串行**。

快在哪（两条都成立，可叠加）
---------------------------
① **连续三角扫描**：一次把弯角从 0 扫到 Θmax 再扫回 0，全程 20 Hz 采样。
   同一弯角下，上升支与下降支的张力差就是 2F ⇒ **一条曲线给出整条 F(Θ)**，
   不用分 6 个角度停 6 次。
② **6 丝并行**：形状由 6 根丝共同决定，但小幅滞回时形状几乎不动，
   6 根丝各自的力传感器同时读 ⇒ 一次扫描同时标 6 根。

   总时间 = 2 次扫描（两个预紧级别，用于分离 μ 与 c）× 60 s ≈ **2 分钟**

本脚本：用已知 (μ_j, c_j) 造连续扫描数据（20 Hz、0.05 N 噪声），
跑"上下支配对 → 斜率 → 分离 μ/c"，报精度与耗时。
"""
import numpy as np

NWIRE = 6
THETA_MAX = np.radians(120.0)
T_SCAN = 60.0        # s，单次扫描时长
FS = 20.0            # Hz
NOISE = 0.05         # N


def truth():
    mu = np.array([0.11, 0.14, 0.09, 0.16, 0.12, 0.13])
    c = np.array([0.10, 0.06, 0.13, 0.05, 0.08, 0.09])
    return mu, c


def scan(mu, c, tau, rng, t_scan=T_SCAN, fs=FS):
    """生成一次三角扫描的 (Theta, tau_meas)。"""
    n = int(t_scan * fs)
    t = np.arange(n) / fs
    # 三角波：0 -> Theta_max (前半) -> 0 (后半)
    frac = t / t_scan
    th = np.where(frac < 0.5, frac * 2, (1 - frac) * 2) * THETA_MAX
    up = frac < 0.5
    F = (mu * tau + c) * th[:, None]            # (n, 6)
    meas = tau + np.where(up[:, None], F, -F) + rng.normal(0, NOISE, (n, NWIRE))
    return th, meas, up


def extract_F(th, meas, up, tau, n_grid=120):
    """把上下支插值到同一 Θ 网格，取半差。"""
    grid = np.linspace(THETA_MAX * 0.05, THETA_MAX * 0.95, n_grid)
    def interp(mask, j):
        x, y = th[mask], meas[mask, j]
        o = np.argsort(x)                       # 下降支的 th 递减，必须排序
        return np.interp(grid, x[o], y[o])
    F = np.array([0.5 * (interp(up, j) - interp(~up, j)) for j in range(NWIRE)]).T
    return grid, F


def main():
    mu_t, c_t = truth()
    rng = np.random.default_rng(11)
    print(f"连续三角扫描：0→{np.degrees(THETA_MAX):.0f}°→0，{T_SCAN:.0f} s，"
          f"{FS:.0f} Hz 采样，6 丝并行\n")

    # 两个预紧级别
    res = {}
    for tau in (8.0, 14.0):
        th, meas, up = scan(mu_t, c_t, tau, rng)
        grid, F = extract_F(th, meas, up, tau)
        k = np.array([np.polyfit(grid, F[:, j], 1)[0] for j in range(NWIRE)])
        res[tau] = k
    k1, k2 = res[8.0], res[14.0]
    # k(τ) = μ·τ + c  ⇒ 用两个 τ 线性插出 μ 与 c
    mu_h = (k2 - k1) / (14.0 - 8.0)
    c_h = k1 - mu_h * 8.0

    print("  丝  |  mu真  mu辨   误差 |  c真   c辨    误差")
    print("  ----+----------------------+--------------------")
    for j in range(NWIRE):
        print(f"  M{j+1}  | {mu_t[j]:.3f}  {mu_h[j]:.3f}  "
              f"{abs(mu_h[j]-mu_t[j])/mu_t[j]*100:4.1f}% |  "
              f"{c_t[j]:.3f}  {c_h[j]:.3f}  {abs(c_h[j]-c_t[j]):5.3f}")
    print(f"\n  mu 平均相对误差 {np.mean(np.abs(mu_h-mu_t)/mu_t)*100:.1f}%")
    print(f"  c  平均绝对误差 {np.mean(np.abs(c_h-c_t)):.3f} N")

    print(f"\n耗时对比：")
    print(f"  原方案（66 次分段往返，逐丝）   ≈ 22 min")
    print(f"  本方案（2 次扫描 × {T_SCAN:.0f} s，6 丝并行） ≈ "
          f"{2*T_SCAN/60:.0f} min")
    print(f"  进一步：单次扫描缩到 30 s（放慢到 8 Hz 采样）=> "
          f"{2*30/60:.0f} min")

    print("\n实现要点：")
    for s in ["· 电机侧 6 路独立力传感（或用电流×常数标定）",
              "· 三角波速度恒定（不可中途变速，否则滞回环变形）",
              "· 每级预紧先静置 3 s 消除蠕变",
              "· 只取 Θ 在 5%–95% 区间的数据（端部换向有加速）",
              "· 上下支配对必须按**同一弯角**插值，不能按时间配对"]:
        print("  " + s)


if __name__ == "__main__":
    main()
