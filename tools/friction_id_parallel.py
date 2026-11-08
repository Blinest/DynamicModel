# -*- coding: utf-8 -*-
"""friction_id_parallel.py — 6 丝并行标定的实现（含"只有 3 根受拉"的约束）

并行标定真正的难点（不只是"同时读 6 路"）
------------------------------------------
① **丝只能拉**：一个方向的弯曲只有距弯曲方位 ±60° 内的 3 根受拉，
   另 3 根松弛、张力 = 预紧 ⇒ 拿不到滞回 ⇒ **必须两个相反方向各扫一次**。
② **Θ_j 逐丝不同**：丝的跨度（A 段丝 200 mm / B 段丝 400 mm）不同，
   累计转角不同 ⇒ 配对时必须用**每根丝自己的 Θ_j**，不能共用一个弯角轴。
③ **同时抖动会耦合**：6 丝同相位抖动会让形状整体晃动 ⇒ 用**错相位**
   （相邻丝差 60°），使合力矩近似恒定。

时间账（并行 + 连续扫描）
    2 个方向 × 60 s = 120 s ⇒ **2 min**，且两个预紧级别 = 4 次扫描 ⇒ 4 min
    （若预紧级别可只做 2 次、每次 45 s ⇒ 3 min）

本脚本：模拟两个方向、6 丝并行、错相位抖动，逐丝反求 (μ_j, c_j)。
"""
# ── 包路径引导（分层目录）────────────────────────────────────
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('geometry', 'stiffness', 'force', 'plot'):
    _p = _os.path.join(_ROOT, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
# ──────────────────────────────────────────────────────────────


import numpy as np

NWIRE = 6
AZIM = np.array([0, 60, 120, 180, 240, 300], float)   # 丝的周向位置(°)
TAUT_HALF = 60.0                                       # 受拉扇区半角
THETA_MAX = np.radians(120.0)
T_SCAN = 60.0
FS = 20.0
NOISE = 0.05
PHASE_SHIFT = 60.0          # 错相位抖动


def truth():
    mu = np.array([0.11, 0.14, 0.09, 0.16, 0.12, 0.13])
    c = np.array([0.10, 0.06, 0.13, 0.05, 0.08, 0.09])
    span = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])   # A/B 段跨度比（此处等同）
    return mu, c, span


def taut_set(phi_deg):
    """给定弯曲方位，返回受拉丝的索引（|ψ−φ| ≤ 60°）。"""
    d = np.abs((AZIM - phi_deg + 180) % 360 - 180)
    return np.where(d <= TAUT_HALF)[0]


def scan_dir(phi_deg, tau, mu, c, span, rng, t_scan=T_SCAN, fs=FS):
    """一个方向的三角扫描；返回 (Theta_j, meas[n,6], up, taut)。"""
    n = int(t_scan * fs)
    t = np.arange(n) / fs
    frac = t / t_scan
    bend = np.where(frac < 0.5, frac * 2, (1 - frac) * 2) * THETA_MAX
    up = frac < 0.5
    taut = taut_set(phi_deg)
    # 每根丝自己的累计转角：受拉丝用 bend×跨度，其余丝松弛（不进数据）
    Theta = bend[:, None] * span[None, :] * np.cos(np.radians(AZIM - phi_deg))[None, :]
    F = (mu[None, :] * tau + c[None, :]) * np.abs(Theta)
    meas = tau + np.where(up[:, None], F, -F) + rng.normal(0, NOISE, (n, NWIRE))
    return np.abs(Theta), meas, up, taut


def extract(theta, meas, up, j, n_grid=100):
    """取第 j 根丝的 F(Θ)：上下支按**该丝自己的 Θ** 插值配对。"""
    th_j = theta[:, j]
    lo, hi = th_j.min(), th_j.max()
    grid = np.linspace(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo), n_grid)

    def interp(mask):
        x, y = th_j[mask], meas[mask, j]
        o = np.argsort(x)                     # 下降支 Θ 递减，必须排序
        return np.interp(grid, x[o], y[o])

    return grid, 0.5 * (interp(up) - interp(~up))


def main():
    mu_t, c_t, span = truth()
    rng = np.random.default_rng(3)
    print("6 丝并行标定：两个相反方向 × 三角扫描 × 错相位抖动\n")
    for phi in (0.0, 180.0):
        print(f"  弯曲方位 φ={phi:.0f}°  → 受拉丝 "
              f"{[f'M{k+1}' for k in taut_set(phi)]}"
              f"（其余松弛，拿不到滞回）")
    print()

    taus = (8.0, 14.0)
    slopes = {tau: np.full(NWIRE, np.nan) for tau in taus}
    t0 = 0.0
    for phi in (0.0, 180.0):
        taut = taut_set(phi)
        for tau in taus:
            theta, meas, up, _ = scan_dir(phi, tau, mu_t, c_t, span, rng)
            for j in taut:
                g, F = extract(theta, meas, up, j)
                slopes[tau][j] = np.polyfit(g, F, 1)[0]
            t0 += T_SCAN
    k1, k2 = slopes[taus[0]], slopes[taus[1]]
    mu_h = (k2 - k1) / (taus[1] - taus[0])
    c_h = k1 - mu_h * taus[0]

    print("  丝  | mu真  mu辨   误差 |  c真   c辨   误差 | 受拉于")
    print("  ----+---------------------+------------------+--------")
    for j in range(NWIRE):
        phi = 0.0 if j in taut_set(0.0) else 180.0
        e = (abs(mu_h[j]-mu_t[j])/mu_t[j]*100) if np.isfinite(mu_h[j]) else float('nan')
        print(f"  M{j+1}  | {mu_t[j]:.3f} {mu_h[j]:.3f} {e:5.1f}% |  "
              f"{c_t[j]:.3f} {c_h[j]:.3f} {abs(c_h[j]-c_t[j]):5.3f} | φ={phi:.0f}°")
    ok = np.isfinite(mu_h)
    print(f"\n  mu 平均相对误差 {np.mean(np.abs(mu_h[ok]-mu_t[ok])/mu_t[ok])*100:.1f}%"
          f"（覆盖 {ok.sum()}/6 根）")
    print(f"  c  平均绝对误差 {np.mean(np.abs(c_h[ok]-c_t[ok])):.3f} N")
    print(f"\n  标定总时长 {t0:.0f} s = {t0/60:.1f} min"
          f"（2 方向 × 2 预紧 × {T_SCAN:.0f} s）")

    print("\n实现要点：")
    for s in ["· 两个相反方向各扫一次（单向只能标 3 根）",
              "· 上下支按**每根丝自己的 Θ_j** 插值配对，且插值前 argsort",
              "· 相邻丝抖动相位差 60°，避免合力矩整体晃动",
              "· 松弛丝的数据直接丢弃（张力≈预紧、无滞回）",
              "· 每级预紧静置 3 s；端部 5% 弯角区间丢弃"]:
        print("  " + s)


if __name__ == "__main__":
    main()
