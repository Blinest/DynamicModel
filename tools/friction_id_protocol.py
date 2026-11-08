# -*- coding: utf-8 -*-
"""friction_id_protocol.py — 用"加载/卸载张力滞回"辨识每根丝的摩擦因数

为什么不能比"绝对张力"
----------------------
τ_meas − τ_ideal 里同时混着：EI 误差（直接改所需力矩）、预紧零漂、重力。
EI 与 μ 无法从一次绝对测量中分开（与"尖端力/EI 退化"同一个病）。

为什么"滞回"可以
----------------
同一形状、**加载方向相反**时：
    拉进（加载）  τ_up = τ_tip · e^{+μΘ}
    放出（卸载）  τ_dn = τ_tip · e^{−μΘ}
两者之差只由摩擦产生，**EI 完全抵消**（同一形状 ⇒ 同一所需力矩）。

    Δτ = τ_up − τ_dn = 2·τ_mean·sinh(μΘ) ≈ 2·μ·Θ·τ_mean   (μΘ ≪ 1)

于是  μ ≈ Δτ / (2·Θ·τ_mean)，Θ 为丝在盘孔处的累计转折角
（可由 κ(s) 积分给出，见 seg12_model / cosserat_lattice）。

本脚本给出：Δτ 随 μ、Θ 的变化，以及在给定力传感器噪声下 μ 的辨识分辨率。
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

TAU_MEAN = 10.0      # N，丝内平均张力（预紧 + 差动）
SENSOR_NOISE = 0.1   # N，电机侧力传感器分辨力（典型值）


def delta_tau(mu, Theta, tau=TAU_MEAN):
    """加载/卸载张力差（N）。"""
    return 2.0 * tau * np.sinh(mu * Theta)


def main():
    print(f"丝内平均张力 τ={TAU_MEAN:.0f} N，力传感器分辨力 {SENSOR_NOISE:.1f} N\n")
    print("Δτ (N) 随 μ 与总转折角 Θ 的变化：")
    Thetas = np.radians([30, 60, 90, 150, 180])
    head = "  μ    |" + "".join(f"  Θ={np.degrees(t):.0f}° " for t in Thetas)
    print(head)
    print("  -----+" + "----------" * len(Thetas))
    for mu in (0.05, 0.10, 0.15, 0.20, 0.30):
        row = f"  {mu:.2f} |"
        for t in Thetas:
            row += f"  {delta_tau(mu, t):6.2f}  "
        print(row)

    print(f"\nμ 的可辨识分辨率（由传感器噪声决定）：")
    print("  Θ      μ_分辨率(±)   μ=0.15 时的相对精度")
    for t in Thetas:
        dmu = SENSOR_NOISE / (2.0 * t * TAU_MEAN)
        print(f"  {np.degrees(t):5.0f}°   ±{dmu:6.4f}      "
              f"{dmu/0.15*100:5.1f}%")

    print("\n结论：")
    dmu90 = SENSOR_NOISE / (2.0 * np.radians(90) * TAU_MEAN)
    print(f"  · 90° 弯曲、0.1 N 传感器 ⇒ μ 精度 ±{dmu90:.4f}"
          f"（μ=0.15 时 {dmu90/0.15*100:.1f}%）——足够")
    print(f"  · 30° 小弯时精度降到 ±{SENSOR_NOISE/(2*np.radians(30)*TAU_MEAN):.4f}"
          f" ⇒ 标定应在**大弯角**下做")
    print("  · Δτ 与 μ 近似线性，但 μ>0.3 后指数项不可忽略（sinh 非线性）")
    print("\n标定流程：")
    steps = [
        "1. 直臂归零，记录每丝 τ_0（预紧 + 零漂）",
        "2. 只驱动一根丝，慢速弯到目标角，静置 2 s，记 τ_up",
        "3. 同速回直，记 τ_dn（整个过程其余丝保持预紧不变）",
        "4. 对 5–8 个弯角重复，得 Δτ(Θ) 曲线",
        "5. 单参数拟合 μ：Δτ = 2τ·sinh(μΘ)，Θ 由 κ(s) 积分给出",
        "6. 6 根丝各做一遍 ⇒ 逐丝 μ_j（或取中位数作为共同 μ）",
    ]
    for s_ in steps:
        print("   " + s_)


if __name__ == "__main__":
    main()
