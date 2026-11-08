# -*- coding: utf-8 -*-
"""tension_scan.py — 增大预紧张力会怎样？（扫 ΣT）

要看清三件事
------------
1. **可达弯角上限**：τ 必须 ≥0 ⇒ 差动幅值 A ≤ T0 − τ_min ⇒
   θ_max ≈ κ1_max·0.3，其中 κ1 = 3·A·r/EI ⇒ **θ_max 与 T0 成正比**
2. **轴向压缩**：δ = ΣT / k_rod（k_rod=7000 N）⇒ 预紧越高臂越短
3. **摩擦衰减**：μ 不变，但总张力高时机端力矩也高，远端相对更弱

用 cosserat_lattice 的两段刚度（段1 0.01499 / 段2 0.00950，段1 筋宽 2mm）
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cosserat_lattice as CL   # noqa: E402

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

R_DISK = 5e-3
L = 0.4
K_ROD = 7000.0
TAU_MIN = 0.5
TAU_MAX_MOTOR = 60.0          # 单根丝上限（电机/丝强度）


def theta_max(segEI, T0):
    """给定基准预紧 T0（每根），最大差动 A = T0 − τ_min，返回 (θ_max, A, ΣT)。"""
    A = T0 - TAU_MIN
    if A <= 0:
        return 0.0, 0.0, 6 * T0
    k1 = 3 * A * R_DISK / segEI["EI"]
    k2 = k1 * segEI["EI"] / CL.stiffness(CL.SEG[1])["EI"]       # 段2 力矩为段1 的 1/2
    th = np.degrees(k1 * 0.2 + k2 * 0.2)
    return th, A, 6 * T0


def main():
    ks = CL.stiffness()
    print(f"段1 EI={ks[0]['EI']:.5f}  段2 EI={ks[1]['EI']:.5f}")
    print("\n%6s %6s %7s %8s %9s %9s %9s" %
          ("T0/根", "ΣT", "A_max", "θ_max", "轴向压缩", "ΔL_A", "ΔL_B"))
    rows = []
    for T0 in (3, 5, 10, 15, 20, 30):
        th, A, sT = theta_max(ks[0], T0)
        k1 = 3 * A * R_DISK / ks[0]["EI"]
        k2 = k1 * ks[0]["EI"] / ks[1]["EI"]
        dlA = R_DISK * k1 * 0.2 * 1e3
        dlB = R_DISK * (k1 * 0.2 + k2 * 0.2) * 1e3
        comp = sT / K_ROD * 1e3
        rows.append((T0, sT, A, th, comp, dlA, dlB))
        print("%6.0f %6.0f %7.2f %7.1f° %8.2fmm %8.2f %8.2f" %
              (T0, sT, A, th, comp, dlA, dlB))

    print("\n要点：")
    print("  · θ_max 与 T0 近似成正比（A ≤ T0−0.5）")
    print("  · 每增加 10 N/根，ΣT +60 N ⇒ 轴向缩短 60/7000 = 8.6 mm")
    print(f"  · 单根上限 {TAU_MAX_MOTOR:.0f} N ⇒ T0 不宜超过其一半（留差动余量）")

    T0s = np.array([r[0] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(T0s, [r[3] for r in rows], "-o", color=RED, ms=5)
    axes[0].set_xlabel("预紧 T0（每根, N）")
    axes[0].set_ylabel("可达弯角上限 (°)")
    axes[0].set_title("θ_max ∝ T0（差动余量 = T0−τ_min）", fontsize=9, color=RED)
    axes[1].plot(T0s, [r[4] for r in rows], "-s", color=BLUE, ms=5)
    axes[1].set_xlabel("预紧 T0（每根, N）")
    axes[1].set_ylabel("轴向压缩 δ (mm)")
    axes[1].set_title("ΣT/k_rod：预紧越高臂越短", fontsize=9, color=BLUE)
    axes[2].plot(T0s, [r[5] for r in rows], "-o", color=RED, ms=4, label="ΔL_A")
    axes[2].plot(T0s, [r[6] for r in rows], "-s", color=GREEN, ms=4, label="ΔL_B")
    axes[2].set_xlabel("预紧 T0（每根, N）")
    axes[2].set_ylabel("最大收丝量 (mm)")
    axes[2].set_title("行程需求随预紧上升", fontsize=9, color=GREEN)
    axes[2].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.25, ls=":")
    fig.suptitle("增大预紧张力的三个后果（段1 筋宽 2mm，EI=0.01499/0.00950）",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_tension_scan.png", dpi=160, bbox_inches="tight")
    print("\nsaved fig_tension_scan.png")


if __name__ == "__main__":
    main()
