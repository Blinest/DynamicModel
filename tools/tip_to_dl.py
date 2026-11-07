# -*- coding: utf-8 -*-
"""tip_to_dl.py — 给定末端位姿，反求驱动丝位移量

重要结论先说
------------
模型是 **2 参数族**（κ1, φ）：
    φ 决定弯曲平面（方位），κ1 决定弯多少（段2 恒为 κ1/2）
⇒ 末端在弯曲平面内的位置只有 **1 个自由度**（沿一条曲线），
  平面内任意 (ρ, z) 一般**不可精确达到**，只能投影到曲线上取最近点。
所以本脚本对每个目标末端位置给出：
    最近的 (κ1 → θ) 、方位 φ、残差 mm、以及对应的 6 路 ΔL

输出 fig_tip_dl.png + 控制台表格
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tdcr_control as C   # noqa: E402

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

R_O = 8.0
L1 = L2 = 200.0


def tip_of(k1, n=200):
    """两段常曲率（κ2=k1/2）的末端位置 (横向, 轴向) mm。"""
    p = np.zeros(2)                 # m
    th = 0.0
    for kap, Lseg in ((k1, L1), (k1 / 2.0, L2)):
        ds = (Lseg / 1000.0) / n     # ⚠ κ 是 1/m ⇒ ds 必须换成米
        for _ in range(n):
            p = p + np.array([np.sin(th), np.cos(th)]) * ds
            th += kap * ds
    return p[0] * 1e3, p[1] * 1e3, np.degrees(th)


def main():
    # 可达曲线：κ1 从 0 扫到 12 1/m（θ 到约 206°）
    k1s = np.linspace(1e-6, 12.0, 600)
    lat, ax_ = [], []
    for k1 in k1s:
        a, b, _ = tip_of(k1)
        lat.append(a); ax_.append(b)
    lat, ax_ = np.array(lat), np.array(ax_)

    # 取**可达曲线上的点**作为"固定末端位姿"（任意 (ρ,z) 不可达，见文件头说明）
    targets = []
    for th_t in (30, 60, 90, 120, 150, 180):
        k1_t = np.radians(th_t) / 0.3
        x_, z_, _ = tip_of(k1_t)
        targets.append((round(x_, 1), round(z_, 1)))
    print("%8s %8s %8s %8s %8s %9s %9s %9s %9s" %
          ("目标x", "目标z", "κ1", "θ", "φ", "残差mm", "ΔL_A", "ΔL_B", "ΣT"))
    rows = []
    for tx, tz in targets:
        d = np.hypot(lat - tx, ax_ - tz)
        i = int(np.argmin(d))
        k1 = k1s[i]
        th_deg = np.degrees(k1 * (0.2 + 0.1))          # θ = κ1(L1 + L2/2)
        phi = np.degrees(np.arctan2(tx, tz))           # 目标方位（相对 z 轴）
        tau, A, _ = C.solve_tension(th_deg, phi)
        _, _, k1c, k2c, dL, sT = C.predict(tau)
        res = d[i]
        print("%8.0f %8.0f %8.2f %8.1f %8.1f %9.1f %9.2f %9.2f %9.0f" %
              (tx, tz, k1c, th_deg, phi, res, dL[0], dL[1], sT))
        rows.append((tx, tz, lat[i], ax_[i], k1c, th_deg, phi, res,
                     dL[0], dL[1]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4),
                                   gridspec_kw=dict(width_ratios=[1, 1]))
    ax1.plot(lat, ax_, color=BLUE, lw=2.2, label="可达曲线（κ1 单参数）")
    for (tx, tz, px, pz, k1c, th_deg, phi, res, la, lb) in rows:
        ax1.plot([tx], [tz], "o", color=RED, ms=6, mfc="none", mew=1.4)
        ax1.plot([px], [pz], "+", color=GREEN, ms=8, mew=1.4)
        ax1.plot([tx, px], [tz, pz], color="0.6", lw=0.7, ls=":")
        ax1.annotate(f"θ={th_deg:.0f}°", (px, pz), textcoords="offset points",
                     xytext=(6, -10), fontsize=7, color=GREEN)
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.2, ls=":")
    ax1.set_xlabel("横向 (mm)"); ax1.set_ylabel("轴向 (mm)")
    ax1.set_title("目标末端取在可达曲线上（空心圈=模型可达点）", fontsize=9, color=RED)
    ax1.legend(fontsize=8)

    xs = np.arange(len(rows))
    ax2.bar(xs - 0.18, [r[8] for r in rows], 0.36, color=RED, label="A 段丝 ΔL")
    ax2.bar(xs + 0.18, [r[9] for r in rows], 0.36, color=GREEN, label="B 段丝 ΔL")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"θ={r[5]:.0f}°" for r in rows], fontsize=8)
    ax2.set_ylabel("驱动丝位移量 (mm)")
    ax2.grid(alpha=0.25, ls=":", axis="y")
    ax2.legend(fontsize=8)
    ax2.set_title("对应 6 路位移（A 段 3 根 / B 段 3 根，各相同）", fontsize=9,
                  color=GREEN)
    fig.suptitle("末端位姿 → 驱动丝位移量（模型仅 2 参数 ⇒ 末端 1 自由度，"
                 "不能任意指定）", fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_tip_dl.png", dpi=160, bbox_inches="tight")
    print("saved fig_tip_dl.png")


if __name__ == "__main__":
    main()
