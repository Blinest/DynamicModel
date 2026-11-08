# -*- coding: utf-8 -*-
"""fig_arm_curves.py — 臂体中线曲线（按曲率着色）

左：4 个方位 φ 下的形状族（θ=0…180°，每 15° 一条），颜色=局部曲率
右：单条曲线放大，标出 20 个间隔盘节点与曲率分段
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.collections import LineCollection
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tdcr_model_v2 as V2

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"


def curve(m, th, ph, F=0.0):
    psi = np.radians(V2.GEOM["psi_deg"])
    Ph = np.radians(ph)
    L = m.g["L_total"]; half = m.N // 2
    comp = (half * m.ds) / m.EI[0] + (m.N - half) * m.ds / m.EI[-1]
    A = np.radians(th) / (3.0 * m.g["r_disk"] * comp)
    tau = np.maximum(0.0, m.T0 + A * np.cos(psi - Ph))
    o = m.solve(tau, F_tip=(np.array([0.0, -F, 0.0]) if F else None))
    return o


def main():
    m = V2.TdcrModelV2(n_sub=4)
    theta_max = 0.6                       # 统一色标上限
    fig, axes = plt.subplots(1, 5, figsize=(21, 5.0))
    for ax, ph in zip(axes[:4], (0, 90, 180, 270)):
        for th in np.arange(0, 181, 15):
            o = curve(m, th, ph)
            P3 = o["p"] * 1e3
            # 弯曲是平面内的 => 投影到 (弯曲方位, z) 平面
            c, sn = np.cos(np.radians(ph)), np.sin(np.radians(ph))
            P = np.column_stack([P3[:, 0] * c + P3[:, 1] * sn, P3[:, 2]])
            k = np.linalg.norm(o["kappa"][:, :2], axis=1)[:len(P)]
            segs = np.stack([P[:-1], P[1:]], axis=1)
            lc = LineCollection(segs, cmap="magma",
                                norm=plt.Normalize(0, 6.0), lw=2.0)
            lc.set_array(k[:-1] * 1.0)
            ax.add_collection(lc)
        ax.plot([0], [0], "s", color="k", ms=6)
        ax.set_aspect("equal")
        ax.set_xlim(-320, 320); ax.set_ylim(-60, 430)
        ax.grid(alpha=0.15, ls=":")
        ax.set_title(f"φ = {ph}°", fontsize=9, color=RED)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("z (mm)", fontsize=8)

    # 最后一栏：单条 90° 曲线 + 20 个盘节点
    ax = axes[4]
    o = curve(m, 90.0, 0.0)
    P3 = o["p"] * 1e3
    P = np.column_stack([P3[:, 0], P3[:, 2]])
    k = np.linalg.norm(o["kappa"][:, :2], axis=1)[:len(P)]
    segs = np.stack([P[:-1], P[1:]], axis=1)
    lc = LineCollection(segs, cmap="magma",
                        norm=plt.Normalize(0, 6.0), lw=3.2)
    lc.set_array(k[:-1])
    ax.add_collection(lc)
    idx = np.linspace(0, len(P) - 1, 21).astype(int)     # 20 段 = 21 节点
    ax.plot(P[idx, 0], P[idx, 1], "o", mfc="none", mec="0.35", ms=5,
            label="间隔盘 20 个")
    ax.plot(P[:1, 0], P[:1, 1], "s", color="k", ms=7)
    ax.plot(P[-1, 0], P[-1, 1], "o", color=RED, ms=7)
    ax.set_aspect("equal")
    ax.set_xlim(-320, 320); ax.set_ylim(-60, 430)
    ax.grid(alpha=0.15, ls=":")
    ax.set_title("θ=90°：20 段节点", fontsize=9, color=RED)
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=7)
    sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, 6.0))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.01,
                 label="曲率 κ (1/m)")
    fig.suptitle("臂体中线曲线族（θ 每 15°，颜色=局部曲率）", fontsize=10)
    fig.savefig("fig_arm_curves.png", dpi=145, bbox_inches="tight")
    print("saved fig_arm_curves.png")
    # 读数
    for th in (30, 90, 150):
        o = curve(m, th, 0.0)
        P = o["p"][-1] * 1e3
        k = np.linalg.norm(o["kappa"][:, :2], axis=1)
        print(f"  θ={th}°  tip=({P[0]:6.1f},{P[2]:6.1f})  "
              f"κ 根/末 = {k[1]:.2f}/{k[-1]:.2f}")


if __name__ == "__main__":
    main()
