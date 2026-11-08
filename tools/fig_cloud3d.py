# -*- coding: utf-8 -*-
"""fig_cloud3d.py — 三维曲线云图（3000 条中线的点云）

把查表里所有中线的节点撒成点云：
  左：按弯角 θ 着色（看可达空间的层次）
  右：按局部曲率 κ 着色（看哪里弯得最狠）
曲率由 τ 反算合力矩 M，再按段 EI 换算：κ = |M|/EI(s)。
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"
PSI = np.radians(np.array([0, 60, 120, 180, 240, 300], float))
EI1, EI2 = 0.01499, 0.01266


def main():
    z = np.load("model_v2/table.npz")
    sel = z["F"] == 0.0
    nodes = z["nodes"][sel]          # (1500, 61, 3)
    th = z["theta"][sel]
    tau = z["tau"][sel]
    n, nn, _ = nodes.shape
    print(f"点云：{n} 条中线 × {nn} 节点 = {n*nn} 点")

    # 每点的合力矩 → κ（分段）
    Mx = -5e-3 * (tau * np.sin(PSI)).sum(1)
    My = 5e-3 * (tau * np.cos(PSI)).sum(1)
    Mg = np.hypot(Mx, My)
    k1 = Mg / EI1
    k2 = Mg / EI2
    # 节点 0..30（s<200mm）用 k1，其余 k2
    kappa = np.where(np.arange(nn)[None, :] <= nn // 2, k1[:, None], k2[:, None])

    P = nodes.reshape(-1, 3)
    C_th = np.repeat(th, nn)
    C_k = kappa.ravel()
    sub = np.arange(0, len(P), 2)          # 抽样减半
    P, C_th, C_k = P[sub], C_th[sub], C_k[sub]
    print(f"  抽样后 {len(P)} 点")

    fig = plt.figure(figsize=(15.0, 7.2))
    for k, (C, name, cmap) in enumerate([(C_th, "弯角 θ (deg)", "turbo"),
                                         (C_k, "曲率 κ (1/m)", "magma")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=C, cmap=cmap,
                        s=0.8, alpha=0.35, linewidths=0)
        fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.08, label=name)
        ax.set_xlabel("x (mm)", fontsize=7)
        ax.set_ylabel("y (mm)", fontsize=7)
        ax.set_zlabel("z (mm)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=16, azim=-60)
        ax.set_title(name, fontsize=9, color=RED)
    fig.suptitle("三维曲线云图（1500 条中线，F_tip=0）", fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_cloud3d.png", dpi=150, bbox_inches="tight")
    print("saved fig_cloud3d.png")
    print(f"  曲率范围 {C_k.min():.2f} … {C_k.max():.2f} 1/m")
    print(f"  包络 x [{P[:,0].min():.0f},{P[:,0].max():.0f}] "
          f"y [{P[:,1].min():.0f},{P[:,1].max():.0f}] "
          f"z [{P[:,2].min():.0f},{P[:,2].max():.0f}] mm")


if __name__ == "__main__":
    main()
