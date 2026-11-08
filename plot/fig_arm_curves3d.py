# -*- coding: utf-8 -*-
"""fig_arm_curves3d.py — 臂体中线的三维曲线族（按曲率着色）

每根曲线 = 一个 (θ, φ) 的姿态；颜色 = 该处曲率。
左：多方位扇形族（φ = 0/60/…/300°，θ = 60/120/180°）
右：同一 φ 下的角度族（θ 每 20°），并标出 20 个间隔盘
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


import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d.art3d import Line3DCollection
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tdcr_model_v2 as V2

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"
KMAX = 8.0


def curve(m, th, ph):
    psi = np.radians(V2.GEOM["psi_deg"]); Ph = np.radians(ph)
    half = m.N // 2
    comp = (half * m.ds) / m.EI[0] + (m.N - half) * m.ds / m.EI[-1]
    A = np.radians(th) / (3.0 * m.g["r_disk"] * comp)
    tau = np.maximum(0.0, m.T0 + A * np.cos(psi - Ph))
    o = m.solve(tau)
    return o["p"] * 1e3, np.linalg.norm(o["kappa"][:, :2], axis=1)


def add3d(ax, P, k, lw=2.2):
    segs = np.stack([P[:-1], P[1:]], axis=1)
    lc = Line3DCollection(segs, cmap="magma", norm=plt.Normalize(0, KMAX),
                          linewidths=lw)
    lc.set_array(k[:-1])
    ax.add_collection3d(lc)
    return lc


def main():
    m = V2.TdcrModelV2(n_sub=4)
    fig = plt.figure(figsize=(15.5, 7.0))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for ph in np.arange(0, 360, 60):
        for th in (60, 120, 180):
            P, k = curve(m, th, ph)
            add3d(ax, P, k, lw=1.8)
    ax.plot([0], [0], [0], "s", color="k", ms=8)
    ax.set_xlabel("x (mm)", fontsize=7); ax.set_ylabel("y (mm)", fontsize=7)
    ax.set_zlabel("z (mm)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=16, azim=-60)
    ax.set_title("六方位 × 三角度（θ=60/120/180°）", fontsize=9, color=RED)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for th in np.arange(0, 181, 20):
        P, k = curve(m, th, 0.0)
        add3d(ax2, P, k, lw=2.2)
    P, k = curve(m, 90.0, 0.0)
    idx = np.linspace(0, len(P) - 1, 21).astype(int)
    ax2.plot(P[idx, 0], P[idx, 1], P[idx, 2], "o", mfc="none", mec="0.3",
             ms=4, label="间隔盘 ×20")
    ax2.plot([0], [0], [0], "s", color="k", ms=8)
    ax2.set_xlabel("x (mm)", fontsize=7); ax2.set_ylabel("y (mm)", fontsize=7)
    ax2.set_zlabel("z (mm)", fontsize=7)
    ax2.tick_params(labelsize=6)
    ax2.view_init(elev=16, azim=-60)
    ax2.set_title("同一方位 φ=0°：θ 每 20°", fontsize=9, color=RED)
    ax2.legend(fontsize=7, loc="upper left")

    sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, KMAX))
    sm.set_array([])
    fig.colorbar(sm, ax=[ax, ax2], fraction=0.015, pad=0.02,
                 label="曲率 κ (1/m)")
    fig.suptitle("臂体中线三维曲线族（颜色=局部曲率）", fontsize=10)
    fig.savefig("fig_arm_curves3d.png", dpi=145, bbox_inches="tight")
    print("saved fig_arm_curves3d.png")
    for th in (60, 120, 180):
        P, k = curve(m, th, 0.0)
        print(f"  θ={th:3d}°  tip=({P[-1,0]:6.1f},{P[-1,1]:6.1f},{P[-1,2]:6.1f})"
              f"  κ {k[1]:.2f}→{k[-1]:.2f}")


if __name__ == "__main__":
    main()
