# -*- coding: utf-8 -*-
"""plot_table_contours.py — 3000 点查表的云图

输出
    fig_table_contour.png   4 张 (θ, φ) 云图：末端 x / 末端 z / 最大张力 / 最大收丝
    fig_table_cloud.png     三维点云：3000 个末端位置（按弯角着色）
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


import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"


def load():
    z = np.load("model_v2/table.npz")
    return {k: z[k] for k in z.files}


def contours(d):
    th, ph, F = d["theta"], d["phi"], d["F"]
    sel = F == 0.0                       # 只看无外力那一半（1500 点）
    th, ph = th[sel], ph[sel]
    tip = d["tip_mm"][sel]
    tmax = d["tau"][sel].max(1)
    dlmax = d["dL_mm"][sel].max(1)

    TH = np.unique(th); PH = np.unique(ph)
    def grid(v):
        return v.reshape(len(TH), len(PH))

    # 曲率：由 tau 反算合力矩 M，再 κ = |M|/EI（μ=0 时 M 沿程恒定，κ 分段恒定）
    psi = np.radians(np.array([0, 60, 120, 180, 240, 300], float))
    tau = d["tau"][sel]
    Mx = -5e-3 * (tau * np.sin(psi)).sum(1)
    My = 5e-3 * (tau * np.cos(psi)).sum(1)
    Mg = np.hypot(Mx, My)
    EI = np.array([0.01499, 0.01266])
    k1 = Mg / EI[0]
    k2 = Mg / EI[1]
    kmean = 0.5 * (k1 + k2)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6))
    ####################################################################
    # 曲率：由 tau 反算合力矩 M，再 κ = |M|/EI（μ=0 时 M 沿程恒定，κ 分段恒定）
    panels = [
        (grid(tip[:, 0]), "末端 x (mm)", "RdBu_r"),
        (grid(tip[:, 2]), "末端 z (mm)", "viridis"),
        (grid(tmax), "最大张力 (N)", "inferno"),
        (grid(dlmax), "最大收丝 (mm)", "cividis"),
        (grid(k1), "根部曲率 kappa_1 (1/m)", "magma"),
        (grid(kmean), "平均曲率 (kappa_1+kappa_2)/2", "plasma"),
    ]
    for ax, (G, name, cmap) in zip(axes.ravel(), panels):
        im = ax.pcolormesh(PH, TH, G, cmap=cmap, shading="auto")
        cs = ax.contour(PH, TH, G, levels=7, colors="w", linewidths=0.6,
                        alpha=0.8)
        ax.clabel(cs, fontsize=6, fmt="%.0f")
        ax.set_xlabel("方位 φ (deg)", fontsize=8)
        ax.set_ylabel("弯角 θ (deg)", fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("v2 查表云图（1500 点，F_tip=0）—— 含曲率", fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_table_contour.png", dpi=150, bbox_inches="tight")
    print("saved fig_table_contour.png")


def cloud(d):
    fig = plt.figure(figsize=(9.5, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    tip = d["tip_mm"]; th = d["theta"]
    sel = d["F"] == 0.0
    sc = ax.scatter(tip[sel, 0], tip[sel, 1], tip[sel, 2], c=th[sel],
                    cmap="turbo", s=5, alpha=0.85)
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.08,
                 label="θ (deg)")
    ax.set_xlabel("x (mm)", fontsize=8); ax.set_ylabel("y (mm)", fontsize=8)
    ax.set_zlabel("z (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=16, azim=-58)
    ax.set_title("末端位置点云（1500 点，F_tip=0）", fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_table_cloud.png", dpi=150, bbox_inches="tight")
    print("saved fig_table_cloud.png")


def main():
    d = load()
    print(f"载入 {len(d['theta'])} 点")
    contours(d)
    cloud(d)


if __name__ == "__main__":
    main()
