# -*- coding: utf-8 -*-
"""fig_curves_cloud.py — 臂体曲线云图（直接由表的 kS/kK 重建 1500 条曲线）

不调用求解器：从 `kS`（弧长）与 `kK`（曲率矢量）积分出中线
    R' = R·hat(κ),  p' = R·e_z
左：按弯角 θ 着色；右：按局部曲率 κ 着色。
每条线半透明、极细 ⇒ 叠成"曲线云"，能看出可达包络。
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
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"


def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rebuild(kS, kK):
    """由 (61,) 弧长与 (61,3) 曲率矢量积分出 (61,3) 中线（mm）。"""
    p = np.zeros(3); R = np.eye(3)
    P = [p.copy()]
    for i in range(len(kS) - 1):
        ds = kS[i + 1] - kS[i]
        kv = 0.5 * (kK[i] + kK[i + 1])
        p = p + (R @ np.array([0, 0, 1.0])) * ds
        R = R @ (np.eye(3) + hat(kv) * ds + 0.5 * hat(kv) @ hat(kv) * ds * ds)
        u, _, vt = np.linalg.svd(R); R = u @ vt
        P.append(p.copy())
    return np.array(P) * 1e3


def main():
    z = np.load("model/vc_table_v2_20.npz")
    sel = z["F"] == 0.0
    kS = z["kS"][sel]; kK = z["kK"][sel]
    th = z["theta"][sel]; kappa = z["kappa"][sel]
    n = len(kS)
    print(f"重建 {n} 条曲线（kS {kS.shape[1]} 节点）")

    P = np.array([rebuild(kS[i], kK[i]) for i in range(n)])
    print(f"  中线形状 {P.shape}  包络 "
          f"x[{P[...,0].min():.0f},{P[...,0].max():.0f}] "
          f"y[{P[...,1].min():.0f},{P[...,1].max():.0f}] "
          f"z[{P[...,2].min():.0f},{P[...,2].max():.0f}] mm")

    fig = plt.figure(figsize=(15.5, 7.2))
    NS = P.shape[1] - 1
    segs = np.stack([P[:, :-1], P[:, 1:]], axis=2)      # (N, NS, 2, 3)
    segs = segs.reshape(-1, 2, 3)                       # (N*NS, 2, 3)

    for k, (name, cmap, cval, unit) in enumerate([
            ("弯角 θ", "turbo", th, "θ (deg)"),
            ("局部曲率 κ", "magma", kappa.mean(1), "κ (1/m)")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        cnorm = plt.Normalize(np.percentile(cval, 2), np.percentile(cval, 98))
        rgba = plt.get_cmap(cmap)(cnorm(cval))
        cols = np.repeat(rgba, NS, axis=0)              # 每条曲线的色重复 NS 段
        lc = Line3DCollection(segs, colors=cols, linewidths=0.35, alpha=0.45)
        ax.add_collection3d(lc)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm); sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.08, label=unit)
        ax.plot([0], [0], [0], "s", color="k", ms=8)
        ax.set_xlabel("x (mm)", fontsize=7)
        ax.set_ylabel("y (mm)", fontsize=7)
        ax.set_zlabel("z (mm)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=16, azim=-60)
        ax.set_title(f"{name}（全部 {n} 条）", fontsize=9, color=RED)
    fig.suptitle(f"臂体曲线云图（全部 {n} 条中线，由 kS/kK 重建）", fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_curves_cloud.png", dpi=150, bbox_inches="tight")
    print("saved fig_curves_cloud.png")


if __name__ == "__main__":
    main()
