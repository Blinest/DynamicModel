# -*- coding: utf-8 -*-
"""plot_cloud.py — 位姿表的"云图"：工作空间点云 + 曲率/弯角着色

    python tools/plot_cloud.py            # 出 20N / 40N 两张
    python tools/plot_cloud.py --tau 40

每张图三格：
  ① x–z 平面点云，颜色 = 平均 |κ|
  ② 同一点云，颜色 = 末端弯角
  ③ |κ| 与末端半径 r 的散点（看"卷"与"伸"的分布）
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TC_NO_JIT", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.tri import Triangulation            # noqa: E402,F401
from ik_table import load_table                     # noqa: E402


def cloud(tau_max, out):
    f = f"model/vc_table_{tau_max:.0f}.npz"
    if not os.path.exists(f):
        f = f"vc_table_{tau_max:.0f}.npz"
    T = load_table(f)
    P = np.array([r["p_tip"] for r in T.recs]) * 1e3          # mm
    R = np.array([r["R_tip"] for r in T.recs])
    K = np.array([np.linalg.norm(r["kcoef"]["K"], axis=1).mean() for r in T.recs])
    ang = np.degrees(np.arccos(np.clip(R[:, 2, 2], -1, 1)))
    r = np.linalg.norm(P[:, :2], axis=1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    s = 26
    for a, c, lab, cm in ((ax[0], K, "mean |κ|  [1/m]", "viridis"),
                          (ax[1], ang, "tip bend angle  [deg]", "plasma")):
        sc = a.scatter(P[:, 0], P[:, 2], c=c, s=s, cmap=cm, edgecolors="none")
        fig.colorbar(sc, ax=a, label=lab, shrink=0.9)
        a.set_xlabel("x  [mm]"); a.set_ylabel("z  [mm]")
        a.set_aspect("equal"); a.grid(alpha=.3)
        a.set_title(f"τmax = {tau_max:.0f} N   (N={len(P)})")
    ax[2].scatter(r, K, c=ang, s=s, cmap="plasma", edgecolors="none")
    ax[2].set_xlabel("tip radius r = |(x,y)|  [mm]")
    ax[2].set_ylabel("mean |κ|  [1/m]")
    ax[2].grid(alpha=.3)
    ax[2].set_title("curvature vs radius (color = bend angle)")
    fig.suptitle(f"TDCR workspace cloud — table τmax = {tau_max:.0f} N "
                 f"| EI={0.1294:.4g} N·m²  k_rod=2907 N", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}   N={len(P)}  |κ| {K.min():.2f}-{K.max():.2f} 1/m  "
          f"弯角 {ang.min():.1f}-{ang.max():.1f}°  r {r.min():.0f}-{r.max():.0f} mm")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, nargs="+", default=[20, 40])
    a = ap.parse_args()
    for t in a.tau:
        cloud(t, f"fig_cloud_{t:.0f}N.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
