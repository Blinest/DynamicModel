# -*- coding: utf-8 -*-
"""check_fillet3d.py — 三维圆角的近距离核对（只渲染根部 3 行）

用途：全臂 400 mm 视图里 5 mm 圆角只有约 8 像素，且 180° 时段2 会盖住段1，
看不出圆角是必然的。本脚本把相机推到根部 3 行（约 34 mm）直看，
如果这里能看到圆角，说明三维几何与平面展开图一致，问题只在视角。

输出：fig_fillet3d_check.png（左：直臂 0°，右：20° 小弯）
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import rcParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TC_NO_JIT", "1")
import anim_bend3d as M   # 复用同一套几何，保证"同一几何"

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

N_ROW = 3


def render(ax, theta_total, title):
    ss, kf = M.kappa_profile(theta_total)
    sg, P, Rm = M.build_frames(kf, ss)
    Bm = np.pi * M.R_OUT * 1e3 / 3.0
    h = M.W_STRUT * 1e3 / 2.0
    L = Bm / np.cos(np.radians(M.ALPHA_A))
    a = L * np.sin(np.radians(M.ALPHA_A))
    quads, cols, caps = [], [], []
    # 行
    svals = [i * a * 1e-3 for i in range(N_ROW + 1)]
    for i in range(N_ROW):
        s0, s1 = svals[i], svals[i + 1]
        dth = Bm / (M.R_OUT * 1e3)
        for j in range(6):
            th0 = j * np.pi / 3
            for dj in (+1, -1):
                th1 = th0 + dj * dth
                q = M.strut(P, Rm, sg, th0, s0, th1, s1, n=10)
                quads += q
                cols += [RED if dj > 0 else "#6E3234"] * len(q)
    # 圆角（与 anim_bend3d 同公式）
    for i in range(N_ROW + 1):
        cy = svals[i]
        for j in range(6):
            th_c = j * np.pi / 3        # 菱形中心与节点同角度
            for (dc, ds), (e1, e2), kind in (
                    ((0.0, +a), ((-Bm, -a), (+Bm, -a)), "tip"),
                    ((0.0, -a), ((-Bm, +a), (+Bm, +a)), "tip"),
                    ((+Bm, 0.0), ((-Bm, -a), (-Bm, +a)), "side"),
                    ((-Bm, 0.0), ((+Bm, -a), (+Bm, +a)), "side")):
                rho = M.R_AX if kind == "tip" else M.R_SD_A
                if rho <= 0:
                    continue
                phi = float(np.arccos(np.clip(
                    np.dot(e1, e2) / np.linalg.norm(e1) / np.linalg.norm(e2), -1, 1)))
                u1 = np.array(e1, float) / np.linalg.norm(e1)
                u2 = np.array(e2, float) / np.linalg.norm(e2)
                bis = (u1 + u2) / np.linalg.norm(u1 + u2)
                V = bis * (h / np.sin(phi / 2))
                t = rho / np.tan(phi / 2)
                d = rho / np.sin(phi / 2)
                P1, P2 = V + u1 * t, V + u2 * t
                O = V + bis * d
                a1 = np.arctan2((P1 - O)[1], (P1 - O)[0])
                a2 = np.arctan2((P2 - O)[1], (P2 - O)[0])
                da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
                arc = [O + rho * np.array([np.cos(a1 + da * q), np.sin(a1 + da * q)])
                       for q in np.linspace(0, 1, 12)]
                poly = [V] + [P1] + arc + [P2]
                ro_r = M.R_OUT + 0.05e-3
                ri_r = M.R_IN - 0.01e-3
                ro = [M.surface(P, Rm, sg, cy + p[1] * 1e-3,
                                th_c + p[0] / (M.R_OUT * 1e3), ro_r) for p in poly]
                ri = [M.surface(P, Rm, sg, cy + p[1] * 1e-3,
                                th_c + p[0] / (M.R_OUT * 1e3), ri_r) for p in poly]
                caps.append(ro)
                caps.append(ri[::-1])
                for kk in range(len(ro)):
                    k2 = (kk + 1) % len(ro)
                    caps.append([ro[kk], ro[k2], ri[k2], ri[kk]])
    pc = Poly3DCollection(quads, edgecolor="none", linewidth=0, alpha=1.0)
    pc.set_facecolor(cols)
    ax.add_collection3d(pc)
    if caps:
        cpc = Poly3DCollection(caps, facecolor=RED, edgecolor="none",
                               alpha=1.0)   # 圆角也是筋, 同色
        ax.add_collection3d(cpc)
    ax.set_xlim(-0.012, 0.012); ax.set_ylim(-0.012, 0.012)
    ax.set_zlim(-0.002, N_ROW * a * 1e-3 + 0.002)
    ax.set_box_aspect((1, 1, (N_ROW * a) / 24.0 + 0.3))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=18, azim=-62)
    ax.set_title(title, fontsize=9, color=RED)


def main():
    fig = plt.figure(figsize=(11, 5.6))
    for k, th in enumerate((0.0, np.radians(20))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        render(ax, th, f"{'直臂' if k == 0 else '20° 小弯'}：根部 {N_ROW} 行，"
                       f"α={M.ALPHA_A}° R轴={M.R_AX}")
    fig.suptitle("三维圆角近距离核对（亮红=圆角板，深红=筋；同一几何）", fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_fillet3d_check.png", dpi=170, bbox_inches="tight")
    print("saved fig_fillet3d_check.png")


if __name__ == "__main__":
    main()
