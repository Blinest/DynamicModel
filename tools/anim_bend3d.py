# -*- coding: utf-8 -*-
"""anim_bend3d.py — 立体动图：臂体弯曲 + 菱形胞变形（3D）

几何（公共节点菱形格，无错位）：
    周向 6 个节点（每 60° 一个），周向间距 b = πR/3 = 8.377 mm
    轴向节点间距 a = pa/2 = 4.167 mm
    筋连对角节点 (i,j)→(i+1,j±1)，长 √(a²+b²) = 9.356 mm 恒定
变形：ε = κ(s)·y，杆长不变 ⇒ a' = a(1+ε)、b' = √(L²−a'²)
曲率：κ(s) = C/(1+μCs)，C = (e^{μΘ}−1)/(μL)  ⇒ 根部最大

只渲染根部 8 行（≈67 mm，ε 最大处），以保证 25 帧在 1 分钟内跑完。
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import rcParams
from PIL import Image

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

EI, R_DISK = 0.01037, 5e-3
R_OUT, R_IN = 8.0, 6.5          # mm
W_STRUT = 1.5
L1 = L2 = 0.2
L_TOT = L1 + L2
MU, NFRAME, AMAX = 0.15, 21, 7.0
N_ROW = 8                        # 轴向行数（窗口）
B_MM = np.pi * R_OUT / 3.0       # 8.377
A_MM = 4.167
L_STRUT = np.hypot(A_MM, B_MM)


def kappa_profile(theta_total, mu=MU, L=L_TOT, n=400):
    ss = np.linspace(0.0, L, n)
    if abs(theta_total) < 1e-9:
        return ss, np.zeros(n)
    C = (np.exp(mu * theta_total) - 1.0) / (mu * L)
    return ss, C / (1.0 + mu * C * ss)


def frame_at(kfun, ss, s_m, n=400):
    """骨架到 s_m 处的位置与姿态（沿 x–z 平面弯曲）。"""
    sg = np.linspace(0.0, L_TOT, n + 1)
    kap = np.interp(sg, ss, kfun)
    p = np.zeros(3)
    R = np.eye(3)
    for i in range(n):
        ds = sg[i + 1] - sg[i]
        if sg[i] >= s_m:
            break
        p = p + (R @ np.array([0.0, 0.0, 1.0])) * ds
        th = kap[i] * ds
        Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0],
                       [-np.sin(th), 0, np.cos(th)]])
        R = R @ Ry
    return p, R


def ab(eps):
    a = float(np.clip(A_MM * (1.0 + eps), 0.3, 0.985 * L_STRUT))
    b = float(np.sqrt(max(L_STRUT ** 2 - a ** 2, 0.0)))
    return a, b


def box(pa, pb, w=W_STRUT, r_in=R_IN, r_out=R_OUT):
    """两点之间的实体筋（4 面）。pa/pb 为 (theta, s) 表面对坐标。"""
    quads = []
    n = 8
    th = np.linspace(pa[0], pb[0], n)
    ss = np.linspace(pa[1], pb[1], n)

    def P(thv, sv, r):
        p, R = frame_at(g_ss, g_kf, sv)
        o = R @ np.array([r * np.cos(thv), r * np.sin(thv), 0.0])
        return p + o

    cor = []
    for i in range(n):
        j0, j1 = max(i - 1, 0), min(i + 1, n - 1)
        t = P(th[j1], ss[j1], r_out) - P(th[j0], ss[j0], r_out)
        t /= max(np.linalg.norm(t), 1e-12)
        nrm = np.array([np.cos(th[i]), np.sin(th[i]), 0.0])
        perp = np.cross(nrm, t)
        perp /= max(np.linalg.norm(perp), 1e-12)
        base = P(th[i], ss[i], 0.0)
        cor.append([base + np.array([0, 0, 0]) + perp * (w / 2)
                    + R_out_dir(th[i]) * r_out,
                    base + perp * (w / 2) + R_out_dir(th[i]) * r_in,
                    base - perp * (w / 2) + R_out_dir(th[i]) * r_in,
                    base - perp * (w / 2) + R_out_dir(th[i]) * r_out])
    cor = np.asarray(cor)
    for i in range(n - 1):
        for k in range(4):
            k2 = (k + 1) % 4
            quads.append([cor[i, k], cor[i + 1, k], cor[i + 1, k2], cor[i, k2]])
    return quads


def R_out_dir(th):
    return np.array([np.cos(th), np.sin(th), 0.0])


g_ss = g_kf = None       # 全局（供 box 内取骨架）


def draw(ax, kfun, ss, A, theta_total):
    global g_ss, g_kf
    g_ss, g_kf = ss, kfun
    ax.clear()
    quads, cols = [], []
    s_cur = 0.0
    rows = []
    for i in range(N_ROW + 1):
        eps = -float(np.interp(s_cur, ss, kfun)) * (R_OUT * 1e-3)
        a, b = ab(eps)
        rows.append((s_cur, a, b, eps))
        s_cur += 2 * a * 1e-3
    for i in range(N_ROW):
        s0, a0, b0, e0 = rows[i]
        s1 = rows[i + 1][0]
        th_step = (b0 / R_OUT)          # 弧度：b/R
        for j in range(6):
            th0 = j * 2 * np.pi / 6
            for dj in (+1, -1):
                th1 = th0 + dj * th_step
                q = box((th0, s0), (th1, s1))
                quads += q
                cols += [RED if dj > 0 else GREEN] * len(q)
    pc = Poly3DCollection(quads, edgecolor="none", linewidth=0, alpha=0.97)
    pc.set_facecolor([tuple(np.array(matplotlib.colors.to_rgb(c))
                            * (1.0 if (k // 4) % 2 else 0.75))
                      for k, c in enumerate(cols)])
    ax.add_collection3d(pc)

    th = np.linspace(0, 2 * np.pi, 48)
    p0, R0 = frame_at(ss, kfun, 0.0)
    for rr, al in ((R_OUT, 0.55), (R_IN, 0.35)):
        pts = np.array([p0 + R0 @ np.array([rr * np.cos(t), rr * np.sin(t), 0])
                        for t in th])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=BLUE, lw=0.7, alpha=al)
    for k in range(6):
        a = np.radians(60 * k)
        p1, R1 = frame_at(ss, kfun, rows[-1][0])
        pA = p0 + R0 @ np.array([R_DISK * np.cos(a), R_DISK * np.sin(a), 0])
        pB = p1 + R1 @ np.array([R_DISK * np.cos(a), R_DISK * np.sin(a), 0])
        ax.plot([pA[0], pB[0]], [pA[1], pB[1]], [pA[2], pB[2]],
                color=RED, lw=0.9, alpha=0.85)

    span = rows[-1][0] * 1e3
    ax.set_xlim(-14, 14); ax.set_ylim(-14, 14); ax.set_zlim(-2, span + 2)
    ax.set_box_aspect((1, 1, max(span / 26, 0.5)))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=15, azim=-60)
    ax.set_title(f"A={A:.2f} N  θ={np.degrees(theta_total):.1f}°  "
                 f"根部 ε={rows[0][3]*100:+.1f}%  (窗口 {span:.0f} mm)",
                 fontsize=9)


def main():
    files, As, TH = [], [], []
    for idx, A in enumerate(np.linspace(0.0, AMAX, NFRAME)):
        k1 = 3 * A * R_DISK / EI
        theta_total = 0.3 * k1
        ss, kfun = kappa_profile(theta_total)
        fig = plt.figure(figsize=(5.6, 6.0))
        ax = fig.add_subplot(111, projection="3d")
        draw(ax, kfun, ss, A, theta_total)
        fig.tight_layout()
        fn = f"/tmp/_b3d_{idx:03d}.png"
        fig.savefig(fn, dpi=95)
        plt.close(fig)
        files.append(fn)
        As.append(A)
        TH.append(np.degrees(theta_total))
        print(f"  帧 {idx:02d}: A={A:5.2f} θ={TH[-1]:6.1f}° "
              f"ε_root={-kfun[0]*R_OUT*1e-3*100:+.1f}%", flush=True)
    ims = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
    ims[0].save("anim_bend3d.gif", save_all=True, append_images=ims[1:],
                duration=180, loop=0, optimize=True)
    print(f"saved anim_bend3d.gif ({len(ims)} 帧)")


if __name__ == "__main__":
    main()
