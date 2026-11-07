# -*- coding: utf-8 -*-
"""anim_bend3d.py — 立体动图：臂体弯曲 + 菱形胞变形（3D）

修复记录
--------
1. **参数顺序 bug**：原 `box()` 里调用 `frame_at(g_ss, g_kf, sv)`，而签名为
   `frame_at(kfun, ss, s_m)` ⇒ 曲率与弧长传反，所有筋落到同一位置，画面只剩一个环。
2. **性能**：原实现每根筋积分 8 次骨架 ⇒ 768 次积分。改为**每帧预计算一次**
   `(s_grid, P, R)`，之后按索引插值。

几何（公共节点菱形格）：周向 6 节点（每 60°）、轴向间距 a = pa/2 = 4.167 mm、
筋连对角节点 (i,j)→(i+1,j±1)，长 √(a²+b²) = 9.356 mm 恒定。
变形：ε = κ(s)·y ⇒ a' = a(1+ε)、b' = √(L²−a'²)；
曲率：κ(s) = C/(1+μCs)，根部最大。
"""
import os

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
R_OUT, R_IN = 8e-3, 6.5e-3      # m（内部一律米制，避免与 P 混单位）
W_STRUT = 1.5e-3
L1 = L2 = 0.2
L_TOT = L1 + L2
MU, NFRAME, AMAX = 0.15, 21, 7.0
N_ROW = 8
B_MM = np.pi * R_OUT * 1e3 / 3.0
A_MM = 4.167
L_STRUT = np.hypot(A_MM, B_MM)
N_INT = 240                      # 骨架积分点数


def kappa_profile(theta_total, mu=MU, L=L_TOT, n=400):
    ss = np.linspace(0.0, L, n)
    if abs(theta_total) < 1e-9:
        return ss, np.zeros(n)
    C = (np.exp(mu * theta_total) - 1.0) / (mu * L)
    return ss, C / (1.0 + mu * C * ss)


def build_frames(kfun, ss, n=N_INT):
    """一次积分得到全臂的 (s_grid, P, R)，后续按索引插值。"""
    sg = np.linspace(0.0, L_TOT, n + 1)
    kap = np.interp(sg, ss, kfun)
    P = np.zeros((n + 1, 3))
    Rm = np.zeros((n + 1, 3, 3))
    p = np.zeros(3)
    R = np.eye(3)
    P[0], Rm[0] = p, R
    for i in range(n):
        ds = sg[i + 1] - sg[i]
        p = p + (R @ np.array([0.0, 0.0, 1.0])) * ds
        th = kap[i] * ds
        Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0],
                       [-np.sin(th), 0, np.cos(th)]])
        R = R @ Ry
        P[i + 1], Rm[i + 1] = p, R
    return sg, P, Rm


def surface(P, Rm, sg, s_m, theta, r):
    i = int(np.clip(np.searchsorted(sg, s_m) - 1, 0, len(sg) - 2))
    t = np.clip((s_m - sg[i]) / max(sg[i + 1] - sg[i], 1e-12), 0.0, 1.0)
    p = (1 - t) * P[i] + t * P[i + 1]
    R = (1 - t) * Rm[i] + t * Rm[i + 1]
    return p + R @ np.array([r * np.cos(theta), r * np.sin(theta), 0.0])


def ab(eps):
    a = float(np.clip(A_MM * (1.0 + eps), 0.3, 0.985 * L_STRUT))
    b = float(np.sqrt(max(L_STRUT ** 2 - a ** 2, 0.0)))
    return a, b


def strut(P, Rm, sg, thA, sA, thB, sB, n=8):
    """一段实体筋：径向 R_IN..R_OUT、周向宽 W_STRUT，返回 4 面四边形。"""
    cor = []
    for i in range(n):
        t = i / (n - 1)
        thv = (1 - t) * thA + t * thB
        sv = (1 - t) * sA + t * sB
        p_out = surface(P, Rm, sg, sv, thv, R_OUT)
        p_in = surface(P, Rm, sg, sv, thv, R_IN)
        d = surface(P, Rm, sg, sv, thv + 1e-4, R_OUT) - p_out
        d /= max(np.linalg.norm(d), 1e-12)
        perp = np.cross(d, p_out - p_in)
        perp /= max(np.linalg.norm(perp), 1e-12)
        o = perp * (W_STRUT / 2)
        cor.append([p_out + o, p_out - o, p_in - o, p_in + o])
    cor = np.asarray(cor)
    quads = []
    for i in range(n - 1):
        for k in range(4):
            k2 = (k + 1) % 4
            quads.append([cor[i, k], cor[i + 1, k], cor[i + 1, k2], cor[i, k2]])
    return quads


def draw(ax, kfun, ss, A, theta_total):
    ax.clear()
    sg, P, Rm = build_frames(kfun, ss)

    # 节点行（沿 s 累积，按变形后节距）
    rows, s_cur = [], 0.0
    for i in range(N_ROW + 1):
        eps = -float(np.interp(s_cur, ss, kfun)) * R_OUT
        a, b = ab(eps)
        rows.append(dict(s=s_cur, a=a, b=b, eps=eps))
        s_cur += 2 * a * 1e-3

    quads, cols = [], []
    for i in range(N_ROW):
        r0, r1 = rows[i], rows[i + 1]
        dth = r0["b"] / (R_OUT * 1e3)    # 弧度：周向半节距（b 为 mm）
        for j in range(6):
            th0 = j * np.pi / 3          # 每 60° 一个节点
            for dj in (+1, -1):
                th1 = th0 + dj * dth
                q = strut(P, Rm, sg, th0, r0["s"], th1, r1["s"])
                quads += q
                cols += [RED if dj > 0 else GREEN] * len(q)
    if not quads:
        return
    pc = Poly3DCollection(quads, edgecolor="none", linewidth=0, alpha=0.97)
    pc.set_facecolor([tuple(np.array(matplotlib.colors.to_rgb(c))
                            * (1.0 if (k // 4) % 2 else 0.72))
                      for k, c in enumerate(cols)])
    ax.add_collection3d(pc)

    # 端面环 + 6 根丝
    th = np.linspace(0, 2 * np.pi, 60)
    for sv, alpha in ((rows[0]["s"], 0.6), (rows[-1]["s"], 0.6)):
        for rr in (R_OUT, R_IN):
            pts = np.array([surface(P, Rm, sg, sv, t, rr) for t in th])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=BLUE, lw=0.7, alpha=alpha)
    for k in range(6):
        a = k * np.pi / 3
        pA = surface(P, Rm, sg, rows[0]["s"], a, R_DISK)
        pB = surface(P, Rm, sg, rows[-1]["s"], a, R_DISK)
        ax.plot([pA[0], pB[0]], [pA[1], pB[1]], [pA[2], pB[2]],
                color=RED, lw=0.9, alpha=0.85)

    span = rows[-1]["s"]
    ax.set_xlim(-0.014, 0.014); ax.set_ylim(-0.014, 0.014)
    ax.set_zlim(-0.001, span + 0.001)
    ax.set_box_aspect((1, 1, max(span / 0.028, 0.45)))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=14, azim=-62)
    ax.set_title(f"A={A:.2f} N  θ={np.degrees(theta_total):.1f}°  "
                 f"ε_root={rows[0]['eps']*100:+.1f}%  窗口 {span*1e3:.0f} mm",
                 fontsize=9)


def main():
    files, TH = [], []
    for idx, A in enumerate(np.linspace(0.0, AMAX, NFRAME)):
        k1 = 3 * A * R_DISK / EI
        theta_total = 0.3 * k1
        ss, kfun = kappa_profile(theta_total)
        fig = plt.figure(figsize=(5.6, 6.2))
        ax = fig.add_subplot(111, projection="3d")
        draw(ax, kfun, ss, A, theta_total)
        fig.tight_layout()
        fn = f"/tmp/_b3d_{idx:03d}.png"
        fig.savefig(fn, dpi=95)
        plt.close(fig)
        files.append(fn)
        TH.append(np.degrees(theta_total))
        print(f"  帧 {idx:02d}: A={A:5.2f} θ={TH[-1]:6.1f}° "
              f"ε_root={-kfun[0]*R_OUT*100:+.1f}%", flush=True)
    ims = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
    ims[0].save("anim_bend3d.gif", save_all=True, append_images=ims[1:],
                duration=180, loop=0, optimize=True)
    print(f"saved anim_bend3d.gif ({len(ims)} 帧)")


if __name__ == "__main__":
    main()
