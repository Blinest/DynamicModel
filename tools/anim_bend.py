# -*- coding: utf-8 -*-
"""anim_bend.py — 臂体静力学仿真 + 弯曲动图（摩擦耦合、根部集中）

曲率分布（闭式，来自 Capstan 摩擦）：
    张紧端张力 τ(s) = τ0·exp(−μ·Θ(s)),  Θ(s) = ∫0^s κ
    弯矩 ∝ τ  ⇒  κ(s) = C·exp(−μΘ(s))
    由 Θ' = C·e^{−μΘ} 积分得  Θ(s) = ln(1+μCs)/μ  ⇒  κ(s) = C/(1+μCs)
    给定总弯角 Θ:  C = (e^{μΘ}−1)/(μL)
⇒ **根部曲率最大、末端最小**，比值 e^{μΘ}；压缩量同理，越靠根部越大。

单胞变形：杆长 L_strut 不变 ⇒ a' = a(1+ε), b' = √(L²−a'²)，ε = κ·y。
节点 N(i,j)=(j·b, i·a)，筋连对角节点 ⇒ 公共节点。

输出：anim_bend.gif（25 帧）+ anim_bend.png（A→θ 与 κ(s)）
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

# ── 参数 ──────────────────────────────────────────────────────────
EI, R_O, R_DISK = 0.01037, 8e-3, 5e-3
L1 = L2 = 0.2
L_TOT = L1 + L2
NSUB = 60
NFRAME = 25
AMAX = 7.0
MU = 0.15

PA_MM = L_TOT / 48.0 * 1e3             # 8.333 mm
PITCH_C_MM = 2 * np.pi * (R_O * 1e3) / 3.0     # 16.755 mm
A0 = PA_MM / 2.0                       # 4.167
B0 = PITCH_C_MM / 2.0                  # 8.377
L_STRUT = np.hypot(A0, B0)             # 9.356 mm 恒定
R_FIL = 0.9


# ── 曲率分布 ──────────────────────────────────────────────────────
def kappa_profile(theta_total, mu=MU, L=L_TOT, n=400):
    """摩擦耦合下的曲率分布：κ(s) = C/(1+μCs)，C=(e^{μΘ}−1)/(μL)。"""
    ss = np.linspace(0.0, L, n)
    if abs(theta_total) < 1e-9:
        return ss, np.zeros(n)
    C = (np.exp(mu * theta_total) - 1.0) / (mu * L)
    return ss, C / (1.0 + mu * C * ss)


def kappa_at(s_m, kfun, ss):
    return float(np.interp(s_m, ss, kfun))


# ── 单胞 ──────────────────────────────────────────────────────────
def ab(eps, exag=1.0):
    a = A0 * (1.0 + eps * exag)
    a = float(np.clip(a, 0.08 * A0, 0.985 * L_STRUT))
    b = float(np.sqrt(max(L_STRUT ** 2 - a ** 2, 0.0)))
    return a, b


def lattice_patch(ax, x0, y0, eps, n_i=3, n_j=3, exag=4.0):
    a, b = ab(eps, exag)
    for i in range(n_i - 2):
        for j in range(1, n_j - 1):
            v = [(x0 + j * b, y0 + i * a), (x0 + (j + 1) * b, y0 + (i + 1) * a),
                 (x0 + j * b, y0 + (i + 2) * a), (x0 + (j - 1) * b, y0 + (i + 1) * a)]
            ax.add_patch(plt.Polygon(v, closed=True, facecolor=BLUE, alpha=0.40,
                                     edgecolor="none", zorder=1))
    for i in range(n_i - 1):
        for j in range(n_j):
            for dj in (+1, -1):
                j2 = j + dj
                if 0 <= j2 < n_j:
                    ax.plot([x0 + j * b, x0 + j2 * b], [y0 + i * a, y0 + (i + 1) * a],
                            color=RED, lw=1.2, solid_capstyle="round", zorder=2)
    for i in range(n_i):
        for j in range(n_j):
            ax.add_patch(plt.Circle((x0 + j * b, y0 + i * a), R_FIL,
                                    facecolor=GREEN, alpha=0.95,
                                    edgecolor="white", lw=0.3, zorder=3))
    return a


# ── 骨架 ──────────────────────────────────────────────────────────
def shape(kfun, ss, n=NSUB):
    sgrid = np.linspace(0.0, ss[-1], n + 1)
    kap = np.interp(sgrid, ss, kfun)
    p = np.zeros(3)
    R = np.eye(3)
    Ps, Rs = [p.copy()], [R.copy()]
    for i in range(n):
        ds = sgrid[i + 1] - sgrid[i]
        p = p + (R @ np.array([0.0, 0.0, 1.0])) * ds
        th = kap[i] * ds
        Ry = np.array([[np.cos(th), 0, np.sin(th)],
                       [0, 1, 0],
                       [-np.sin(th), 0, np.cos(th)]])
        R = R @ Ry
        Ps.append(p.copy())
        Rs.append(R.copy())
    return np.array(Ps) * 1e3, np.array(Rs)


def draw_tube(ax, kfun, ss, A, theta_total):
    ax.clear()
    P, R = shape(kfun, ss)
    nvec = R[:, :, 0][:, :2]
    cen = P[:, [0, 2]]
    up = cen + nvec * (R_O * 1e3)
    dn = cen - nvec * (R_O * 1e3)
    ax.fill(np.concatenate([up[:, 0], dn[::-1, 0]]),
            np.concatenate([up[:, 1], dn[::-1, 1]]),
            color=BLUE, alpha=0.26, lw=0)
    ax.plot(up[:, 0], up[:, 1], color=BLUE, lw=1.4)
    ax.plot(dn[:, 0], dn[:, 1], color=BLUE, lw=1.4)
    ax.plot(cen[:, 0], cen[:, 1], color="0.5", lw=0.6, ls=":")
    for k in range(6):
        a = np.radians(60 * k)
        off = R @ np.array([R_DISK * np.cos(a), R_DISK * np.sin(a), 0.0])
        w = P[:, [0, 2]] + off[:, [0, 2]] * 1e3
        ax.plot(w[:, 0], w[:, 1], color=RED if k in (0, 2, 4) else GREEN,
                lw=0.8, alpha=0.8)
    ax.plot(cen[:1, 0], cen[:1, 1], "s", color="k", ms=8)
    ax.plot(cen[-1, 0], cen[-1, 1], "o", color=RED, ms=8)
    ax.set_title(f"A = {A:.2f} N    θ = {np.degrees(theta_total):.1f}°    "
                 f"κ: 根 {kfun[0]:.2f} → 端 {kfun[-1]:.2f} 1/m", fontsize=10)
    ax.set_xlim(-300, 300)
    ax.set_ylim(-80, 470)
    ax.set_aspect("equal")
    ax.grid(alpha=0.18, ls=":")
    ax.set_xlabel("横向 x (mm)")
    ax.set_ylabel("高度 z (mm)")


def main():
    files, As, TH = [], [], []
    stations = [0.03, 0.13, 0.24, 0.35]
    for idx, A in enumerate(np.linspace(0.0, AMAX, NFRAME)):
        k1 = 3 * A * R_DISK / EI
        theta_total = 0.3 * k1
        ss, kfun = kappa_profile(theta_total)
        fig, (ax, axc) = plt.subplots(
            1, 2, figsize=(13.5, 5.4), gridspec_kw=dict(width_ratios=[1.45, 1]))
        draw_tube(ax, kfun, ss, A, theta_total)
        axc.clear()
        for m, st in enumerate(stations):
            eps = -kappa_at(st, kfun, ss) * R_O
            x0 = m * 2.45 * B0
            lattice_patch(axc, x0, 0.0, eps)
            axc.text(x0, -3.5, f"s={st*1e3:.0f} mm\nε={eps*100:+.1f}%",
                     fontsize=7.5, ha="center", va="top",
                     color=RED if eps < -0.05 else GREEN)
        axc.set_xlim(-4, 4 * 2.45 * B0)
        axc.set_ylim(-11, 4.2 * A0)
        axc.set_aspect("equal")
        axc.axis("off")
        axc.set_title("内侧壁单胞：压缩量从根部向末端递减（变形 ×4 放大）",
                      fontsize=9, color=GREEN)
        fig.tight_layout()
        fn = f"/tmp/_bend_{idx:03d}.png"
        fig.savefig(fn, dpi=100)
        plt.close(fig)
        files.append(fn)
        As.append(A)
        TH.append(np.degrees(theta_total))
        if idx % 6 == 0 or idx == NFRAME - 1:
            print(f"  帧 {idx:02d}: A={A:5.2f}N θ={TH[-1]:6.2f}° "
                  f"κ_root={kfun[0]:5.2f} κ_tip={kfun[-1]:5.2f}", flush=True)

    ims = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
    ims[0].save("anim_bend.gif", save_all=True, append_images=ims[1:],
                duration=150, loop=0, optimize=True)
    print(f"saved anim_bend.gif ({len(ims)} 帧)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(As, TH, "-o", color=RED, ms=4)
    axes[0].set_xlabel("幅值 A (N)")
    axes[0].set_ylabel("总弯角 θ (°)")
    axes[0].set_title(f"A → θ（μ={MU}）", color=RED)
    axes[0].grid(alpha=0.25, ls=":")
    ss, kfun = kappa_profile(np.radians(TH[-1]))
    axes[1].plot(ss * 1e3, kfun, color=BLUE, lw=2)
    axes[1].set_xlabel("弧长 s (mm)")
    axes[1].set_ylabel("曲率 κ (1/m)")
    axes[1].set_title(f"末帧 κ(s)：{kfun[0]:.2f} → {kfun[-1]:.2f} "
                      f"(比值 {kfun[0]/kfun[-1]:.2f})", color=BLUE)
    axes[1].grid(alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig("anim_bend.png", dpi=140)
    print("saved anim_bend.png")


if __name__ == "__main__":
    main()
