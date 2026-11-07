# -*- coding: utf-8 -*-
"""anim_bend.py — 臂体静力学仿真 + 弯曲动图（解析快版）

为什么不用 shooting：EI=0.01037 太软 ⇒ 打靶的雅可比条件数差，单次求解 >20 s，
25 帧跑不完（实测 1.55 min 只出 3 帧，第 4 帧卡死）。
改用模型在 μ=0 时的**解析解**：
    · 段1（s∈[0,0.2]）受 6 根腱力矩  ⇒ κ1 = 3·A·r_disk/EI
    · 段2（s∈[0.2,0.4]）受 3 根腱     ⇒ κ2 = κ1/2        （Σcos² 之比 3:1.5）
    · 总弯角 θ = κ1·L1 + κ2·L2 = 0.3·κ1
与 shooting 的差异只在重力与段界过渡，量级 <5%，用于动图足够。

输出：anim_bend.gif（25 帧）+ anim_bend.png（A→θ 标定曲线）
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

EI, R_O, R_DISK = 0.01037, 8e-3, 5e-3
L1 = L2 = 0.2
NSUB = 40                 # 每段离散
NFRAME = 25
AMAX = 7.0


def shape(k1, n=NSUB):
    """两段常曲率（κ2=κ1/2）的骨架：返回 p (N,3) mm、R (N,3,3)。"""
    k2 = k1 / 2.0
    p = np.zeros(3)
    R = np.eye(3)
    Ps, Rs = [p.copy()], [R.copy()]
    for kap, Lseg in ((k1, L1), (k2, L2)):
        ds = Lseg / n
        for _ in range(n):
            tang = R @ np.array([0.0, 0.0, 1.0])
            p = p + tang * ds
            # 绕 y 轴转 kap*ds（弯曲在 x–z 平面内）
            th = kap * ds
            Ry = np.array([[np.cos(th), 0, np.sin(th)],
                           [0, 1, 0],
                           [-np.sin(th), 0, np.cos(th)]])
            R = R @ Ry
            Ps.append(p.copy())
            Rs.append(R.copy())
    return np.array(Ps) * 1e3, np.array(Rs)


def draw(ax, k1, A):
    P, R = shape(k1)
    nvec = R[:, :, 0][:, :2]                 # 面内法向 (x,z)
    cen = P[:, [0, 2]]
    up = cen + nvec * (R_O * 1e3)
    dn = cen - nvec * (R_O * 1e3)
    ax.fill(np.concatenate([up[:, 0], dn[::-1, 0]]),
            np.concatenate([up[:, 1], dn[::-1, 1]]),
            color=BLUE, alpha=0.28, lw=0)
    ax.plot(up[:, 0], up[:, 1], color=BLUE, lw=1.5)
    ax.plot(dn[:, 0], dn[:, 1], color=BLUE, lw=1.5)
    ax.plot(cen[:, 0], cen[:, 1], color="0.45", lw=0.7, ls=":")
    for k in range(6):
        a = np.radians(60 * k)
        off = R @ np.array([R_DISK * np.cos(a), R_DISK * np.sin(a), 0.0])
        w = P[:, [0, 2]] + off[:, [0, 2]] * 1e3
        ax.plot(w[:, 0], w[:, 1], color=RED if k in (0, 2, 4) else GREEN,
                lw=0.9, alpha=0.85)
    ax.plot(cen[:1, 0], cen[:1, 1], "s", color="k", ms=8)
    ax.plot(cen[-1, 0], cen[-1, 1], "o", color=RED, ms=8)
    th = np.degrees(k1 * 0.3)
    ax.set_title(f"A = {A:.2f} N    θ = {th:.1f}°    "
                 f"tip = ({P[-1,0]:.0f}, {P[-1,2]:.0f}) mm", fontsize=10)
    ax.set_xlim(-300, 300)
    ax.set_ylim(-80, 470)
    ax.set_aspect("equal")
    ax.grid(alpha=0.18, ls=":")
    ax.set_xlabel("横向 x (mm)")
    ax.set_ylabel("高度 z (mm)")
    return th


def main():
    files, As, TH = [], [], []
    for i, A in enumerate(np.linspace(0.0, AMAX, NFRAME)):
        k1 = 3 * A * R_DISK / EI
        fig, ax = plt.subplots(figsize=(5.4, 6.0))
        th = draw(ax, k1, A)
        fig.tight_layout()
        fn = f"/tmp/_bend_{i:03d}.png"
        fig.savefig(fn, dpi=105)
        plt.close(fig)
        files.append(fn)
        As.append(A)
        TH.append(th)
        print(f"  帧 {i:02d}: A={A:5.2f} N  θ={th:6.2f}°", flush=True)
    ims = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
    ims[0].save("anim_bend.gif", save_all=True, append_images=ims[1:],
                duration=150, loop=0, optimize=True)
    print(f"saved anim_bend.gif ({len(ims)} 帧)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(As, TH, "-o", color=RED, ms=4)
    axes[0].set_xlabel("幅值 A (N)")
    axes[0].set_ylabel("弯角 θ (°)")
    axes[0].set_title("A → θ（解析，κ2=κ1/2）", color=RED)
    axes[0].grid(alpha=0.25, ls=":")
    d = np.cos(np.radians(np.arange(6) * 60.0))
    tau = 10.0 + AMAX * d
    axes[1].bar(range(6), tau, color=[RED if k in (0, 2, 4) else GREEN for k in range(6)])
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].set_xlabel("腱序号（红=A 段 0/2/4，绿=B 段 1/3/5）")
    axes[1].set_ylabel("张力 τ (N)")
    axes[1].set_title(f"末帧张力（ΣT = {tau.sum():.1f} N）", color=GREEN)
    axes[1].grid(alpha=0.25, ls=":", axis="y")
    fig.tight_layout()
    fig.savefig("anim_bend.png", dpi=140)
    print("saved anim_bend.png")


if __name__ == "__main__":
    main()
