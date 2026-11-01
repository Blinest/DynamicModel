# -*- coding: utf-8 -*-
"""make_figs_v2.py — 用固定三色生成：位形图 / 分段曲率图 / 电机位移图

配色（基底）：  #A34D4E 红棕   #516CA3 蓝   #8DAE7F 绿
工况：  水平弯曲 0→90→0    垂直(向上)弯曲 0→90→0
参数：  EI=0.01556（2026-10-31 预校准后重标）、EA=7e3、r_disk=5mm、L=0.4m、水平安装 g=(-9.81,0,0)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TC_NO_JIT", "1")
from tendon_coupling import (build_arm, shooting, _wire_geom_len_batch,  # noqa: E402
                            _L0_wire, ALPHA_DEG)

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"
G = (-9.81, 0.0, 0.0)            # 水平安装：重力沿 -x，臂沿 +z
ALPHA = np.radians(np.array(ALPHA_DEG, float))
T0 = 10.0
EI = 0.01556


def make_params():
    ca, cb, p = build_arm(EI=EI, EA=7e3, r_i=0.008, r_disk=0.005,
                          n_circ_a=3, n_circ_b=3, n_long=136, E=2e8,
                          t_strut=1.5e-3, rhoA=0.020 / 0.4, n_p=20, n_d=20,
                          lattice=False)
    p.g_world = G
    return p


def bend_angle(o):
    return float(np.degrees(np.arccos(np.clip(o["R_curve"][-1][2, 2], -1.0, 1.0))))


def solve_at(p, phi_deg, target_deg):
    """二分幅值 A 使总弯角 = target_deg（τ_i = T0 + A·cos(α_i − φ)）。"""
    d = np.cos(ALPHA - np.radians(phi_deg))
    lo, hi, best = 0.0, T0 - 0.2, None
    for _ in range(7):
        A = 0.5 * (lo + hi)
        tau = T0 + A * d
        o = shooting(tau, p, F_tip=np.zeros(3), g_world=G, tol=1e-9, max_iter=100)
        if not o["converged"]:
            hi = A
            continue
        ang = bend_angle(o)
        if ang < target_deg:
            lo = A
        else:
            hi = A
        best = (tau, o, ang)
    return best


def dL_mm(o, p, L0):
    return (_wire_geom_len_batch(o["p_curve"][None], o["R_curve"][None],
                                 r_wire=p.r_disk, L1_geo=p.L1,
                                 L_total=p.L_total)[0] - L0) * 1e3


def main():
    p = make_params()
    L0 = _L0_wire(p)
    angles = list(range(0, 91, 10))
    cases = [("水平弯曲", 90.0, 1), ("垂直向上弯曲", 0.0, 0)]   # (名称, 方位, 投影轴)

    data = {}
    for name, phi, axis in cases:
        rows = []
        for t in angles:
            if t == 0:
                o = shooting(np.full(6, T0), p, F_tip=np.zeros(3), g_world=G,
                             tol=1e-9, max_iter=100)
                rows.append(dict(t=t, o=o, ang=0.0, tau=np.full(6, T0),
                                 dL=dL_mm(o, p, L0)))
                continue
            r = solve_at(p, phi, t)
            if r is None:
                continue
            tau, o, ang = r
            rows.append(dict(t=t, o=o, ang=ang, tau=tau, dL=dL_mm(o, p, L0)))
        data[name] = rows
        print(f"[{name}] 解出 {len(rows)}/{len(angles)} 档，末档 {rows[-1]['ang']:.1f}°", flush=True)

    # ── 图1 位形图 ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (name, phi, axis) in zip(axes, cases):
        for i, r in enumerate(data[name]):
            P = np.asarray(r["o"]["p_curve"]) * 1e3
            lat = P[:, 1] if axis == 1 else P[:, 0]
            c = RED if i % 2 == 0 else BLUE
            ax.plot(lat, P[:, 2], color=c, lw=2.0, solid_capstyle="round")
            if i % 2 == 0:
                ax.text(lat[-1], P[:, 2][-1] - 14, f"{r['ang']:.0f}°",
                        color=c, fontsize=8, ha="center")
        ax.plot([0], [0], "o", color=GREEN, ms=9, zorder=5)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25, ls=":")
        ax.set_xlabel("横向偏移 (mm)")
        ax.set_ylabel("轴向高度 (mm)")
        ax.set_title(f"{name}  0→90°（EI={EI}）", color=RED)
    fig.suptitle("臂体位形（红/蓝交替为 10° 步进）", fontsize=11)
    fig.tight_layout()
    fig.savefig("fig_v2_shape.png", dpi=160)
    plt.close(fig)

    # ── 图2 分段曲率 ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    for j, (name, phi, axis) in enumerate(cases):
        for k, t in enumerate((30, 60, 90)):
            row = [r for r in data[name] if abs(r["t"] - t) < 1e-6]
            if not row:
                continue
            o = row[0]["o"]
            s = np.asarray(o["s_curve"]) * 1e3
            u = np.asarray(o["u_curve"])
            comp = u[:, 1] if axis == 1 else u[:, 0]
            ls = "-" if j == 0 else "--"
            ax.plot(s, np.abs(comp), ls, color=[RED, BLUE, GREEN][k], lw=1.8,
                    label=f"{'水平' if j == 0 else '垂直'} {t}°")
    ax.grid(alpha=0.25, ls=":")
    ax.set_xlabel("弧长 s (mm)")
    ax.set_ylabel("曲率 |κ| (1/m)")
    ax.set_title("沿程曲率（红30°/蓝60°/绿90°，实线水平 虚线垂直）", fontsize=9)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    w = 0.35
    xpos = np.arange(len(angles))
    for j, (name, phi, axis) in enumerate(cases):
        s1, s2 = [], []
        for r in data[name]:
            s = np.asarray(r["o"]["s_curve"])
            u = np.asarray(r["o"]["u_curve"])
            comp = np.abs(u[:, 1] if axis == 1 else u[:, 0])
            m1 = s <= 0.2
            s1.append(comp[m1].mean() if m1.any() else np.nan)
            s2.append(comp[~m1].mean() if (~m1).any() else np.nan)
        ax.bar(xpos + (j - 0.5) * w, s1, w, color=RED if j == 0 else BLUE,
               alpha=0.9, label=f"{name} 段1")
        ax.bar(xpos + (j - 0.5) * w, s2, w, bottom=s1, color=GREEN,
               alpha=0.85, label=f"{name} 段2")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{a}" for a in angles], fontsize=8)
    ax.set_xlabel("目标弯角 (°)")
    ax.set_ylabel("平均曲率 (1/m)")
    ax.set_title("段1(红/蓝) / 段2(绿) 平均曲率", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, ls=":", axis="y")
    fig.tight_layout()
    fig.savefig("fig_v2_kappa.png", dpi=160)
    plt.close(fig)

    # ── 图3 电机位移 ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    styles = ["-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1))]
    cols = [RED, BLUE, GREEN, RED, BLUE, GREEN]
    for ax, (name, phi, axis) in zip(axes, cases):
        D = np.array([r["dL"] for r in data[name]])
        T = np.array([r["t"] for r in data[name]])
        for w in range(6):
            lab = f"M{w+1}" + ("(段A)" if w in (0, 2, 4) else "(段B)")
            ax.plot(T, D[:, w], ls=styles[w], color=cols[w], lw=1.9,
                    label=lab, marker="o", ms=2.5)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.grid(alpha=0.25, ls=":")
        ax.set_xlabel("目标弯角 (°)")
        ax.set_title(name, color=RED if axis == 1 else BLUE)
    axes[0].set_ylabel("电机位移 ΔL (mm)  ＋收丝 / －放丝")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("电机位移（0→90°）", fontsize=11)
    fig.tight_layout()
    fig.savefig("fig_v2_dL.png", dpi=160)
    plt.close(fig)
    print("saved fig_v2_shape.png / fig_v2_kappa.png / fig_v2_dL.png")


if __name__ == "__main__":
    main()
