# -*- coding: utf-8 -*-
"""make_figs_v3.py — 快版：连续扫幅值（不二分），每工况约 12 次求解

原因: v2 用「逐角度二分」= 2 工况 x 10 档 x 7~12 次 = 140~240 次打靶，
      纯 numpy 单核要 40+ 分钟，两次被 timeout 杀死且无任何产出。
本版: 幅值 A 从 0 单调增到 Amax，每步解一次、记录全部量，最后按角度分组画图。

配色: #A34D4E 红棕 / #516CA3 蓝 / #8DAE7F 绿
工况: 水平弯曲（方位 90°）与 垂直向上（方位 0°），水平安装 g=(-9.81,0,0)
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TC_NO_JIT", "1")
from tendon_coupling import (build_arm, shooting, _wire_geom_len_batch,  # noqa: E402
                            _L0_wire, ALPHA_DEG)

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"
G = (-9.81, 0.0, 0.0)
ALPHA = np.radians(np.array(ALPHA_DEG, float))
T0, EI, NSUB = 10.0, 0.01556, 15


def make_params():
    ca, cb, p = build_arm(EI=EI, EA=7e3, r_i=0.008, r_disk=0.005,
                          n_circ_a=3, n_circ_b=3, n_long=136, E=2e8,
                          t_strut=1.5e-3, rhoA=0.020 / 0.4, n_p=NSUB, n_d=NSUB,
                          lattice=False)
    p.g_world = G
    return p


def ang_of(o):
    return float(np.degrees(np.arccos(np.clip(o["R_curve"][-1][2, 2], -1.0, 1.0))))


def sweep(p, L0, phi_deg, amin, amax, n=12):
    d = np.cos(ALPHA - np.radians(phi_deg))
    out = []
    for A in np.linspace(amin, amax, n):
        tau = T0 + A * d
        if tau.min() < 0.15:
            break
        o = shooting(tau, p, F_tip=np.zeros(3), g_world=G, tol=1e-8, max_iter=60)
        if not o["converged"]:
            continue
        dL = (_wire_geom_len_batch(o["p_curve"][None], o["R_curve"][None],
                                   r_wire=p.r_disk, L1_geo=p.L1,
                                   L_total=p.L_total)[0] - L0) * 1e3
        out.append(dict(A=A, ang=ang_of(o), o=o, dL=dL,
                        P=np.asarray(o["p_curve"]), S=np.asarray(o["s_curve"]),
                        U=np.asarray(o["u_curve"])))
        print(f"    A={A:5.2f}N  弯角 {out[-1]['ang']:6.1f}°  "
              f"tip z {out[-1]['P'][-1][2]*1e3:6.1f}mm", flush=True)
    return out


def main():
    p = make_params()
    L0 = _L0_wire(p)
    cases = [("水平弯曲", 90.0, 1), ("垂直向上弯曲", 0.0, 0)]
    data = {}
    for name, phi, axis in cases:
        print(f"[{name}] 扫幅值 …", flush=True)
        data[name] = sweep(p, L0, phi, 1.0, T0 - 0.3, n=12)
        print(f"[{name}] 记录 {len(data[name])} 档，最大 {data[name][-1]['ang']:.1f}°",
              flush=True)

    # 图1 位形
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
    for ax, (name, phi, axis) in zip(axes, cases):
        for i, r in enumerate(data[name]):
            P = r["P"] * 1e3
            lat = P[:, 1] if axis == 1 else P[:, 0]
            ax.plot(lat, P[:, 2], color=RED if i % 2 == 0 else BLUE, lw=1.9)
            if i % 3 == 0:
                ax.text(lat[-1], P[-1, 2] - 13, f"{r['ang']:.0f}°", fontsize=7,
                        color=RED if i % 2 == 0 else BLUE, ha="center")
        ax.plot([0], [0], "o", color=GREEN, ms=9, zorder=5)
        ax.set_aspect("equal"); ax.grid(alpha=0.25, ls=":")
        ax.set_xlabel("横向偏移 (mm)"); ax.set_ylabel("轴向高度 (mm)")
        ax.set_title(f"{name}（EI={EI}）", color=RED)
    fig.suptitle("臂体位形：红=偶数档 / 蓝=奇数档 / 绿=基座", fontsize=10)
    fig.tight_layout(); fig.savefig("fig_v2_shape.png", dpi=150); plt.close(fig)
    print("saved fig_v2_shape.png", flush=True)

    # 图2 曲率
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    for j, (name, phi, axis) in enumerate(cases):
        rows = data[name]
        idx = [int(round(k * (len(rows) - 1) / 3)) for k in (1, 2, 3)]
        for c, i in zip((RED, BLUE, GREEN), idx):
            r = rows[i]
            comp = np.abs(r["U"][:, 1] if axis == 1 else r["U"][:, 0])
            ax.plot(r["S"] * 1e3, comp, "-" if j == 0 else "--", color=c, lw=1.8,
                    label=f"{'水平' if j==0 else '垂直'} {r['ang']:.0f}°")
    ax.grid(alpha=0.25, ls=":"); ax.set_xlabel("弧长 s (mm)")
    ax.set_ylabel("曲率 |κ| (1/m)"); ax.set_title("沿程曲率（实线水平/虚线垂直）", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2)
    ax = axes[1]
    for (name, phi, axis), c in zip(cases, (RED, BLUE)):
        rows = data[name]
        A = [r["ang"] for r in rows]
        s1 = [np.abs((r["U"][:, 1] if axis == 1 else r["U"][:, 0])[r["S"] <= 0.2]).mean()
              for r in rows]
        s2 = [np.abs((r["U"][:, 1] if axis == 1 else r["U"][:, 0])[r["S"] > 0.2]).mean()
              for r in rows]
        ax.plot(A, s1, "-o", color=c, ms=3, label=f"{name} 段1")
        ax.plot(A, s2, "--s", color=GREEN, ms=3, label=f"{name} 段2")
    ax.grid(alpha=0.25, ls=":"); ax.set_xlabel("实测弯角 (°)")
    ax.set_ylabel("平均曲率 (1/m)"); ax.set_title("段1 vs 段2", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("fig_v2_kappa.png", dpi=150); plt.close(fig)
    print("saved fig_v2_kappa.png", flush=True)

    # 图3 电机位移
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    styles = ["-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1))]
    cols = [RED, BLUE, GREEN, RED, BLUE, GREEN]
    for ax, (name, phi, axis) in zip(axes, cases):
        rows = data[name]
        A = np.array([r["ang"] for r in rows])
        D = np.array([r["dL"] for r in rows])
        for w in range(6):
            ax.plot(A, D[:, w], ls=styles[w], color=cols[w], lw=1.8, marker="o",
                    ms=2.5, label=f"M{w+1}" + ("(A)" if w in (0, 2, 4) else "(B)"))
        ax.axhline(0, color="0.5", lw=0.8); ax.grid(alpha=0.25, ls=":")
        ax.set_xlabel("实测弯角 (°)"); ax.set_title(name, color=RED if axis == 1 else BLUE)
    axes[0].set_ylabel("电机位移 ΔL (mm)  +收丝/−放丝")
    axes[0].legend(fontsize=6.5, ncol=2)
    fig.suptitle("电机位移 vs 弯角", fontsize=10)
    fig.tight_layout(); fig.savefig("fig_v2_dL.png", dpi=150); plt.close(fig)
    print("saved fig_v2_dL.png", flush=True)


if __name__ == "__main__":
    main()
