# -*- coding: utf-8 -*-
"""draw_arm_structure.py — 按最终几何重建**整根 400 mm 臂体结构**（平面展开）

两段变体（2026-11-08 定案）
    段1  s ∈ [0, 200]   α=34°   轴向尖角 R=5.0   侧向不倒（保持尖）
    段2  s ∈ [200, 400] α=20°   轴向尖角 R=5.0   侧向 R=1.0
共同: 周向 3 个菱形/圈（半节距 B=8.377）、筋宽 w=1.5、孔边从筋中心线内缩 h=0.75

输出：fig_arm_structure.png（横=周向 50.3 mm，纵=轴向 0–400 mm）
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


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

CIRC = 2 * np.pi * 8.0
B = CIRC / 6.0
W_STRUT = 1.5
H_OFF = W_STRUT / 2
L_SEG = 200.0
L_TOT = 400.0

SEG = [dict(alpha=34.0, rho_ax=5.0, rho_sd=0.0),
       dict(alpha=20.0, rho_ax=5.0, rho_sd=1.0)]


def seg_at(s):
    return SEG[0] if s <= L_SEG else SEG[1]


def ab_of(alpha):
    L = B / np.cos(np.radians(alpha))
    return L * np.sin(np.radians(alpha)), L


def band(p0, p1, w=W_STRUT):
    d = np.asarray(p1, float) - np.asarray(p0, float)
    L = np.linalg.norm(d)
    if L < 1e-9:
        return None
    n = np.array([-d[1], d[0]]) / L * (w / 2)
    return np.array([p0 + n, p1 + n, p1 - n, p0 - n])


def node_fillets(cx, cy, a, rho_ax, rho_sd, n=9):
    dirs = [np.array([+B, +a]), np.array([-B, +a]),
            np.array([-B, -a]), np.array([+B, -a])]
    order = np.argsort([np.arctan2(d[1], d[0]) for d in dirs])
    dirs = [dirs[k] / np.linalg.norm(dirs[k]) for k in order]
    Vs, wedges = [], []
    for k in range(4):
        u1, u2 = dirs[k], dirs[(k + 1) % 4]
        phi = float(np.arccos(np.clip(u1 @ u2, -1, 1)))
        bis = u1 + u2
        bis /= np.linalg.norm(bis)
        is_ax = abs(bis[1]) > abs(bis[0])
        rho = rho_ax if is_ax else rho_sd
        V = np.array([cx, cy]) + bis * (H_OFF / np.sin(phi / 2))
        Vs.append(V)
        if rho <= 0:
            wedges.append(None)
            continue
        t = rho / np.tan(phi / 2)
        d = rho / np.sin(phi / 2)
        P1, P2 = V + u1 * t, V + u2 * t
        O = V + bis * d
        a1 = np.arctan2((P1 - O)[1], (P1 - O)[0])
        a2 = np.arctan2((P2 - O)[1], (P2 - O)[0])
        da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
        arc = [O + rho * np.array([np.cos(a1 + da * q), np.sin(a1 + da * q)])
               for q in np.linspace(0, 1, n)]
        wedges.append(np.array([V] + [P1] + arc + [P2]))
    return np.array(Vs), wedges


def main():
    fig, ax = plt.subplots(figsize=(6.2, 13.0))
    ax.add_patch(Rectangle((0, 0), CIRC, L_TOT, facecolor="white",
                           edgecolor="0.85", lw=0.6, zorder=0))

    # ── 节点行：沿 s 累积，每行的 a 由所在段的 α 决定；跨段时重新对齐
    rows, s, seg_idx = [], 0.0, 0
    while s < L_TOT - 1e-6:
        S = seg_at(s + 1e-6)
        a, L = ab_of(S["alpha"])
        rows.append(dict(s=s, a=a, L=L, **S))
        s += a
    for i, r in enumerate(rows):
        if i == 0:
            continue
        r0 = rows[i - 1]
        if abs(r["a"] - r0["a"]) > 1e-9:      # 跨段：跳过一步保证对齐
            continue
    # 筋
    for i in range(len(rows) - 1):
        r0, r1 = rows[i], rows[i + 1]
        if abs(r0["a"] - r1["a"]) > 1e-9:
            continue                          # 段界不连（结构上本来就换了）
        a = r0["a"]
        for m in range(-1, int(CIRC / B) + 3):
            for j in range(3):
                cx = (2 * j + 1) * B
                x0 = cx + (m - 1) * B
                for dx, dn in ((-B, 0), (+B, 0)):
                    pass
        # 用节点网格画：节点 (m·B, n·a)，n 对应行号
        n = i
        for m in range(-2, int(CIRC / B) + 3):
            if (m - n) % 2:
                continue
            p0 = np.array([m * B, r0["s"]])
            for dm in (+1, -1):
                # 下一行的节点在 (m±1)·B 处（n+1 的奇偶相反，故相邻）
                p1 = np.array([(m + dm) * B, r1["s"]])
                bd = band(p0, p1)
                if bd is not None:
                    ax.add_patch(Polygon(bd, closed=True, facecolor=RED,
                                         edgecolor="none", zorder=2))
    # 节点圆角
    for i, r in enumerate(rows):
        n = i
        for m in range(-2, int(CIRC / B) + 3):
            if (m - n) % 2:
                continue
            cx, cy = m * B, r["s"]
            Vs, wedges = node_fillets(cx, cy, r["a"], r["rho_ax"], r["rho_sd"])
            ax.add_patch(Polygon(Vs, closed=True, facecolor=RED,
                                 edgecolor="none", zorder=3))
            for wd in wedges:
                if wd is not None:
                    ax.add_patch(Polygon(wd, closed=True, facecolor=RED,
                                         edgecolor="none", zorder=3))
    # 段界
    ax.axhline(L_SEG, color=GREEN, lw=1.6, ls="--", zorder=5)
    ax.text(CIRC * 0.5, L_SEG + 6, "段界 s = 200 mm", color=GREEN, fontsize=8,
            ha="center")
    ax.set_xlim(0, CIRC)
    ax.set_ylim(0, L_TOT)
    ax.set_aspect("equal")
    ax.set_xticks([0, CIRC / 3, 2 * CIRC / 3, CIRC])
    ax.set_xticklabels(["0", "16.8", "33.5", "50.3"], fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("周向展开 (mm)  3 个菱形/圈", fontsize=8)
    ax.set_ylabel("轴向 s (mm)", fontsize=8)
    ax.set_title("臂体结构（平面展开，全长 400 mm）\n"
                 "段1 α=34° 轴向R=5.0/侧向尖　|　段2 α=20° 轴向R=5.0/侧向R=1.0",
                 fontsize=9, color=RED)
    fig.tight_layout()
    fig.savefig("fig_arm_structure.png", dpi=170, bbox_inches="tight")
    print("saved fig_arm_structure.png")
    for tag, S in (("段1", SEG[0]), ("段2", SEG[1])):
        a, L = ab_of(S["alpha"])
        print(f"  {tag}: α={S['alpha']}° a={a:.2f} L={L:.2f} 节距={2*a:.2f} "
              f"行数={L_SEG/(2*a):.1f} R轴={S['rho_ax']} R侧={S['rho_sd']}")


if __name__ == "__main__":
    main()
