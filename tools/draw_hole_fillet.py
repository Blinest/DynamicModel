# -*- coding: utf-8 -*-
"""draw_hole_fillet.py — 放大对比两段菱形孔的圆角

段1（α=34°）：只在**轴向尖角**加 R=2.0 的切线圆角 ⇒ 孔"两头圆、两侧尖"
段2（α=28°）：**四角全加** ⇒ 孔"圆润菱形"

画法：孔边界 = 菱形四角按 ρ 倒圆（切点 t=ρ/tan(φ/2)、圆心距 d=ρ/sin(φ/2)），
      真实孔边还要从筋的中心线内缩 h=w/2=0.75 mm。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

B = np.pi * 8.0 / 3.0        # 周向半节距 8.377 mm
W = 1.5
H_OFF = W / 2.0              # 孔边相对中心线的内缩
RHO = 2.0


def geom(alpha_deg):
    a = np.cos(np.radians(alpha_deg))
    L = B / a
    return L * np.sin(np.radians(alpha_deg)), B, L


def rounded_hole(alpha_deg, fillet_corners, rho=RHO, n=10):
    """返回孔边界点列（逆时针）。corner 顺序：右 / 上 / 左 / 下。"""
    a, b, L = geom(alpha_deg)
    a -= H_OFF / np.sin(np.radians(alpha_deg)) * 0   # 内缩在下面按角处理
    corners = np.array([[b, 0.0], [0.0, a], [-b, 0.0], [0.0, -a]])
    pts = []
    for i, c in enumerate(corners):
        prev = corners[(i - 1) % 4]
        nxt = corners[(i + 1) % 4]
        u1 = (prev - c) / np.linalg.norm(prev - c)
        u2 = (nxt - c) / np.linalg.norm(nxt - c)
        phi = float(np.arccos(np.clip(u1 @ u2, -1, 1)))
        # 孔角 = 两条内缩边线的交点
        bis = (u1 + u2) / np.linalg.norm(u1 + u2)
        V = c + bis * (H_OFF / np.sin(phi / 2))
        if fillet_corners[i]:
            t = rho / np.tan(phi / 2)
            d = rho / np.sin(phi / 2)
            P1, P2 = V + u1 * t, V + u2 * t
            O = V + bis * d
            a1 = np.arctan2(*(P1 - O)[::-1])
            a2 = np.arctan2(*(P2 - O)[::-1])
            da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
            arc = [O + rho * np.array([np.cos(a1 + da * q), np.sin(a1 + da * q)])
                   for q in np.linspace(0, 1, n)]
            pts += [P1] + arc + [P2]
        else:
            pts.append(V)
    return np.array(pts), a, b


def panel(ax, alpha, fillets, title):
    pts, a, b = rounded_hole(alpha, fillets)
    # 材料：外框 - 孔
    ax.add_patch(Polygon(pts, closed=True, facecolor=BLUE, alpha=0.55,
                         edgecolor="none", zorder=2))
    ax.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]),
            color=RED, lw=2.0, zorder=3)
    # 未倒圆的参考菱形（虚线）
    ref = np.array([[b, 0], [0, a], [-b, 0], [0, -a], [b, 0]])
    ax.plot(ref[:, 0], ref[:, 1], ls="--", color="0.55", lw=1.0, zorder=4)
    ax.plot(0, 0, "+", color="0.3", ms=8, zorder=5)
    ax.set_aspect("equal")
    ax.grid(alpha=0.15, ls=":")
    ax.set_xlim(-b * 1.15, b * 1.15)
    ax.set_ylim(-a * 1.35, a * 1.35)
    ax.set_title(f"{title}\nα={alpha}°  a={a:.2f}  b={b:.2f}  R={RHO}",
                 fontsize=9, color=RED)
    ax.set_xlabel("周向 (mm)", fontsize=7)
    ax.set_ylabel("轴向 (mm)", fontsize=7)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    # 角顺序: 右(侧) / 上(尖) / 左(侧) / 下(尖)
    panel(axes[0], 34.0, [False, True, False, True],
          "① 段1：只倒轴向尖角（上下圆、左右尖）")
    panel(axes[1], 28.0, [True, True, True, True],
          "② 段2：四角全倒（圆润菱形）")
    fig.suptitle(f"菱形孔圆角放大对比（切线圆弧 R={RHO} mm，孔边已从筋中心线内缩 {H_OFF} mm）",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_hole_fillet.png", dpi=170, bbox_inches="tight")
    print("saved fig_hole_fillet.png")
    for al, tag in ((34.0, "段1"), (28.0, "段2")):
        a, b, L = geom(al)
        for kind, phi in (("尖角", np.pi - 2 * np.arctan(a / b)),
                          ("侧角", 2 * np.arctan(a / b))):
            print(f"  {tag} {kind} φ={np.degrees(phi):5.1f}°  "
                  f"t={RHO/np.tan(phi/2):.2f}  d={RHO/np.sin(phi/2):.2f}")


if __name__ == "__main__":
    main()
