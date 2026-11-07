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
MU, NFRAME = 0.15, 15
ROW_STEP = 1          # 每隔几行画一次筋（1=全画）
B_MM = np.pi * R_OUT * 1e3 / 3.0        # 8.377 周向半节距（由 N_circ=3 锁定）
# ── 两种变体（2026-11-08 用户描述）──────────────────────────────
#   段1（靠基座, s∈[0,200]）：α 增大（菱形更"高"）、只给**轴向尖角**加圆角
#   段2（s∈[200,400]）     ：α 减小（菱形更"扁"）、**所有角**都加圆角
ALPHA_A, ALPHA_B = 34.0, 28.0           # deg（待实测确认）
R_FIL_A, R_FIL_B = 1.2, 1.2             # mm
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


def seg_alpha(s_m):
    return ALPHA_A if s_m <= L1 else ALPHA_B


def seg_L(s_m):
    """该段的斜撑长（由 α 与固定周向半节距决定）。"""
    return B_MM / np.cos(np.radians(seg_alpha(s_m)))


def ab(eps, s_m):
    """返回 (轴向半对角 a, 周向半对角 b, 该段斜撑长 L)。"""
    L = seg_L(s_m)
    a0 = L * np.sin(np.radians(seg_alpha(s_m)))
    a = float(np.clip(a0 * (1.0 + eps), 0.3, 0.985 * L))
    b = float(np.sqrt(max(L ** 2 - a ** 2, 0.0)))
    return a, b, L


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


def draw(ax, kfun, ss, A, theta_total, lims=None):
    ax.clear()
    sg, P, Rm = build_frames(kfun, ss)

    # 节点行（沿 s 累积，按变形后节距）
    rows, s_cur = [], 0.0
    while s_cur < L_TOT - 1e-6:
        eps = -float(np.interp(s_cur, ss, kfun)) * R_OUT
        a, b, Ls = ab(eps, s_cur)
        rows.append(dict(s=s_cur, a=a, b=b, eps=eps, seg=(0 if s_cur <= L1 else 1),
                         L=Ls, alpha=seg_alpha(s_cur)))
        s_cur += 2 * a * 1e-3
    rows.append(dict(**{**rows[-1], "s": L_TOT}))

    quads, cols = [], []
    for i in range(0, len(rows) - 1, ROW_STEP):
        r0, r1 = rows[i], rows[i + 1]
        dth = r0["b"] / (R_OUT * 1e3)    # 弧度：周向半节距（b 为 mm）
        for j in range(6):
            th0 = j * np.pi / 3          # 每 60° 一个节点
            for dj in (+1, -1):
                th1 = th0 + dj * dth
                q = strut(P, Rm, sg, th0, r0["s"], th1, r1["s"])
                quads += q
                cols += [RED if dj > 0 else "#6E3234"] * len(q)   # 两组方向用明暗区分
    if not quads:
        return
    pc = Poly3DCollection(quads, edgecolor="#4A2425", linewidth=0.22,
                          alpha=1.0)
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
    # 驱动丝：沿臂身弯曲（r = R_DISK 恒定，全程在臂体内部），不是直线
    svals = np.linspace(0.0, rows[-1]["s"], 120)
    for k in range(6):
        thk = k * np.pi / 3
        pts = np.array([surface(P, Rm, sg, sv, thk, R_DISK) for sv in svals])
        # ⚠ matplotlib 的 3D 深度排序对 Line3D vs Poly3DCollection 不可靠，
        #   给丝设高 zorder 会盖在筋上面。改为不提 zorder + 低透明度：
        #   丝只在孔洞处隐约可见，绝不压住臂体。
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=GREEN, lw=0.7, alpha=0.40,
                solid_capstyle="round")

    # 固定基座：s=0 处一块方形夹持板（向下 10 mm），并用剖面线示意"固定"
    b = R_OUT * 1.55
    zb0, zb1 = -0.010, 0.0
    bx = [(-b, -b), (b, -b), (b, b), (-b, b)]
    top = [(x, y, zb1) for x, y in bx]
    bot = [(x, y, zb0) for x, y in bx]
    base_quads = [top[::-1], bot] + [[top[i], top[(i + 1) % 4],
                                      bot[(i + 1) % 4], bot[i]] for i in range(4)]
    bpc = Poly3DCollection(base_quads, facecolor="#8A8F98", edgecolor="0.35",
                           linewidth=0.4, alpha=0.95)
    ax.add_collection3d(bpc)
    ax.text(0, 0, zb0 - 0.004, "固定基座", fontsize=7, color="0.25",
            ha="center", zorder=7)

    # ── 圆角：孔的两条边线的**切线圆弧**（凹圆角），填角上的楔形
    def wedge(u1, u2, rho, h=W_STRUT * 1e3 / 2.0):
        """孔的切线圆角楔形。

        ⚠ 关键：孔的**边界是筋的侧面**，不是筋的中心线。两侧各内缩 h=w/2，
           两条内缩边线的交点沿角平分线移动  h/sin(φ/2)。
           之前漏了这一步 ⇒ 圆角被画进孔中间。
        """
        u1 = np.asarray(u1, float); u2 = np.asarray(u2, float)
        u1 /= np.linalg.norm(u1); u2 /= np.linalg.norm(u2)
        phi = float(np.arccos(np.clip(u1 @ u2, -1, 1)))
        if phi < 1e-4 or phi > np.pi - 1e-4:
            return None
        bis = u1 + u2
        bis /= np.linalg.norm(bis)
        V = bis * (h / np.sin(phi / 2))          # 真实孔角（内缩后的交点）
        t = rho / np.tan(phi / 2)
        d = rho / np.sin(phi / 2)
        P1, P2 = V + u1 * t, V + u2 * t
        O = V + bis * d
        a1 = np.arctan2((P1 - O)[1], (P1 - O)[0])
        a2 = np.arctan2((P2 - O)[1], (P2 - O)[0])
        da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
        arc = [O + rho * np.array([np.cos(a1 + da * q), np.sin(a1 + da * q)])
               for q in np.linspace(0, 1, 7)]
        return [V] + [P1] + arc + [P2]

    caps = []
    for i in range(len(rows) - 1):
        r = rows[i]
        a_mm, b_mm = r["a"], r["b"]
        rho = (R_FIL_A if r["seg"] == 0 else R_FIL_B)
        # 该行的菱形：中心在 (s+a, θ_c)，四角 = 上下尖角 + 左右侧角
        for jc in range(6):
            th_c = jc * np.pi / 3
            corners = [
                # (角相对中心 (dθ, ds), 两条邻边方向)
                ((0.0, +a_mm), ((-b_mm, -a_mm), (+b_mm, -a_mm)), "tip"),
                ((0.0, -a_mm), ((-b_mm, +a_mm), (+b_mm, +a_mm)), "tip"),
                ((+b_mm, 0.0), ((-b_mm, -a_mm), (-b_mm, +a_mm)), "side"),
                ((-b_mm, 0.0), ((+b_mm, -a_mm), (+b_mm, +a_mm)), "side"),
            ]
            for (dc, ds), (e1, e2), kind in corners:
                if r["seg"] == 0 and kind != "tip":
                    continue                  # 段1：只填轴向尖角
                w = wedge(e1, e2, rho)
                if w is None:
                    continue
                sv0 = r["s"] + a_mm * 1e-3 + ds * 1e-3
                th0 = th_c + dc / (R_OUT * 1e3)
                ring = [surface(P, Rm, sg, sv0 + q[1] * 1e-3,
                                th0 + q[0] / (R_OUT * 1e3), R_OUT + 0.02e-3)
                        for q in w]
                caps.append(ring)
    if caps:
        cpc = Poly3DCollection(caps, facecolor="#8E4446", edgecolor="#4A2425",
                               linewidth=0.12, alpha=1.0)
        ax.add_collection3d(cpc)

    span = rows[-1]["s"]
    # 坐标范围：所有帧共用同一套（否则相机跟着包围盒走 ⇒ 看起来整体平移）
    if lims is None:
        pmin, pmax = P.min(0), P.max(0)
        ctr = 0.5 * (pmin + pmax)
        half = 0.5 * float(np.max(pmax - pmin)) + 0.012
        lims = ((ctr[0] - half, ctr[0] + half), (ctr[1] - half, ctr[1] + half),
                (ctr[2] - half, ctr[2] + half))
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=14, azim=-62)
    ax.set_title(f"A={A:.2f} N  θ={np.degrees(theta_total):.1f}°  "
                 f"ε_root={rows[0]['eps']*100:+.1f}%\n红=臂体菱形筋  绿=驱动丝  灰=基座",
                 fontsize=9)


def compute_lims(theta=np.pi):
    """用最大弯曲(180°)的骨架包围盒作为全帧固定视野。"""
    ss, kf = kappa_profile(theta)
    sg, P, Rm = build_frames(kf, ss)
    pmin, pmax = P.min(0), P.max(0)
    ctr = 0.5 * (pmin + pmax)
    half = 0.5 * float(np.max(pmax - pmin)) + 0.012
    return ((ctr[0] - half, ctr[0] + half), (ctr[1] - half, ctr[1] + half),
            (ctr[2] - half, ctr[2] + half))


def main():
    files, TH = [], []
    LIMS = compute_lims()
    # 按目标角度驱动（末帧正好 180°）：θ = 0.9·A·r/EI
    for idx, theta_total in enumerate(np.linspace(0.0, np.pi, NFRAME)):
        k1 = theta_total / 0.3
        A = k1 * EI / (3 * R_DISK)
        ss, kfun = kappa_profile(theta_total)
        fig = plt.figure(figsize=(6.6, 7.0))
        ax = fig.add_subplot(111, projection="3d")
        draw(ax, kfun, ss, A, theta_total, LIMS)
        fig.tight_layout()
        fn = f"/tmp/_b3d_{idx:03d}.png"
        fig.savefig(fn, dpi=150)
        plt.close(fig)
        files.append(fn)
        TH.append(np.degrees(theta_total))
        print(f"  帧 {idx:02d}: A={A:5.2f} θ={TH[-1]:6.1f}° "
              f"ε_root={-kfun[0]*R_OUT*100:+.1f}%", flush=True)
    ims = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
    ims[0].save("anim_bend3d.gif", save_all=True, append_images=ims[1:],
                duration=180, loop=0, optimize=True)
    print(f"saved anim_bend3d.gif ({len(ims)} 帧)")

    # 末帧（180°）高分辨率 PNG：GIF 只有 256 色，细描边会被量化掉
    k1 = np.pi / 0.3
    A = k1 * EI / (3 * R_DISK)
    ss, kfun = kappa_profile(np.pi)
    fig = plt.figure(figsize=(8.0, 8.4))
    ax = fig.add_subplot(111, projection="3d")
    draw(ax, kfun, ss, A, np.pi, LIMS)
    fig.tight_layout()
    fig.savefig("anim_bend3d_final.png", dpi=220)
    plt.close(fig)
    print("saved anim_bend3d_final.png (180°, dpi 220)")


if __name__ == "__main__":
    main()
