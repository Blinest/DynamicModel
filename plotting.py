# -*- coding: utf-8 -*-
"""
plotting.py — 绘图封装 (从 plot_workspace.py / plot_tau_compare.py 抽离)
======================================================================
把原先两个**脚本**里的绘图逻辑抽成模块, 对外只暴露 **两个函数**:

    plot_workspace(...)     臂体可达空间: κ(s) 重建臂形族 + 末端点云
                            → fig_workspace[_<tag>].png
                            → fig_workspace[_<tag>]_orient.png

    plot_tau_compare(...)   不同张力上限的可达空间对比
                            → fig_tau_compare.png

用法
----
    # 函数调用 (推荐)
    from plotting import plot_workspace, plot_tau_compare
    plot_workspace("40")                       # 单表 → 2 张图
    plot_tau_compare(("20", "30", "40"))       # 多表对比 → 1 张图

    # 命令行 (等价)
    python plotting.py workspace 40
    python plotting.py tau-compare 20 30 40

契约 (两个函数统一)
------------------
    参数   tag/file 选表, out 输出前缀, dpi/alpha/lw/n_draw 控制画质,
           style 选配色字体 ("paper" | "default"), show 结尾是否 plt.show()
    返回   dict: 含各图的统计量(样本数/半径范围/倾角/末端重建误差等)
    副作用 只写 PNG 到当前工作目录; 不修改输入表
"""
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")                       # 服务器无显示环境; 需交互请自行 set backend
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from ik_table import load_table
from tendon_coupling import _E3, L
from tendon_coupling import TendonCouplingModel      # ★ 统一模型入口

__all__ = ["plot_workspace", "plot_tau_compare", "PALETTE", "CMAP_SEQ",
           "CMAP_NEG", "CMAP_DIV"]


# ══════════════════════════════════════════════════════════════════════
# §0  共用: 配色 / 样式
# ══════════════════════════════════════════════════════════════════════
# 低饱和高级配色 (鼠尾草绿 / 靛蓝 / 砖红 / 低饱和金 / 灰紫)
PALETTE = ['#8DAE7F', '#516CA3', '#A34D4E', '#C4A06A', '#7A6A8A']
# 连续量: 绿 → 蓝 → 红 (低 → 高)
CMAP_SEQ = LinearSegmentedColormap.from_list('sage', ['#8DAE7F', '#516CA3', '#A34D4E'])
# ΔL 负值(收丝)深浅: 浅粉 → 砖红 → 深红。**按数据实际范围铺满**, 不用对称 ±vmax
CMAP_NEG = LinearSegmentedColormap.from_list(
    'sage_neg', ['#F5E4E0', '#D89A96', '#A34D4E', '#6E2F30', '#3E1517'])
# 数据跨零时回退发散色图
CMAP_DIV = LinearSegmentedColormap.from_list('sage_div', ['#516CA3', '#F2F0EA', '#A34D4E'])

L_ARM = 0.45          # 臂长 (m)
N_DRAW = 100000       # 默认画多少条臂形线 (>=N 即全画)

_STYLES = {
    # 论文风格: 中英混排, 加粗, 透明底 (原 plot_workspace.py)
    "paper": dict(
        font_family=['Times New Roman', 'SimHei'],
        font_sans=['Times New Roman', 'SimHei'],
        weight='bold', size=15, transparent=True,
    ),
    # 文档/屏幕风格: 微软雅黑 (原 plot_tau_compare.py)
    "default": dict(
        font_family='sans-serif',
        font_sans=['Microsoft YaHei', 'SimHei'],
        weight=None, size=10, transparent=False,
    ),
}


def _apply_style(style="paper"):
    """统一设置 rcParams (对应原两脚本各自的样式块)."""
    cfg = _STYLES.get(style, _STYLES["paper"])
    plt.rcParams['font.family'] = cfg["font_family"]
    plt.rcParams['font.sans-serif'] = cfg["font_sans"]
    if cfg["weight"]:
        plt.rcParams['font.weight'] = cfg["weight"]
        plt.rcParams['axes.labelweight'] = cfg["weight"]
        plt.rcParams['axes.titleweight'] = cfg["weight"]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = cfg["size"]
    if cfg["transparent"]:
        plt.rcParams['savefig.transparent'] = True
        plt.rcParams['figure.facecolor'] = 'none'
        plt.rcParams['axes.facecolor'] = 'none'
    return cfg


def _resolve_prefix(tag, out):
    """输出前缀: 20N 表不带后缀 (向后兼容)."""
    return out or ("fig_workspace" if str(tag) == "20" else f"fig_workspace_{tag}")


def _tilt(R):
    """末端切线与竖直夹角 (°)."""
    return np.degrees(np.arccos(np.clip(R[:, :, 2] @ _E3, -1, 1)))


# ══════════════════════════════════════════════════════════════════════
# §1  函数一: plot_workspace
# ══════════════════════════════════════════════════════════════════════
def _load_workspace_data(fn, verbose=True):
    """载表 → **经 TendonCouplingModel** 批量解算 → 细骨架重建.

    ⚠ params=None → 从 npz 读 n_p/n_d 并沿用 default_params() 的刚度。
      若显式传 default_params(), 会把建表时的网格 (refine 后 n=300) 覆盖回 n=30,
      于是批量积只走 61 个节点, 与表里 kS/kK 的 601 节点对不上。

    ⚠ 重建必须**带 v 场**: p'=R·v, k_rod 有限时 v_z 实测 0.85~0.90 (轴向压缩
      10~15%)。只用 κ (v=e₃) 会让骨架凭空长 40~58mm。
    """
    T = load_table(fn, params=None)
    recs = T.recs
    P = np.array([r["p_tip"] for r in recs])
    R = np.array([r["R_tip"] for r in recs])
    TAU = np.array([r["tau"] for r in recs])
    X = np.array([r["x"] for r in recs])

    M = TendonCouplingModel(params=T.params)          # ← 统一模型
    cv, _ = M.curve_batch(TAU, x=X)                   # 解网格曲线场 (复用表内打靶解)
    PS, RS, _ = M.build_shapes_batch(TAU, nsub=600, x=X, curve=cv)
    Uc, Vc, S = cv[2], cv[6], cv[1]                   # 曲率 / 线应变场 / 弧长

    d = dict(T=T, M=M, recs=recs, N=len(recs), P=P, R=R, TAU=TAU, X=X,
             Uc=Uc, Vc=Vc, PS=PS, RS=RS, S=S,
             bend=np.linalg.norm(Uc, axis=2).mean(axis=1),   # 平均弯曲 |κ|
             rad=np.linalg.norm(P, axis=1),
             tilt=_tilt(R),
             e_rec=np.linalg.norm(PS[:, -1] - P, axis=1) * 1000)
    if verbose:
        print(f"[{os.path.basename(fn)}] N={d['N']}  τmax={TAU.max():.0f}N  "
              f"r∈[{d['rad'].min()*1000:.0f},{d['rad'].max()*1000:.0f}]mm  "
              f"倾角max={d['tilt'].max():.1f}°")
        print(f"[{os.path.basename(fn)}] κ(s) 重建 vs 全阶解: 末端误差 "
              f"中位={np.median(d['e_rec']):.3f} mm "
              f"90%={np.percentile(d['e_rec'], 90):.3f} mm")
    return d


def _draw_workspace_main(d, prefix, n_draw=N_DRAW, alpha=0.18, lw=0.6,
                         dpi=150, transparent=True, close=True):
    """主图: 1 行 4 个 3D 臂形族 (各按一变量着色) + 1 行 3 个投影."""
    N, PS, P, TAU, tilt, RS = d["N"], d["PS"], d["P"], d["TAU"], d["tilt"], d["RS"]
    sel = np.arange(N) if n_draw >= N else np.linspace(0, N - 1, n_draw).astype(int)
    projs = [("XY", 0, 1), ("XZ", 0, 2), ("YZ", 1, 2)]
    proj_colors = CMAP_SEQ(np.random.default_rng(0).random(len(sel)))

    panels = [
        (np.linalg.norm(TAU, axis=1), r"$\|\tau\|$ (N)", "%.1f"),
        (TAU.max(axis=1), r"$\max\tau_i$ (N)", "%.1f"),
        (np.abs(d["Uc"][:, :, 2]).mean(axis=1), r"$|\kappa_z|$ (1/m)", "%.2f"),
        (tilt, r"$\theta_{tip}$ (°)", "%.0f"),
    ]

    # ⚠ GridSpec: 两行必须用**同一套**列划分 (2×4 与 2×3 混用会报
    # `integer indices must be multiples` 且 tight_layout 失效)。
    fig = plt.figure(figsize=(22, 11))
    gs = GridSpec(2, 12, figure=fig)
    allpts = PS[sel].reshape(-1, 3)
    xlim = (allpts[:, 0].min(), allpts[:, 0].max())
    ylim = (allpts[:, 1].min(), allpts[:, 1].max())

    for k, (vals, label, fmt) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 3 * k:3 * (k + 1)], projection="3d")
        norm = Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
        ax.add_collection3d(Line3DCollection(
            [PS[i] for i in sel], colors=CMAP_SEQ(norm(vals[sel])),
            linewidths=lw, alpha=alpha))
        ax.scatter(P[sel, 0], P[sel, 1], P[sel, 2], c=vals[sel], cmap=CMAP_SEQ,
                   s=6, alpha=0.55, depthshade=False)
        ax.scatter([0], [0], [0], c='#3A3A3A', s=70, marker="o", zorder=10)
        ax.plot([0, 0], [0, 0], [0, L_ARM], "--", c='#9A9A9A', lw=1.1)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(0, L_ARM)
        ax.set_box_aspect((1, 1, 1.2))
        ax.set_xlabel("x (m)", fontsize=13)
        ax.set_ylabel("y (m)", fontsize=13)
        ax.set_zlabel("z (m)", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.set_title(label, fontsize=24, pad=8)
        ax.view_init(elev=20, azim=-60)
        fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP_SEQ, norm=norm), ax=ax,
                     shrink=0.55, pad=0.02, format=fmt).ax.tick_params(labelsize=11)

    # 第 2 行: 三视图 (颜色与物理量无关, 只为区分单条曲线)
    for i, (nm, a, b) in enumerate(projs):
        axp = fig.add_subplot(gs[1, 4 * i:4 * (i + 1)])
        axp.add_collection(LineCollection(
            [PS[j][:, [a, b]] for j in sel],
            colors=proj_colors, linewidths=lw, alpha=alpha))
        axp.scatter(P[sel, a], P[sel, b], c=proj_colors, s=5, alpha=0.35)
        axp.scatter([0], [0], c='#3A3A3A', s=60, marker="o", zorder=10)
        axp.set_xlabel(f"{'xyz'[a]} (m)", fontsize=13)
        axp.set_ylabel(f"{'xyz'[b]} (m)", fontsize=13)
        axp.tick_params(labelsize=11)
        axp.set_title(nm, fontsize=24); axp.grid(alpha=0.20)
        axp.set_aspect("equal")
        axp.autoscale_view()

    fig.suptitle(rf"$\tau_{{max}}$ = {TAU.max():.0f} N", fontsize=32)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    path = f"{prefix}.png"
    fig.savefig(path, dpi=dpi, transparent=transparent)
    if close:
        plt.close(fig)
    print(f"saved: {path}")
    return path


def _draw_workspace_orient(d, prefix, dpi=150, transparent=True, close=True):
    """副图: 12 条代表性臂形 + 各自驱动丝位移 ΔL 热图 + 末端朝向分布.

    符号约定 (与固件 SDM.c 一致): 拉丝侧 ΔL 为负。基准取**零张力直臂**,
    未含预紧 ⇒ 对称加压时六丝全负 (纯轴向压缩), 弯曲工况下对面丝才出现正值。
    """
    N, recs, PS, tilt, bend = d["N"], d["recs"], d["PS"], d["tilt"], d["bend"]
    order = np.argsort(bend)
    picks = order[np.linspace(0, N - 1, 12).astype(int)]
    DL = np.array([recs[i]["dL_mm"] for i in picks])          # (12, 6) mm
    colors = CMAP_SEQ(np.linspace(0.0, 1.0, len(picks)))

    fig2 = plt.figure(figsize=(19, 6))

    # (a) 代表性臂形
    ax2 = fig2.add_subplot(1, 3, 1, projection="3d")
    for c, i in zip(colors, picks):
        ax2.plot(PS[i][:, 0], PS[i][:, 1], PS[i][:, 2], lw=2.2, c=c,
                 label=rf"$|\kappa|$={bend[i]:.2f}")
    ax2.scatter([0], [0], [0], c='#3A3A3A', s=80, marker="o")
    ax2.plot([0, 0], [0, 0], [0, L_ARM], "--", c='#9A9A9A', lw=1)
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)"); ax2.set_zlabel("z (m)")
    ax2.set_title(r"Arm shapes", fontsize=24)
    ax2.legend(fontsize=11, loc="upper left", framealpha=0.85)
    ax2.set_box_aspect((1, 1, 1.2))

    # (b) ΔL 热图: 行=代表臂形, 列=6 根丝
    axDL = fig2.add_subplot(1, 3, 2)
    if DL.min() >= 0:
        cmap_dl, vmin_dl, vmax_dl = CMAP_DIV, -max(abs(DL.max()), 1e-9), DL.max()
    elif DL.max() <= 0:
        cmap_dl, vmin_dl, vmax_dl = CMAP_NEG, DL.min(), 0.0
    else:
        m = float(np.abs(DL).max())
        cmap_dl, vmin_dl, vmax_dl = CMAP_DIV, -m, m
    im = axDL.imshow(DL, aspect="auto", cmap=cmap_dl, vmin=vmin_dl, vmax=vmax_dl)
    axDL.set_xticks(range(6))
    axDL.set_xticklabels([rf"$w_{{{w}}}$" + ("A" if w in (0, 2, 4) else "B")
                          for w in range(6)], fontsize=13)
    axDL.set_yticks(range(len(picks)))
    axDL.set_yticklabels([f"{bend[i]:.2f}" for i in picks], fontsize=16)
    for r in range(len(picks)):
        for c in range(6):
            axDL.text(c, r, f"{DL[r, c]:.0f}", ha="center", va="center",
                      fontsize=11, color='#2A2A2A')
    axDL.set_title(r"$\Delta L$ (mm)", fontsize=26)
    axDL.set_xlabel("wire", fontsize=14)
    axDL.set_ylabel(r"$|\kappa|$ (1/m)", fontsize=14)
    fig2.colorbar(im, ax=axDL, shrink=0.8, label=r"$\Delta L$ (mm)")

    # (c) 末端朝向分布
    ax3 = fig2.add_subplot(1, 3, 3)
    ax3.hist(tilt, bins=40, color='#8DAE7F', alpha=0.9,
             edgecolor='#516CA3', lw=0.4)
    ax3.set_xlabel(r"$\theta_{tip}$ (°)", fontsize=14)
    ax3.set_ylabel("count", fontsize=14)
    ax3.set_title(r"$\theta_{tip}$", fontsize=26)
    ax3.grid(alpha=0.20)

    fig2.tight_layout()
    path = f"{prefix}_orient.png"
    fig2.savefig(path, dpi=dpi, transparent=transparent)
    if close:
        plt.close(fig2)
    print(f"saved: {path}")
    print(f"  代表臂形 ΔL 范围 [{DL.min():.1f}, {DL.max():.1f}] mm")
    return path, DL


def plot_workspace(tag="40", file=None, out=None, n_draw=N_DRAW, alpha=0.18,
                   lw=0.6, dpi=150, style="paper", show=False, verbose=True):
    """★ 函数一: 画臂体可达空间 (原 plot_workspace.py).

    参数
    ----
    tag    : str       张力上限标签 → vc_table_<tag>.npz
    file   : str|None  直接指定 npz 路径 (覆盖 tag)
    out    : str|None  输出前缀, 默认 fig_workspace[_<tag>]
    n_draw : int       画多少条臂形线 (>=N 即全画)
    alpha/lw : float   线透明/线宽
    dpi    : int       输出分辨率
    style  : str       "paper"(透明底+Times) | "default"(雅黑)
    show   : bool      结尾 plt.show()
    verbose: bool      是否打印统计

    返回 dict: N, tau_max, r_min_mm, r_max_mm, tilt_max_deg,
               rec_err_med_mm, rec_err_p90_mm, main_png, orient_png, dl_mm
    """
    fn = file or f"vc_table_{tag}.npz"
    if not os.path.exists(fn):
        raise FileNotFoundError(
            f"表不存在: {fn}\n(旧模型的表已归档到 old_model_2026-09-19/, "
            f"用 file= 指定, 或使用当前 vc_table_40.npz)")
    transparent = True
    _apply_style(style)
    transparent = bool(_STYLES.get(style, _STYLES["paper"])["transparent"])
    prefix = _resolve_prefix(tag, out)

    d = _load_workspace_data(fn, verbose=verbose)
    main_png = _draw_workspace_main(d, prefix, n_draw=n_draw, alpha=alpha, lw=lw,
                                    dpi=dpi, transparent=transparent)
    orient_png, DL = _draw_workspace_orient(d, prefix, dpi=dpi,
                                            transparent=transparent)
    if show:
        plt.show()
    return dict(N=d["N"], tau_max=float(d["TAU"].max()),
                r_min_mm=float(d["rad"].min() * 1000),
                r_max_mm=float(d["rad"].max() * 1000),
                tilt_max_deg=float(d["tilt"].max()),
                rec_err_med_mm=float(np.median(d["e_rec"])),
                rec_err_p90_mm=float(np.percentile(d["e_rec"], 90)),
                main_png=main_png, orient_png=orient_png, dl_mm=DL)


# ══════════════════════════════════════════════════════════════════════
# §2  函数二: plot_tau_compare
# ══════════════════════════════════════════════════════════════════════
def _load_tau_data(tag, verbose=True):
    """单表 → 点云 + 臂形族 + 半径/倾角指标 (经 TendonCouplingModel); 缺表返回 None."""
    fn = f"vc_table_{tag}.npz"
    if not os.path.exists(fn):
        return None
    T = load_table(fn)
    recs = T.recs
    P = np.array([r["p_tip"] for r in recs])
    R = np.array([r["R_tip"] for r in recs])
    TAU = np.array([r["tau"] for r in recs])
    X = np.array([r["x"] for r in recs])
    M = TendonCouplingModel(params=T.params)
    cv, _ = M.curve_batch(TAU, x=X)
    # ⚠ 必须带 v 场 (同 §1)
    PS, _, _ = M.build_shapes_batch(TAU, nsub=600, x=X, curve=cv)
    return dict(recs=recs, M=M, P=P, R=R, TAU=TAU, PS=PS, T=T,
                r=np.linalg.norm(P, axis=1) * 1000,
                tilt=_tilt(R))


def plot_tau_compare(tags=("20", "30", "40"), out="fig_tau_compare.png",
                     cmap=None, alpha=0.16, lw=0.5, dpi=150, style="default",
                     show=False, verbose=True):
    """★ 函数二: 不同张力上限的可达空间对比 (原 plot_tau_compare.py).

    参数
    ----
    tags   : 序列      张力上限标签, 如 ("20","30","40")
    out    : str       输出 PNG 路径
    cmap   : dict|None 各 tag 的颜色, 默认 tab:blue/orange/red 循环
    alpha/lw : float   臂形线透明/线宽
    style  : str       "default"(雅黑) | "paper"
    返回 dict: 每 tag 的 N/r_min/r_med/r_p10/tilt_max/r_lt300_pct, png, have
    """
    tags = list(tags)
    colors = cmap or {"20": "tab:blue", "30": "tab:orange", "40": "tab:red"}
    _apply_style(style)

    D = {t: _load_tau_data(t, verbose=verbose) for t in tags}
    have = [t for t in tags if D[t] is not None]
    if not have:
        raise FileNotFoundError(
            "没有任何可用表: " + ", ".join(f"vc_table_{t}.npz" for t in tags))
    print("载入: " + ", ".join(f"{t}N(N={len(D[t]['recs'])})" for t in have))

    def col(t):
        return colors.get(t, "tab:gray")

    fig = plt.figure(figsize=(16, 10))

    # ---------- 上排: 每个 τmax 的 3D 点云 + 臂形族 ----------
    ncol = 3
    for k, t in enumerate(tags):
        ax = fig.add_subplot(2, ncol, k + 1, projection="3d")
        d = D[t]
        if d is None:
            ax.set_title(f"τmax={t}N (表不存在)", fontsize=11)
            ax.axis("off")
            continue
        NN = len(d["recs"])
        sub = (np.arange(NN) if NN <= 1500
               else np.linspace(0, NN - 1, 1500).astype(int))
        ax.add_collection3d(Line3DCollection(
            [d["PS"][i] for i in sub], colors=col(t), linewidths=lw, alpha=alpha))
        ax.scatter(d["P"][sub, 0], d["P"][sub, 1], d["P"][sub, 2],
                   c=col(t), s=4, alpha=0.35, depthshade=False)
        ax.scatter([0], [0], [0], c="k", s=70, marker="o")
        ax.plot([0, 0], [0, 0], [0, L], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.set_title(f"τmax={float(t):.0f}N   N={NN}\n"
                     f"r∈[{d['r'].min():.0f},{d['r'].max():.0f}]mm  "
                     f"倾角max={d['tilt'].max():.0f}°", fontsize=11)
        ax.set_box_aspect((1, 1, 1.2)); ax.view_init(elev=18, azim=-60)
        ax.set_zlim(0, L)

    # ---------- 下排 (1): 半径分布 ----------
    ax = fig.add_subplot(2, ncol, ncol + 1)
    for t in have:
        ax.hist(D[t]["r"], bins=34, range=(100, 460), alpha=0.55,
                color=col(t), label=f"τmax={t}N")
    ax.axvline(2 * L / np.pi * 1000, color="k", ls="--", lw=1)
    ax.text(2 * L / np.pi * 1000 + 4, ax.get_ylim()[1] * 0.8, "半圆极限 2L/π",
            fontsize=8)
    ax.set_xlabel("末端到基座距离 r (mm)"); ax.set_ylabel("样本数")
    ax.set_title("可达半径分布 (越靠左=工作空间越大)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ---------- 下排 (2): 倾角分布 ----------
    ax = fig.add_subplot(2, ncol, ncol + 2)
    for t in have:
        ax.hist(D[t]["tilt"], bins=34, range=(0, 180), alpha=0.55,
                color=col(t), label=f"τmax={t}N")
    ax.set_xlabel("末端切线与竖直夹角 (°)"); ax.set_ylabel("样本数")
    ax.set_title("末端倾角分布 (越大=越能弯)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ---------- 下排 (3): 包络指标 ----------
    ax = fig.add_subplot(2, ncol, ncol + 3)
    xs = [float(t) for t in have]
    ax.plot(xs, [D[t]["r"].min() for t in have], "o-", color="tab:red",
            label="最小半径 r_min", lw=2, ms=7)
    ax.plot(xs, [np.median(D[t]["r"]) for t in have], "s-", color="tab:blue",
            label="半径中位", lw=2, ms=7)
    ax.plot(xs, [np.percentile(D[t]["r"], 10) for t in have], "^--",
            color="tab:purple", label="半径 10%", lw=1.5, ms=6)
    ax.set_xlabel("τmax (N)"); ax.set_ylabel("半径 (mm)")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(xs, [D[t]["tilt"].max() for t in have], "D--", color="tab:green",
             label="最大倾角", lw=1.5, ms=6)
    ax2.set_ylabel("最大倾角 (°)", color="tab:green")
    ax2.tick_params(axis="y", colors="tab:green")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="center right")
    ax.set_title("包络指标 vs 张力上限")

    fig.suptitle("张力上限对可达空间的影响 (τ 空间 LHS 均匀采样, 臂长 450mm)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    print(f"saved: {out}")

    # ---------- 控制台汇总 ----------
    stats = {}
    print(f"\n{'τmax':>6} {'N':>6} {'r_min':>7} {'r中位':>7} {'倾角max':>8} {'r<300占比':>10}")
    for t in have:
        d = D[t]
        stats[t] = dict(N=len(d["recs"]), r_min=float(d["r"].min()),
                        r_med=float(np.median(d["r"])),
                        r_p10=float(np.percentile(d["r"], 10)),
                        tilt_max=float(d["tilt"].max()),
                        r_lt300_pct=float((d["r"] < 300).mean() * 100))
        print(f"{float(t):>6.0f} {stats[t]['N']:>6} {stats[t]['r_min']:>7.0f} "
              f"{stats[t]['r_med']:>7.0f} {stats[t]['tilt_max']:>8.1f} "
              f"{stats[t]['r_lt300_pct']:>9.1f}%")
    return dict(png=out, have=have, stats=stats)


# ══════════════════════════════════════════════════════════════════════
# §3  命令行 (可选, 等价于函数调用)
# ══════════════════════════════════════════════════════════════════════
def _main(argv=None):
    import argparse
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="绘图封装: workspace | tau-compare")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("workspace", help="臂体可达空间")
    p1.add_argument("tag", nargs="?", default="40")
    p1.add_argument("--file", default=None)
    p1.add_argument("--out", default=None)
    p1.add_argument("--n-draw", type=int, default=N_DRAW)
    p1.add_argument("--dpi", type=int, default=150)
    p1.add_argument("--style", default="paper", choices=list(_STYLES))

    p2 = sub.add_parser("tau-compare", help="张力上限对比")
    p2.add_argument("tags", nargs="*", default=["20", "30", "40"])
    p2.add_argument("--out", default="fig_tau_compare.png")
    p2.add_argument("--dpi", type=int, default=150)
    p2.add_argument("--style", default="default", choices=list(_STYLES))

    a = ap.parse_args(argv)
    if a.cmd == "workspace":
        plot_workspace(a.tag, file=a.file, out=a.out, n_draw=a.n_draw,
                       dpi=a.dpi, style=a.style)
    else:
        plot_tau_compare(a.tags, out=a.out, dpi=a.dpi, style=a.style)


if __name__ == "__main__":
    _main()
