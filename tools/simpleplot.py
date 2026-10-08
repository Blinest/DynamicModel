# -*- coding: utf-8 -*-
"""simpleplot.py — 极简绘图后端 (只用 Pillow, 不需要 matplotlib)
==========================================================================================
给 `tools/plot_friction.py` 在没有 matplotlib 的环境里兜底出图 (如 CI / 精简容器)。

只实现本项目用到的最小 API (仿 matplotlib 的一小撮):
    fig, ax = subplots(1, 2, figsize=(11, 4.2))
    ax[i].plot(x, y, "o-", label="...")
    ax[i].axvline(x0, color="k", ls=":")
    ax[i].fill_between(x, y1, y2, color="purple", alpha=.25, label="...")
    ax[i].set_xlabel(...) / set_ylabel(...) / set_title(...) / grid() / legend()
    fig.suptitle(...) / savefig("x.png", dpi=150)

渲染方式: 先按 SS=2 超采样画, 再 LANCZOS 缩小 → 线条平滑。
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

SS = 2                                   # 超采样倍率
C = {"b": "#1f77b4", "r": "#d62728", "g": "#2ca02c", "o": "#ff7f0e",
     "p": "#9467bd", "k": "#000000", "gray": "#808080", "c0": "#1f77b4",
     "c1": "#ff7f0e", "c3": "#d62728"}


def _norm(col, default="#1f77b4"):
    """把 matplotlib 风格色名 (C0/C1/c0/tab:blue...) 归一化到 PIL 能认的颜色."""
    if col is None:
        return default
    if not isinstance(col, str):
        return col
    key = col.lower()
    if key in C:
        return C[key]
    if key.startswith("c") and key[1:].isdigit():
        return C.get("c" + key[1:], default)
    if key.startswith("tab:"):
        return {"tab:blue": "#1f77b4", "tab:red": "#d62728", "tab:green": "#2ca02c",
                "tab:orange": "#ff7f0e", "tab:purple": "#9467bd"}.get(key, default)
    return col


def _font(size=13):
    try:
        return ImageFont.load_default(size=size * SS)
    except Exception:
        return ImageFont.load_default()


class Axes:
    def __init__(self, fig, rect):
        self.fig = fig
        self.rect = rect                    # (x0, y0, x1, y1) 像素 (已含 SS)
        self.series = []                    # dict(kind, x, y, color, ls, marker, lw, label)
        self.xlabel = ""; self.ylabel = ""; self.title = ""
        self.do_grid = False; self._legend = False
        self.ylog = False

    # ── 数据 ──
    def plot(self, x, y, fmt="-", label=None, color=None, lw=1.8, ms=4.0,
             alpha=1.0, ls=None):
        col = _norm(color, "#1f77b4")
        if fmt:
            if "o" in fmt: mk = "o"
            elif "s" in fmt: mk = "s"
            elif "^" in fmt: mk = "^"
            else: mk = None
            if "--" in fmt: ls = ls or "--"
            elif ":" in fmt: ls = ls or ":"
            else: ls = ls or "-"
        else:
            mk = None
        self.series.append(dict(kind="line", x=list(x), y=list(y), color=col,
                                ls=ls or "-", marker=mk, lw=lw, ms=ms,
                                alpha=alpha, label=label))

    def axvline(self, x0, color="k", ls=":", lw=1.0):
        self.series.append(dict(kind="vline", x0=x0, color=_norm(color, "#000000"),
                                ls=ls, lw=lw))

    def fill_between(self, x, y1, y2, color="purple", alpha=.25, label=None):
        self.series.append(dict(kind="fill", x=list(x), y1=list(y1), y2=list(y2),
                                color=_norm(color, "#9467bd"), alpha=alpha, label=label))

    # ── 装饰 ──
    def text(self, x, y, s, **k):
        self.series.append(dict(kind="text", x=x, y=y, s=s,
                                color=_norm(k.get("color"), "#333333")))

    def set_xlabel(self, s, **k): self.xlabel = s
    def set_ylabel(self, s, **k): self.ylabel = s
    def set_title(self, s, **k): self.title = s
    def grid(self, *a, **k): self.do_grid = True
    def legend(self, *a, **k): self._legend = True
    def set_yscale(self, *a, **k): pass     # 仅支持线性, 静默忽略

    # ── 渲染 ──
    def _limits(self):
        xs, ys = [], []
        for s in self.series:
            if s["kind"] in ("line", "fill"):
                xs += s.get("x", []); xs += s.get("x0", []) if "x0" in s else []
                if s["kind"] == "line":
                    ys += [v for v in s["y"] if np.isfinite(v)]
                else:
                    ys += [v for v in (s["y1"] + s["y2"]) if np.isfinite(v)]
        if not xs:
            return 0, 1, 0, 1
        x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
        px = 0.05 * (x1 - x0 or 1); py = 0.08 * (y1 - y0 or 1)
        return x0 - px, x1 + px, y0 - py, y1 + py

    def draw(self, d):
        X0, Y0, X1, Y1 = self.rect
        m_l, m_r, m_t, m_b = 78 * SS, 14 * SS, 30 * SS, 42 * SS
        if self._legend:
            m_r = 132 * SS              # 给图例留位置, 避免文字被裁
        px0, px1 = X0 + m_l, X1 - m_r
        py0, py1 = Y0 + m_t, Y1 - m_b
        x0, x1, y0, y1 = self._limits()
        f = _font(12); ft = _font(13); fl = _font(12)

        def sx(v): return px0 + (v - x0) / (x1 - x0 or 1) * (px1 - px0)
        def sy(v): return py1 - (v - y0) / (y1 - y0 or 1) * (py1 - py0)

        # 网格 + 刻度
        if self.do_grid:
            for i in range(6):
                v = x0 + (x1 - x0) * i / 5
                d.line([(sx(v), py0), (sx(v), py1)], fill="#dddddd", width=SS)
                v2 = y0 + (y1 - y0) * i / 5
                d.line([(px0, sy(v2)), (px1, sy(v2))], fill="#dddddd", width=SS)
        d.rectangle([px0, py0, px1, py1], outline="#333333", width=SS)
        for i in range(6):
            v = x0 + (x1 - x0) * i / 5
            d.line([(sx(v), py1), (sx(v), py1 + 4 * SS)], fill="#333333", width=SS)
            t = f"{v:.4g}"
            d.text((sx(v) - d.textlength(t, font=f) / 2, py1 + 6 * SS), t,
                   fill="#222222", font=f)
            v2 = y0 + (y1 - y0) * i / 5
            d.line([(px0 - 4 * SS, sy(v2)), (px0, sy(v2))], fill="#333333", width=SS)
            t = f"{v2:.4g}"
            d.text((px0 - 6 * SS - d.textlength(t, font=f), sy(v2) - 7 * SS), t,
                   fill="#222222", font=f)

        # 数据
        for s in self.series:
            if s["kind"] == "vline":
                d.line([(sx(s["x0"]), py0), (sx(s["x0"]), py1)],
                       fill=s["color"], width=max(SS, int(s["lw"] * SS)))
            elif s["kind"] == "text":
                d.text((sx(s["x"]) + 3 * SS, sy(s["y"])), s["s"],
                       fill=s["color"], font=f)
            elif s["kind"] == "fill":
                pairs = [(a, b1, b2) for a, b1, b2 in zip(s["x"], s["y1"], s["y2"])
                         if np.isfinite(b1) and np.isfinite(b2)]
                if len(pairs) < 3:          # 全部/几乎全为 NaN → 跳过, 不画垃圾面
                    continue
                pts = [(sx(a), sy(b1)) for a, b1, _ in pairs] + \
                      [(sx(a), sy(b2)) for a, _, b2 in pairs[::-1]]
                rgb = ImageColor.getrgb(_norm(s["color"], "#9467bd"))
                rgba = Image.new("RGBA", d._image.size, (0, 0, 0, 0))
                ImageDraw.Draw(rgba).polygon(pts, fill=rgb + (int(255 * s["alpha"]),))
                base = d._image.convert("RGBA")
                base.alpha_composite(rgba)
                d._image.paste(base.convert("RGB"))
                d._image = self.fig.img
            else:
                if s["ls"] in ("-", "--"):
                    pts = [(sx(a), sy(b)) for a, b in zip(s["x"], s["y"])
                           if np.isfinite(a) and np.isfinite(b)]
                    step = 1
                    for i in range(len(pts) - 1):
                        if s["ls"] == "--" and (i // 2) % 2 == 1:
                            continue
                        d.line([pts[i], pts[i + 1]], fill=s["color"],
                               width=max(SS, int(s["lw"] * SS)))
                if s["marker"]:
                    r = s["ms"] * SS / 2
                    for a, b in zip(s["x"], s["y"]):
                        if not (np.isfinite(a) and np.isfinite(b)):
                            continue
                        cx, cy = sx(a), sy(b)
                        if s["marker"] == "s":
                            d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=s["color"])
                        elif s["marker"] == "^":
                            d.polygon([(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)],
                                      fill=s["color"])
                        else:
                            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=s["color"])

        # 文字
        if self.title:
            d.text(((px0 + px1) / 2 - d.textlength(self.title, font=ft) / 2,
                    py0 - 24 * SS), self.title, fill="#111111", font=ft)
        if self.xlabel:
            d.text(((px0 + px1) / 2 - d.textlength(self.xlabel, font=fl) / 2,
                    Y1 - 22 * SS), self.xlabel, fill="#111111", font=fl)
        if self.ylabel:
            d.text((8 * SS, (py0 + py1) / 2), self.ylabel, fill="#111111", font=fl)
        if self._legend:
            ly = py0 + 6 * SS
            for s in self.series:
                if not s.get("label"):
                    continue
                lx = px1 - 12 * SS
                d.line([(lx - 26 * SS, ly + 6 * SS), (lx, ly + 6 * SS)],
                       fill=s["color"], width=2 * SS)
                t = s["label"]
                d.text((lx + 4 * SS, ly), t, fill="#111111", font=f)
                ly += 18 * SS


class Fig:
    def __init__(self, w_in, h_in, dpi):
        self.W = int(w_in * dpi * SS); self.H = int(h_in * dpi * SS)
        self.img = Image.new("RGB", (self.W, self.H), "white")
        self.d = ImageDraw.Draw(self.img)
        self.axes = []; self.supt = ""

    def add(self, rect):
        a = Axes(self, rect); self.axes.append(a); return a

    def suptitle(self, s, **k): self.supt = s

    def tight_layout(self, *a, **k):
        """布局已固定, 空实现 (接口兼容)."""
        return None

    def savefig(self, path, **kw):
        if self.supt:
            self.d.text((12 * SS, 6 * SS), self.supt, fill="#111111", font=_font(12))
        for a in self.axes:
            a.draw(self.d)
        out = self.img.resize((self.W // SS, self.H // SS), Image.LANCZOS)
        out.save(path)
        return path


def subplots(nrows=1, ncols=1, figsize=(11, 4.2), dpi=150, **kw):
    w, h = figsize
    fig = Fig(w, h, dpi)
    for j in range(ncols):
        gap = 0.06 * w
        cw = (w - gap * (ncols + 1)) / ncols
        x0 = gap + j * (cw + gap)
        # 像素坐标 (含超采样), y 轴留出标题空间
        fig.add((int(x0 * dpi * SS), int(0.13 * h * dpi * SS),
                 int((x0 + cw) * dpi * SS), int(0.97 * h * dpi * SS)))
    return fig, (fig.axes if ncols > 1 else fig.axes[0])


def close(fig=None):
    pass
