# -*- coding: utf-8 -*-
"""kappa_from_points.py — 从**三维坐标**重算曲率（替代坏掉的 kx/ky 列）

诊断背景：CSV 里 `kx` 与 `ky` 恒等于 1.000、曲率值成对重复、`净方向`只有 2 个值、
曲率积分比真实弯角小 5–11 倍 ⇒ 那几列不能用。本脚本只吃**节点坐标**重算。

用法
----
    python tools/kappa_from_points.py selftest            # 自检：已知圆弧必须还原
    python tools/kappa_from_points.py csv 数据.csv        # 从 CSV 里找 x*/y*/z* 三列
    python tools/kappa_from_points.py xyz 点.csv          # 三列 x,y,z（单位 mm）

公式
----
    κ_vec = (p' × p'') / |p'|³          （空间曲线曲率矢量）
    θ     = ∫ |κ| ds                     （总弯曲角，rad）
    R     = 1/|κ|                        （曲率半径）

⚠ 二阶差分对噪声极敏感（噪声 ∝ 1/Δs²）。实测点建议：
  1) 节点间距 ≥ 20 mm；2) 坐标噪声 < 0.5 mm；3) 或多给几个点做平滑
"""
from __future__ import annotations

import sys
import numpy as np


def arclen(P):
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def curvature_from_points(P, s=None):
    """三点外接圆求 |κ|（对圆弧精确），再用 d1×d2 定方向。

    返回 (kvec (n,3), |k| (n,), s (n,))。
    |κ|_i = 4A / (a·b·c)，A 为 i-1,i,i+1 三点三角形面积。
    端点用邻值补齐（外接圆只覆盖内部 n-2 点）。
    """
    P = np.asarray(P, float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("需要 (n,3) 的点坐标")
    if s is None:
        s = arclen(P)
    n = len(P)
    kap = np.zeros(n)
    for i in range(1, n - 1):
        a = P[i] - P[i - 1]
        b = P[i + 1] - P[i]
        c = P[i + 1] - P[i - 1]
        la, lb, lc = (np.linalg.norm(v) for v in (a, b, c))
        den = la * lb * lc
        if den > 1e-18:
            kap[i] = 2.0 * np.linalg.norm(np.cross(a, c)) / den   # = 4A/(abc)
    if n > 2:
        kap[0] = kap[1]
        kap[-1] = kap[-2]
    # 方向：d1 × d2
    d1 = np.gradient(P, s, axis=0)
    d2 = np.gradient(d1, s, axis=0)
    ax = np.cross(d1, d2)
    nn = np.linalg.norm(ax, axis=1)
    dirs = np.divide(ax, np.maximum(nn, 1e-18)[:, None])
    return dirs * kap[:, None], kap, s


def total_bend(kap, s):
    """梯形积分 θ = ∫|κ| ds（rad）。"""
    if len(s) < 2:
        return 0.0
    return float(np.sum(0.5 * (kap[1:] + kap[:-1]) * np.diff(s)))


def _arc(theta_deg, R_mm=141.7, n=61):
    th = np.radians(theta_deg)
    s = np.linspace(0, R_mm * th, n)
    phi = s / R_mm
    return np.stack([R_mm * (1 - np.cos(phi)), np.zeros_like(phi),
                     R_mm * np.sin(phi)], axis=1)


def selftest():
    ok = True
    for th in (30, 90, 180, 270):
        P = _arc(th)
        kvec, kap, s = curvature_from_points(P)
        R = 1.0 / np.median(kap[1:-1])
        theta = np.degrees(total_bend(kap, s))
        e_c = abs(R - 141.7)
        e_t = abs(theta - th)
        flag = "PASS" if (e_c < 0.3 and e_t < 0.5) else "FAIL"
        ok &= flag == "PASS"
        print(f"  θ={th:3d}°  还原 R={R:8.2f} mm (真 141.70, 差 {e_c:.3f})  "
              f"θ={theta:7.2f}° (差 {e_t:.3f})  {flag}")
    # 直线
    Lz = np.stack([np.zeros(50), np.zeros(50), np.linspace(0, 400, 50)], axis=1)
    _, kap2, s2 = curvature_from_points(Lz)
    b2 = np.degrees(total_bend(kap2, s2))
    flag = "PASS" if b2 < 0.1 else "FAIL"
    ok &= flag == "PASS"
    print(f"  直线: θ={b2:.4f}° (应 0)  {flag}")
    # 平面外（绕 +x 弯，即 y-z 平面）
    th = np.radians(120.0)
    s = np.linspace(0, 141.7 * th, 61)
    phi = s / 141.7
    Pyz = np.stack([np.zeros_like(phi), 141.7 * (1 - np.cos(phi)),
                    141.7 * np.sin(phi)], axis=1)
    _, kap3, s3 = curvature_from_points(Pyz)
    b3 = np.degrees(total_bend(kap3, s3))
    flag = "PASS" if abs(b3 - 120) < 0.5 else "FAIL"
    ok &= flag == "PASS"
    print(f"  y-z 平面 120°: θ={b3:.2f}°  {flag}")
    print("\n自检结果:", "全部 PASS" if ok else "有 FAIL")
    return 0 if ok else 1


def from_csv(path):
    import csv
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    if not rows:
        raise SystemExit("CSV 为空")
    keys = list(rows[0].keys())
    # 找 x/y/z 三列：优先 x1/x2… 其次 x 单列
    def find(prefix):
        c = [k for k in keys if k.strip().lower().startswith(prefix)]
        return c
    xs, ys, zs = find("x"), find("y"), find("z")
    if not (xs and ys and zs):
        raise SystemExit(f"没找到 x/y/z 列。现有列前 20 个: {keys[:20]}")
    n = min(len(xs), len(ys), len(zs))
    print(f"用列: {xs[:n]} / {ys[:n]} / {zs[:n]}  共 {n} 点")
    out = []
    for r in rows:
        P = np.array([[float(r[xs[i]]), float(r[ys[i]]), float(r[zs[i]])]
                      for i in range(n)])
        if not np.isfinite(P).all():
            continue
        kv, kap, s = curvature_from_points(P)
        out.append((np.degrees(total_bend(kap, s)), 1.0 / max(np.median(kap[1:-1]), 1e-12)))
    for i, (th, R) in enumerate(out):
        print(f"  样本{i+1}: 总弯角 {th:7.2f}°   中位曲率半径 {R:8.2f} mm")
    return 0


def main(argv):
    if len(argv) < 2 or argv[1] == "selftest":
        return selftest()
    if argv[1] == "csv":
        return from_csv(argv[2])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
