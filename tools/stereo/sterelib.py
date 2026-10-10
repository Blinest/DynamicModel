# -*- coding: utf-8 -*-
"""sterelib.py — 双目 3D 重建 / 曲率评价 的纯 numpy 数学核心
=====================================================================================
只依赖 numpy（不依赖 cv2 / scipy），便于在无 OpenCV 的机器上跑自检与离线评价。
所有 cv2 相关（棋盘标定、ArUco 检测、去畸变）都在调用方脚本里，且用 try-import 隔离。

单位约定（与 `ik_table` 的表一致）
--------------------------------
    长度 m   角 rad   曲率 1/m
    模型表: kS (N,601) 弧长 s [m], 均匀 ds = L/600;  kK (N,601,3) 曲率矢量 [1/m]
    末端 tip 存 p_tip (N,3) [m], 世界系 Z 朝上、基座原点、无外力时臂沿 +Z

坐标/相机约定
------------
    相机 1 为本体坐标系: X_cam2 = R @ X_cam1 + t   (与 cv2.stereoCalibrate 一致)
    传入 `triangulate` 的必须是**归一化像坐标** (x=(u-cx)/fx, y=(v-cy)/fy)，
    即已经 cv2.undistortPoints 过、且除了内参；这样 DLT 不必再管内参。

曲率定义
--------
    取离散点列的连续三点做外接圆，曲率矢量 kappa = kappa * n_hat，方向指向曲率中心
    (= 材料坐标下的 dt/ds = kappa n)，大小 1/R。四点以上时逐点滑窗，端点复制相邻值。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "rodrigues", "inv_rodrigues", "project", "triangulate",
    "curvature_from_points", "curvature_valid_mask", "resample_to_s",
    "kabsch", "fit_kappa_gain", "estimate_s_offset",
    "load_points_csv", "save_points_csv",
]


# ══════════════════════════════════════════════════════════════════════
# 1. 旋转 / 投影
# ══════════════════════════════════════════════════════════════════════
def rodrigues(rvec):
    """旋转矢量 (3,) → 3x3 旋转矩阵 (Rodrigues, 纯 numpy)。"""
    r = np.asarray(rvec, float).reshape(3)
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def inv_rodrigues(R):
    """3x3 旋转矩阵 → 旋转矢量 (3,)。"""
    R = np.asarray(R, float)
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    th = float(np.arccos(c))
    if th < 1e-12:
        return np.zeros(3)
    if abs(np.pi - th) < 1e-6:                      # th≈π: 由 R+I 的非对角提取轴
        A = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.clip(np.diag(A), 0.0, None))
        j = int(np.argmax(k))
        k = A[:, j] / max(k[j], 1e-12)
        k = k / max(np.linalg.norm(k), 1e-12)
        return k * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2.0 * np.sin(th)) * th


def project(X, K, R, t):
    """世界点投影到像素。

    X:(N,3) 世界系 [m]  K:(3,3)  R:(3,3)  t:(3,)
    返回 (uv:(N,2) 像素, z:(N,) 光轴深度 [m]，z<=0 表示点在相机后方)
    """
    X = np.atleast_2d(np.asarray(X, float))
    Xc = X @ np.asarray(R, float).T + np.asarray(t, float).reshape(1, 3)
    z = Xc[:, 2]
    zz = np.where(np.abs(z) < 1e-12, 1e-12, z)
    uv = (Xc @ np.asarray(K, float).T)[:, :2] / zz[:, None]
    return uv, z


def triangulate(uv1, uv2, R, t):
    """归一化像坐标 → 相机 1 系 3D 点 (DLT + SVD)。

    uv1,uv2 : (N,2) 归一化像坐标（已去畸变、已除内参）
    R,t     : X_cam2 = R @ X_cam1 + t
    返回    : (N,3) 相机 1 系坐标 [m]（与输入同单位，默认 m）
    """
    uv1 = np.atleast_2d(np.asarray(uv1, float))
    uv2 = np.atleast_2d(np.asarray(uv2, float))
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([np.asarray(R, float), np.asarray(t, float).reshape(3, 1)])
    out = np.empty((len(uv1), 3))
    for i in range(len(uv1)):
        a, b = uv1[i], uv2[i]
        A = np.vstack([a[0] * P1[2] - P1[0],
                       a[1] * P1[2] - P1[1],
                       b[0] * P2[2] - P2[0],
                       b[1] * P2[2] - P2[1]])
        _, _, Vt = np.linalg.svd(A)
        Xh = Vt[-1]
        out[i] = Xh[:3] / Xh[3]
    return out


# ══════════════════════════════════════════════════════════════════════
# 2. 离散点 → 曲率(s)
# ══════════════════════════════════════════════════════════════════════
def _kappa_3pt(A, B, C):
    """三点外接圆 → (kappa_vec, R)。严格精确（对圆弧无离散偏差），但噪声放大 ∝ 1/h²。"""
    a = C - B
    b = A - B
    n = np.cross(a, b)
    nn = float(np.linalg.norm(n))
    if nn < 1e-15:                                       # 共线 → 直
        return np.zeros(3), np.inf
    e3 = n / nn
    t = C - A
    nt = float(np.linalg.norm(t))
    if nt < 1e-15:
        return np.zeros(3), np.inf
    e1 = t / nt
    e2 = np.cross(e3, e1)
    ne2 = float(np.linalg.norm(e2))
    if ne2 < 1e-15:
        return np.zeros(3), np.inf
    e2 = e2 / ne2
    ua, va = float((A - B) @ e1), float((A - B) @ e2)
    uc, vc = float((C - B) @ e1), float((C - B) @ e2)
    # 圆过 (0,0),(ua,va),(uc,vc) ⇒ u²+v² = 2(Uc·u + Vc·v)
    M = np.array([[ua, va], [uc, vc]])
    rhs = 0.5 * np.array([ua * ua + va * va, uc * uc + vc * vc])
    if abs(np.linalg.det(M)) < 1e-18:
        return np.zeros(3), np.inf
    Uc, Vc = np.linalg.solve(M, rhs)
    R = float(np.hypot(Uc, Vc))
    if R < 1e-12:
        return np.zeros(3), np.inf
    return (Uc * e1 + Vc * e2) / (R * R), R               # 指向中心, |κ| = 1/R


def curvature_from_points(P, window=3, bias_correct=True):
    """有序 3D 点列 → (s, kappa_vec, R_curv)。

    P      : (m,3) 沿臂从基座到末端排列的点 [m]
    window : 1 = 严三点外接圆（精确、但噪声放大 ∝ 1/h²）
             k>=2 = 取 2k+1 点做局部二次拟合（噪声被平均，误差 ∝ 1/h²/√k）
    bias_correct : 只对 window>=2 生效。局部二次拟合圆弧会**系统性高估** κ，
             相对偏差 = 5U²/(28R²)（U=窗半宽）。该式解析可解，迭代 2 次消掉。

    返回
      s        : (m,) 累积弦长 [m]（弧长近似，点密时误差 O(h³)）
      kappa_vec: (m,3) 曲率矢量 [1/m]，指向曲率中心
      R_curv   : (m,) 曲率半径 [m]，直线处 inf

    ⚠ 精度权衡（本仓实测, L=445 mm, 点噪声 σ=0.2 px @0.55 m ≈ 0.20 mm,
       统计 = 窗完整且**窗内不含段界 L1** 的点, 25 次中位）：
       点数 h(mm) | w=2 单帧/30帧(可用点) | w=3 单帧/30帧(可用点)
         12  38.2 |   1.09% / 0.69% ( 4)   |     —  /   —   ( 0) ← 点太少, 无点可用
         20  22.1 |   2.26% / 0.56% (12)   |  1.17% / 0.69% ( 8) ← 推荐
         32  13.5 |   7.08% / 1.66% (24)   |  2.97% / 1.01% (20)
      单帧噪声 ∝ 1/h²（点越密越差），多帧平均 ∝ 1/√N 抵消之。

      ⇒ **推荐：20 个 marker（h≈22 mm）+ window=3 + 单点 30 帧平均 → κ 误差 ~0.7%**，
        且两段各有 ~4 个可用点。
      ⚠ 段界 L1 处的剔除必须按**窗跨度**（window·h），不能按固定带宽：不剔时偏差
        ~4.4% 且**多帧平均消不掉**（是偏差不是噪声）。用 `curvature_valid_mask`。
    """
    P = np.asarray(P, float)
    m = len(P)
    if m < 3:
        raise ValueError("至少需要 3 个点才能定曲率")
    window = int(max(1, window))
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    kap = np.zeros((m, 3))
    Rc = np.full(m, np.inf)

    if window == 1:
        for i in range(1, m - 1):
            kv, R = _kappa_3pt(P[i - 1], P[i], P[i + 1])
            kap[i], Rc[i] = kv, R
    else:
        for i in range(m):
            lo, hi = max(0, i - window), min(m - 1, i + window)
            Q = P[lo:hi + 1]
            if len(Q) < 3:
                kv, R = _kappa_3pt(P[max(0, i - 1)], P[i], P[min(m - 1, i + 1)])
                kap[i], Rc[i] = kv, R
                continue
            # 局部平面: 法向 e3 取相邻叉积之和（与严三点同手性: 入弦 × 出弦）
            nrm = np.zeros(3)
            for j in range(1, len(Q) - 1):
                nrm = nrm + np.cross(Q[j] - Q[j - 1], Q[j + 1] - Q[j])
            nn = float(np.linalg.norm(nrm))
            if nn < 1e-15:                               # 退化（直线/拐点）→ 严三点
                kv, R = _kappa_3pt(P[max(0, i - 1)], P[i], P[min(m - 1, i + 1)])
                kap[i], Rc[i] = kv, R
                continue
            e3 = nrm / nn
            c = Q[-1] - Q[0]
            nc = float(np.linalg.norm(c))
            if nc < 1e-15:
                continue
            e1 = c / nc
            e2 = np.cross(e3, e1)
            e2 = e2 / float(np.linalg.norm(e2))
            u = (Q - P[i]) @ e1
            v = (Q - P[i]) @ e2
            cfs = np.polyfit(u, v, 2)                    # v = c0 + c1 u + c2 u²
            c1, c2 = float(cfs[1]), float(cfs[0])
            U = float(max(abs(u[0]), abs(u[-1])))
            # κ = |v''|/(1+v'²)^{3/2}；矢量方向 = (-v',1)/√(1+v'²)
            k_mag = abs(2.0 * c2) / (1.0 + c1 ** 2) ** 1.5
            if bias_correct and k_mag > 1e-9:
                for _ in range(2):                       # 解 κ_c = κ_est/(1+5U²κ_c²/28)
                    k_mag = k_mag / (1.0 + 5.0 * U ** 2 * k_mag ** 2 / 28.0)
            n2 = np.array([-c1, 1.0]) / np.sqrt(1.0 + c1 ** 2)
            kap[i] = k_mag * (n2[0] * e1 + n2[1] * e2)
            Rc[i] = np.inf if k_mag < 1e-12 else 1.0 / k_mag

    if window == 1:                                      # 端点复制相邻
        kap[0], kap[-1] = kap[1], kap[-2]
        Rc[0], Rc[-1] = Rc[1], Rc[-2]
    else:
        kap[0], kap[-1] = kap[1], kap[-2]
        Rc[0], Rc[-1] = Rc[1], Rc[-2]
    return s, kap, Rc


def curvature_valid_mask(m, window=2, s=None, seg_splits=(), band=None):
    """哪些点的曲率可信：**窗完整** ∩ **窗内不含段界**。

    段界处 κ 有物理跃变，只要滑窗跨过它，该点的拟合就被污染 —— 这与"点到段界的
    距离"无关，必须按**窗跨度**剔除。实测：不剔时偏差 ~4.4%，且**多帧平均也消不掉**
    （那是偏差不是噪声）；正确剔除后 30 帧平均可到 0.7%。

    m          : 点数
    window     : 与 curvature_from_points 一致
    s          : (m,) 弧长；给出时按窗跨度剔除段界附近
    seg_splits : 段界弧长列表 [m]（如 (L1,)）
    band       : 覆盖默认的剔除半宽 [m]；None → 自动取 window·h（h=中位点距）
    返回 bool 掩码 (m,)
    """
    m = int(m)
    ok = np.ones(m, bool)
    if window >= 1:
        ok[:window] = False
        ok[max(0, m - window):] = False
    if s is not None and len(np.atleast_1d(seg_splits)) and m >= 2:
        s = np.asarray(s, float)
        h = float(np.median(np.diff(s)))
        b = (window * h) if band is None else float(band)
        for sp in np.atleast_1d(np.asarray(seg_splits, float)):
            ok &= np.abs(s - sp) > b
    elif s is not None and band:
        s = np.asarray(s, float)
        for sp in np.atleast_1d(np.asarray(seg_splits, float)):
            ok &= np.abs(s - sp) > float(band)
    return ok


def estimate_s_offset(s_meas, kap_meas, s_model, kap_model, max_off=0.05, n=201):
    """实测弧长与模型弧长的**原点偏移** δ（profile: 每个 δ 内先最小二乘定增益 g）。

    实测 s 从"第一个 marker"起算，模型 s 从基座起算 —— 两者差一个未知常数 δ。
    ⚠ 弱可辨识：**同一常曲率段内** κ 的模是常数，只有 κ 的**方向**随 s 转，
    所以 δ 的可辨识性来自方向旋转率（≈|κ|）。若两段都近直，δ 基本测不出来 ——
    工程上应直接把 marker0 贴在基座夹持面上（δ≈0）或手工量出 δ 输入。

    返回 (delta, g, rmse)：rmse 为对齐后的曲率矢量 RMSE [1/m]。
    """
    s_meas = np.asarray(s_meas, float)
    kap_meas = np.asarray(kap_meas, float)
    s_model = np.asarray(s_model, float)
    best = None
    for d in np.linspace(-abs(max_off), abs(max_off), int(n)):
        sq = s_meas + d
        m = (sq >= s_model[0]) & (sq <= s_model[-1])
        if m.sum() < 4:
            continue
        km = resample_to_s(s_model, kap_model, sq[m])
        kr = kap_meas[m]
        den = float((km * km).sum())
        if den <= 1e-30:
            continue
        g = float((kr * km).sum()) / den
        r = float(np.sqrt(np.mean(((kr - g * km) ** 2).sum(axis=1))))
        if best is None or r < best[2]:
            best = (float(d), g, r)
    return best if best is not None else (0.0, float("nan"), float("nan"))


def resample_to_s(s_src, arr, s_q):
    """把 (s_src, arr) 重采样到 s_q（arr 可为 (m,) 或 (m,k)）。"""
    s_src = np.asarray(s_src, float)
    arr = np.asarray(arr, float)
    if s_src.shape != (len(arr),):
        raise ValueError("s_src 长度须等于 arr 第一维")
    if arr.ndim == 1:
        return np.interp(s_q, s_src, arr)
    return np.column_stack([np.interp(s_q, s_src, arr[:, k]) for k in range(arr.shape[1])])


# ══════════════════════════════════════════════════════════════════════
# 3. 基座系对齐 (Kabsch) 与增益拟合
# ══════════════════════════════════════════════════════════════════════
def kabsch(A, B, W=None):
    """求 R,t 使 R@A + t ≈ B（最小二乘，带反射修正）。

    A,B : (N,3) N>=3（不共线）
    W   : (N,) 可选权重
    返回 (R:(3,3), t:(3,), rms:float, resid:(N,))
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if len(A) < 3:
        raise ValueError("Kabsch 至少 3 点")
    w = np.ones(len(A)) if W is None else np.asarray(W, float)
    w = w / w.sum()
    ca = (A * w[:, None]).sum(0)
    cb = (B * w[:, None]).sum(0)
    H = (A - ca).T @ ((B - cb) * w[:, None])
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    resid = np.linalg.norm(A @ R.T + t - B, axis=1)
    return R, t, float(np.sqrt((w * resid ** 2).sum())), resid


def fit_kappa_gain(kappa_real, kappa_model, s=None, seg_split=None):
    """标量增益 g: kappa_real ≈ g · kappa_model（最小二乘，逐分量）。

    参数
      kappa_real, kappa_model : (m,3) 同一 s 网格上的曲率矢量 [1/m]
      s                       : (m,) 弧长 [m]，给定时算分段增益
      seg_split               : float，段界弧长 L1 [m]；给定时输出 gA/gB

    返回 dict:
      g, rmse_raw, rmse_fit (均为 1/m),
      gA, gB (无 seg_split 时为 nan),
      ei_scale = 1/g  ← 直接乘到模型的 EI0/EI 上（kappa≈M/EI，κ 偏大说明实际更软）
      mean_kappa, R_mean [m]
    """
    kr = np.asarray(kappa_real, float)
    km = np.asarray(kappa_model, float)
    num = float((kr * km).sum())
    den = float((km * km).sum())
    g = num / den if den > 1e-30 else float("nan")
    rmse_raw = float(np.sqrt(np.mean(((kr - km) ** 2).sum(axis=1))))
    rmse_fit = float(np.sqrt(np.mean(((kr - g * km) ** 2).sum(axis=1))))
    gA = gB = float("nan")
    if (s is not None) and (seg_split is not None):
        s = np.asarray(s, float)
        mA = s <= seg_split
        mB = ~mA
        for tag, msk in (("A", mA), ("B", mB)):
            d = float((km[msk] ** 2).sum())
            v = float((kr[msk] * km[msk]).sum()) / d if d > 1e-30 else float("nan")
            if tag == "A":
                gA = v
            else:
                gB = v
    mk = float(np.linalg.norm(km, axis=1).mean())
    return {
        "g": g, "gA": gA, "gB": gB,
        "ei_scale": (1.0 / g if np.isfinite(g) and abs(g) > 1e-30 else float("nan")),
        "rmse_raw": rmse_raw, "rmse_fit": rmse_fit,
        "mean_kappa": mk, "R_mean": (1.0 / mk if mk > 0 else float("inf")),
    }


# ══════════════════════════════════════════════════════════════════════
# 4. 点列 CSV 读写（长表: 每行一个 marker）
# ══════════════════════════════════════════════════════════════════════
POINT_COLS = ["frame", "t", "marker_id", "x_mm", "y_mm", "z_mm", "n_cam"]


def save_points_csv(path, rows, frame=None, t=None):
    """写长表 CSV。rows: 可迭代 (marker_id, x_mm, y_mm, z_mm[, n_cam])。"""
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(POINT_COLS)
        for i, r in enumerate(rows):
            r = list(r)
            fid = r[4] if len(r) > 4 else (i if frame is None else frame)
            tt = r[5] if len(r) > 5 else (0.0 if t is None else t)
            w.writerow([fid, tt, r[0], f"{r[1]:.6f}", f"{r[2]:.6f}", f"{r[3]:.6f}",
                        (r[6] if len(r) > 6 else 2)])
    return path


def load_points_csv(path, frame=None):
    """读长表 CSV → (marker_id, X:(n,3) [mm])；frame=None 取全部行。"""
    import csv
    ids, pts = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            if frame is not None and int(float(row["frame"])) != frame:
                continue
            ids.append(int(float(row["marker_id"])))
            pts.append([float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])])
    return np.array(ids, int), np.array(pts, float)
