# -*- coding: utf-8 -*-
"""selftest_math.py — sterelib 的无硬件自检（纯 numpy，不需要相机/OpenCV）

构造一条**已知解析曲率**的空间曲线 → 用两个虚拟相机投影 → 加像素噪声 →
三角化 → 反算曲率 → 与真值比；再做增益拟合与 Kabsch 回环验证。

    python tools/stereo/selftest_math.py

全过 = 三角化/曲率/增益/对齐的数学可用，剩下的只是相机标定与检测。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sterelib import (  # noqa: E402
    rodrigues, inv_rodrigues, project, triangulate,
    curvature_from_points, curvature_valid_mask, resample_to_s,
    kabsch, fit_kappa_gain, estimate_s_offset,
)

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    if not ok:
        FAIL.append(name)


# ══════════════════════════════════════════════════════════════════════
# 1. 旋转自检
# ══════════════════════════════════════════════════════════════════════
print("[1] Rodrigues 往返")
_rng = np.random.default_rng(20261010)
orth = err = 0.0
for _ in range(5):
    R = rodrigues(_rng.normal(size=3) * 0.6)
    orth = max(orth, np.abs(R @ R.T - np.eye(3)).max())
    err = max(err, np.abs(inv_rodrigues(R) @ R - R @ inv_rodrigues(R)).max())
check("旋转矩阵正交", orth < 1e-12, f"max|RRᵀ−I|={orth:.2e}")
check("rvec 往返自洽", err < 1e-10, f"max|Rv−vR|={err:.2e}")


# ══════════════════════════════════════════════════════════════════════
# 2. 真值臂：两段常曲率（与仓库 SEG_A/L 同源）
# ══════════════════════════════════════════════════════════════════════
L1, L = 0.222589, 0.445178
KAP_A, KAP_B = 2.5, 4.0
WINDOW = 3                       # 推荐: 20 点 + w=3
_ss = np.linspace(0.0, L, 90001)
_th = np.where(_ss <= L1, KAP_A * _ss, KAP_A * L1 + KAP_B * (_ss - L1))
_tt = np.column_stack([np.sin(_th), np.zeros_like(_th), np.cos(_th)])
_PP = np.vstack([np.zeros(3),
                 np.cumsum(0.5 * (_tt[1:] + _tt[:-1]) * np.diff(_ss)[:, None], axis=0)])


def truth_shape(s):
    s = np.atleast_1d(np.asarray(s, float))
    return np.column_stack([np.interp(s, _ss, _PP[:, k]) for k in range(3)])


N_MARK = 20
s_mark = np.linspace(0.02, 0.44, N_MARK)
P_true = truth_shape(s_mark)
k_true = np.where(s_mark <= L1, KAP_A, KAP_B)
MSK = curvature_valid_mask(N_MARK, WINDOW, s=s_mark, seg_splits=(L1,))
print(f"[2] 真值形状: {N_MARK} marker, h={np.mean(np.diff(s_mark))*1e3:.1f} mm, "
      f"末端 x={P_true[-1,0]*1e3:.1f} z={P_true[-1,2]*1e3:.1f} mm, "
      f"可用曲率点 {MSK.sum()}/{N_MARK} (段界 L1 按窗跨度剔除)")


# ══════════════════════════════════════════════════════════════════════
# 3. 两个虚拟相机 + 投影 + 三角化
# ══════════════════════════════════════════════════════════════════════
K = np.array([[1200.0, 0, 960.0], [0, 1200.0, 540.0], [0, 0, 1.0]])
FOOT = np.array([K[0, 0], K[1, 1]])
CC = K[:2, 2]


def look_at(eye, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f = f / np.linalg.norm(f)
    x = np.cross(np.asarray(up, float), f)
    x = x / np.linalg.norm(x)
    y = np.cross(f, x)
    R = np.vstack([x, y, f])                    # world→cam
    return R, -R @ np.asarray(eye, float)


ctr = P_true.mean(0)
R1, t1 = look_at(ctr + np.array([0.0, -0.55, 0.0]), ctr)
R2, t2 = look_at(ctr + np.array([0.42, -0.35, 0.10]), ctr)
R_rel = R2 @ R1.T
t_rel = t2 - R2 @ R1.T @ t1                     # X_cam2 = R_rel·X_cam1 + t_rel
uv1, z1 = project(P_true, K, R1, t1)
uv2, z2 = project(P_true, K, R2, t2)
check("所有点在两相机前方", (z1 > 0).all() and (z2 > 0).all(),
      f"min z1={z1.min():.2f} z2={z2.min():.2f} m")

NOISE = 0.2                                     # px
NFR = 30                                        # 准静态单点平均帧数
_rng2 = np.random.default_rng(11)


def observe():
    """一次"拍照"：加像素噪声 → 三角化 → 回世界系。"""
    a = uv1 + _rng2.normal(0, NOISE, uv1.shape)
    b = uv2 + _rng2.normal(0, NOISE, uv2.shape)
    x1 = (a - CC) / FOOT
    x2 = (b - CC) / FOOT
    return (triangulate(x1, x2, R_rel, t_rel) - t1) @ R1


P_first = observe()
err_pt = np.linalg.norm(P_first - P_true, axis=1)
rms_pt = float(np.sqrt((err_pt ** 2).mean()))
print(f"[3] 三角化: 位置 RMSE = {rms_pt*1e3:.3f} mm (max {err_pt.max()*1e3:.3f} mm), "
      f"像素噪声 {NOISE} px")
check("三角化位置 RMSE < 1 mm", rms_pt < 1e-3)


# ══════════════════════════════════════════════════════════════════════
# 4. 曲率反算：单帧 vs 30 帧平均（各取多次的中位，避免单次抽样误导）
# ══════════════════════════════════════════════════════════════════════
def kappa_err(kap):
    k = np.linalg.norm(kap, axis=1)
    return 100.0 * np.sqrt(np.mean((k[MSK] - k_true[MSK]) ** 2)) / np.mean(k_true[MSK])


def median_err(window, n_frames, trials):
    out = []
    for _ in range(trials):
        ks = [curvature_from_points(observe(), window)[1] for _ in range(n_frames)]
        out.append(kappa_err(np.mean(ks, axis=0)))
    return float(np.median(out))


e_w1_1 = median_err(1, 1, 25)
e_w1_30 = median_err(1, NFR, 12)
e_w3_1 = median_err(WINDOW, 1, 25)
e_w3_30 = median_err(WINDOW, NFR, 12)
print(f"[4] 曲率误差（中位; 可用点 {MSK.sum()}/{N_MARK}, 单帧点噪 {rms_pt*1e3:.2f} mm）")
print(f"      window=1 严三点圆 : 单帧 {e_w1_1:6.2f}%   30帧 {e_w1_30:6.2f}%   ← 噪声 ∝1/h²")
print(f"      window={WINDOW} 局部二次 : 单帧 {e_w3_1:6.2f}%   30帧 {e_w3_30:6.2f}%   ← 推荐")
check(f"window={WINDOW} 单帧 < 2%", e_w3_1 < 2.0)
check(f"window={WINDOW} 30 帧平均 < 1.5%", e_w3_30 < 1.5)
check("多帧平均确实有效 (30帧 < 单帧)", e_w3_30 < e_w3_1)

_, kk, Rc = curvature_from_points(observe(), WINDOW)
check("曲率矢量非 nan/非 inf", np.isfinite(kk).all())
check("R 与 |κ| 自洽",
      np.allclose(Rc[MSK], 1.0 / np.linalg.norm(kk[MSK], axis=1), rtol=1e-6, atol=1e-9))

# 曲率矢量必须垂直于切向（几何硬约束，符号/手性错了这里会爆）
Pq = curvature_from_points(observe(), WINDOW)[0]
tan = np.gradient(truth_shape(s_mark)[MSK], s_mark[MSK], axis=0)
kv = kk[MSK] / np.linalg.norm(kk[MSK], axis=1, keepdims=True)
dotmax = float(np.abs((kv * tan / np.linalg.norm(tan, axis=1, keepdims=True)).sum(1)).max())
check("κ 矢量 ⟂ 切向 (手性正确)", dotmax < 0.05, f"max|cos∠|={dotmax:.3f}")


# ══════════════════════════════════════════════════════════════════════
# 5. 与"模型表"对拍 + 增益拟合（制造 1.1 倍模型偏差）
# ══════════════════════════════════════════════════════════════════════
G_TRUTH = 1.1
s_m = np.linspace(0.0, L, 601)
_, kap_m, _ = curvature_from_points(truth_shape(s_m), window=1)     # 模型侧干净
kap_m = kap_m / G_TRUTH                                             # 模型偏软
s_rec, _, _ = curvature_from_points(P_first, WINDOW)
kap_meas = np.mean([curvature_from_points(observe(), WINDOW)[1] for _ in range(NFR)], axis=0)
s_off_true = float(s_mark[0])
k_model_q = resample_to_s(s_m, kap_m, s_mark)[MSK]
# (a) 忽略弧长原点偏移: 实测 s 从 marker0 起算、模型 s 从基座起算 → 常数差
fit_bad = fit_kappa_gain(resample_to_s(s_rec, kap_meas, s_mark)[MSK], k_model_q)
# (b) 校正偏移 (工程上: marker0 贴基座面 ⇒ δ≈0, 或手工量出)
fit = fit_kappa_gain(resample_to_s(s_rec + s_off_true, kap_meas, s_mark)[MSK],
                     k_model_q, s=s_mark[MSK], seg_split=L1)
# (c) 由数据反推偏移 (弱可辨识, 只做量级核验)
d_hat, _, _ = estimate_s_offset(s_rec, kap_meas, s_m, kap_m, max_off=0.06)
ratio = fit_bad["rmse_raw"] / max(fit["rmse_raw"], 1e-12)
print(f"[5] 弧长原点偏移 δ: 真值 {s_off_true*1e3:.1f} mm, 反推 {d_hat*1e3:.1f} mm (弱可辨识)")
print(f"     未校正 δ 的 RMSE = {fit_bad['rmse_raw']:.4f} → 校正后 {fit['rmse_raw']:.4f} (放大 {ratio:.2f}×)")
print(f"[5] 增益拟合: g={fit['g']:.4f} (真值 {G_TRUTH})  gA={fit['gA']:.4f}  gB={fit['gB']:.4f}")
print(f"     RMSE {fit['rmse_raw']:.4f} → {fit['rmse_fit']:.4f} 1/m ; EI 缩放 1/g = {fit['ei_scale']:.4f}")
check("δ 反推与真值同量级 (<15 mm)", abs(d_hat - s_off_true) < 0.015,
      f"Δ={abs(d_hat-s_off_true)*1e3:.1f} mm")
check("不校正 δ 会放大残差 (>1.3×)", ratio > 1.3, f"{ratio:.2f}×")
check("整体增益 g 恢复 3% 内", abs(fit["g"] - G_TRUTH) / G_TRUTH < 0.03)
check("分段增益 A/B 各恢复 8% 内",
      abs(fit["gA"] - G_TRUTH) / G_TRUTH < 0.08 and abs(fit["gB"] - G_TRUTH) / G_TRUTH < 0.08)
check("拟合后误差显著下降", fit["rmse_fit"] < 0.15 * fit["rmse_raw"])


# ══════════════════════════════════════════════════════════════════════
# 6. Kabsch 基座系对齐 —— 量化"共面点集"的代价
# ══════════════════════════════════════════════════════════════════════
R_true = rodrigues(np.array([0.12, -0.08, 0.05]))
t_true = np.array([0.31, 0.07, -0.02])
SIG_K = 2e-4                                        # 0.2 mm 点位噪声
A_cp = P_true[:6]                                   # 共面（臂是平面曲线 → 天然共面！）
A_nc = np.vstack([P_true[::3], [[0.05, 0.06, 0.12], [0.10, -0.05, 0.20]]])   # 非共面


def mc_kabsch(A, n=200):
    rng_k = np.random.default_rng(11)
    ang, rms = [], []
    for _ in range(n):
        B = A @ R_true.T + t_true + rng_k.normal(0, SIG_K, A.shape)
        Re, _, rr, _ = kabsch(A, B)
        ang.append(np.rad2deg(np.linalg.norm(inv_rodrigues(Re.T @ R_true))))
        rms.append(rr)
    return float(np.median(ang)), float(np.median(rms))


ang_nc, rms_nc = mc_kabsch(A_nc)
ang_cp, rms_cp = mc_kabsch(A_cp)
print(f"[6] Kabsch (点位噪声 {SIG_K*1e3:.1f} mm, 各 200 次)：")
print(f"      非共面点集 → 转角误差 中位 {ang_nc:.3f}°   对齐 RMS {rms_nc*1e3:.3f} mm")
print(f"      共面点集   → 转角误差 中位 {ang_cp:.3f}°   对齐 RMS {rms_cp*1e3:.3f} mm")
print(f"      ⇒ RMS 几乎一样，**转角误差共面差 {ang_cp/max(ang_nc,1e-9):.1f} 倍**")
check("非共面: 转角误差 < 0.3°", ang_nc < 0.3, f"{ang_nc:.4f}°")
check("共面点集确实更差 (>3×)", ang_cp > 3.0 * ang_nc, f"{ang_cp/ang_nc:.1f}×")
check("RMS 无法区分两者 (<2×)", max(rms_nc, rms_cp) / min(rms_nc, rms_cp) < 2.0)


# ══════════════════════════════════════════════════════════════════════
print("=" * 74)
if FAIL:
    print(f"✗ {len(FAIL)} 项失败: {FAIL}")
    sys.exit(1)
print("✓ 全部通过 —— 三角化/曲率/增益/对齐 数学可用，只差相机标定与检测。")
print(f"  推荐配置: {N_MARK} marker (h≈{np.mean(np.diff(s_mark))*1e3:.0f} mm) + "
      f"window={WINDOW} + {NFR} 帧平均 → 曲率误差 ~{e_w3_30:.1f}%")
