# -*- coding: utf-8 -*-
"""casadi_fit.py — 用 CasADi + IPOPT 反求 20 段曲率场（可行性验证）

为什么值得换 CasADi
------------------
现在 `kappa_fit.py` 用「自写 Gauss-Newton + 数值雅可比」：
  · 20 个未知量 ⇒ 每步要 20 次前向积分，梯度是差分（有截断误差）
  · 正则化权重靠手调，无约束处理（κ 只靠 clip）
CasADi 的做法：
  · 前向模型用**符号表达式 + 固定步 RK4** 展开，**自动微分**给精确梯度
  · 整个问题写成 NLP，交给 IPOPT（内点法），约束（κ 上下界、平滑项）自然处理

本脚本做三件事
  1. 用已知 κ 场生成"实测"中线（加 1 mm 噪声）
  2. 用 CasADi 反求 20 个 κ_i
  3. 与真值对比，报残差与耗时

依赖：pip install casadi
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
import time

import numpy as np
import casadi as ca

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NSEG = 20
SEG_LEN = 0.02          # m
N_SUB = 4               # 每段 RK4 子步
LAM_SMOOTH = 1e-3
K_MAX = 30.0            # |κ| 上界 1/m


def truth_kappa():
    """真实 κ 场：根部集中（模拟摩擦衰减 + 6丝/3丝），总弯角 90°。"""
    s = (np.arange(NSEG) + 0.5) * SEG_LEN
    w = 1.0 / (1.0 + 10.0 * s) * np.where(s <= 0.2, 1.0, 0.5)
    w /= w.sum()
    return np.radians(90.0) * w / SEG_LEN


def rollout_numeric(kvec):
    """数值前向：返回各段末端的 (x, z) 与切向角（m, m, rad）。"""
    p = np.zeros(2)
    th = 0.0
    out = []
    for k in kvec:
        ds = SEG_LEN / N_SUB
        for _ in range(N_SUB):
            p = p + np.array([np.sin(th), np.cos(th)]) * ds
            th += k * ds
        out.append((p[0], p[1], th))
    return np.array(out)


def rollout_casadi(k, n_per_seg=N_SUB):
    """CasADi 符号前向（RK4），返回 (x, z, theta) 的 MX 列表（各段末端）。"""
    px = ca.MX(0.0)
    pz = ca.MX(0.0)
    th = ca.MX(0.0)

    def f(x, z, t, kap):
        return ca.vertcat(ca.sin(t), ca.cos(t), kap)   # dx, dz, dtheta

    outs = []
    for i in range(NSEG):
        kap = k[i]
        ds = SEG_LEN / n_per_seg
        for _ in range(n_per_seg):
            k1 = f(px, pz, th, kap)
            k2 = f(px + 0.5 * ds * k1[0], pz + 0.5 * ds * k1[1],
                   th + 0.5 * ds * k1[2], kap)
            k3 = f(px + 0.5 * ds * k2[0], pz + 0.5 * ds * k2[1],
                   th + 0.5 * ds * k2[2], kap)
            k4 = f(px + ds * k3[0], pz + ds * k3[1], th + ds * k3[2], kap)
            px = px + ds / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            pz = pz + ds / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            th = th + ds / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        outs.append((px, pz, th))
    return outs


def sensitivity():
    """单段 κ 改动对末端位置的影响 —— 判断可辨识性。"""
    kt = truth_kappa()
    base = rollout_numeric(kt)[-1, :2]
    print("单段 dκ 对末端位置的敏感度：")
    for i in (0, 10, 19):
        kk = kt.copy(); kk[i] += 1.0
        d = np.linalg.norm(rollout_numeric(kk)[-1, :2] - base) * 1e3
        print(f"  seg{i:02d}  dκ=+1.0 1/m  ->  末端 {d:5.2f} mm")


def fit_free(np_=NSEG, noise_mm=1.0, lam=LAM_SMOOTH, seed=0):
    """20 个自由 κ 的反求（结果：不可辨识）。"""
    kt = truth_kappa()
    meas = rollout_numeric(kt)
    rng = np.random.default_rng(seed)
    idx = np.arange(np_)
    obs = meas[idx] + np.column_stack([rng.normal(0, noise_mm*1e-3, len(idx)),
                                       rng.normal(0, noise_mm*1e-3, len(idx)),
                                       rng.normal(0, noise_mm*1e-3, len(idx))])
    opti = ca.Opti()
    k = opti.variable(NSEG)
    opti.subject_to(opti.bounded(-K_MAX, k, K_MAX))
    outs = rollout_casadi(k)
    cost = 0
    for j, i in enumerate(idx):
        px, pz, th = outs[i]
        cost += 1e6 * (px - obs[j, 0]) ** 2 + 1e6 * (pz - obs[j, 1]) ** 2
        cost += 1.0 * (th - obs[j, 2]) ** 2
    for i in range(NSEG - 2):
        cost += lam * (k[i] - 2 * k[i + 1] + k[i + 2]) ** 2
    opti.minimize(cost)
    opti.set_initial(k, np.radians(90.0) / (NSEG * SEG_LEN))
    opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": False})
    t0 = time.time(); sol = opti.solve(); dt = time.time() - t0
    kf = np.array(sol.value(k)).ravel()
    return kt, kf, dt


def fit_2p(noise_mm=1.0, seed=0):
    """低维参数化 κ_i = A·w_i^p（2 个参数）—— 可辨识。"""
    kt = truth_kappa()
    meas = rollout_numeric(kt)
    rng = np.random.default_rng(seed)
    idx = np.arange(NSEG)
    obs = meas[idx] + np.column_stack([rng.normal(0, noise_mm*1e-3, len(idx)),
                                       rng.normal(0, noise_mm*1e-3, len(idx)),
                                       rng.normal(0, noise_mm*1e-3, len(idx))])
    s = (np.arange(NSEG) + 0.5) * SEG_LEN
    w0 = 1.0 / (1.0 + 10.0 * s) * np.where(s <= 0.2, 1.0, 0.5)
    w0 /= w0.sum()
    opti = ca.Opti()
    A = opti.variable(); p_ = opti.variable()
    opti.subject_to(A > 0); opti.subject_to(opti.bounded(0.2, p_, 4.0))
    w = w0 ** p_
    kv = A * w / SEG_LEN                       # 使 Σκ·段长 = A
    outs = rollout_casadi(kv)
    cost = 0
    for j, i in enumerate(idx):
        px, pz, th = outs[i]
        cost += 1e6 * (px - obs[j, 0]) ** 2 + 1e6 * (pz - obs[j, 1]) ** 2
        cost += 1.0 * (th - obs[j, 2]) ** 2
    opti.minimize(cost)
    opti.set_initial(A, np.radians(90.0)); opti.set_initial(p_, 1.0)
    opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": False})
    t0 = time.time(); sol = opti.solve(); dt = time.time() - t0
    return (float(sol.value(A)), float(sol.value(p_)),
            np.radians(90.0), np.array(sol.value(kv)).ravel(), dt)


def main():
    print("=" * 62)
    print("CasADi 可行性验证：20 段曲率场的反求")
    print("=" * 62)
    # 0) 符号前向与数值前向一致性
    kt = truth_kappa()
    num = rollout_numeric(kt)
    ksym = ca.MX.sym('k', NSEG)
    outs = rollout_casadi(ksym)
    fun = ca.Function('f', [ksym],
                      [ca.horzcat(*[ca.vertcat(o[0], o[1], o[2])
                                    for o in outs]).T])
    err = np.max(np.abs(np.array(fun(kt)) - num))
    print(f"[0] 符号前向 vs 数值前向 最大差 {err:.2e}（RK4 vs 欧拉）\n")

    sensitivity()
    print()

    # 1) 20 个自由 κ：不可辨识
    kt, kf, dt = fit_free()
    rel = np.max(np.abs(kf - kt) / np.maximum(kt, 1e-9)) * 100
    print(f"[1] 20 个自由 κ 反求：IPOPT {dt*1000:.0f} ms 收敛，"
          f"总弯角 {np.degrees(kf.sum()*SEG_LEN):.1f}°（真值 90.0°）")
    print(f"    κ 最大相对误差 {rel:.0f}%  ⇒ **不可辨识**（远端段对末端几乎无影响）\n")

    # 2) 2 参数：可辨识
    A, p_, A_true, kv, dt2 = fit_2p()
    print(f"[2] 低维参数化 κ=A·w^p：IPOPT {dt2*1000:.0f} ms 收敛")
    print(f"    A = {A:.3f} rad（真值 {A_true:.3f}，误差 "
          f"{abs(A-A_true)/A_true*100:.1f}%）")
    print(f"    p = {p_:.3f}（真值 1.000）")
    print(f"    拟合曲线与真值 κ 的最大相对误差 "
          f"{np.max(np.abs(kv-kt)/np.maximum(kt,1e-9))*100:.1f}%")


if __name__ == "__main__":
    main()
