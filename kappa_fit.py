# -*- coding: utf-8 -*-
"""kappa_fit.py — 直接以曲率场为参数拟合臂形（不经过 θ / φ 分解）

模型
----
把臂体沿弧长分成 12 段，每段一个曲率 κ_i（1/m），在弯曲平面内：
    p' = [sinθ, cosθ],  θ' = κ(s)
⇒ 3 个可观测量（末端横向 x、轴向 z、末端切向角 θ_tip）对 12 个未知量
⇒ **欠定**，需要正则：最小化

    J(κ) = ‖r(κ)‖² + λ·‖D₂κ‖²      （D₂ 二阶差分 = 曲率场平滑）

用 Gauss-Newton 求解（数值雅可比，12 列）。

输出：
    · 拟合残差（mm / °）
    · 12 段曲率与曲率图
    · 对应 6 路驱动丝位移量（A 段丝只看前 6 段，B 段丝看全 12 段）
"""
import numpy as np

EI = 0.01037
R_DISK = 5e-3
L = 0.4
L1 = 0.2
NSEG = 12
SEG_LEN = L / NSEG
N_SUB = 8


def shape(kvec):
    p = np.zeros(2)
    th = 0.0
    P = [p.copy()]
    for k in kvec:
        ds = SEG_LEN / N_SUB
        for _ in range(N_SUB):
            p = p + np.array([np.sin(th), np.cos(th)]) * ds
            th += k * ds
            P.append(p.copy())
    return np.array(P), th


def obs(kvec):
    """可观测：末端 (x, z) mm + 末端切向角 ° + 总弯角 °。"""
    P, th = shape(kvec)
    return np.array([P[-1, 0] * 1e3, P[-1, 1] * 1e3, np.degrees(th)])


def D2(n=NSEG):
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i], D[i, i + 1], D[i, i + 2] = 1.0, -2.0, 1.0
    return D


def fit(target, lam=1e-3, k0=None, iters=60, verbose=False):
    """target = (x_mm, z_mm, theta_tip_deg)，返回 (κ12, 残差)。"""
    k = np.full(NSEG, np.radians(target[2]) / 0.3 / NSEG / SEG_LEN * NSEG
                if k0 is None else 0.0)
    if k0 is not None:
        k = np.asarray(k0, float).copy()
    else:
        # 初始：均匀曲率，使总弯角 = target 的切向角
        k = np.full(NSEG, np.radians(target[2]) / L)
    D = D2()
    tgt = np.asarray(target, float)
    for it in range(iters):
        r0 = obs(k)
        e = r0 - tgt
        # 数值雅可比（12 列）
        J = np.zeros((3, NSEG))
        h = 1e-4
        for i in range(NSEG):
            kp = k.copy()
            kp[i] += h
            J[:, i] = (obs(kp) - r0) / h
        # 增广系统：[J; sqrt(lam) D] Δ = [-e; -sqrt(lam) D k]
        A = np.vstack([J, np.sqrt(lam) * D])
        b = np.concatenate([-e, -np.sqrt(lam) * (D @ k)])
        try:
            dk = np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        k = k + dk
        if np.linalg.norm(dk) < 1e-9:
            break
        if verbose and it % 10 == 0:
            print(f"    it{it:03d} ‖e‖={np.linalg.norm(e):.4f} λ={lam:g}")
    res = obs(k) - tgt
    return k, res


def dl_of(kvec):
    """6 路 ΔL：A 段丝只累计前 6 段的转角，B 段丝累计全部。"""
    th_seg = kvec * SEG_LEN
    th_A = np.sum(th_seg[:6])
    th_B = np.sum(th_seg)
    return R_DISK * th_A * 1e3, R_DISK * th_B * 1e3


if __name__ == "__main__":
    print("=== 目标：末端 (300, 240) mm、切向 60° ===")
    tgt = (300.0, 240.0, 60.0)
    for lam in (1e-1, 1e-2, 1e-3, 1e-4):
        k, res = fit(tgt, lam=lam)
        a, b = dl_of(k)
        print(f"  λ={lam:6.0e}  残差 x={res[0]:+7.2f} z={res[1]:+7.2f} "
              f"θ={res[2]:+6.2f}°   ‖κ‖={np.linalg.norm(k):6.2f}  "
              f"ΔL_A={a:5.2f} ΔL_B={b:5.2f} mm")
    print("\n=== 取 λ=1e-3 的 12 段曲率 ===")
    k, res = fit(tgt, lam=1e-3)
    print("  κ (1/m):", np.round(k, 2))
    print("  总弯角 %.1f°，A 段(前6) %.1f°，B 段(全12) %.1f°" %
          (np.degrees(k.sum() * SEG_LEN),
           np.degrees(k[:6].sum() * SEG_LEN),
           np.degrees(k.sum() * SEG_LEN)))
    a, b = dl_of(k)
    print(f"  ΔL：A 段丝 {a:.2f} mm，B 段丝 {b:.2f} mm（比值 {b/max(a,1e-9):.2f}）")
    print("\n=== 对照：2 参数模型（θ、均匀分布）能到的最近点 ===")
    best = None
    for deg in np.linspace(1, 200, 400):
        ku = np.full(NSEG, np.radians(deg) / L)
        o = obs(ku)
        d = np.hypot(o[0] - tgt[0], o[1] - tgt[1])
        if best is None or d < best[0]:
            best = (d, deg, o)
    print(f"  最接近 θ={best[1]:.0f}° → 末端 ({best[2][0]:.1f},{best[2][1]:.1f})，"
          f"残差 {best[0]:.1f} mm")
