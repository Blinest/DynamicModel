# -*- coding: utf-8 -*-
"""varjac.py — 打靶雅可比 ∂res/∂x₀ 的解析（变分方程）实现【原型·单样本】

背景
----
`shooting_batch` 现在用**有限差分**求 J = ∂res/∂x₀：每个 LM 迭代要 6 次（前向）
或 12 次（中心）**完整正解**（601 子步）。README 统计它占端到端耗时约 90%。

解析化思路（变分方程）
--------------------
打靶未知量 y = [v;u] ∈ R⁶，状态 ODE 为

    y'(s) = f(y) = M(y)⁻¹ · r(y)

其中 M = [[Kse+A, G],[B, Kbt+H]]、r = [d;c]，而 A,B,G,H,c,d 全部由
`intermed_w` 的代数给出（O(W) 次 3×3 运算，与状态数无关）。于是

    ∂f/∂y = M⁻¹ · ( ∂r/∂y − (∂M/∂y) · f )          (1)

敏感度矩阵 Φ = ∂y(s)/∂y(0) 满足 Φ' = (∂f/∂y)·Φ，Φ(0)=I。RK2 中点格式下

    Φ_next = [ I + ds·(∂f/∂y)_mid ] · [ I + 0.5·ds·(∂f/∂y) ] · Φ    (2)

所以**唯一的难点**是解析地写出 ∂(A,B,G,H,c,d)/∂(u,v) —— 本文件就做这件事。

本文件的两个可验证闭环
--------------------
1. `intermed()` 必须与 `tendon_coupling._intermed_b_vec` 逐位一致（数学搬移）
2. `intermed_jac()` 必须与 `intermed()` 的中心差分逐元素一致（解析导数正确）

⚠ 边界: μ≠0 时 tau 依赖累积包角 φ(u)，本文件按 μ=0 推导（现有表都是 μ=0）。
   该情形下 ∂tau/∂y ≠ 0，属后续项。
"""
from __future__ import annotations

import numpy as np

_E3 = np.array([0.0, 0.0, 1.0])


# ══════════════════════════════════════════════════════════════════════
# 基础
# ══════════════════════════════════════════════════════════════════════
def hat(p):
    """3-向量 → 反对称矩阵 (p×)."""
    p = np.asarray(p, float).reshape(3)
    return np.array([[0.0, -p[2], p[1]],
                     [p[2], 0.0, -p[0]],
                     [-p[1], p[0], 0.0]])


def hat_basis():
    """∂hat(p)/∂p_j = hat(e_j) —— hat 是线性的, 导数就是基底的反对称矩阵."""
    return np.stack([hat(np.eye(3)[j]) for j in range(3)], axis=0)   # (3,3,3): [j,i,k]


def d_hat2_dp(p):
    """d(hat(p)²)/dp_j = e_j pᵀ + p e_jᵀ − 2 p_j I   (因为 hat(p)² = p pᵀ − |p|² I).

    返回 (3,3,3): [i,j,m] = ∂(hat(p)²)[i,j] / ∂p[m]
    """
    p = np.asarray(p, float).reshape(3)
    out = np.zeros((3, 3, 3))
    for m in range(3):
        em = np.eye(3)[m]
        out[:, :, m] = np.outer(em, p) + np.outer(p, em) - 2.0 * p[m] * np.eye(3)
    return out


# ══════════════════════════════════════════════════════════════════════
# 1) κ(u) 的割线刚度 Kbt(u) 及其导数
# ══════════════════════════════════════════════════════════════════════
def kbt(u, lat, sEI, sGJ, sLs):
    if lat:
        th = float(np.linalg.norm(np.asarray(u, float)[:2]) * sLs)
        e = sEI if th < 1e-9 else sEI * np.sin(th) / th
    else:
        e = sEI
    return np.diag([e, e, sGJ])


def kbt_jac(u, lat, sEI, sGJ, sLs):
    """dKbt/du → (3,3,3). 只有前两维奇异 (e 依赖 |u_xy|)."""
    u = np.asarray(u, float)
    d = np.zeros((3, 3, 3))
    if not lat:
        return d
    nxy = float(np.hypot(u[0], u[1]))
    th = nxy * sLs
    if th < 1e-9:
        de = 0.0
    else:
        # e(th) = sEI·sin(th)/th ; de/dth = sEI·(th·cos th − sin th)/th²
        de = sEI * (th * np.cos(th) - np.sin(th)) / th ** 2 * sLs
    if nxy < 1e-12:
        return d
    g = np.array([u[0] / nxy, u[1] / nxy, 0.0]) * de
    d[0, 0, :] = g
    d[1, 1, :] = g
    return d


# ══════════════════════════════════════════════════════════════════════
# 2) A,B,G,H,c,d（单样本, 与 tendon_coupling._intermed_b_vec 同数学）
# ══════════════════════════════════════════════════════════════════════
def intermed(u, v, Kse, Kbt, tau_w, ri_w, fe_b):
    """tau_w:(W,) 本节点各丝张力; ri_w:(W,3) 本段参与丝的孔位偏移."""
    u = np.asarray(u, float); v = np.asarray(v, float)
    W = len(tau_w)
    A = np.zeros((3, 3)); B = np.zeros((3, 3))
    G = np.zeros((3, 3)); H = np.zeros((3, 3))
    av = np.zeros(3); bv = np.zeros(3)
    uh = hat(u)
    for w in range(W):
        ri = ri_w[w]
        rih = hat(ri)
        pb = uh @ ri + v
        nrm = max(float(np.linalg.norm(pb)), 1e-12)
        P2 = hat(pb) @ hat(pb)
        Ai = (-(tau_w[w] / nrm ** 3)) * P2
        Bi = rih @ Ai
        A += Ai; B += Bi
        G -= Ai @ rih; H -= Bi @ rih
        ai = Ai @ (uh @ pb)
        av += ai
        bv += rih @ ai
    vv = v - _E3
    Kv = Kse @ vv
    c = -(uh @ (Kbt @ u)) - (hat(v) @ Kv) - fe_b - bv
    d = -(uh @ Kv) - fe_b - av
    return A, B, G, H, c, d


def intermed_jac(u, v, Kse, Kbt, tau_w, ri_w, fe_b, dKbt_dy=None):
    """解析 ∂(A,B,G,H,c,d)/∂y, y=[u(3);v(3)].

    返回 (dA,dB,dG,dH:(3,3,6), dc,dd:(3,6))
    dKbt_dy: (3,3,6) 可选, 由 kbt_jac 嵌入 y 坐标 (前 3 列对应 u)。
    """
    u = np.asarray(u, float); v = np.asarray(v, float)
    W = len(tau_w)
    y = np.concatenate([u, v])
    HY = hat_basis()                                      # (3,3,3)
    dA = np.zeros((3, 3, 6)); dB = np.zeros((3, 3, 6))
    dG = np.zeros((3, 3, 6)); dH = np.zeros((3, 3, 6))
    dav = np.zeros((3, 6)); dbv = np.zeros((3, 6))
    uh = hat(u)
    for w in range(W):
        ri = ri_w[w]; rih = hat(ri)
        pb = uh @ ri + v
        nrm = max(float(np.linalg.norm(pb)), 1e-12)
        P2 = hat(pb) @ hat(pb)
        coef = -(tau_w[w] / nrm ** 3)
        Ai = coef * P2
        Bi = rih @ Ai
        # ── 链式: y → pb → nrm → coef ──
        dpb_dy = np.zeros((3, 6))
        dpb_dy[:, :3] = -rih                               # ∂(hat(u)·ri)/∂u = −hat(ri)
        dpb_dy[:, 3:] = np.eye(3)                          # ∂pb/∂v = I
        dnrm_dy = (pb / nrm) @ dpb_dy                      # (6,)
        dcoef_dy = (3.0 * tau_w[w] / nrm ** 4) * dnrm_dy   # coef = −τ/nrm³
        dP2_dpb = d_hat2_dp(pb)                            # (3,3,3)
        # dAi[i,j,k] = P2[i,j]·dcoef[k] + coef·Σ_m dP2_dpb[i,j,m]·dpb_dy[m,k]
        dAi = P2[:, :, None] * dcoef_dy[None, None, :] \
            + coef * np.einsum('ijm,mk->ijk', dP2_dpb, dpb_dy)
        dBi = np.einsum('im,mjk->ijk', rih, dAi)
        dA += dAi; dB += dBi
        dG -= np.einsum('imk,mj->ijk', dAi, rih)
        dH -= np.einsum('imk,mj->ijk', dBi, rih)
        # ── av = Σ Ai·(hat(u)·pb) ──
        uhpb = uh @ pb
        duhpb_dy = np.zeros((3, 6))
        for k in range(3):                                  # ∂u_k: hat(e_k)·pb + hat(u)·∂pb/∂u_k
            duhpb_dy[:, k] = HY[k] @ pb + uh @ dpb_dy[:, k]
        for k in range(3, 6):
            duhpb_dy[:, k] = uh @ dpb_dy[:, k]
        dai_dy = np.einsum('ijk,j->ik', dAi, uhpb) + Ai @ duhpb_dy
        dav += dai_dy
        dbv += rih @ dai_dy
    # ── c, d ──
    vv = v - _E3
    Kv = Kse @ vv
    w_u = Kbt @ u
    uhKbtu = uh @ w_u
    vhKsev = hat(v) @ Kv
    uhKsev = uh @ Kv
    dKbt = np.zeros((3, 3, 6)) if dKbt_dy is None else np.asarray(dKbt_dy, float)
    # d(hat(u)@w)/dy, w = Kbt@u
    d_uhKbtu = np.zeros((3, 6))
    for k in range(3):
        dw_duk = dKbt[:, :, k] @ u + Kbt @ np.eye(3)[k]
        d_uhKbtu[:, k] = HY[k] @ w_u + uh @ dw_duk
    # d(hat(v)@Kv)/dy  (只在 v 上有非零)
    d_vhKsev = np.zeros((3, 6))
    for k in range(3):
        d_vhKsev[:, 3 + k] = HY[k] @ Kv + hat(v) @ (Kse @ np.eye(3)[k])
    d_uhKsev = np.zeros((3, 6))
    for k in range(3):
        d_uhKsev[:, k] = HY[k] @ Kv
    for k in range(3):
        d_uhKsev[:, 3 + k] = uh @ (Kse @ np.eye(3)[k])
    dc = -d_uhKbtu - d_vhKsev - dbv
    dd = -d_uhKsev - dav
    return dA, dB, dG, dH, dc, dd


# ══════════════════════════════════════════════════════════════════════
# 3) 阶段函数 f(y) = M⁻¹r 及其解析雅可比 ∂f/∂y
# ══════════════════════════════════════════════════════════════════════
def _assemble(Kse, Kbt, A, B, G, H, c, d):
    M = np.zeros((6, 6))
    M[:3, :3] = Kse + A; M[:3, 3:] = G
    M[3:, :3] = B; M[3:, 3:] = Kbt + H
    r = np.concatenate([d, c])
    return M, r


def stage_f_jac(y, Kse, tau_w, ri_w, fe_b, lat, sEI, sGJ, sLs):
    """返回 (f(6,), df_dy(6,6))。y = [u;v]（⚠ 注意本文件内部 y 的前 3 维是 u）。"""
    u = np.asarray(y, float)[:3]; v = np.asarray(y, float)[3:]
    Kbt = kbt(u, lat, sEI, sGJ, sLs)
    A, B, G, H, c, d = intermed(u, v, Kse, Kbt, tau_w, ri_w, fe_b)
    M, r = _assemble(Kse, Kbt, A, B, G, H, c, d)
    f = np.linalg.solve(M, r)
    dKt = kbt_jac(u, lat, sEI, sGJ, sLs)
    dKbt_dy = np.zeros((3, 3, 6)); dKbt_dy[:, :, :3] = dKt
    dA, dB, dG, dH, dc, dd = intermed_jac(u, v, Kse, Kbt, tau_w, ri_w, fe_b, dKbt_dy)
    dM = np.zeros((6, 6, 6))
    dM[:3, :3, :] = dA; dM[:3, 3:, :] = dG
    dM[3:, :3, :] = dB; dM[3:, 3:, :] = dKbt_dy + dH
    dr = np.zeros((6, 6))
    dr[:3, :] = dd; dr[3:, :] = dc
    # ∂f/∂y = M⁻¹ (∂r/∂y − (∂M/∂y)·f)
    for k in range(6):
        rhs = dr[:, k] - dM[:, :, k] @ f
        dr[:, k] = np.linalg.solve(M, rhs)
    return f, dr
