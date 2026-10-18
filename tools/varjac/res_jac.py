# -*- coding: utf-8 -*-
"""res_jac.py — 打靶残差与其解析雅可比 ∂res/∂x₀【阶段 2·单样本·μ=0】

把 `tendon_coupling.forward_batch_ref` 的积分镜像一遍，同时按变分方程传播
   Φ   = ∂(v,u)/∂x₀      (6,6)
   dR  = ∂R/∂x₀          (3,3,6)
   dLz = ∂Lz/∂x₀         (6,)
   dp  = ∂tip/∂x₀        (3,6)
最终给出
   Jres = ∂res/∂x₀       (ncol,6)     ← `shooting_batch` 里 J 就是这个
   Jtip = ∂tip/∂x₀       (3,6)

已知近似（本文件里显式标注）：
  ★ `_orthonormalize_b` 的雅可比在 `ortho_jac()` 里**解析给出**。
    （2026-10-16 修：原实现把 ∂|a| 多除了一次范数 ⇒ 只在 n0=1（严格正交）时无误差，
      所以"直杆"对拍通过、一偏离就错。修后 ∂res/∂x₀ 相对差 1.3e-3 → 4.9e-9。）

μ≠0 未支持：此时 τ_w 依赖累积包角 φ(u)，∂τ/∂y≠0。
⚠ nsub 取 2–8 时状态会撞 ±3/±80 钳位，导数不连续 + 混沌放大 ⇒ 该工况下对拍无意义
   （实测 nsub=5 时 res 值本身差 1.9e3，不是本文件的 bug）。对拍须用 nsub≥15。
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from varjac import kbt, kbt_jac, stage_f_jac3, stage_f_dtau  # noqa: E402

_E3 = np.array([0.0, 0.0, 1.0])


def hat(p):
    p = np.asarray(p, float).reshape(3)
    return np.array([[0.0, -p[2], p[1]],
                     [p[2], 0.0, -p[0]],
                     [-p[1], p[0], 0.0]])


def dhat_cols(V):
    """V:(3,K) 每列一个向量 → (3,3,K) 每片的 hat(列)."""
    K = V.shape[1]
    out = np.zeros((3, 3, K))
    for k in range(K):
        out[:, :, k] = hat(V[:, k])
    return out


def ortho_jac(Rtmp, dRtmp):
    """`tendon_coupling._orthonormalize_b`（Gram–Schmidt）的值 + 雅可比。

    Rtmp:(3,3)  dRtmp:(3,3,K) = ∂Rtmp/∂x₀  →  返回 (R_out, dR_out)，同形。

    ⚠ 这个正交化**不能省**：R ← R(I+ds·hat u) 的偏离是 O(ds²|u|²)，
      nsub 粗时 ds·|u| 可达 0.5 ⇒ 单步 R 就偏离正交 ~0.1，值路径直接错。
      （实测: nsub=8 时省略它, res 相对误差 8.5e2 —— 不是小偏差, 是量级错误。）
    """
    a0 = Rtmp[:, 0]; a1 = Rtmp[:, 1]; a2 = Rtmp[:, 2]
    d0 = dRtmp[:, 0, :]; d1 = dRtmp[:, 1, :]; d2 = dRtmp[:, 2, :]
    K = dRtmp.shape[2]
    n0 = float(np.linalg.norm(a0))
    n0g = n0 if n0 >= 1e-12 else 1.0
    b0 = a0 / n0g
    # ⚠ ∂|a0| = (a0·∂a0)/|a0| = b0·∂a0  —— **不能再除一次 |a0|**。
    #   原实现写成 (b0 @ d0)/n0g, 多除一个范数 ⇒ 每步都错; 只有在 n0=1(严格正交) 时
    #   才恰好无误差, 所以"直杆"(R≡I) 的对拍会通过、一偏离就暴露。
    dn0 = (b0 @ d0) if n0 >= 1e-12 else np.zeros(K)
    db0 = d0 / n0g - np.outer(b0, dn0) / n0g
    c = float(a1 @ b0)
    dc = b0 @ d1 + a1 @ db0
    b1p = a1 - c * b0
    db1p = d1 - np.outer(b0, dc) - c * db0
    n1 = float(np.linalg.norm(b1p))
    n1g = n1 if n1 >= 1e-12 else 1.0
    b1 = b1p / n1g
    dn1 = (b1 @ db1p) if n1 >= 1e-12 else np.zeros(K)   # 同上: 不再 /n1g
    db1 = db1p / n1g - np.outer(b1, dn1) / n1g
    b2 = np.cross(b0, b1)
    db2 = np.cross(db0.T, b1.T).T + np.cross(b0.T, db1.T).T
    R_out = np.stack([b0, b1, b2], axis=1)
    dR_out = np.stack([db0, db1, db2], axis=1)          # (3,3,K)
    return R_out, dR_out


# ══════════════════════════════════════════════════════════════════════
# 锚点集中力/矩（世界系）及其对 x₀ 的导数
# ══════════════════════════════════════════════════════════════════════
def anchor_jac(u, v, R, dR, du, dv, anchor, tau6, ri_all):
    """Fs/Ls 及其对 x₀ 的导数。

    u,v   : (3,)        当前状态
    dR    : (3,3,6)     ∂R/∂x₀
    du,dv : (3,6)       ∂(v,u)/∂x₀
    返回 Fs(3,), Ls(3,), dFs(3,6), dLs(3,6)
    """
    Fs = np.zeros(3); Ls = np.zeros(3)
    dFs = np.zeros((3, 6)); dLs = np.zeros((3, 6))
    uh = hat(u)
    for i in anchor:
        ri = np.asarray(ri_all[i], float)
        rih = hat(ri)
        pb = uh @ ri + v
        psp = R @ pb
        nrm = max(float(np.linalg.norm(psp)), 1e-12)
        tp = R @ ri
        tph = hat(tp)
        g = tph @ psp
        ti = float(tau6[i])
        Fs -= ti * psp / nrm
        Ls -= ti * g / nrm
        # ∂pb/∂x₀ = −hat(ri)·∂u/∂x₀ + ∂v/∂x₀
        dpb = -rih @ du + dv                                     # (3,6)
        dpsp = np.einsum('ijk,j->ik', dR, pb) + R @ dpb          # (3,6)
        dnrm = (psp / nrm) @ dpsp                                # (6,)
        dFs -= ti * (dpsp / nrm - np.outer(psp, dnrm) / nrm ** 2)
        dtp = np.einsum('ijk,j->ik', dR, ri)                     # ∂(R·ri)/∂x₀ (3,6)
        dg = np.einsum('ijk,j->ik', dhat_cols(dtp), psp) + tph @ dpsp
        dLs -= ti * (dg / nrm - np.outer(g, dnrm) / nrm ** 2)
    return Fs, Ls, dFs, dLs


# ══════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════
def res_jac(x0, tau6, params, F_tip=None, trace=False):
    """返回 (res:(ncol,), tip:(3,), Jres:(ncol,6), Jtip:(3,6))。trace 时多返回步进记录。"""
    tr = []
    from tendon_coupling import L, L1, L2, SEG_A, SEG_B, _wire_r_vec
    x0 = np.asarray(x0, float).reshape(6)
    tau6 = np.asarray(tau6, float).reshape(6)
    v = x0[:3].copy(); u = x0[3:].copy()
    R = np.eye(3)
    p = np.zeros(3)
    Phi = np.eye(6)                 # ∂[v;u]/∂x₀
    dR = np.zeros((3, 3, 6))        # ∂R/∂x₀
    dp = np.zeros((3, 6))
    Lz = 0.0; dLz = np.zeros(6)

    ri_all = _wire_r_vec(params.r_disk)
    rho_total = params.rhoA + params.disk_mass / L
    fe_w = np.array([0.0, 0.0, -rho_total * 9.81])
    rod_on = bool(getattr(params, "rod_constraint", False))
    lat = bool(getattr(params, "lattice", False))
    cells = getattr(params, "cells", None)

    def _seg(cn):
        c = (cells[0] if cn == 0 else cells[-1]) if cells else None
        if c is not None:
            return (c.EI0, c.C0, c.L_strut, c.GJ, c.k_rod, params.GA)
        from tendon_coupling import C0_DEFAULT, L_STRUT_DEFAULT
        m = 2.0 if cn == 0 else 1.0
        return (m * params.EI, C0_DEFAULT, L_STRUT_DEFAULT, m * params.GJ,
                m * params.k_rod, m * params.GA)

    for k in (0, 1):
        Lseg = L1 if k == 0 else L2
        nsub = params.n_p if k == 0 else params.n_d
        sEI, sC0, sLs, sGJ, sKR, sGA = _seg(k)
        Kse = np.diag([sGA, sGA, sKR])
        wires = list(range(6)) if k == 0 else list(SEG_B)
        tau_w = tau6[list(wires)]
        ri_w = ri_all[list(wires)]
        ds = Lseg / nsub

        for _ in range(nsub):
            # fe_b = Rᵀ fe_w 与其导数
            fe_b = R.T @ fe_w
            dfe_b = np.einsum('jik,j->ik', dR, fe_w)
            # ── 第一级 ──
            y = np.concatenate([u, v])                    # varjac 约定 [u;v]
            f, df_dy, df_dfe = stage_f_jac3(y, Kse, tau_w, ri_w, fe_b, lat, sEI, sGJ, sLs)
            df_dx = df_dy[:, [3, 4, 5, 0, 1, 2]]          # 列序改回 [v;u] = x
            k1 = f.copy()                                  # [v'; u'] 顺序 → 与 x 一致
            # ★ 必须加 df_dfe·dfe_b：fe_b = Rᵀ·fe_w 依赖 R、R 依赖 x₀
            dk1 = df_dx @ Phi + df_dfe @ dfe_b
            u_m = u + 0.5 * ds * k1[3:]
            v_m = v + 0.5 * ds * k1[:3]
            du_m = Phi[3:, :] + 0.5 * ds * dk1[3:, :]
            dv_m = Phi[:3, :] + 0.5 * ds * dk1[:3, :]
            # ── 中点级 ──
            ym = np.concatenate([u_m, v_m])              # varjac 约定 [u;v]
            f2, df2_dy, df2_dfe = stage_f_jac3(ym, Kse, tau_w, ri_w, fe_b,
                                               lat, sEI, sGJ, sLs)
            # ⚠ 这里**不能**把 df2_dy 的列序换成 [v;u]：
            #   `dym` 的行序是 [u_m;v_m]（∂[u_m;v_m]/∂x₀），与 df2_dy 的列序 [u;v]
            #   是一致的；再换列序就变成 (∂f/∂x|_{x=[v;u]}) · (∂[u;v]/∂x₀) —— 行列错配。
            #   错配的后果不是小偏差：Φ 会被错误特征方向指数放大（实测发散到 1e78，
            #   而差分只有 2.9e3）。第一级之所以要换序，是因为 Phi 的行序本来就是 [v;u]。
            dym = np.vstack([du_m, dv_m])                 # ∂[u_m;v_m]/∂x₀
            # fe_b 在同一子步内不变 ⇒ 中点也用同一个 dfe_b
            dk2 = df2_dy @ dym + df2_dfe @ dfe_b
            k2 = f2.copy()
            # ── 状态推进 ──
            Lz += ds * v_m[2]
            dLz += ds * dv_m[2, :]
            v_new = v + ds * k2[:3]
            u_new = u + ds * k2[3:]
            dPhi = np.vstack([Phi[:3, :] + ds * dk2[:3, :],
                              Phi[3:, :] + ds * dk2[3:, :]])
            # 盒约束 (钳位处导数为 0)
            mask = np.ones(6)
            for a in range(3):
                if v_new[a] > 3.0 or v_new[a] < -3.0:
                    mask[a] = 0.0
                if u_new[a] > 80.0 or u_new[a] < -80.0:
                    mask[3 + a] = 0.0
            v_new = np.clip(v_new, -3.0, 3.0)
            u_new = np.clip(u_new, -80.0, 80.0)
            Phi_new = mask[:, None] * dPhi
            # ── p, R 推进 ──
            Tm = np.eye(3) + 0.5 * ds * hat(u_m)
            Rmid = R @ Tm
            p = p + ds * (Rmid @ v_m)
            dp = dp + ds * (np.einsum('ijk,j->ik', dR, (Tm @ v_m))
                            + np.einsum('ij,jk->ik', Rmid, dv_m)
                            + 0.5 * ds * np.einsum('ij,jk->ik', R,
                                                   np.einsum('ijk,j->ik',
                                                             dhat_cols(du_m), v_m)))
            # R ← ortho(R(I + ds·hat(u_new)))  —— 值与雅可比都要
            T = np.eye(3) + ds * hat(u_new)
            Rtmp = R @ T
            dRtmp = np.einsum('ijk,jl->ilk', dR, T) \
                + ds * np.einsum('ij,jlk->ilk', R, dhat_cols(Phi_new[3:, :]))
            R, dR = ortho_jac(Rtmp, dRtmp)
            if trace:
                tr.append((float(np.max(np.abs(Phi))), float(np.max(np.abs(dR))),
                           float(np.max(np.abs(u))), float(np.max(np.abs(Rtmp @ Rtmp.T - np.eye(3))))))
            v, u, Phi = v_new, u_new, Phi_new
        if k == 0:                                   # 段界锚丝修正
            Fs, Ls, dFs, dLs = anchor_jac(u, v, R, dR, Phi[3:, :], Phi[:3, :],
                                          list(SEG_A), tau6, ri_all)
            RFs = R.T @ Fs
            RLs = R.T @ Ls
            dRFs = np.einsum('jik,j->ik', dR, Fs) + R.T @ dFs
            dRLs = np.einsum('jik,j->ik', dR, Ls) + R.T @ dLs
            # v -= Kse⁻¹ RFs
            dv_corr = np.linalg.solve(Kse, dRFs)
            v = v - np.linalg.solve(Kse, RFs)
            Phi = Phi.copy()
            Phi[:3, :] -= dv_corr
            # u -= Kbt(u)⁻¹ RLs   (Kbt 也依赖 u ⇒ 要链式)
            Kb = kbt(u, lat, sEI, sGJ, sLs)
            dKb = kbt_jac(u, lat, sEI, sGJ, sLs)
            du_cur = Phi[3:, :]
            dKb_dx = np.einsum('ijm,mk->ijk', dKb, du_cur)       # (3,3,6)
            w = np.linalg.solve(Kb, RLs)
            # ∂w = Kb⁻¹(∂RLs − dKb_dx·w)
            rhs = dRLs - np.einsum('ijk,j->ik', dKb_dx, w)
            du_corr = np.linalg.solve(Kb, rhs)
            u = u - w
            Phi[3:, :] -= du_corr

    # ── 末端残差 ──
    sEI, sC0, sLs, sGJ, sKR, sGA = _seg(1)
    Kse = np.diag([sGA, sGA, sKR])
    Kb = kbt(u, lat, sEI, sGJ, sLs)
    dKb = kbt_jac(u, lat, sEI, sGJ, sLs)
    dKb_dx = np.einsum('ijm,mk->ijk', dKb, Phi[3:, :])
    vv = v - _E3
    n_e = R @ (Kse @ vv)
    m_e = R @ (Kb @ u)
    dn_e = np.einsum('ijk,j->ik', dR, (Kse @ vv)) + R @ (Kse @ Phi[:3, :])
    dm_e = np.einsum('ijk,j->ik', dR, (Kb @ u)) \
        + R @ (np.einsum('ijk,j->ik', dKb_dx, u) + Kb @ Phi[3:, :])
    Fs, Ls, dFs, dLs = anchor_jac(u, v, R, dR, Phi[3:, :], Phi[:3, :],
                                  list(SEG_B), tau6, ri_all)
    ft = np.zeros(3) if F_tip is None else np.asarray(F_tip, float).reshape(3)
    res = np.concatenate([n_e - Fs - ft, m_e - Ls])
    Jres = np.vstack([dn_e - dFs, dm_e - dLs])
    if rod_on:
        res = np.concatenate([res, [Lz / params.rod_free_len - 1.0]])
        Jres = np.vstack([Jres, (dLz / params.rod_free_len)[None, :]])
    if trace:
        return res, p.copy(), Jres, dp, tr
    return res, p.copy(), Jres, dp


def res_only(x0, tau6, params, F_tip=None):
    return res_jac(x0, tau6, params, F_tip)[:2]


# ══════════════════════════════════════════════════════════════════════
# ∂res/∂τ —— 逆解里位姿雅可比 J=∂e/∂τ 的解析来源（现在还是差分算）
# ══════════════════════════════════════════════════════════════════════
def _anchor_direct_tau(u, v, R, anchor, ri_all):
    """锚点对 τ 的**直接**偏导：(dF(3,6), dL(3,6))。

    间接项（R、(u,v) 随 τ 变）由 dRt / Psi 那条链承担，这里只算
    F = −Σ τ_i·psp/|psp| 里那个显式的 τ_i。
    """
    dF = np.zeros((3, 6))
    dL = np.zeros((3, 6))
    uh = hat(u)
    for i in anchor:
        ri = ri_all[i]
        pb = uh @ ri + v
        psp = R @ pb
        nrm = max(float(np.linalg.norm(psp)), 1e-12)
        dF[:, i] += -psp / nrm
        dL[:, i] += -(hat(R @ ri) @ psp) / nrm
    return dF, dL


def res_jac_tau(x0, tau6, params, F_tip=None):
    """返回 (res, tip, Jtau)，Jtau = ∂res/∂τ 形如 (ncol,6)。

    变分链：Ψ = ∂[v;u]/∂τ (6,6)，dRt = ∂R/∂τ (3,3,6)，dLzt = ∂Lz/∂τ。
    与 `res_jac`（对 x₀ 求导）结构完全平行，只是初值不同：Ψ(0)=0、dRt(0)=0。
    同一套 ∂f/∂y，外加 τ 直接进 `intermed` 的那一项 ∂f/∂τ_w。
    """
    from tendon_coupling import L, L1, L2, SEG_A, SEG_B, _wire_r_vec
    x0 = np.asarray(x0, float).reshape(6)
    tau6 = np.asarray(tau6, float).reshape(6)
    v = x0[:3].copy(); u = x0[3:].copy()
    R = np.eye(3)
    p = np.zeros(3)
    Psi = np.zeros((6, 6))          # ∂[v;u]/∂τ
    dRt = np.zeros((3, 3, 6))       # ∂R/∂τ
    Lz = 0.0; dLzt = np.zeros(6)

    ri_all = _wire_r_vec(params.r_disk)
    rho_total = params.rhoA + params.disk_mass / L
    fe_w = np.array([0.0, 0.0, -rho_total * 9.81])
    rod_on = bool(getattr(params, "rod_constraint", False))
    lat = bool(getattr(params, "lattice", False))
    cells = getattr(params, "cells", None)

    def _seg(cn):
        c = (cells[0] if cn == 0 else cells[-1]) if cells else None
        if c is not None:
            return (c.EI0, c.C0, c.L_strut, c.GJ, c.k_rod, params.GA)
        from tendon_coupling import C0_DEFAULT, L_STRUT_DEFAULT
        m = 2.0 if cn == 0 else 1.0
        return (m * params.EI, C0_DEFAULT, L_STRUT_DEFAULT, m * params.GJ,
                m * params.k_rod, m * params.GA)

    for k in (0, 1):
        Lseg = L1 if k == 0 else L2
        nsub = params.n_p if k == 0 else params.n_d
        sEI, sC0, sLs, sGJ, sKR, sGA = _seg(k)
        Kse = np.diag([sGA, sGA, sKR])
        wires = list(range(6)) if k == 0 else list(SEG_B)
        tau_w = tau6[list(wires)]
        ri_w = ri_all[list(wires)]
        S = np.zeros((len(wires), 6))
        for i, w in enumerate(wires):
            S[i, w] = 1.0
        ds = Lseg / nsub

        for _ in range(nsub):
            fe_b = R.T @ fe_w
            dfe_bt = np.einsum('jik,j->ik', dRt, fe_w)
            y = np.concatenate([u, v])
            f, df_dy, df_dfe = stage_f_jac3(y, Kse, tau_w, ri_w, fe_b, lat, sEI, sGJ, sLs)
            df_dx = df_dy[:, [3, 4, 5, 0, 1, 2]]
            df_dtau = stage_f_dtau(y, Kse, tau_w, ri_w, fe_b, lat, sEI, sGJ, sLs) @ S
            k1 = f.copy()
            dk1 = df_dx @ Psi + df_dfe @ dfe_bt + df_dtau
            u_m = u + 0.5 * ds * k1[3:]
            v_m = v + 0.5 * ds * k1[:3]
            Psi_m = Psi + 0.5 * ds * dk1             # 行序 [v;u]
            du_m = Psi_m[3:, :]
            dv_m = Psi_m[:3, :]

            ym = np.concatenate([u_m, v_m])
            f2, df2_dy, df2_dfe = stage_f_jac3(ym, Kse, tau_w, ri_w, fe_b,
                                               lat, sEI, sGJ, sLs)
            df2_dtau = stage_f_dtau(ym, Kse, tau_w, ri_w, fe_b,
                                    lat, sEI, sGJ, sLs) @ S
            dym = np.vstack([du_m, dv_m])            # 行序 [u;v] 与 df2_dy 的列序一致
            dk2 = df2_dy @ dym + df2_dfe @ dfe_bt + df2_dtau
            k2 = f2.copy()

            Lz += ds * v_m[2]
            dLzt += ds * dv_m[2, :]
            v_new = v + ds * k2[:3]
            u_new = u + ds * k2[3:]
            dPsi = np.vstack([Psi[:3, :] + ds * dk2[:3, :],
                              Psi[3:, :] + ds * dk2[3:, :]])
            mask = np.ones(6)
            for a in range(3):
                if v_new[a] > 3.0 or v_new[a] < -3.0:
                    mask[a] = 0.0
                if u_new[a] > 80.0 or u_new[a] < -80.0:
                    mask[3 + a] = 0.0
            v_new = np.clip(v_new, -3.0, 3.0)
            u_new = np.clip(u_new, -80.0, 80.0)
            Psi_new = mask[:, None] * dPsi

            Tm = np.eye(3) + 0.5 * ds * hat(u_m)
            Rmid = R @ Tm
            p = p + ds * (Rmid @ v_m)
            dpt = ds * (np.einsum('ijk,j->ik', dRt, (Tm @ v_m))
                        + np.einsum('ij,jk->ik', Rmid, dv_m)
                        + 0.5 * ds * np.einsum('ij,jk->ik', R,
                                               np.einsum('ijk,j->ik',
                                                         dhat_cols(du_m), v_m)))

            T = np.eye(3) + ds * hat(u_new)
            Rtmp = R @ T
            dRtmp = np.einsum('ijk,jl->ilk', dRt, T) \
                + ds * np.einsum('ij,jlk->ilk', R, dhat_cols(Psi_new[3:, :]))
            R, dRt = ortho_jac(Rtmp, dRtmp)

            v, u, Psi = v_new, u_new, Psi_new
        if k == 0:
            Fs, Ls, dFs, dLs = anchor_jac(u, v, R, dRt, Psi[3:, :], Psi[:3, :],
                                          list(SEG_A), tau6, ri_all)
            dFdir, dLdir = _anchor_direct_tau(u, v, R, list(SEG_A), ri_all)
            dFs = dFs + dFdir; dLs = dLs + dLdir
            dRFs = np.einsum('jik,j->ik', dRt, Fs) + R.T @ dFs
            dRLs = np.einsum('jik,j->ik', dRt, Ls) + R.T @ dLs
            dv_corr = np.linalg.solve(Kse, dRFs)
            v = v - np.linalg.solve(Kse, R.T @ Fs)
            Psi = Psi.copy()
            Psi[:3, :] -= dv_corr
            Kb = kbt(u, lat, sEI, sGJ, sLs)
            dKb = kbt_jac(u, lat, sEI, sGJ, sLs)
            dKb_dt = np.einsum('ijm,mk->ijk', dKb, Psi[3:, :])
            w = np.linalg.solve(Kb, R.T @ Ls)
            rhs = dRLs - np.einsum('ijk,j->ik', dKb_dt, w)
            u = u - w
            Psi[3:, :] -= np.linalg.solve(Kb, rhs)

    sEI, sC0, sLs, sGJ, sKR, sGA = _seg(1)
    Kse = np.diag([sGA, sGA, sKR])
    Kb = kbt(u, lat, sEI, sGJ, sLs)
    dKb = kbt_jac(u, lat, sEI, sGJ, sLs)
    dKb_dt = np.einsum('ijm,mk->ijk', dKb, Psi[3:, :])
    vv = v - _E3
    n_e = R @ (Kse @ vv)
    m_e = R @ (Kb @ u)
    dn_e = np.einsum('ijk,j->ik', dRt, (Kse @ vv)) + R @ (Kse @ Psi[:3, :])
    dm_e = np.einsum('ijk,j->ik', dRt, (Kb @ u)) \
        + R @ (np.einsum('ijk,j->ik', dKb_dt, u) + Kb @ Psi[3:, :])
    Fs, Ls, dFs, dLs = anchor_jac(u, v, R, dRt, Psi[3:, :], Psi[:3, :],
                                  list(SEG_B), tau6, ri_all)
    dFdir, dLdir = _anchor_direct_tau(u, v, R, list(SEG_B), ri_all)
    dFs = dFs + dFdir; dLs = dLs + dLdir
    ft = np.zeros(3) if F_tip is None else np.asarray(F_tip, float).reshape(3)
    res = np.concatenate([n_e - Fs - ft, m_e - Ls])
    Jtau = np.vstack([dn_e - dFs, dm_e - dLs])
    if rod_on:
        res = np.concatenate([res, [Lz / params.rod_free_len - 1.0]])
        Jtau = np.vstack([Jtau, (dLzt / params.rod_free_len)[None, :]])
    return res, p.copy(), Jtau
