# -*- coding: utf-8 -*-
"""S2: `integrate_all` —— **一次 JIT 调用**内完成 N 样本 × 601 子步的全积分.

设计见 `_VEC_PLAN.md`。要点:
  · 样本循环 + 子步循环**都在 numba 内** ⇒ 只有 1 次 Python→JIT 调用
    (S1 曾试"逐样本调 JIT", 实测反而慢 0.5× —— 见计划"错误路径作废")
  · 复用 `_jit_core` 的 5 个 `inline="always"` 标量核
  · **不改动** `tendon_coupling.forward_batch`（S3 才切换）
  · 三处历史坑已按原版照搬: C 用端点 u / 段界按段取刚度 / C 与 κ 同节点

跑对拍:  python _jit_integrate.py
"""
import sys

import numpy as np
from numba import njit

from _jit_core import hat3s, solve6s, ortho3s, kbt3s, intermed_s  # noqa: F401


# ═════ 5') 六丝耦合 —— 可按 `wires` 取子集 (段B 只用 SEG_B) ═════
@njit(cache=True, inline="always")
def intermed_w(u, v, Kse, Kbt, tau6, ri_all, wires, W, fe_b, E3,
               A, B, G, H, c, d):
    """`intermed_s` 的"可 w 子集"版. A,B,G,H,c,d 为输出缓冲."""
    uh = np.empty((3, 3))
    hat3s(u[0], u[1], u[2], uh)
    for i in range(3):
        for j in range(3):
            A[i, j] = 0.0; B[i, j] = 0.0; G[i, j] = 0.0; H[i, j] = 0.0
    av = np.zeros(3); bv = np.zeros(3)
    for wi in range(W):
        w = wires[wi]
        rih = np.empty((3, 3))
        hat3s(ri_all[w, 0], ri_all[w, 1], ri_all[w, 2], rih)
        uri = uh @ ri_all[w]
        pb = uri + v
        nrm = np.sqrt(pb[0] ** 2 + pb[1] ** 2 + pb[2] ** 2)
        if nrm < 1e-12:
            nrm = 1e-12
        pbh = np.empty((3, 3))
        hat3s(pb[0], pb[1], pb[2], pbh)
        pb2 = pbh @ pbh
        coef = -(tau6[w] / nrm ** 3)
        Ai = coef * pb2
        Bi = rih @ Ai
        Ar = Ai @ rih
        Br = Bi @ rih
        for a in range(3):
            for b in range(3):
                A[a, b] += Ai[a, b]; B[a, b] += Bi[a, b]
                G[a, b] -= Ar[a, b]; H[a, b] -= Br[a, b]
        uhpb = uh @ pb
        ai = Ai @ uhpb
        for a in range(3):
            av[a] += ai[a]
        bi = rih @ ai
        for a in range(3):
            bv[a] += bi[a]
    vv = np.empty(3)
    for a in range(3):
        vv[a] = v[a] - E3[a]
    Kv = Kse @ vv
    uhKbtu = uh @ (Kbt @ u)
    vh = np.empty((3, 3))
    hat3s(v[0], v[1], v[2], vh)
    vhKsev = vh @ Kv
    uhKsev = uh @ Kv
    for a in range(3):
        c[a] = -uhKbtu[a] - vhKsev[a] - fe_b[a] - bv[a]
        d[a] = -uhKsev[a] - fe_b[a] - av[a]


# ═════ 5'') 6×6 解 —— 用 LAPACK (定位"手写消元 vs LAPACK"的数值差异) ═════
@njit(cache=True, inline="always")
def solve6_lapack(M, rhs, out):
    """与 `_safe_solve6` 同语义, 但内部走 `np.linalg.solve` (LAPACK) ⇒ 数值与原版一致."""
    if (not np.isfinite(M).all()) or (not np.isfinite(rhs).all()):
        for k in range(6):
            out[k] = 0.0
        return
    try:
        sol = np.linalg.solve(M, rhs)
        for k in range(6):
            val = sol[k]
            if not np.isfinite(val):
                val = 0.0
            elif val > 1e3:
                val = 1e3
            elif val < -1e3:
                val = -1e3
            out[k] = val
    except Exception:
        for k in range(6):
            out[k] = 0.0


# ═════ 6) 锚丝集中力/矩 (对应 `_anchor_b`, 单样本) ═════
@njit(cache=True, inline="always")
def anchor_s(u, v, R, anchor, na, tau6, ri_all, F, Lm):
    for a in range(3):
        F[a] = 0.0; Lm[a] = 0.0
    uh = np.empty((3, 3))
    hat3s(u[0], u[1], u[2], uh)
    for ai_ in range(na):
        i = anchor[ai_]
        ri = ri_all[i]
        pb = (uh @ ri) + v
        psp = R @ pb
        nrm = np.sqrt(psp[0] ** 2 + psp[1] ** 2 + psp[2] ** 2)
        if nrm < 1e-12:
            nrm = 1e-12
        Rri = R @ ri
        Rrih = np.empty((3, 3))
        hat3s(Rri[0], Rri[1], Rri[2], Rrih)
        Rrih_psp = Rrih @ psp
        for a in range(3):
            F[a] -= tau6[i] * psp[a] / nrm
            Lm[a] -= tau6[i] * Rrih_psp[a] / nrm


# ═════════════════════ S2: 全积分 (双层循环都在 JIT 内) ═════════════════════
import os
_FM = os.environ.get("JIT_FASTMATH", "1") == "1"   # 诊断用: JIT_FASTMATH=0 关掉


@njit(cache=True, fastmath=_FM)
def integrate_all(x, tau, ri_all, segA, segB, E3, few, L1, L2, n_p, n_d,
                  lat, rod_on, rod_free_len, wiresA, wiresB,
                  anchorA, anchorB, want_curve):
    """x:(N,6) tau:(N,6);  segA/segB:(6,)=[EI0,C0,L_strut,GJ,k_rod,GA].

    返回 res:(N,6|7), tip:(N,3), 以及 want_curve 时的 6 个曲线数组 + S.
    """
    N = x.shape[0]
    ncol = 7 if rod_on else 6
    res = np.zeros((N, ncol))
    tip = np.zeros((N, 3))
    ntot = n_p + n_d
    if want_curve:
        Pc = np.zeros((N, ntot + 1, 3))
        Uc = np.zeros((N, ntot + 1, 3))
        Cc = np.zeros((N, ntot + 1, 3))
        Rc = np.zeros((N, ntot + 1, 3, 3))
        Nc = np.zeros((N, ntot + 1, 3))
        Vc = np.zeros((N, ntot + 1, 3))
        S = np.linspace(0.0, L1 + L2, ntot + 1)
    else:
        Pc = np.zeros((1, 1, 3)); Uc = np.zeros((1, 1, 3))
        Cc = np.zeros((1, 1, 3)); Rc = np.zeros((1, 1, 3, 3))
        Nc = np.zeros((1, 1, 3)); Vc = np.zeros((1, 1, 3)); S = np.zeros(1)

    Kse = np.zeros((3, 3)); Kbt = np.zeros((3, 3)); Kbtm = np.zeros((3, 3))
    A = np.zeros((3, 3)); B = np.zeros((3, 3))
    G = np.zeros((3, 3)); H = np.zeros((3, 3))
    c = np.zeros(3); d = np.zeros(3)
    A2 = np.zeros((3, 3)); B2 = np.zeros((3, 3))
    G2 = np.zeros((3, 3)); H2 = np.zeros((3, 3))
    c2 = np.zeros(3); d2 = np.zeros(3)
    M = np.zeros((6, 6)); rhs = np.zeros(6); vu = np.zeros(6)
    M2 = np.zeros((6, 6)); rhs2 = np.zeros(6); vu2 = np.zeros(6)
    hh = np.zeros((3, 3)); hmid = np.zeros((3, 3))
    Rmid = np.zeros((3, 3)); Tmp = np.zeros((3, 3)); Rtmp = np.zeros((3, 3))
    fe_b = np.zeros(3)
    Fs = np.zeros(3); Ls = np.zeros(3)
    vm = np.zeros(3); um = np.zeros(3)
    I3 = np.eye(3)

    for n in range(N):
        v = np.empty(3); u = np.empty(3)
        for a in range(3):
            v[a] = x[n, a]; u[a] = x[n, 3 + a]
        R = np.eye(3)
        p = np.zeros(3)
        Lz = 0.0
        idx = 0
        if want_curve:
            e0 = segA[0]
            if lat:
                th0 = np.sqrt(u[0] ** 2 + u[1] ** 2) * segA[2]
                if th0 >= 1e-9:
                    e0 = segA[0] * np.sin(th0) / th0
            Cc[n, 0, 0] = e0 * u[0]; Cc[n, 0, 1] = e0 * u[1]
            Cc[n, 0, 2] = segA[3] * u[2]
            Nc[n, 0, 0] = segA[5] * v[0]; Nc[n, 0, 1] = segA[5] * v[1]
            Nc[n, 0, 2] = segA[4] * (v[2] - 1.0)
            Uc[n, 0, :] = u; Vc[n, 0, :] = v; Rc[n, 0, :, :] = R
            idx = 1

        for k in range(2):
            if k == 0:
                Lseg = L1; nsub = n_p
                sEI = segA[0]; sGJ = segA[3]; sKR = segA[4]
                sGA = segA[5]; sLs = segA[2]
                wires = wiresA; W = wiresA.shape[0]
            else:
                Lseg = L2; nsub = n_d
                sEI = segB[0]; sGJ = segB[3]; sKR = segB[4]
                sGA = segB[5]; sLs = segB[2]
                wires = wiresB; W = wiresB.shape[0]
            for a in range(3):
                for b in range(3):
                    Kse[a, b] = 0.0
            Kse[0, 0] = sGA; Kse[1, 1] = sGA; Kse[2, 2] = sKR
            ds = Lseg / nsub
            for _ in range(nsub):
                # ⚠⚠ fe_b = Rᵀ·fe_w 必须**每子步重算**: 它依赖当前 R, 而 R 在子步间变化。
                #   原实现把它提到"段循环内、子步循环外"(每段只算一次) ⇒ 段B 从第 1 步
                #   就与 numpy 版分岔 (实测 Pc 节点31 d~1e-10 → 累积到 tip ~3.8e-8)。
                #   (子步**内** R 不变, 故中点用同一 fe_b 是对的。)
                for a in range(3):
                    fe_b[a] = R[0, a] * few[0] + R[1, a] * few[1] + R[2, a] * few[2]
                kbt3s(u, lat, sEI, sGJ, sLs, Kbt)
                intermed_w(u, v, Kse, Kbt, tau[n], ri_all, wires, W, fe_b, E3,
                           A, B, G, H, c, d)
                for a in range(3):
                    for b in range(3):
                        M[a, b] = Kse[a, b] + A[a, b]
                        M[a, 3 + b] = G[a, b]
                        M[3 + a, b] = B[a, b]
                        M[3 + a, 3 + b] = Kbt[a, b] + H[a, b]
                    rhs[a] = d[a]; rhs[3 + a] = c[a]
                solve6_lapack(M, rhs, vu)
                for a in range(3):
                    vm[a] = v[a] + 0.5 * ds * vu[a]
                    um[a] = u[a] + 0.5 * ds * vu[3 + a]
                kbt3s(um, lat, sEI, sGJ, sLs, Kbtm)
                intermed_w(um, vm, Kse, Kbtm, tau[n], ri_all, wires, W, fe_b, E3,
                           A2, B2, G2, H2, c2, d2)
                for a in range(3):
                    for b in range(3):
                        M2[a, b] = Kse[a, b] + A2[a, b]
                        M2[a, 3 + b] = G2[a, b]
                        M2[3 + a, b] = B2[a, b]
                        M2[3 + a, 3 + b] = Kbtm[a, b] + H2[a, b]
                    rhs2[a] = d2[a]; rhs2[3 + a] = c2[a]
                solve6_lapack(M2, rhs2, vu2)
                Lz += ds * vm[2]
                for a in range(3):
                    v[a] += ds * vu2[a]
                    u[a] += ds * vu2[3 + a]
                    if v[a] > 3.0:
                        v[a] = 3.0
                    elif v[a] < -3.0:
                        v[a] = -3.0
                    if u[a] > 80.0:
                        u[a] = 80.0
                    elif u[a] < -80.0:
                        u[a] = -80.0
                # p 更新 (中点) + R 更新 (端点)
                hat3s(um[0], um[1], um[2], hmid)
                for a in range(3):
                    for b in range(3):
                        Tmp[a, b] = I3[a, b] + 0.5 * ds * hmid[a, b]
                for a in range(3):
                    for b in range(3):
                        Rmid[a, b] = R[a, 0] * Tmp[0, b] + R[a, 1] * Tmp[1, b] \
                            + R[a, 2] * Tmp[2, b]
                for a in range(3):
                    p[a] += ds * (Rmid[a, 0] * vm[0] + Rmid[a, 1] * vm[1]
                                  + Rmid[a, 2] * vm[2])
                hat3s(u[0], u[1], u[2], hh)
                for a in range(3):
                    for b in range(3):
                        Tmp[a, b] = I3[a, b] + ds * hh[a, b]
                for a in range(3):
                    for b in range(3):
                        Rtmp[a, b] = R[a, 0] * Tmp[0, b] + R[a, 1] * Tmp[1, b] \
                            + R[a, 2] * Tmp[2, b]
                ortho3s(Rtmp, R)
                # 曲线 (⚠ C 用**端点 u**; C 与 κ 同节点)
                if want_curve:
                    Pc[n, idx, :] = p; Uc[n, idx, :] = u
                    Rc[n, idx, :, :] = R; Vc[n, idx, :] = v
                    kbt3s(u, lat, sEI, sGJ, sLs, Kbtm)
                    for a in range(3):
                        Cc[n, idx, a] = (Kbtm[a, 0] * u[0] + Kbtm[a, 1] * u[1]
                                         + Kbtm[a, 2] * u[2])
                        Nc[n, idx, a] = (Kse[a, 0] * (v[0] - E3[0])
                                         + Kse[a, 1] * (v[1] - E3[1])
                                         + Kse[a, 2] * (v[2] - E3[2]))
                    idx += 1

            if k == 0:                       # 段A→段B 界面的锚丝修正
                anchor_s(u, v, R, anchorA, anchorA.shape[0], tau[n], ri_all, Fs, Ls)
                for a in range(3):           # v -= Kse⁻¹ (Rᵀ Fs)
                    rr = R[0, a] * Fs[0] + R[1, a] * Fs[1] + R[2, a] * Fs[2]
                    v[a] -= rr / sGA if a < 2 else rr / sKR
                kbt3s(u, lat, sEI, sGJ, sLs, Kbt)
                # ⚠⚠ e 必须**先算一次**(用修正前的 u): numpy 版是 `solve(_kbt(u), RᵀLs)`,
                #   `_kbt` 只求值一次 ⇒ 两个分量共用同一个 e。
                #   若在循环内按分量重算 th, a=1 时会用到已被 a=0 改过的 u[0]
                #   ⇒ 只有 lat=True 时分岔 (lat=False 时 e≡sEI 恒定), 实测 d~1e-10 起累积到 3.8e-8。
                ee = Kbt[0, 0]
                for a in range(3):           # u -= Kbt⁻¹ (Rᵀ Ls)
                    rr = R[0, a] * Ls[0] + R[1, a] * Ls[1] + R[2, a] * Ls[2]
                    u[a] -= rr / ee if a < 2 else rr / sGJ
        # ── 末端残差 (Kse 为**段B**的) ──
        ne = np.empty(3); me = np.empty(3)
        vv = np.empty(3)
        for a in range(3):
            vv[a] = v[a] - E3[a]
        nv = Kse @ vv
        for a in range(3):
            ne[a] = R[a, 0] * nv[0] + R[a, 1] * nv[1] + R[a, 2] * nv[2]
        kbt3s(u, lat, segB[0], segB[3], segB[2], Kbt)
        mu = Kbt @ u
        for a in range(3):
            me[a] = R[a, 0] * mu[0] + R[a, 1] * mu[1] + R[a, 2] * mu[2]
        anchor_s(u, v, R, anchorB, anchorB.shape[0], tau[n], ri_all, Fs, Ls)
        for a in range(3):
            res[n, a] = ne[a] - Fs[a]
            res[n, 3 + a] = me[a] - Ls[a]
        if rod_on:
            res[n, 6] = Lz / rod_free_len - 1.0
        for a in range(3):
            tip[n, a] = p[a]

    return res, tip, Pc, Uc, Cc, Rc, Nc, Vc, S


# ═════════════════ 参数展平 (Python 侧, 一次性) ═════════════════
def pack_params(params):
    """把 `params` 展平成 numba 可吃的标量/数组 (与 `forward_batch` 的取值一致)."""
    from tendon_coupling import (L, L1, L2, SEG_A, SEG_B, _E3, _wire_r_vec,
                                 L_STRUT_DEFAULT, C0_DEFAULT)
    ri_all = np.ascontiguousarray(_wire_r_vec(params.r_disk))
    rho_total = params.rhoA + params.disk_mass / L
    few = np.array([0.0, 0.0, -rho_total * 9.81])
    cells = getattr(params, "cells", None)

    def _seg(cn):
        c = (cells[0] if cn == 0 else cells[-1]) if cells else None
        if c is not None:
            return (c.EI0, c.C0, c.L_strut, c.GJ, c.k_rod, params.GA)
        m = 2.0 if cn == 0 else 1.0
        return (m * params.EI, C0_DEFAULT, L_STRUT_DEFAULT, m * params.GJ,
                m * params.k_rod, m * params.GA)

    segA = np.array(_seg(0), float)
    segB = np.array(_seg(1), float)
    wiresA = np.arange(6, dtype=np.int64)
    wiresB = np.asarray(list(SEG_B), dtype=np.int64)
    anchorA = np.asarray(list(SEG_A), dtype=np.int64)
    anchorB = np.asarray(list(SEG_B), dtype=np.int64)
    return dict(ri_all=ri_all, segA=segA, segB=segB,
                E3=np.asarray(_E3, float), few=few,
                L1=float(L1), L2=float(L2),
                n_p=int(params.n_p), n_d=int(params.n_d),
                lat=bool(getattr(params, "lattice", False)),
                rod_on=bool(getattr(params, "rod_constraint", False)),
                rod_free_len=float(getattr(params, "rod_free_len", 1.0)),
                wiresA=wiresA, wiresB=wiresB,
                anchorA=anchorA, anchorB=anchorB)


def integrate(x, tau, params, want_curve=False):
    """numba 版前向 (与 `forward_batch` 接口对齐的薄封装)."""
    x = np.asarray(x, float)
    tau = np.asarray(tau, float)
    if x.ndim == 1:
        x = x[None, :]
    if tau.ndim == 1:
        tau = tau[None, :]
    pr = pack_params(params)
    out = integrate_all(x, tau, pr["ri_all"], pr["segA"], pr["segB"], pr["E3"],
                        pr["few"], pr["L1"], pr["L2"], pr["n_p"], pr["n_d"],
                        pr["lat"], pr["rod_on"], pr["rod_free_len"],
                        pr["wiresA"], pr["wiresB"], pr["anchorA"], pr["anchorB"],
                        bool(want_curve))
    if not want_curve:
        return out[0], out[1]
    Pc, Uc, Cc, Rc, Nc, Vc, S = out[2:]
    return out[0], out[1], (Pc, S, Uc, Cc, Rc, Nc, Vc)


# ═════════════════════════ S2 对拍 ═════════════════════════
def _selfcheck_s2(n_rep=6, seed=0, tol=1e-8):
    from tendon_coupling import forward_batch, TendonCouplingModel
    from ik_table import load_table
    M = TendonCouplingModel()
    P = M.params
    T = load_table("vc_table_60.npz")
    rng = np.random.default_rng(seed)
    print("=== S2 对拍: integrate_all  vs  forward_batch ===")
    keys = ["res", "tip", "Pc", "Rc", "Uc", "Cc", "Nc", "Vc"]
    worst = {k: 0.0 for k in keys}
    for rep in range(n_rep):
        N = 1 if rep % 2 == 0 else 13
        idx = rng.integers(0, len(T.recs), size=N)
        x = np.array([T.recs[i]["x"] for i in idx], float)
        tau = np.array([T.recs[i]["tau"] for i in idx], float)
        x = x + 0.0
        r0, p0, c0 = forward_batch(x, tau, P, want_curve=True)
        r1, p1, c1 = integrate(x, tau, P, want_curve=True)
        worst["res"] = max(worst["res"], np.abs(r0 - r1).max())
        worst["tip"] = max(worst["tip"], np.abs(p0 - p1).max())
        # c0/c1 = (Pc,S,Uc,Cc,Rc,Nc,Vc) —— 逐项比
        pairs = [("Pc", 0), ("Uc", 2), ("Cc", 3), ("Rc", 4), ("Nc", 5), ("Vc", 6)]
        for k, ci in pairs:
            worst[k] = max(worst[k], np.abs(np.asarray(c0[ci]) - np.asarray(c1[ci])).max())
    ok = True
    for k in keys:
        flag = "✅" if worst[k] < tol else "❌"
        ok &= (worst[k] < tol)
        print("  %-5s max|Δ| = %.3e   %s" % (k, worst[k], flag))
    print("S2 对拍:", "ALL PASS ✅" if ok else "FAIL ❌")
    return ok


if __name__ == "__main__":
    import time
    ok = _selfcheck_s2()
    # 性能
    from tendon_coupling import forward_batch, TendonCouplingModel
    from ik_table import load_table
    M = TendonCouplingModel(); P = M.params
    T = load_table("vc_table_60.npz")
    x = np.array([T.recs[500]["x"]], float)
    tau = np.array([T.recs[500]["tau"]], float)
    integrate(x, tau, P)                       # 预热/编译
    t0 = time.time()
    for _ in range(20):
        integrate(x, tau, P)
    t1 = time.time() - t0
    t0 = time.time()
    for _ in range(20):
        forward_batch(x, tau, P)
    t2 = time.time() - t0
    print("\n=== 性能 (N=1, 20 次平均) ===")
    print("  numba integrate : %7.3f ms" % (t1 / 20 * 1e3))
    print("  numpy forward   : %7.3f ms" % (t2 / 20 * 1e3))
    print("  加速比          : %.1f×" % (t2 / t1))
    sys.exit(0 if ok else 1)
