# -*- coding: utf-8 -*-
"""S1: numba 标量核 —— 与 `tendon_coupling` 的 numpy 批量版**数学一致**.

设计见 `_VEC_PLAN.md`。本文件只提供标量核 + 自检(与 numpy 版逐项对拍);
**不改动** `tendon_coupling.forward_batch`(S3 才切换, 且必须先对拍通过)。

跑自检:  python _jit_core.py
"""
import numpy as np
from numba import njit

# ═══════════════════════ 1) 3×3 反对称 ═══════════════════════
@njit(cache=True, inline="always")
def hat3s(v0, v1, v2, H):
    """对应 `hat_b` 的单样本版 (H 为输出缓冲, 避免分配)."""
    H[0, 0] = 0.0; H[0, 1] = -v2;  H[0, 2] = v1
    H[1, 0] = v2;  H[1, 1] = 0.0;  H[1, 2] = -v0
    H[2, 0] = -v1; H[2, 1] = v0;   H[2, 2] = 0.0


# ═══════════════ 2) 6×6 解 (对应 `_safe_solve6`) ═══════════════
@njit(cache=True, inline="always")
def solve6s(M, rhs, out):
    """高斯消元 + 部分选主元. 奇异/非有限 → 0; 末尾 clip(±1e3) (与原版一致)."""
    a = np.empty((6, 7))
    for i in range(6):
        for j in range(6):
            v = M[i, j]
            if not np.isfinite(v):
                for k in range(6):
                    out[k] = 0.0
                return
            a[i, j] = v
        rv = rhs[i]
        if not np.isfinite(rv):
            for k in range(6):
                out[k] = 0.0
            return
        a[i, 6] = rv
    for col in range(6):
        piv = col; mx = abs(a[col, col])
        for r in range(col + 1, 6):
            if abs(a[r, col]) > mx:
                mx = abs(a[r, col]); piv = r
        if mx < 1e-300:
            for k in range(6):
                out[k] = 0.0
            return
        if piv != col:
            for j in range(col, 7):
                t = a[col, j]; a[col, j] = a[piv, j]; a[piv, j] = t
        p = a[col, col]
        for r in range(col + 1, 6):
            f = a[r, col] / p
            if f != 0.0:
                for j in range(col, 7):
                    a[r, j] -= f * a[col, j]
    for i in range(5, -1, -1):
        s = a[i, 6]
        for j in range(i + 1, 6):
            s -= a[i, j] * out[j]
        out[i] = s / a[i, i]
    for k in range(6):
        if not np.isfinite(out[k]):
            out[k] = 0.0
        elif out[k] > 1e3:
            out[k] = 1e3
        elif out[k] < -1e3:
            out[k] = -1e3


# ═════════ 3) 正交化 (对应 `_orthonormalize_b`) ═════════
@njit(cache=True, inline="always")
def ortho3s(R, out):
    """Gram-Schmidt + cross; 非有限 → 单位阵."""
    n0 = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2 + R[2, 0] ** 2)
    if n0 < 1e-12:
        n0 = 1.0
    b0 = np.empty(3)
    for i in range(3):
        b0[i] = R[i, 0] / n0
    d = R[0, 1] * b0[0] + R[1, 1] * b0[1] + R[2, 1] * b0[2]
    b1 = np.empty(3)
    for i in range(3):
        b1[i] = R[i, 1] - d * b0[i]
    n1 = np.sqrt(b1[0] ** 2 + b1[1] ** 2 + b1[2] ** 2)
    if n1 < 1e-12:
        n1 = 1.0
    for i in range(3):
        b1[i] = b1[i] / n1
    b2 = np.cross(b0, b1)
    for i in range(3):
        out[i, 0] = b0[i]; out[i, 1] = b1[i]; out[i, 2] = b2[i]
    ok = True
    for i in range(3):
        for j in range(3):
            if not np.isfinite(out[i, j]):
                ok = False
    if not ok:
        for i in range(3):
            for j in range(3):
                out[i, j] = 1.0 if i == j else 0.0


# ═════ 4) 菱形晶胞割线刚度 (对应 `forward_batch._kbt`) ═════
@njit(cache=True, inline="always")
def kbt3s(u, lat, sEI, sGJ, sLs, M):
    """EI_eq = EI0·sinc(|κ_xy|·L_strut); κ→0 退化为 EI0. M 为 3×3 输出."""
    if lat:
        th = np.sqrt(u[0] ** 2 + u[1] ** 2) * sLs
        if th < 1e-9:
            e = sEI
        else:
            e = sEI * np.sin(th) / th
    else:
        e = sEI
    for i in range(3):
        for j in range(3):
            M[i, j] = 0.0
    M[0, 0] = e; M[1, 1] = e; M[2, 2] = sGJ


# ═════ 5) 六丝耦合 (对应 `_intermed_b_vec` / `_intermed_b`) ═════
@njit(cache=True, inline="always")
def intermed_s(u, v, Kse, Kbt, tau6, ri_all, fe_b, E3, A, B, G, H, c, d):
    """A,B,G,H,c,d 为输出缓冲. 数学与 `_intermed_b_vec` 完全一致(6 丝显式循环)."""
    uh = np.empty((3, 3))
    hat3s(u[0], u[1], u[2], uh)
    for i in range(3):
        for j in range(3):
            A[i, j] = 0.0; B[i, j] = 0.0; G[i, j] = 0.0; H[i, j] = 0.0
    av = np.zeros(3); bv = np.zeros(3)
    for w in range(6):
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
        Ar = Ai @ rih                      # 每丝只算一次 (原版: G -= Ai @ rih)
        Br = Bi @ rih                      #             H -= Bi @ rih
        for a in range(3):
            for b in range(3):
                A[a, b] += Ai[a, b]; B[a, b] += Bi[a, b]
                G[a, b] -= Ar[a, b]
                H[a, b] -= Br[a, b]
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


# ═══════════════════════ 自检: 与 numpy 版对拍 ═══════════════════════
def _selfcheck(n_rep=200, seed=0, tol=1e-9):
    from tendon_coupling import (_intermed_b_vec, _safe_solve6, _orthonormalize_b,
                                 hat_b, _E3, _wire_r_vec, TendonCouplingModel)
    rng = np.random.default_rng(seed)
    _prm = TendonCouplingModel().params
    ri_all = _wire_r_vec(_prm.r_disk)
    print("=== S1 对拍 (n=%d, tol=%.0e) ===" % (n_rep, tol))
    res = {}

    # ① hat3s
    e_hat = 0.0
    for _ in range(n_rep):
        vv = rng.normal(size=(1, 3))
        H0 = hat_b(vv)[0]
        H1 = np.empty((3, 3)); hat3s(vv[0, 0], vv[0, 1], vv[0, 2], H1)
        e_hat = max(e_hat, np.abs(H0 - H1).max())
    res["hat3s"] = e_hat

    # ② solve6s
    e_slv = 0.0
    for _ in range(n_rep):
        M = rng.normal(size=(1, 6, 6))
        r = rng.normal(size=(1, 6))
        o0 = _safe_solve6(M, r)[0]
        o1 = np.empty(6); solve6s(M[0], r[0], o1)
        e_slv = max(e_slv, np.abs(o0 - o1).max())
    res["solve6s"] = e_slv

    # ③ ortho3s
    e_ort = 0.0
    for _ in range(n_rep):
        R = rng.normal(size=(1, 3, 3))
        o0 = _orthonormalize_b(R)[0]
        o1 = np.empty((3, 3)); ortho3s(R[0], o1)
        e_ort = max(e_ort, np.abs(o0 - o1).max())
    res["ortho3s"] = e_ort

    # ④ intermed_s (含 Kbt 一致性)
    from tendon_coupling import L1, L, SEG_A  # noqa
    e_int = 0.0
    for _ in range(n_rep):
        u = rng.normal(size=(1, 3)); v = 1.0 + 0.01 * rng.normal(size=(1, 3))
        Kse = np.zeros((1, 3, 3)); Kse[0, 0, 0] = Kse[0, 1, 1] = 5e3; Kse[0, 2, 2] = 8e3
        K = np.zeros((1, 3, 3)); K[0, 0, 0] = K[0, 1, 1] = 3e3; K[0, 2, 2] = 4e3
        tau = rng.uniform(1, 50, size=(1, 6))
        fe = rng.normal(size=(1, 3)) * 1e-2
        r0 = _intermed_b_vec(u, v, Kse, K, np.arange(6), tau, ri_all, fe)
        bufs = [np.zeros((3, 3)) for _ in range(4)] + [np.zeros(3), np.zeros(3)]
        intermed_s(u[0], v[0], Kse[0], K[0], tau[0], ri_all, fe[0],
                   np.asarray(_E3, float), *bufs)
        for a, b in zip(r0, bufs):
            e_int = max(e_int, np.abs(np.asarray(a)[0] - b).max())
    res["intermed_s"] = e_int

    ok = True
    for k, e in res.items():
        flag = "✅" if e < tol else "❌"
        ok &= (e < tol)
        print("  %-12s max|Δ| = %.3e   %s" % (k, e, flag))
    print("S1 对拍:", "ALL PASS ✅" if ok else "FAIL ❌")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selfcheck() else 1)
