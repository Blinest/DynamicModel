# -*- coding: utf-8 -*-
"""check_stage_jac.py — 解析变分方程（stage 级）与有限差分逐元素对拍

三级验证：
  ① intermed() 与 tendon_coupling._intermed_b_vec 一致（数学搬移没搬错）
  ② intermed_jac() 与 intermed() 的中心差分一致（∂A/∂y … ∂d/∂y 解析式正确）
  ③ stage_f_jac() 与 stage f 的中心差分一致（含 ∂f/∂y = M⁻¹(∂r/∂y − ∂M/∂y·f)）

跑: TC_NO_JIT=1 python tools/varjac/check_stage_jac.py [--n 5]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
os.environ.setdefault("TC_NO_JIT", "1")

from varjac import (  # noqa: E402
    intermed, intermed_jac, kbt, kbt_jac, stage_f_jac, _assemble,
)
from tendon_coupling import (  # noqa: E402
    TendonCouplingModel, _intermed_b_vec, _E3, _wire_r_vec,
    L_STRUT_DEFAULT, C0_DEFAULT,
)

FAIL = []


def pack_params(params):
    """本地版参数展平（`_jit_integrate.pack_params` 的等价子集）。

    ⚠ 不能直接 import `_jit_integrate`：它在模块顶层无条件 `from numba import njit`，
       无 numba 环境直接 ImportError（本沙箱就是这种情况）。此处只取本检查需要的字段。
    """
    cells = getattr(params, "cells", None)

    def _seg(cn):
        c = (cells[0] if cn == 0 else cells[-1]) if cells else None
        if c is not None:
            return (c.EI0, c.C0, c.L_strut, c.GJ, c.k_rod, params.GA)
        m = 2.0 if cn == 0 else 1.0
        return (m * params.EI, C0_DEFAULT, L_STRUT_DEFAULT, m * params.GJ,
                m * params.k_rod, m * params.GA)

    return dict(
        ri_all=np.ascontiguousarray(_wire_r_vec(params.r_disk)),
        segA=np.array(_seg(0), float), segB=np.array(_seg(1), float),
        E3=np.asarray(_E3, float),
        lat=bool(getattr(params, "lattice", False)),
        wiresA=list(range(6)),
        n_p=int(params.n_p), n_d=int(params.n_d),
    )


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    if not ok:
        FAIL.append(name)


def rel(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = np.max(np.abs(a - b))
    s = max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-30)
    return d, d / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="用例数")
    ap.add_argument("--h", type=float, default=1e-6, help="中心差分步长")
    a = ap.parse_args()

    M = TendonCouplingModel()
    P = M.params
    pr = pack_params(P)
    ri_all = pr["ri_all"]; segA = pr["segA"]; segB = pr["segB"]
    lat = pr["lat"]; E3 = pr["E3"]
    wiresA = list(pr["wiresA"])
    sEI, sGJ = segA[0], segA[3]
    sLs = segA[2]
    print(f"参数: lat={lat}  EI0={sEI:.6g}  GJ={sGJ:.6g}  L_strut={sLs:.6g}  "
          f"W={len(wiresA)}  n_p={pr['n_p']} n_d={pr['n_d']}")

    rng = np.random.default_rng(20261013)
    tau = rng.uniform(1.0, 60.0, size=6)

    e1_int = e2_int = e3_int = e4_kbt = 0.0
    for trial in range(a.n):
        u = rng.normal(0, 0.6, size=3)
        if trial == 0:
            u = np.array([1.2, -0.4, 0.2])          # 含 z 分量, 检验 lat 只对 xy 有效
        v = np.array([0.0, 0.0, 1.0]) + rng.normal(0, 0.02, size=3)
        Kse = np.diag([segA[5], segA[5], segA[4]])   # [GA,GA,k_rod]
        fe_b = rng.normal(0, 0.01, size=3)
        y = np.concatenate([u, v])

        # ── ① 与向量化参考实现一致 ──
        Kbt = kbt(u, lat, sEI, sGJ, sLs)
        mine = intermed(u, v, Kse, Kbt, tau[list(wiresA)], ri_all[list(wiresA)], fe_b)
        ref = _intermed_b_vec(u[None, :], v[None, :], Kse, Kbt[None, :, :],
                              wiresA, tau[None, :], ri_all, fe_b[None, :])
        d, r = 0.0, 0.0
        for i in range(6):
            aa = np.asarray(mine[i], float)
            bb = np.asarray(ref[i], float).reshape(aa.shape)
            dd, rr = rel(aa, bb)
            r = max(r, rr)
        e1_int = max(e1_int, r)

        # ── ② intermed_jac vs 中心差分 ──
        dKt = kbt_jac(u, lat, sEI, sGJ, sLs)
        dKbt_dy = np.zeros((3, 3, 6)); dKbt_dy[:, :, :3] = dKt
        jac = intermed_jac(u, v, Kse, Kbt, tau[list(wiresA)],
                           ri_all[list(wiresA)], fe_b, dKbt_dy)
        h = a.h
        shap = [np.shape(x) for x in mine]
        fd33 = np.zeros((4, 3, 3, 6))          # [comp][i][j][k] for A,B,G,H
        fd3 = np.zeros((2, 3, 6))              # [comp][i][k]      for c,d
        for k in range(6):
            yp = y.copy(); ym = y.copy()
            yp[k] += h; ym[k] -= h
            up, vp = yp[:3], yp[3:]
            um_, vm_ = ym[:3], ym[3:]
            Kp = kbt(up, lat, sEI, sGJ, sLs); Km = kbt(um_, lat, sEI, sGJ, sLs)
            op = intermed(up, vp, Kse, Kp, tau[list(wiresA)], ri_all[list(wiresA)], fe_b)
            om = intermed(um_, vm_, Kse, Km, tau[list(wiresA)], ri_all[list(wiresA)], fe_b)
            for comp in range(6):
                num = (np.asarray(op[comp], float) - np.asarray(om[comp], float)) / (2 * h)
                if comp < 4:
                    fd33[comp, :, :, k] = num
                else:
                    fd3[comp - 4, :, k] = num
        for comp in range(6):
            ana = np.asarray(jac[comp], float)          # (3,3,6) 或 (3,6)
            num = fd33[comp] if comp < 4 else fd3[comp - 4]
            _, r = rel(ana, num)
            e2_int = max(e2_int, r)

        # ── ③ stage_f_jac vs 中心差分 ──
        f0, J = stage_f_jac(y, Kse, tau[list(wiresA)], ri_all[list(wiresA)],
                            fe_b, lat, sEI, sGJ, sLs)

        def stage(yy):
            uu, vv = yy[:3], yy[3:]
            Kb = kbt(uu, lat, sEI, sGJ, sLs)
            A, B, G, H, c, d = intermed(uu, vv, Kse, Kb, tau[list(wiresA)],
                                        ri_all[list(wiresA)], fe_b)
            Mm, rr = _assemble(Kse, Kb, A, B, G, H, c, d)
            return np.linalg.solve(Mm, rr)

        Jfd = np.zeros((6, 6))
        for k in range(6):
            yp = y.copy(); ym = y.copy()
            yp[k] += h; ym[k] -= h
            Jfd[:, k] = (stage(yp) - stage(ym)) / (2 * h)
        d, r = rel(J, Jfd)
        e3_int = max(e3_int, r)

        # ── ④ kbt_jac vs 中心差分 ──
        # ⚠ 归一化用 **Kbt 本身** 的尺度, 不能用导数矩阵自身的 max:
        #   ∂Kbt/∂u 只有 ~1e-5 量级, 而 FD 噪声底 = eps·|Kbt|/(2h) ≈ 5.7e-11
        #   ⇒ 除以导数自身会把噪声放大成"1e-5 的相对误差"假警报。
        scale_kbt = max(np.max(np.abs(Kbt)), 1e-30)
        for k in range(3):
            up = u.copy(); um_ = u.copy()
            up[k] += h; um_[k] -= h
            num = (kbt(up, lat, sEI, sGJ, sLs) - kbt(um_, lat, sEI, sGJ, sLs)) / (2 * h)
            e4_kbt = max(e4_kbt, np.max(np.abs(dKt[:, :, k] - num)) / scale_kbt)

    print(f"[①] intermed vs _intermed_b_vec : max 相对差 {e1_int:.3e}")
    check("① 数学搬移一致 (<1e-12)", e1_int < 1e-12)
    print(f"[②] intermed_jac vs 中心差分    : max 相对差 {e2_int:.3e}")
    check("② ∂(A,B,G,H,c,d)/∂y 解析正确 (<1e-7)", e2_int < 1e-7)
    print(f"[③] stage_f_jac vs 中心差分      : max 相对差 {e3_int:.3e}")
    check("③ ∂f/∂y 解析正确 (<1e-6)", e3_int < 1e-6)
    print(f"[④] kbt_jac vs 中心差分         : max 相对差 {e4_kbt:.3e}")
    check("④ ∂Kbt/∂u 解析正确 (<1e-7)", e4_kbt < 1e-7)

    print("=" * 74)
    if FAIL:
        print(f"✗ {len(FAIL)} 项失败: {FAIL}")
        return 1
    print("✓ 全部通过 —— ∂f/∂y 解析式可用，下一步接变分方程传播出 ∂res/∂x₀。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
