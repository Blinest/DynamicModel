# -*- coding: utf-8 -*-
"""check_reachability_floor.py — 可达性判据 / 残差地板 推导的数值验证
==========================================================================================
验证对象: `inverse_solver.LookupInverseSolver.reachability()` 的**线性化残差地板**

    当前实现 (判据 A):  沿最小范数方向 dtau = -J⁺e 取**等比缩放**的最大可行步长 s*,
                        地板 = (1 - s*) * ||e||
    推导给出的真值 (B): 箱约束最小二乘的真最优
                        floor* = min_{tau' ∈ [0, τmax]^6} || e + J (tau' - τ) ||

本脚本随机生成 (J, e, τ, τmax) 大量用例, 检查:
  ① 判据 A 是 **保守上界**:  floor_A ≥ floor*   (恒成立, 否则推导或实现有错)
  ② s* 闭式公式 vs 一维暴力网格扫描一致
  ③ A 与 B 的差距有多大 —— 即"判据 A 会不会把可达目标误判为不可达"
     (对应 inverse_solver 里注释所说的"局部一阶判据误判"反例)

box-QP 真值用**穷举 3^6 = 729 个活跃集 + KKT 校验**求得 (6 维下严格精确),
再用投影梯度下降独立复核, 两者必须一致。

跑: python tools/check_reachability_floor.py   (只需 numpy)
"""
from __future__ import annotations

import itertools
import sys

import numpy as np


# ═══════════════════════════ 判据 A (当前实现) ═══════════════════════════
def criterion_A(J, e, tau, tau_max, tol_frac=0.5):
    """复刻 `reachability()`: 最小范数方向 + 等比步长 + (1-s)*||e|| 地板."""
    tau = np.asarray(tau, float).reshape(6)
    e = np.asarray(e, float).reshape(-1)
    S = J @ J.T + 1e-10 * np.eye(6)
    try:
        dtau = -J.T @ np.linalg.solve(S, e)
    except np.linalg.LinAlgError:
        dtau = -J.T @ (np.linalg.pinv(S) @ e)
    n0 = float(np.linalg.norm(e))
    eps = 1e-12
    s = 1.0
    for k in range(6):
        dk = float(dtau[k])
        if dk > eps:
            s = min(s, (tau_max - tau[k]) / dk)
        elif dk < -eps:
            s = min(s, tau[k] / (-dk))
    s = float(max(0.0, min(1.0, s)))
    floor = float((1.0 - s) * n0)
    return {"dtau": dtau, "s": s, "floor": floor, "now": n0,
            "feasible": bool(floor <= tol_frac * n0 + 1e-12)}


# ═════════════════ 判据 B: 箱约束最小二乘真值 (穷举活跃集 + KKT) ═════════════════
def criterion_B_exact(J, e, tau, tau_max):
    """floor* = min_{d ∈ [lo-τ, hi-τ]} ||e + J d|| 的**精确**解.

    6 维箱-QP: 最优点的每个分量要么贴下界/上界, 要么自由。
    枚举 3^6 模式, 对每种模式解等式约束 LS, 再用 KKT 号条件判定其是否真是最优。
    """
    lo = np.zeros(6) - np.asarray(tau, float).reshape(6)
    hi = np.full(6, float(tau_max)) - np.asarray(tau, float).reshape(6)
    e = np.asarray(e, float).reshape(-1)
    best = None
    for pat in itertools.product((0, 1, 2), repeat=6):      # 0=自由 1=下界 2=上界
        free = [k for k in range(6) if pat[k] == 0]
        d = np.zeros(6)
        for k in range(6):
            if pat[k] == 1:
                d[k] = lo[k]
            elif pat[k] == 2:
                d[k] = hi[k]
        if free:
            Jf = J[:, free]
            r = e + J @ d
            try:
                dz = -np.linalg.lstsq(Jf, r, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            for i, k in enumerate(free):
                d[k] = dz[i]
        # 可行性
        if np.any(d < lo - 1e-9) or np.any(d > hi + 1e-9):
            continue
        g = J.T @ (e + J @ d)                                # ∂(||·||²)/∂d / 2
        ok = True
        for k in range(6):
            if pat[k] == 1 and g[k] < -1e-7:                 # 在下界但梯度指向更小 → 应离开
                ok = False
            elif pat[k] == 2 and g[k] > 1e-7:
                ok = False
        if not ok:
            continue
        val = float(np.linalg.norm(e + J @ d))
        if best is None or val < best[0]:
            best = (val, d.copy())
    if best is None:                                          # 数值兜底
        d = np.clip(-np.linalg.pinv(J) @ e, lo, hi)
        return {"floor": float(np.linalg.norm(e + J @ d)), "d": d}
    return {"floor": best[0], "d": best[1]}


# ═════════════ 判据 B 的独立复核: 投影梯度下降 (只用于交叉验证) ═════════════
def criterion_B_pgd(J, e, tau, tau_max, iters=200000):
    lo = np.zeros(6) - np.asarray(tau, float).reshape(6)
    hi = np.full(6, float(tau_max)) - np.asarray(tau, float).reshape(6)
    e = np.asarray(e, float).reshape(-1)
    L = float(np.linalg.norm(J.T @ J, 2)) or 1.0
    d = np.clip(-np.linalg.pinv(J) @ e, lo, hi)
    step = 1.0 / L
    best = float(np.linalg.norm(e + J @ d))
    for i in range(iters):
        g = J.T @ (e + J @ d)
        d = np.clip(d - step * g, lo, hi)
        v = float(np.linalg.norm(e + J @ d))
        if v < best:
            best = v
    return best


# ═══════════════════════════════ 验证主流程 ═══════════════════════════════
def main(n_rep: int = 300, seed: int = 0, verbose: bool = True) -> bool:
    rng = np.random.default_rng(seed)
    ok_all = True
    worst_gap = 0.0
    n_conservative = 0          # A 判不可达、B 判可达 (误判)
    n_any_gap = 0
    # ① 判据 A 必须是保守上界; ② 与独立复核一致
    for rep in range(n_rep):
        J = rng.normal(size=(6, 6))
        if rep % 3 == 0:                                    # 制造病态 (贴近真实: cond≈2850)
            U, _, Vt = np.linalg.svd(J)
            sv = np.array([23.4, 21.8, 1.51, 1.12, 0.146, 0.0082])
            J = U @ np.diag(sv) @ Vt
        e = rng.normal(size=6) * (10.0 ** rng.uniform(-1, 2))
        tau_max = 60.0
        tau = rng.uniform(0.0, tau_max, size=6)
        A = criterion_A(J, e, tau, tau_max)
        B = criterion_B_exact(J, e, tau, tau_max)
        if A["floor"] < B["floor"] - 1e-8 * max(1.0, A["now"]):
            print(f"  ❌ rep{rep}: 判据 A 地板 {A['floor']:.6g} < 真值 {B['floor']:.6g} "
                  f"→ A 不是上界 (推导/实现有错)")
            ok_all = False
            continue
        gap = A["floor"] - B["floor"]
        if gap > 1e-9:
            n_any_gap += 1
        worst_gap = max(worst_gap, gap / max(A["now"], 1e-12))
        if (not A["feasible"]) and (B["floor"] <= 0.5 * A["now"]):
            n_conservative += 1
    # 独立复核: 投影梯度 vs 活跃集枚举 (小样本, 慢)
    for rep in range(12):
        J = rng.normal(size=(6, 6))
        e = rng.normal(size=6) * 10.0
        tau = rng.uniform(0, 60, size=6)
        B1 = criterion_B_exact(J, e, tau, 60.0)["floor"]
        B2 = criterion_B_pgd(J, e, tau, 60.0, iters=6000)
        if abs(B1 - B2) > 1e-4 * max(1.0, B1):
            print(f"  ❌ box-QP 复核不一致: 活跃集 {B1:.6g} vs PGD {B2:.6g}")
            ok_all = False
    # ③ s* 闭式 vs 一维暴力扫描
    for rep in range(60):
        J = rng.normal(size=(6, 6))
        e = rng.normal(size=6) * 10.0
        tau = rng.uniform(0, 60, size=6)
        A = criterion_A(J, e, tau, 60.0)
        grid = np.linspace(0, 1, 20001)
        feas = np.ones_like(grid, bool)
        for k in range(6):
            d = A["dtau"][k]
            lo_k, hi_k = -tau[k], 60.0 - tau[k]
            feas &= (grid * d >= lo_k - 1e-12) & (grid * d <= hi_k + 1e-12)
        s_brute = float(grid[feas].max()) if feas.any() else 0.0
        if abs(s_brute - A["s"]) > 1e-4:
            print(f"  ❌ s* 闭式 {A['s']:.6f} vs 暴力 {s_brute:.6f}")
            ok_all = False

    if verbose:
        print("=" * 74)
        print(f"用例数 {n_rep} | A 为保守上界: {'✅ 恒成立' if ok_all else '❌ 失败'}")
        print(f"地板相对差距 (A-B)/||e|| 最大 {worst_gap:.3e} | 有差距的用例 "
              f"{n_any_gap}/{n_rep} ({100.0*n_any_gap/n_rep:.1f}%)")
        print(f"★ 判据 A 把**可达**目标误判为不可达: {n_conservative}/{n_rep} "
              f"({100.0*n_conservative/n_rep:.1f}%)  ← 这就是保守性代价")
        print("=" * 74)
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
