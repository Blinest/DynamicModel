# -*- coding: utf-8 -*-
"""check_shoot_jac.py — 解析雅可比接进 `shooting_batch` 的对拍与计时

三项：
  ① 同一 x₀ 处，解析 J vs 有限差分 J（前向 / 中心）逐元素相对差
  ② 端到端：同一 τ、同一起点，`TC_JAC_MODE=analytic|forward|central` 三条路径的
     converged / n_iter / x / res / tip 对比
  ③ 单次 J 计算的墙钟计时（解析 vs 前向 vs 中心）

跑: TC_JAC_MODE=analytic python tools/varjac/check_shoot_jac.py [--nsub 30]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
os.environ.setdefault("TC_NO_JIT", "1")
os.environ["TC_JAC_MODE"] = "analytic"          # 必须在 import 之前设

import tendon_coupling as tc  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    if not ok:
        FAIL.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsub", type=int, default=30)
    ap.add_argument("--mu", type=float, default=0.0)
    a = ap.parse_args()

    M = tc.TendonCouplingModel()
    P = M.params
    P.n_p = a.nsub; P.n_d = a.nsub
    P.mu_friction = a.mu
    BASE = np.array([40.0, 20.0, 40.0, 20.0, 40.0, 10.0])

    print(f"工况: nsub={a.nsub}  mu={a.mu}  TC_JAC_MODE={tc._JAC_MODE}  "
          f"res_jac={'OK' if tc._load_res_jac() else '不可用'}  "
          f"can_analytic={tc._can_analytic(P, 1)}")

    # ── 参考解（中心差分冷启动） ──
    tc._JAC_MODE = "central"
    o_ref = tc.shooting_batch(BASE[None, :], P, tol=1e-8, max_iter=20)
    x_ref = np.asarray(o_ref["x"][0], float)
    print(f"参考解: converged={bool(o_ref['converged'][0])}  "
          f"n_iter={int(o_ref['n_iter'][0])}  |res|={np.linalg.norm(o_ref['res'][0]):.3e}")

    # ── ① 同一 x₀ 处比 J ──
    rj = tc._load_res_jac()
    J_ana = np.asarray(rj(x_ref, BASE, P)[2], float)
    J_fwd = tc._jac_diff(x_ref[None, :], BASE[None, :], np.asarray(o_ref["res"][0], float)[None, :],
                         P, mode="forward")[0]
    J_cen = tc._jac_diff(x_ref[None, :], BASE[None, :], np.asarray(o_ref["res"][0], float)[None, :],
                         P, mode="central")[0]

    def rel(A, B):
        d = np.max(np.abs(A - B))
        s = max(np.max(np.abs(B)), 1e-30)
        return d, d / s

    print(f"[①] 同一 x₀ 处 J 对比（|J|max = {np.max(np.abs(J_cen)):.4e}）")
    for tag, Jx in (("前向差分", J_fwd), ("中心差分", J_cen)):
        d, r = rel(J_ana, Jx)
        print(f"    解析 vs {tag}: max|Δ|={d:.3e}  相对={r:.3e}")
    _, r_cen = rel(J_ana, J_cen)
    check("① 解析 vs 中心差分 < 1e-6", r_cen < 1e-6, f"相对 {r_cen:.3e}")

    # ── ② 端到端三条路径 ──
    print("[②] 端到端（冷启动, 直臂初值）")
    res_out = {}
    for mode in ("analytic", "forward", "central"):
        tc._JAC_MODE = mode
        t0 = time.perf_counter()
        o = tc.shooting_batch(BASE[None, :], P, tol=1e-8, max_iter=20)
        dt = time.perf_counter() - t0
        res_out[mode] = (o, dt)
        print(f"    {mode:9s}: converged={bool(o['converged'][0])} "
              f"n_iter={int(o['n_iter'][0])} |res|={np.linalg.norm(o['res'][0]):.3e} "
              f"tip_z={float(o['tip'][0, 2])*1e3:.6f} mm  墙钟 {dt:.2f} s")

    oa, ta_ = res_out["analytic"]
    ok_conv = all(bool(res_out[m][0]["converged"][0]) for m in res_out)
    check("② 三条路径都收敛", ok_conv)
    if ok_conv:
        dx_f = float(np.max(np.abs(oa["x"][0] - res_out["forward"][0]["x"][0])))
        dx_c = float(np.max(np.abs(oa["x"][0] - res_out["central"][0]["x"][0])))
        dt_f = float(np.max(np.abs(oa["tip"][0] - res_out["forward"][0]["tip"][0]))) * 1e3
        dt_c = float(np.max(np.abs(oa["tip"][0] - res_out["central"][0]["tip"][0]))) * 1e3
        dr_f = abs(float(np.linalg.norm(oa["res"][0])) - float(np.linalg.norm(res_out["forward"][0]["res"][0])))
        dr_c = abs(float(np.linalg.norm(oa["res"][0])) - float(np.linalg.norm(res_out["central"][0]["res"][0])))
        print(f"    解析 vs 前向: max|Δx|={dx_f:.3e}  Δtip={dt_f:.2e} mm  Δ|res|={dr_f:.2e}")
        print(f"    解析 vs 中心: max|Δx|={dx_c:.3e}  Δtip={dt_c:.2e} mm  Δ|res|={dr_c:.2e}")
        check("② 解析 vs 中心 Δtip < 1e-2 mm", dt_c < 1e-2, f"{dt_c:.3e} mm")
        check("② 解析 vs 中心 Δ|res| < 1e-8", dr_c < 1e-8, f"{dr_c:.2e}")

    # ── ③ 单次 J 计时 ──
    print("[③] 单次 J 计算计时（中位, 5 次）")

    def t_of(fn, n=5):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    r_arr = np.asarray(o_ref["res"][0], float)[None, :]
    t_ana = t_of(lambda: rj(x_ref, BASE, P))
    t_fwd = t_of(lambda: tc._jac_diff(x_ref[None, :], BASE[None, :], r_arr, P, mode="forward"))
    t_cen = t_of(lambda: tc._jac_diff(x_ref[None, :], BASE[None, :], r_arr, P, mode="central"))
    print(f"    解析    : {t_ana*1e3:8.2f} ms")
    print(f"    前向差分: {t_fwd*1e3:8.2f} ms   → 解析快 {t_fwd/t_ana:.2f}×")
    print(f"    中心差分: {t_cen*1e3:8.2f} ms   → 解析快 {t_cen/t_ana:.2f}×")
    check("③ 解析快于中心差分", t_ana < t_cen, f"{t_cen/t_ana:.2f}×")

    tc._JAC_MODE = "analytic"
    print("=" * 74)
    if FAIL:
        print(f"✗ {len(FAIL)} 项失败: {FAIL}")
        return 1
    print("✓ 全部通过 —— 解析 J 可接进 shooting_batch（单样本、µ=0、无 F_tip）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
