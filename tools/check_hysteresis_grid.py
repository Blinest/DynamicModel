# -*- coding: utf-8 -*-
"""check_hysteresis_grid.py — 滞回环"跳变"的根因复核（离散误差 vs 求解未收敛）
==========================================================================================
背景: `tools/plot_friction.py` 的第 (6) 图（环宽 vs τ 缩放）在 τ 缩放 ≈0.75 附近非单调
（0.14 → 0.68 → 0.90 mm）。粗网格下无法区分两种原因:

    (a) 离散误差: 子步太粗, 加载/卸载两支各自算偏;
    (b) 求解未收敛: 某些 (τ, 支) 点上打靶没收敛, 环宽被垃圾解污染。

本脚本对每个 nsub 同时跑三组**批量**求解（批量才能在一轮里覆盖整条 τ 扫描）:

    L      = 加载支 (+μ), 直臂冷启动
    U_cold = 卸载支 (-μ), 直臂冷启动
    U_warm = 卸载支 (-μ), **以 L 的解为初值** (物理上两支对应同一配置族, 是最自然的初值)

输出每个 τ 缩放下: tip_z、环宽、两支 x 解之差、converged 标志、残差范数,
从而判断跳变究竟来自网格还是来自未收敛点, 以及热启动能否消除它。

跑: python tools/check_hysteresis_grid.py [--mu 0.15] [--grids 30,60,100] [--points 7]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _has_numba() -> bool:
    try:
        import numba  # noqa: F401
        return True
    except Exception:
        return False


if not _has_numba():
    os.environ["TC_NO_JIT"] = "1"

from tendon_coupling import (  # noqa: E402
    TendonCouplingModel, forward_batch, shooting_batch, shooting_batch_retry,
)

BASE_TAU = np.array([40.0, 20.0, 40.0, 20.0, 40.0, 10.0])


def solve_batch(taus, mu, branch, nsub, x0=None, max_iter=12, retry=False):
    """一次批量求解 N 个 τ 样本 (同一 μ / 同一支)。"""
    M = TendonCouplingModel()
    P = M.params
    P.n_p = nsub; P.n_d = nsub
    P.mu_friction = float(mu); P.friction_branch = float(branch)
    taus = np.atleast_2d(np.asarray(taus, float))
    if retry:
        out = shooting_batch_retry(taus, P, tol=1e-8, max_iter=max_iter)
    else:
        out = shooting_batch(taus, P, x0=x0, tol=1e-8, max_iter=max_iter)
    res, tip = forward_batch(out["x"], taus, P)
    return out, res, tip


def _row(tag, s, tip, out, res, ref_x=None):
    z = float(tip[0, 2]) * 1000.0
    rn = float(np.linalg.norm(res[0, :6]))
    dx = (float(np.max(np.abs(out["x"][0] - ref_x))) if ref_x is not None else float("nan"))
    return dict(tag=tag, scale=s, z=z, res=rn, conv=bool(out["converged"][0]),
                nit=int(out["n_iter"][0]), dx=dx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=0.15)
    ap.add_argument("--grids", type=str, default="30,60,100")
    ap.add_argument("--points", type=int, default=7)
    a = ap.parse_args()
    grids = [int(v) for v in a.grids.split(",")]
    scales = np.linspace(0.35, 1.0, a.points)
    TAU = BASE_TAU[None, :] * scales[:, None]

    print(f"μ={a.mu}  点数={a.points}  nsub={grids}    (全部用批量求解)")
    for n in grids:
        oL, rL, tL = solve_batch(TAU, a.mu, +1.0, n)                 # 加载
        oUc, rUc, tUc = solve_batch(TAU, a.mu, -1.0, n)              # 卸载·冷启动
        oUw, rUw, tUw = solve_batch(TAU, a.mu, -1.0, n, x0=oL["x"])  # 卸载·热启动
        print(f"\n──── nsub={n} ────")
        print(f"{'scale':>6} | {'zL':>9} {'zU冷':>9} {'zU热':>9} | "
              f"{'环宽冷':>8} {'环宽热':>8} | {'convL/cold/warm':>16} | {'resU冷':>9} {'resU热':>9}")
        w_cold, w_warm = [], []
        for i, s in enumerate(scales):
            zL = float(tL[i, 2]) * 1000.0
            zc = float(tUc[i, 2]) * 1000.0
            zw = float(tUw[i, 2]) * 1000.0
            wc, ww = zL - zc, zL - zw
            w_cold.append(wc); w_warm.append(ww)
            print(f"{s:>6.3f} | {zL:>9.3f} {zc:>9.3f} {zw:>9.3f} | "
                  f"{wc:>8.4f} {ww:>8.4f} | "
                  f"{int(bool(oL['converged'][i]))}/{int(bool(oUc['converged'][i]))}/"
                  f"{int(bool(oUw['converged'][i])):>12} | "
                  f"{np.linalg.norm(rUc[i, :6]):>9.2e} {np.linalg.norm(rUw[i, :6]):>9.2e}")
        w_cold = np.asarray(w_cold); w_warm = np.asarray(w_warm)
        # 只看双方都收敛的点
        ok = (oL["converged"] & oUw["converged"])
        print(f"  · 冷启动: 环宽 max|w|={np.max(np.abs(w_cold)):.4f} mm, "
              f"最大跳变={np.max(np.abs(np.diff(w_cold))):.4f} mm, "
              f"未收敛点={int((~oUc['converged']).sum())}/{len(scales)}")
        print(f"  · 热启动: 环宽 max|w|={np.max(np.abs(w_warm)):.4f} mm, "
              f"最大跳变={np.max(np.abs(np.diff(w_warm))):.4f} mm, "
              f"未收敛点={int((~oUw['converged']).sum())}/{len(scales)}")
        if ok.sum() >= 3:
            ww_ok = w_warm[ok]
            mono = np.all(np.diff(ww_ok) > -1e-3)
            print(f"  · 双方收敛的 {int(ok.sum())} 个点: 环宽 "
                  f"{np.round(ww_ok, 4).tolist()} → 单调递增={bool(mono)}")
        print(f"  · 卸载支打靶迭代数: 冷 {oUc['n_iter'].tolist()} / 热 {oUw['n_iter'].tolist()}")
    print("""
判读:
  · 若"未收敛点"恰好落在跳变位置, 且热启动后这些点全部收敛、环宽变单调
    ⇒ 跳变根因是**求解未收敛**(b), 修法是把卸载支热启动(或 retry), 并在出图前按
      converged / 残差阈值过滤 —— 与离散网格粗细无关。
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
