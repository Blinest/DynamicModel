# -*- coding: utf-8 -*-
"""build_real_arm.py — 用**实机臂参数**建位姿表（表与标定同源）

现有 `vc_table_20/40.npz` 是按**默认臂**建的（`EI=0.1294`、`r_disk=20 mm`、
α=12°），而 `tdcr_model.CALIB` 描述的是**实机臂**（`EI=0.045`、`EA=7e3`、
`r_disk=5 mm`、Φ16、α=5°）。两者不是同一套模型 —— `tdcr_pack.py info`
会明确报警。

本脚本用 `build_arm()` 生成的**实机 params**去建表，存成
`vc_table_real_<tau>.npz`，与 `CALIB` 同源。

    python tools/build_real_arm.py --tau 20 --n 120 --no-save     # 试算验证
    python tools/build_real_arm.py --tau 20 40 --n 900            # 正式建

与 `main.py build` 的区别只有一处：`params` 来自 `build_arm` 而不是
`default_params()`。其余（LHS 采样 / 16 级延拓 / refine / 版本键）完全一致。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TC_NO_JIT", "1")

from ik_table import build_table, save_table          # noqa: E402
from tendon_coupling import build_arm                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, nargs="+", default=[20.0])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--no-save", action="store_true", help="只跑不存盘（试算）")
    a = ap.parse_args()

    ca, cb, P = build_arm()
    print(f"实机臂: EI={P.EI:.4g} N·m²  EA(k_rod)={P.k_rod:.4g} N  "
          f"L={P.L_total*1e3:.0f}mm  r_disk={P.r_disk*1e3:.0f}mm  "
          f"α={ca.alpha_deg:.3f}°  N_long={ca.N_long:.0f}", flush=True)
    print(f"        EI0 段A/段B = {ca.EI0:.5f} / {cb.EI0:.5f}  "
          f"locked={ca.locked}", flush=True)

    for t in a.tau:
        t0 = time.time()
        recs = build_table(N=a.n, tau_max=t, params=P, seed=a.seed,
                           batch_size=a.batch,
                           continuation=tuple(np.linspace(0.1, 1.0, 16)),
                           verbose=True)
        dt = (time.time() - t0) / 60
        print(f"  τmax={t:g}N: 入库 {len(recs)}/{a.n}  用时 {dt:.1f} min  "
              f"({dt*60/max(len(recs),1):.1f} s/点)", flush=True)
        if a.no_save:
            continue
        path = f"vc_table_real_{t:g}.npz"
        save_table(path, recs, params=P)
        print(f"  已存 {path}  {os.path.getsize(path)/1e6:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
