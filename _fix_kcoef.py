# -*- coding: utf-8 -*-
"""就地重算表的 kcoef (κ/ν 节点) —— 无需重建整张表.

背景: `_kappa_nodes` 修了 4 处 (段A×2 硬编码 / 段界浮点容差 / 正弦本构精确逆变换 / C 采样节点),
现有表是修复前建的, 其 kcoef 的 κ 有 ≤1.9e-2 残差。用 `curve_batch`(正确口径)重算并回写即可。
"""
import sys, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from ik_table import load_table, save_table
from tendon_coupling import TendonCouplingModel as M

for fn in ("vc_table_40.npz", "vc_table_60.npz"):
    t0 = time.time()
    T = load_table(fn)
    recs = T.recs
    taus = np.array([np.asarray(r["tau"], float) for r in recs])
    xs = np.array([np.asarray(r["x"], float) for r in recs])
    model = M(params=T.params)
    cv, _ = model.curve_batch(taus, x=xs)
    S = np.asarray(cv[1], float)
    Uc = np.asarray(cv[2], float)
    Vv = np.asarray(cv[6], float)
    for i, r in enumerate(recs):
        kc = {"S": S, "K": Uc[i].copy()}
        kc["V"] = Vv[i].copy()
        r["kcoef"] = kc
    save_table(fn, recs, params=T.params)
    print(f"{fn}: 已重写 kcoef  (N={len(recs)}, {time.time()-t0:.1f}s)")
print("DONE")
