# -*- coding: utf-8 -*-
"""tdcr_model.py — TDCR（腱驱动连续体机器人）标定模型的统一入口

把 2026-10-19 那一整轮推导收敛下来的东西包成**一个对象**：

    臂:  PEBA 空心薄壁管 Φ16 (壁厚 1.5 mm) + 菱形开孔 + 6 根 NiTi 丝 (分布 Φ10)
    长:  400 mm, 两段各 200 mm, 周向 3 个菱形/段, 轴向 136 个/段
    刚度: 用**实测反标**的截面刚度 EI / EA, 不走晶胞几何推导

设计原则（见 docs/derivations/09 与思源《臂体刚度模型推导与标定》）:

1. `EI`、`EA` 是**截面属性**（材料 × 几何），台架可直测 ⇒ 只保留它们作标定参数
2. `α` 由几何闭合反解（`tanα = L·N_circ/(2π·r·N_long)`），不是自由参数
3. 本构线性 + 几何大转动：Cosserat 的 `R'=R·hat(κ)` 精确积分 90°，`EI` 无需随 κ 变
4. 适用区间 `κ·r < 0.1`（贴死阈值 `κ≈125 1/m`，工作区仅 ~4）

用法
----
    from tdcr_model import TdcrModel
    m = TdcrModel()                       # 用标定值 EI=0.045, EA=7e3
    tip, Rtip = m.pose([28,28,0,0,0,0])   # τ → 末端位姿
    dL = m.delta_L([28,28,0,0,0,0])       # τ → 六丝位移 (mm)
    m.info()                              # 打印标定与自检
    m.save("tdcr_calib.npz"); TdcrModel.load("tdcr_calib.npz")

自检（`python tdcr_model.py`）：两个实测锚点
    · 水平装、自重 20 g + 端载 20 g ⇒ 下垂 ≈ 20°
    · 竖直装、单侧拉至 90°      ⇒ 丝位移 ≈ 10 mm
"""
from __future__ import annotations

import numpy as np

from tendon_coupling import build_arm, shooting, _wire_geom_len_batch, _L0_wire

__all__ = ["TdcrModel", "CALIB"]

# ── 标定值（2026-10-19，由两个实测量反标）──────────────────────────
# ── 2026-10-30 实测标定（30° 四方位，真机数据）────────────────────────────
# 角度: EI 已按实测反标(0.045→0.102), 模型角与实测角 1:1, 不再需要角度增益。
# 方位: 实测方向 120/180/250/320 对应指令 0/90/180/270
#       ⇒ φ_实测 = 0.744·φ_指令 + 117.0° (残差 ≤4°)
#   纯旋转在模型里表达不了(只改丝方位角 ⇒ 增益恒为 1), 所以放在这一层做指令映射。
CAL_DIR_GAIN = 0.744        # dφ_实测 / dφ_指令
CAL_DIR_OFFSET = 117.0      # deg


def dir_to_cmd(dir_target_deg):
    """目标方位(实测口径) → 下发方位(模型口径)。"""
    return (float(dir_target_deg) - CAL_DIR_OFFSET) / CAL_DIR_GAIN


def dir_to_actual(dir_cmd_deg):
    """下发方位(模型口径) → 预期实测方位。"""
    return float(dir_cmd_deg) * CAL_DIR_GAIN + CAL_DIR_OFFSET


CALIB = dict(
    EI=0.0824,         # N·m²  抗弯（2026-11-01 回退：0.01556 的依据是“实测曲率积分 34°”，
                       #   而该曲率链路有 bug(kx/ky≡1、数值成对重复、积分比真实小 5–11 倍)，作废。
                       #   0.0824 由 30° 四方位位姿/目测角度反标，不依赖曲率积分）
