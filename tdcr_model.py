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
    EI=0.01556,        # N·m²  抗弯（2026-10-31 预校准后重标：指令 34° ⇒ 实测 180°，
                       #   0.0824×34/180。10-30 的 0.0824 未计预校准初态位移，作废）
    EA=7.0e3,          # N     抗轴（由 90° 时 10.5 mm 反标）
    L1=0.200, L2=0.200,
    r_o=0.008,         # 外半径 ⇒ 外径 16 mm
    t_wall=1.5e-3,     # 壁厚
    r_disk=0.005,      # 丝分布半径 ⇒ 分布直径 10 mm
    n_circ=3,
    n_long=136,
    E_mat=2.0e8,       # PEBA
    mass_total=0.020,  # 臂自重 20 g
    n_p=30, n_d=30,
)


class TdcrModel:
    """TDCR 标定模型（薄壁开孔管 + 6 丝）。

    Args:
        EI, EA : 截面刚度; 缺省用 CALIB
        rhoA   : 线密度 [kg/m]; 缺省按 `mass_total/L` 由总质量换算
        其余    : 透传给 `build_arm`
    """

    def __init__(self, EI=None, EA=None, rhoA=None, **kw):
        self.EI = float(CALIB["EI"] if EI is None else EI)
        self.EA = float(CALIB["EA"] if EA is None else EA)
        L = CALIB["L1"] + CALIB["L2"]
        self.rhoA = float(CALIB["mass_total"] / L if rhoA is None else rhoA)
        args = dict(
            L1=CALIB["L1"], L2=CALIB["L2"], r_i=CALIB["r_o"],
            r_disk=CALIB["r_disk"], n_circ_a=CALIB["n_circ"],
            n_circ_b=CALIB["n_circ"], n_long=CALIB["n_long"],
            E=CALIB["E_mat"], t_strut=CALIB["t_wall"],
            rhoA=self.rhoA, EI=self.EI, EA=self.EA,
            n_p=CALIB["n_p"], n_d=CALIB["n_d"], lattice=False,
        )
        args.update(kw)
        self.ca, self.cb, self.params = build_arm(**args)
        self._L0 = _L0_wire(self.params)

    # ── 几何/派生量 ──────────────────────────────────────────────
    @property
    def alpha_deg(self):
        """斜撑角（由几何闭合反解，不是输入）。"""
        return self.ca.alpha_deg

    @property
    def L(self):
        return self.params.L_total

    @property
    def kappa_lock(self):
        """贴死阈值 κ [1/m]。

        由 `Δα = α` 的几何给出: 单胞角变化 Δα = κ·r·tanα，令 Δα=α
        ⇒ `κ_lock = α/(r·tanα) ≈ 1/r`（α 小）。**用晶胞半径 r_o，不是丝分布半径 r_disk。**
        """
        return 1.0 / CALIB["r_o"]

    # ── 正解 ────────────────────────────────────────────────────
    def solve(self, tau, g_world=(0.0, 0.0, -9.81), F_tip=None, x0=None):
        """τ → 完整解（tip / R_tip / dL / 曲线）。

        ⚠ `F_tip=None` 会让 `shooting` 走 7 维残差（含杆弧长约束），
          而它内部雅可比按 6 维写死 ⇒ `ValueError: shape (7,) into (6,)`。
          这里统一补零向量规避（见 docs/derivations/09 的遗留项）。
        """
        tau = np.asarray(tau, float).reshape(6)
        ft = np.zeros(3) if F_tip is None else np.asarray(F_tip, float).reshape(3)
        return shooting(tau, self.params, F_tip=ft, g_world=g_world, x0=x0)

    def pose(self, tau, **kw):
        """τ → (tip[m], R_tip)。"""
        o = self.solve(tau, **kw)
        return o["p_curve"][-1], o["R_curve"][-1]

    def bend_angle(self, tau, **kw):
        """末端切向相对基座 +Z 的夹角 [deg]。"""
        _, R = self.pose(tau, **kw)
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))

    def delta_L(self, tau, r_wire=None, **kw):
        """τ → 六丝位移 ΔL [mm]（负 = 收丝）。"""
        o = self.solve(tau, **kw)
        r = CALIB["r_disk"] if r_wire is None else float(r_wire)
        Lg = _wire_geom_len_batch(o["p_curve"][None], o["R_curve"][None],
                                  r_wire=r, L1_geo=self.params.L1,
                                  L_total=self.params.L_total)[0]
        return (Lg - self._L0) * 1000.0

    def kappa_profile(self, tau, **kw):
        """τ → (s[m], |κ|[1/m])，用于看曲率分布。"""
        o = self.solve(tau, **kw)
        return np.asarray(o["s_curve"], float), np.linalg.norm(o["u_curve"], axis=1)

    # ── 工具 ────────────────────────────────────────────────────
    def info(self, verbose=True):
        d = dict(EI=self.EI, EA=self.EA, alpha_deg=self.alpha_deg,
                 L=self.L, L_strut=self.ca.L_strut, l_ax=self.ca.l_ax,
                 N_long=self.ca.N_long, r_disk=CALIB["r_disk"],
                 kappa_lock=self.kappa_lock, rhoA=self.rhoA)
        if verbose:
            print("═" * 58)
            print("TDCR 标定模型")
            print("═" * 58)
            print(f"  EI={d['EI']:.4f} N·m²   EA={d['EA']:.0f} N   "
                  f"（均由实测量反标）")
            print(f"  L={d['L']*1e3:.0f} mm  两段各 {CALIB['L1']*1e3:.0f} mm")
            print(f"  外径 {2*CALIB['r_o']*1e3:.0f} mm / 壁厚 "
                  f"{CALIB['t_wall']*1e3:.1f} mm / 丝分布 Φ{CALIB['r_disk']*2e3:.0f}")
            print(f"  周向 {CALIB['n_circ']} 胞 · 轴向 {d['N_long']:.0f} 胞/段 ⇒ "
                  f"α={d['alpha_deg']:.3f}° (几何反解)")
            print(f"  L_strut={d['L_strut']*1e3:.3f} mm  l_ax={d['l_ax']*1e3:.3f} mm")
            print(f"  贴死阈值 κ≈{d['kappa_lock']:.0f} 1/m；适用区间 κ·r<0.1")
            print(f"  自重 {CALIB['mass_total']*1e3:.0f} g (rhoA={d['rhoA']:.4f} kg/m)")
        return d

    def save(self, path):
        np.savez(path, **{k: np.array(v) for k, v in CALIB.items()},
                 EI=self.EI, EA=self.EA)
        return path

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(EI=float(z["EI"]), EA=float(z["EA"]))


# ══════════════════════════════════════════════════════════════════
def _selftest():
    m = TdcrModel()
    m.info()
    print("\n自检 A: 水平装, 自重 20 g + 端载 20 g ⇒ 下垂角")
    o = m.solve(np.full(6, 0.01), g_world=(-9.81, 0, 0),
                F_tip=np.array([-0.020 * 9.81, 0.0, 0.0]))
    ang = float(np.degrees(np.arccos(np.clip(o["R_curve"][-1, 2, 2], -1, 1))))
    print(f"    下垂 = {ang:.2f}°   (实测约 20°)")
    ok_a = abs(ang - 20.0) < 2.0

    print("\n自检 B: 竖直装, 单侧拉至 90° ⇒ 丝位移")
    best = None
    for T in (20, 24, 28, 32):
        tau = T * np.array([1.0, 1.0, 0, 0, 0, 0])
        b = m.bend_angle(tau, g_world=(0, 0, 0))
        d = np.abs(m.delta_L(tau, g_world=(0, 0, 0))).max()
        print(f"    T={T:2d} N 弯角={b:6.2f}°  max|dL|={d:5.2f} mm")
        if best is None or abs(b - 90) < abs(best[0] - 90):
            best = (b, d)
    print(f"    最接近 90°: 弯角={best[0]:.2f}°  max|dL|={best[1]:.2f} mm  (实测 10–11)")
    ok_b = best[1] < 12.5

    print("\n" + "═" * 58)
    print(f"{'✓' if ok_a else '✗'} 20° 下垂: {ang:.2f}°")
    print(f"{'✓' if ok_b else '✗'} 90° 丝位移: {best[1]:.2f} mm")
    return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
