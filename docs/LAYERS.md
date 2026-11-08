# 分层：几何 / 刚度 / 力学 解耦

## 依赖方向（单向，不允许回头）

```
geom_params.py        纯几何：尺寸、孔形、筋宽、材料率、间隔盘数
      │  （不懂得力学）
      ▼
cosserat_lattice.py   几何 → 每段 EI / EA / GJ（含唯一标定常数 c）
      │  ★ 数据契约：arrays() → dict(EI=(N,), EA=(N,), GJ=(N,), L, NSEG)
      ▼
tdcr_model_v2.py      力学：力矩驱动 + 积分 + 腱 + 摩擦 + 轴向耦合
                       （不认识几何，只吃上面的刚度表）
```

## 各层职责

| 层 | 文件 | 输入 | 输出 | 不该出现的东西 |
|---|---|---|---|---|
| 几何 | `geom_params.py` | 无 | `SEG[]`、`R_O`、`NSEG`、`frac()` | EI / 力矩 / 腱 |
| 刚度 | `cosserat_lattice.py` | 几何 | `EI/EA/GJ` 每段 + `arrays()` | 积分 / 求解器 |
| 力学 | `tdcr_model_v2.py` | 刚度表 + `(τ, T0, μ, ψ)` | 形状、`ΔL`、`κ(s)` | `α` / `f` / `R_O` |

## 换东西只改一处

| 想改 | 只改 | 影响 |
|---|---|---|
| 孔形（α、筋宽、圆角） | `geom_params.SEG` | 刚度 → 形状（自动传导） |
| 材料 E / ν | `geom_params.E_MOD/NU` | 同上 |
| 标定常数 c | `cosserat_lattice.C_BEND` | 只改刚度幅值 |
| 力学模型 | `tdcr_model_v2` | 与几何无关 |

## 刚度的两种来源（可互换）

```python
# 1) 由几何推（默认）
m = TdcrModelV2()

# 2) 实测标定后注入（力学层独立验证）
st = dict(EI=np.full(20, 0.01037), EA=np.full(20, 7000.), GJ=np.full(20, 0.045))
m = TdcrModelV2(stiffness=st)
```

回归已验证：注入 `EI×2` ⇒ 90° 所需张力比 **1.99**（线性，正确）。

## 已修的一个耦合残留 bug

`TdcrModelV2.solve_bend` 原来把非驱动丝置 `τ=0`（等于**撤掉预紧**），
轴向净载荷变成 `+Σ T0` ⇒ 虚假拉伸、`ΔL` 被少算。
现改为余弦律 `τ = max(0, T0 + A·cosψ)`，`Σ(τ−T0)=0` ⇒ `θ=0` 时 `ΔL=0`。
