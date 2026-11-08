# 代码结构

```
geometry/     几何（纯尺寸/孔形，不含力学）
  geom_params.py     唯一事实源：R_O/R_I/NSEG/SEG(α,w,圆角)/frac/派生 B、L_strut

stiffness/    几何 → 截面刚度
  cosserat_lattice.py  EI=c·E·I·f·sin²αcos²α；arrays() 是给力学层的**数据契约**

force/        力学（只吃刚度，不认识几何）
  tdcr_model_v2.py     20 段力矩驱动 + Capstan 摩擦 + 轴向耦合 + 尖端力
  seg12_model.py       20 段曲率场生成/反解
  kappa_fit.py         由实测形状反求曲率场

plot/         绘图（全部可独立运行）
  fig_* / anim_* / draw_* / fillet_mesh* / plot_table_contours

tools/        集成与流程
  sample_table_v2.py       建表（3000 点，约 160 s）
  tipforce_estimator.py    实时外力估计（离线建 J 表 + 在线 1.6 µs 矩阵乘）
  tipforce_from_tension.py 可观测性分析（秩/条件数/灵敏度）
  run_cosserat_bvp.py      完整 Cosserat 打靶对拍
  shoot_lattice.py         打靶接入菱形晶胞（失败诊断，留档）
  friction_id_*.py         摩擦标定协议（4 个）
  casadi_fit.py / casadi_tipforce.py   CasADi + IPOPT 参数辨识

legacy（留在根目录，未并入分层）
  tendon_coupling.py   完整 Cosserat 求解器（2293 行，numba 缺失时 17 s/点）
  tdcr_model.py        v1 标定入口（CALIB）
  ik_table.py / inverse_solver.py / tdcr_control.py   查表逆解与控制
```

## 依赖方向（单向）

```
geometry ──▶ stiffness ──▶ force ──▶ tools / plot
```

## 各目录的路径引导

每个 .py 顶部有统一引导块，自动把 `geometry/stiffness/force/plot` 加入 `sys.path`，
因此任意目录下 `python plot/xxx.py` 都能直接跑。

## 数据契约

```python
from stiffness.cosserat_lattice import arrays
st = arrays()        # dict(EI=(20,), EA=(20,), GJ=(20,), L, NSEG)

from force.tdcr_model_v2 import TdcrModelV2
m = TdcrModelV2(stiffness=st)     # 注入：力学与几何完全解耦
```
