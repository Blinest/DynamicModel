# 可推导清单（按收益/难度排序）

> 生成日期：2026-10-05 ｜ 依据：`_TODO.md`、`_VEC_PLAN.md`、源码内标注的遗留项
> 状态：**① 已完成**（本目录 `01_reachability_floor.md` + `tools/check_reachability_floor.py`），②–⑥ 待做

| # | 题目 | 为什么值得推 | 难度 | 代码位置 | 验证方式 |
|---|------|--------------|------|----------|----------|
| ① | **线性化可达判据与残差地板** | 现判据把 29.3% 的可达目标误判为不可达（实测，见 01 文档 §5） | 低 | `inverse_solver.reachability()` | ✅ 已用 `tools/check_reachability_floor.py` 与箱-QP 真值对拍通过 |
| ② | **解析灵敏度（变分方程）`∂res/∂x₀`、`∂res/∂τ`、`∂tip/∂x₀`** | 现在打靶雅可比靠 12 次全量正解差分，占端到端耗时约 90%；解析化可再降一个量级 | 高 | `tendon_coupling.shooting_batch`、`_jit_integrate.integrate_all` | 与现有中心差分逐元素对拍（难度全在 `A,B,G,H,c,d` 对 `(v,u)` 求导） |
| ③ | **POD 的 `a → τ` 回映** | `_TODO.md §3` 明确列为遗留：大偏离时不可靠、`pod_refine` 默认关 | 中 | `inverse_solver.PODReducedOrderModel`、`fit_tau_predictor` | 用表记录做闭环比：由 τ 生成 a，再回映 τ̂，看 ‖τ̂−τ‖ 随偏离量的变化 |
| ④ | **腱鞘摩擦/滞回项嵌入平衡方程** | 目前模型无摩擦，是解析精度与实验偏差最大的单项（Capstan 指数摩擦 + 加/卸载双支） | 中 | `tendon_coupling.intermedquant`（新增 `τ_w(ΔL_w)` 本构） | 预测 τ–ΔL 滞回环；与 `calibration.py` 标定数据比残差 |
| ⑤ | **六丝耦合矩阵的解析梯度 `∂ΔL/∂τ`** | 现为几何差分；解析式可让在线查表逆解的初值估计更准 | 中高 | `TendonCouplingModel.coupling_matrix` | 与差分版逐元素对拍，误差应只来自曲线离散 |
| ⑥ | **打靶边值问题的奇异方向与适定性** | 起点 σ=[23.4…0.0082]（cond≈2850）⇒ 弱方向 122 N/mm；这是"薄流形"结论的几何来源 | 中 | `_TODO.md §1.5`、`inverse_solver.pose_jacobian_batch` | 用 SVD 谱随 (τ, 位形) 的漂移画出"条件数地图" |

## 优先级建议

1. **②（解析灵敏度）**：唯一能把建表/逆解再提速一个量级的推导，且能与已完成的并行化、前向差分叠加。
2. **③（POD 回映）**：直接解锁 `pod_refine` 这个被默认关掉的加速开关。
3. **①（已完成）**的落地：把 `reachability()` 的地板换成箱-QP 真值（见 01 文档 §6），减少误判提前退出。
4. **④（摩擦）**：从"算得快"转向"算得准"，但需要实验数据才能验证。
