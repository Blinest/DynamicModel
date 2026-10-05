# 待办 / 待更新清单（2026-09-26 15:30）

> 等「重建 40N+60N 表」跑完后执行。重建入口：`python main.py build 40 60 --force`
> 日志 `tools/_rebuild.log`；播报任务 `重建表进度播报`（每 10 min，完成时自动校验版本键 + kcoef 对拍）。

## 0. 重建完成时的自动校验（播报任务已做）
- `python main.py info` → 两张表 **版本键 = True**
- kcoef 对拍：`kcoef["K"]` vs `model.curve_batch()` 的 κ 最大差 **须 ≈ 0**
  （验证 `_kappa_nodes` 的「段A×2」硬编码 bug 已修）

## 1. 思源笔记更新（本项 = 用户说的"更新"）
文档：`/仿真验证/柔性臂动力学建模/完整 Cosserat 静力平衡(VC 式)曲率与弯矩分布推导`
（id `20260905192133-9ofyreu`，内核 `http://127.0.0.1:6806`，token `znljckm9j6ndypko`）

要改的：
- **§10.5 查表**：`_interp_dL` 已修为**过点插值**（精确命中短路 + Shepard/IDW 残差修正）
  + `pos_res_mm` 口径修正（改用**最近表点**度量，新增 `nn_pos_mm`）
- **§11** 补一节「重建后校验结果」：两表版本键、kcoef 对拍、r 范围/倾角
- **§11.6** 补：批量雅可比（`pose_jacobian_batch`，一次 `shooting_batch` 同时给基态解+雅可比）
- **§11.x** 补：POD 降阶精化的现状与局限（a-空间牛顿收敛但 a→τ 回映不可靠）

## 1.5 ★ 可达判据验证结果（2026-09-26 18:25，重大发现）

`inverse_solver.reachability(tau, e, J)` 已实现并接入 `_newton_pose`（it0 计算一次，
零额外开销；结果附 `o.feasible` / `o.floor_mm` / `o.need_N` / `sol["reachability"]`）。

判据算法（修正版）：沿最小范数方向 `dτ = -J⁺e`（满足 `J·dτ = -e`）求**箱约束下可行的
最大步长比 s**（等比缩放保方向；**不可用 `clip(τ+dτ)`**，当 need≫τmax 时六个分量全被削掉、
方向扭曲，会算出 floor>now 的荒谬值）。预测残差 `floor = (1-s)·‖e‖`。

**实测 4 例**

| 点 | 扰动 | need | room | s | floor | now | feasible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| #500（贴边 τ₂=59.47） | 0mm | 0.0 N | 0.53 N | 1.000 | 0.0000 | 0.0000 | **True** |
| #500（贴边） | +5mm | **732.3 N** | 0.53 N | 0.081 | **6.497** | 7.071 | **False** |
| #309（**最内**，room=25.27N） | 0mm | 0.0 N | 25.27 N | 1.000 | 0.0000 | 0.0000 | **True** |
| #309（**最内**） | +5mm | **311.0 N** | 25.27 N | 0.122 | **6.207** | 7.071 | **False** |

**★ 重大结论**：连**最内点**做 +5mm 扰动都需要 **311 N**（= 5.2× τ_max=60 N）⇒
**近表区的可达集是一张"薄流形"**：约 5mm 的位姿偏移在本机构上**基本不可达**。
根因：6 绳 × 6 DOF 的位姿映射**近奇异**（起点 σ_min=0.0082 ⇒ 弱方向 ~122 N/mm）。

⇒ **"+5mm 应该很容易"这个前提是错的**；多轮"+5mm 收敛不了"**不是算法问题，是执行器/机构极限**。
（算法侧已打通：0mm → **0.0235 mm**）

**下一步（待做）**
- 用途①：`feasible=False` 时**提前退出**（省 ~500 s），直接报"超出可达集 + 残差下限"
- 用途②：**投影解** —— 返回最近可达位姿及其 τ/ΔL
- 用途③：**主动集法**替代暴力 `np.clip`
- ⚠ 笔记待补：§11.19「可达性判据 + 薄流形结论」，并**修正 §11.17/§11.18 的"算法问题"定性**


- 全阶牛顿**批量版 + 自适应 LM 阻尼**：目标 ~30 s / 0.05 mm
  （`solve_pose(pod=False)`；已加 `pose_jacobian_batch` + 最优解回退）
- POD-a 精化（口径已诚实）：`solve_pose(pod=True, verify=True)`

## 3. 已知遗留
- POD-a 的 `a→τ` 回映在大偏离时不可靠（默认 `pod_refine=False`，需 `pod=True` 显式开启）
- 表格点间距（NN 中位 4.6 mm）是"表外查询"精度的**真实下限**（非算法问题）
- 旧文件可清理：`old_model_2026-09-19/`（198 MB）

## 4. 工具/环境备忘
- 工作目录：`D:\Continuum robot\control\tdcr_control\tools`（6 个 .py）
  `main.py`(入口) / `tendon_coupling.py`(内核+正解) / `ik_table.py`(表库+CLI)
  `inverse_solver.py`(逆解+POD) / `plotting.py` / `calibration.py`
- 命令：`python main.py {build|info|report|csv|plot|solve} ...`
- shell 需管理员（已开）
