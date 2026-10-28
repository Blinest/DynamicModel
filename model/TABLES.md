# 位姿表清单（实机臂）

> 更新：2026-10-28 ｜ 参数来源 `tdcr_model.CALIB`
> 建表命令里的 `--real` **不可省**（否则新样本按默认臂 `EI=0.1294` 算，会污染表）

## 当前表

| 文件 | 点数 | NN 中位 | p90 | 弯角 | 大小 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| `vc_table_real_20.npz` | 1376 | **4.87 mm** | 10.64 | 0.3–67.4° | 3.09 MB | 达标（中位 <5 mm） |
| `vc_table_real_40.npz` | 861 | **11.03 mm** | 24.92 | 0.6–144.0° | 1.9 MB | 偏稀，补采样进行中 |

## 共同的力学参数（表内 `ver`）

```
EI   = 0.045 N·m²      由 20° 自重下垂反标
k_rod= 7000  N         由 90° 时 10.5 mm 丝位移反标
L1 = L2 = 0.200 m      L = 0.400 m
r_o = 8 mm (Φ16)       t_wall = 1.5 mm
r_disk = 5 mm (Φ10)    n_circ = 3   n_long = 136
E = 200 MPa (PEBA)     A = 1.5 mm²  α = 5.016°（几何闭合反解）
locked = True          lattice = False
```

## 建表 / 补表 / 打包

```bash
# 从零建（实机臂）
python tools/build_real_arm.py --tau 40 --n 1200

# 补采样（换种子 + 剔除近邻 + 合并去重）
python tools/supplement_table.py --tau 40 --prefix vc_table_real_ --real \
       --add 1200 --seed 31 --min-dist 0.005

# 打包（一个 τmax 一个包：段 id 是固定的 KTB1/PTB1/KSB1/KKB1，多表会同名冲突）
python tdcr_build/make_td.py --tau 20 --prefix vc_table_real_ --kk-order planar \
       --out tdcr_model_20N.tdcrmodel
```

## 已知取舍

1. **节点数 61**（7.4 mm/节点）：查表逆解只用 `tau/dL/p_tip/R_tip`，不受影响；κ 剖面较粗
2. **一个包一张表**：容器段 id 固定，两张表放进同一个包会出现两份同名段，后端无法区分
3. **40N 接受率 ~46%**：`tip z` 探到 0.023 m（深弯区），该区求解代价高、失败多
4. 补采样**无 checkpoint**：中途被杀会丢全部新点（盘上旧表不受损），块大小按 `add × 5 s` 留 30% 余量
