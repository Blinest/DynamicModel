# 实机臂位姿表清单（2026-10-29）

标定：`EI=0.045 N·m²`、`EA(k_rod)=7000 N`、`L=400 mm`（两段各 200）、
`r_disk=5 mm`（丝分布 Φ10）、`N_circ=3/3`、`α=5.016°`（几何反解）、`locked=True`、
`lattice=False`、61 节点。

| 表 | 点数 | NN 中位 | p90 | 弯角 | 文件 |
|---|---|---|---|---|---|
| `vc_table_real_20.npz` | **1376** | **4.87 mm** ✓ | 10.64 | 0.3–67.4° | 3.09 MB |
| `vc_table_real_40.npz` | **1432** | **9.58 mm** | 20.06 | 0.6–144.0° | 3.19 MB |

- 20N 达标（≤5 mm）；40N 距 8 mm 目标还差约 **620 点**（按 `NN∝1/√N` 估算）
- 40N 接受率约 46–49%（深弯区求解代价高，~5–13 s/点）
- `max NN`：20N 60.2 mm、40N 107.6 mm ⇒ 两表都有明显空洞，未被中位数反映

## 命令

```bash
# 建表（实机臂，自动带 CALIB）
python tools/build_real_arm.py --tau 20 --n 1200

# 补采样（⚠ 必须带 --real，否则新样本按默认臂 EI=0.1294 算）
python tools/supplement_table.py --tau 40 --prefix vc_table_real_ --real --add 600 --seed 31 --min-dist 0.005

# 打包（一表一包；容器段 id 固定，多表会同名冲突）
python tdcr_build/make_td.py --tau 20 --prefix vc_table_real_ --kk-order planar --out tdcr_model_20N.tdcrmodel
```

## 已知取舍

1. **一包一表**：`KTB1/PTB1/KSB1/KKB1` 段 id 固定，同一包里放两张表会出现同名段
2. **61 节点**：601→61 抽样，离散化损失 0.015 mm（实测），够用；但看不出单胞尺度（`L_strut=8.4 mm`）的曲率起伏
3. **补采样无 checkpoint**：脚本只在全部解完后合并存盘，中途被杀会丢全部新点（盘上旧表不受损）
4. **`--real` 不能省**：漏了会用 `default_params()`，产生“同名不同臂”的污染表
