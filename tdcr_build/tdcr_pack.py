# -*- coding: utf-8 -*-
"""tdcr_pack.py — 建表 → 打包 `.tdcrmodel` 的一键工具

把"建表 + 标定 + 校验信息"收进**一个自包含文件**，换机器/交付时只带这一个。

## 用法

    # ① 只打包（表已经建好）
    python tdcr_build/tdcr_pack.py pack --tau 20 40 --out tdcr_v1.tdcrmodel

    # ② 一键：缺表就建，建完打包（最常用）
    python tdcr_build/tdcr_pack.py make --tau 20 40 --n 900 --out tdcr_v1.tdcrmodel

    # ③ 查看包内容
    python tdcr_build/tdcr_pack.py info tdcr_v1.tdcrmodel

    # ④ 解包（把表还原到当前目录）
    python tdcr_build/tdcr_pack.py unpack tdcr_v1.tdcrmodel -o .

## `.tdcrmodel` 内部结构（普通 zip，可直接用解压工具打开）

    manifest.json          元信息：格式版本 / 创建时间 / 标定 / git HEAD / 表清单+sha256
    calib.json             截面刚度与几何（EI, EA, L1, L2, r_o, r_disk, n_circ, n_long ...）
    tables/vc_table_20.npz 位姿表（原样存放）
    tables/vc_table_40.npz

## 设计原则

1. **不改动仓库现有入口**：建表仍走 `main.py build`，本工具只做"调它 + 打包"
2. **自校验**：manifest 里存每个表的 `sha256` 与点数，`info` 会重算并比对
3. **可复现**：记录 git HEAD、numpy 版本、建表参数（`--n`、`τmax`）
4. 表本身已是压缩 npz，zip 用 `ZIP_STORED` 避免二次压缩白花时间
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone

FORMAT = "tdcrmodel/1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ────────────────────────────── 小工具 ──────────────────────────────
def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10
                              ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _calib():
    """取标定字典；导入失败则返回最小占位。"""
    try:
        from tdcr_model import CALIB
        return {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in CALIB.items()}
    except Exception:
        return {}


def _table_stats(path):
    """点数 + 曲率/弯角摘要（不需要建表，只读 npz）。"""
    import numpy as np
    z = np.load(path)
    tau = z["tau"]
    out = dict(n=int(tau.shape[0]))
    if "kK" in z.files:
        k = np.linalg.norm(z["kK"], axis=2).mean(axis=1)
        out["kappa_mean_median"] = round(float(np.median(k)), 4)
        out["kappa_mean_max"] = round(float(k.max()), 4)
    if "R_tip" in z.files:
        ang = np.degrees(np.arccos(np.clip(z["R_tip"][:, 2, 2], -1, 1)))
        out["bend_min_deg"] = round(float(ang.min()), 2)
        out["bend_max_deg"] = round(float(ang.max()), 2)
    out["tau_comp_max"] = round(float(tau.max()), 2)
    # ★ 表内版本键（EI / C0 / k_rod / r_disk ...）：用于校验"标定 vs 表"是否同一根臂
    if "ver" in z.files:
        try:
            out["ver"] = json.loads(str(z["ver"]))
        except Exception:
            pass
    return out


# ────────────────────────────── pack ──────────────────────────────
def cmd_pack(a):
    taus = [float(t) for t in a.tau]
    files = []
    for t in taus:
        p = os.path.join(ROOT, f"vc_table_{t:g}.npz")
        if not os.path.exists(p):
            sys.exit(f"缺表 {p}；先跑 `python main.py build {t}` 或用 `make` 子命令")
        files.append((t, p))

    manifest = {
        "format": FORMAT,
        "created": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "git_head": _git_head(),
        "python": sys.version.split()[0],
        "numpy": __import__("numpy").__version__,
        "calib": _calib(),
        "tables": [],
    }
    for t, p in files:
        m = dict(tau_max=float(t), file=f"tables/{os.path.basename(p)}",
                 size=int(os.path.getsize(p)), sha256=_sha256(p))
        m.update(_table_stats(p))
        manifest["tables"].append(m)

    with zipfile.ZipFile(a.out, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        z.writestr("calib.json", json.dumps(manifest["calib"], ensure_ascii=False, indent=2))
        for _, p in files:
            z.write(p, f"tables/{os.path.basename(p)}")

    print(f"[pack] {a.out}  {os.path.getsize(a.out)/1e6:.1f} MB")
    for m in manifest["tables"]:
        print(f"   τmax={m['tau_max']:.0f}N  {m['n']:4d} 点  弯角 "
              f"{m.get('bend_min_deg','?')}-{m.get('bend_max_deg','?')}°  "
              f"|κ|均中位 {m.get('kappa_mean_median','?')}")
    return 0


# ────────────────────────────── make ──────────────────────────────
def cmd_make(a):
    for t in a.tau:
        p = os.path.join(ROOT, f"vc_table_{t:g}.npz")
        if os.path.exists(p) and not a.force:
            print(f"[make] τmax={t}N 表已存在，跳过建表（--force 强制重建）")
            continue
        cmd = [sys.executable, "-u", "main.py", "build", str(t), "--force"]
        if a.n:
            cmd += ["--n", str(a.n)]
        print(f"[make] {' '.join(cmd)}  （纯 numpy 约 1.8 s/点，20N/900 点 ≈35 min）",
              flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"建表失败 τmax={t}（exit {r.returncode}）")
    return cmd_pack(a)


# ────────────────────────────── info / unpack ──────────────────────────────
def cmd_info(a):
    with zipfile.ZipFile(a.pkg) as z:
        m = json.loads(z.read("manifest.json"))
        print(f"包   : {a.pkg}  ({os.path.getsize(a.pkg)/1e6:.1f} MB)")
        print(f"格式 : {m['format']}   创建: {m['created']}   git: {m['git_head']}")
        c = m.get("calib") or {}
        if c:
            print(f"代码标定(CALIB): EI={c.get('EI')} N·m²  EA={c.get('EA')} N  "
                  f"r_disk={c.get('r_disk')} m  n_circ={c.get('n_circ')}")
            print("         ↑ 来自 tdcr_model.py，描述**实机臂**；"
                  "表的力学以各表 `ver` 为准")
        print("表   :")
        ok = True
        for t in m["tables"]:
            name = t["file"]
            if name in z.namelist():
                h = hashlib.sha256(z.read(name)).hexdigest()
                same = (h == t["sha256"])
                ok &= same
                flag = "✓" if same else "✗ sha256 不符"
            else:
                flag = "✗ 缺文件"; ok = False
            print(f"   τmax={t['tau_max']:.0f}N  {t['n']:4d} 点  "
                  f"弯角 {t.get('bend_min_deg','?')}-{t.get('bend_max_deg','?')}°  {flag}")
        # ── 标定 vs 表内版本键：必须是同一根臂 ───────────────────────
        warn = False
        for t in m["tables"]:
            v = t.get("ver") or {}
            ei_t = v.get("EI")
            kr_t = ((v.get("seg") or [[None] * 5])[0] or [None] * 5)[4]
            if ei_t is None or not c:
                continue
            ei_c = c.get("EI")
            if ei_c:
                rel = abs(ei_t - ei_c) / max(abs(ei_c), 1e-30)
                if rel > 0.01:
                    warn = True
                    print(f"   ⚠ τmax={t['tau_max']:.0f}N 表内 EI={ei_t:.4g} N·m²，"
                          f"标定 EI={ei_c:.4g} —— 差 {rel*100:.0f}%，**该表不是按这套标定建的**")
            print(f"     表内: EI={ei_t:.4g}  C0={v.get('C0', float('nan')):.4g}  "
                  f"k_rod={kr_t}  r_disk={v.get('r_disk')}")
        print("校验 :", "全部通过" if (ok and not warn) else
              ("sha256 通过，但**标定与表不一致**" if ok else "有问题"))
    return 0 if (ok and not warn) else 1


def cmd_unpack(a):
    with zipfile.ZipFile(a.pkg) as z:
        z.extractall(a.out)
        names = [n for n in z.namelist() if n.startswith("tables/")]
    for n in names:
        src = os.path.join(a.out, n)
        dst = os.path.join(ROOT, os.path.basename(n))
        os.replace(src, dst) if os.path.exists(dst) else os.rename(src, dst)
        print(f"[unpack] {os.path.basename(n)} -> {dst}")
    print("[unpack] 表已就位，可直接 `python main.py info` 验收")
    return 0


# ────────────────────────────── CLI ──────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="建表 + 打包 .tdcrmodel")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("pack", "make"):
        s = sub.add_parser(name)
        s.add_argument("--tau", type=float, nargs="+", default=[20.0, 40.0],
                       help="各表的 τmax（N），默认 20 40")
        s.add_argument("--out", default="tdcr_model.tdcrmodel")
        if name == "make":
            s.add_argument("--n", type=int, default=900, help="每张表的采样数")
            s.add_argument("--force", action="store_true", help="强制重建已有表")

    s = sub.add_parser("info")
    s.add_argument("pkg")
    s = sub.add_parser("unpack")
    s.add_argument("pkg")
    s.add_argument("-o", "--out", default=".")

    a = ap.parse_args()
    return {"pack": cmd_pack, "make": cmd_make,
            "info": cmd_info, "unpack": cmd_unpack}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
