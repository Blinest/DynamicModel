# -*- coding: utf-8 -*-
"""tdcr_pack.py — 建表 → 打包 `.tdcrmodel` 的一键工具（支持两种容器）

`.tdcrmodel` 有**两种**容器，读取时按 magic 自动识别：

  ┌─ bin（默认，二进制）─────────────────────────────────────────┐
  │ 8B  magic = b"TDCRMDL1"                                     │
  │ 4B  header_len   (uint32, little-endian)                    │
  │ NB  header       (UTF-8 JSON: 标定/表清单/offset/size/sha256)│
  │ ... payload      (各表的 .npz **原始字节**依次拼接)          │
  └─────────────────────────────────────────────────────────────┘
  zip 容器：manifest.json + calib.json + tables/*.npz（旧版，仍可读）

    # 打包（默认 bin）
    python tdcr_build/tdcr_pack.py pack --tau 20 --prefix vc_table_real_ --out m.tdcrmodel
    python tdcr_build/tdcr_pack.py pack --tau 20 40 --format zip --out m.tdcrmodel

    # 一键：缺表先建再打包
    python tdcr_build/tdcr_pack.py make --tau 20 --n 900 --out m.tdcrmodel

    # 查看 / 解包
    python tdcr_build/tdcr_pack.py info  m.tdcrmodel
    python tdcr_build/tdcr_pack.py unpack m.tdcrmodel -o .

⚠ bin 容器里 payload 就是原样的 `.npz` 字节 ⇒ 取出后可直接 `np.load`，
  不需要先解压，也不依赖 `zipfile`。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys

MAGIC = b"TDCRMDL1"
FORMAT_BIN = "tdcrmodel-bin/2"
FORMAT_ZIP = "tdcrmodel/1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ────────────────────────────── 小工具 ──────────────────────────────
def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10
                              ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _calib():
    try:
        from tdcr_model import CALIB
        return {k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in CALIB.items()}
    except Exception:
        return {}


def _table_ver(raw):
    """从 npz 字节里取出表内 `ver`（版本键）并解析成 dict。"""
    import io
    import numpy as np
    try:
        z = np.load(io.BytesIO(raw))
        if "ver" in z.files:
            return json.loads(str(z["ver"].item() if z["ver"].ndim == 0
                                  else z["ver"][0]))
    except Exception:
        pass
    return {}


def _table_stats(raw):
    import io
    import numpy as np
    z = np.load(io.BytesIO(raw))
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
    return out


def _read_container(path):
    """→ (manifest dict, get(name)->bytes, kind)。按 magic 自动识别。"""
    with open(path, "rb") as f:
        head = f.read(8)
        f.seek(0)
        if head == MAGIC:
            f.read(8)
            (hlen,) = struct.unpack("<I", f.read(4))
            man = json.loads(f.read(hlen).decode("utf-8"))
            base = f.tell()
            f.seek(0, 2)
            total = f.tell()

            def get(name, _p=path, _base=base):
                rec = next(t for t in man["tables"] if t["file"] == name)
                with open(_p, "rb") as g:
                    g.seek(_base + rec["offset"])
                    return g.read(rec["size"])
            man["_total"] = total
            return man, get, "bin"

        import zipfile
        z = zipfile.ZipFile(path)
        man = json.loads(z.read("manifest.json"))
        return man, (lambda name, _z=z: _z.read(name)), "zip"


def _write_bin(path, manifest, blobs):
    """blobs: [(name, bytes)]；header 里记 offset/size。"""
    off = 0
    entries = []
    for name, b in blobs:
        entries.append(dict(manifest["tables"][len(entries)], file=name,
                            offset=off, size=len(b), sha256=_sha256_bytes(b)))
        off += len(b)
    man = dict(manifest, tables=entries, container=FORMAT_BIN)
    hdr = json.dumps(man, ensure_ascii=False, indent=2).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(hdr)))
        f.write(hdr)
        for _, b in blobs:
            f.write(b)


def _write_zip(path, manifest, blobs):
    import zipfile
    man = dict(manifest, container=FORMAT_ZIP)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED,
                         allowZip64=True) as z:
        z.writestr("manifest.json", json.dumps(man, ensure_ascii=False, indent=2))
        z.writestr("calib.json", json.dumps(man.get("calib", {}),
                                           ensure_ascii=False, indent=2))
        for name, b in blobs:
            z.writestr(name, b)


# ────────────────────────────── pack ──────────────────────────────
def cmd_pack(a):
    prefix = getattr(a, "prefix", "vc_table_")
    blobs, metas = [], []
    for t in [float(x) for x in a.tau]:
        p = os.path.join(ROOT, f"{prefix}{t:g}.npz")
        if not os.path.exists(p):
            sys.exit(f"缺表 {p}；先跑 `python main.py build {t:g}` 或用 `make` 子命令")
        raw = open(p, "rb").read()
        name = f"tables/{os.path.basename(p)}"
        blobs.append((name, raw))
        m = dict(tau_max=t, file=name, size=len(raw),
                 sha256=_sha256_bytes(raw), ver=_table_ver(raw))
        m.update(_table_stats(raw))
        metas.append(m)

    manifest = {
        "format": FORMAT_BIN if a.format == "bin" else FORMAT_ZIP,
        "created": __import__("datetime").datetime.now().astimezone()
                    .isoformat(timespec="seconds"),
        "git_head": _git_head(),
        "python": sys.version.split()[0],
        "numpy": __import__("numpy").__version__,
        "calib": _calib(),
        "tables": metas,
    }
    (_write_bin if a.format == "bin" else _write_zip)(a.out, manifest, blobs)
    print(f"[pack] {a.out}  {os.path.getsize(a.out)/1e6:.2f} MB  容器={a.format}")
    for m in metas:
        print(f"   τmax={m['tau_max']:g}N  {m['n']:4d} 点  弯角 "
              f"{m.get('bend_min_deg','?')}-{m.get('bend_max_deg','?')}°  "
              f"|κ|均中位 {m.get('kappa_mean_median','?')}")
    return 0


# ────────────────────────────── make ──────────────────────────────
def cmd_make(a):
    for t in a.tau:
        p = os.path.join(ROOT, f"{getattr(a, 'prefix', 'vc_table_')}{t:g}.npz")
        if os.path.exists(p) and not a.force:
            print(f"[make] τmax={t:g}N 表已存在，跳过（--force 强制重建）")
            continue
        cmd = [sys.executable, "-u", "main.py", "build", f"{t:g}", "--force"]
        if a.n:
            cmd += ["--n", str(a.n)]
        print(f"[make] {' '.join(cmd)}", flush=True)
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            sys.exit(f"建表失败 τmax={t:g}")
    return cmd_pack(a)


# ────────────────────────────── info / unpack ──────────────────────────────
def cmd_info(a):
    man, get, kind = _read_container(a.pkg)
    print(f"包   : {a.pkg}  ({os.path.getsize(a.pkg)/1e6:.2f} MB)  容器={kind}")
    print(f"格式 : {man.get('container') or man.get('format')}   "
          f"创建: {man['created']}   git: {man['git_head']}")
    c = man.get("calib") or {}
    if c:
        print(f"代码标定(CALIB): EI={c.get('EI')} N·m²  EA={c.get('EA')} N  "
              f"r_disk={c.get('r_disk')} m  n_circ={c.get('n_circ')}")
        print("         ↑ 来自 tdcr_model.py，描述**实机臂**；表的力学以各表 `ver` 为准")
    ok = True
    print("表   :")
    for t in man["tables"]:
        try:
            raw = get(t["file"])
        except Exception as e:
            print(f"   τmax={t['tau_max']:g}N  ✗ 读取失败 {e}"); ok = False; continue
        same = _sha256_bytes(raw) == t["sha256"]
        ok &= same
        v = t.get("ver") or {}
        print(f"   τmax={t['tau_max']:g}N  {t['n']:4d} 点  弯角 "
              f"{t.get('bend_min_deg','?')}-{t.get('bend_max_deg','?')}°  "
              f"{'✓' if same else '✗ sha256 不符'}")
        if v:
            print(f"     表内: EI={v.get('EI')}  C0={v.get('C0')}  "
                  f"k_rod={v.get('k_rod')}  r_disk={v.get('r_disk')}")
            if c:
                de = abs(float(v.get("EI", 0)) / float(c.get("EI", 1)) - 1)
                if de > 0.02:
                    print(f"     ⚠ 表内 EI 与 CALIB 差 {de*100:.0f}% "
                          f"⇒ 表与标定**不是同一根臂**"); ok = False
    print("校验 :", "全部通过" if ok else "有问题")
    return 0 if ok else 1


def cmd_unpack(a):
    man, get, kind = _read_container(a.pkg)
    for t in man["tables"]:
        raw = get(t["file"])
        dst = os.path.join(ROOT, os.path.basename(t["file"]))
        with open(dst, "wb") as f:
            f.write(raw)
        print(f"[unpack] {os.path.basename(t['file'])} -> {dst}")
    print("[unpack] 表已就位，可直接 `python main.py info` 验收")
    return 0


# ────────────────────────────── CLI ──────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="建表 + 打包 .tdcrmodel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pack", "make"):
        s = sub.add_parser(name)
        s.add_argument("--tau", type=float, nargs="+", default=[20.0, 40.0])
        s.add_argument("--out", default="tdcr_model.tdcrmodel")
        s.add_argument("--prefix", default="vc_table_",
                       help="表名前缀；实机臂用 vc_table_real_")
        s.add_argument("--format", choices=("bin", "zip"), default="bin",
                       help="容器格式（默认 bin 二进制）")
        if name == "make":
            s.add_argument("--n", type=int, default=900)
            s.add_argument("--force", action="store_true")
    for name in ("info", "unpack"):
        s = sub.add_parser(name)
        s.add_argument("pkg")
        if name == "unpack":
            s.add_argument("-o", "--out", default=".")
    a = ap.parse_args()
    return {"pack": cmd_pack, "make": cmd_make,
            "info": cmd_info, "unpack": cmd_unpack}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
