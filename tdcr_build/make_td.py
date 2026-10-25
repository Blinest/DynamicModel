# -*- coding: utf-8 -*-
"""make_td.py — 按**后端契约**打 `.tdcrmodel`（TDCRMOD1 v1 二进制容器）

容器布局（全部小端）：

    [0..8]     "TDCRMOD1"
    [8..12]    version = 1                  u32
    [12..16]   n_sections                   u32
    [16..]     n × { id[4], len:u32, offset:u32 }    索引表（每项 12 B）
    [offset..] 各段字节

段 id 与载荷（本工具实现的布局，**行主序 float32 LE**）：

    KTB1 = u32 n | tau(n×6) f32 | dL_mm(n×6) f32
    PTB1 = u32 n | p_tip(n×3) f32 | R_tip(n×9) f32
    META = UTF-8 JSON（标定/几何/点数/弯角/源表名，供人读，后端可忽略）

用法：

    python tdcr_build/make_td.py --tau 20 --prefix vc_table_real_ --out m.tdcrmodel
    python tdcr_build/make_td.py --tau 20 40 --out m.tdcrmodel
    python tdcr_build/make_td.py --dump m.tdcrmodel        # 只读索引与段信息

⚠ `KTB1` / `PTB1` 的**段内布局**是按“n + 行主序 f32”实现的；若后端另有约定
  （例如带时间戳、double、或列优先），改 `_sections()` 一处即可。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys

import numpy as np

MAGIC = b"TDCRMOD1"
VERSION = 1
# KKB1 每样本 183 个 float32 的排列方式：
#   "interleaved" = A: kx0,ky0,kz0, kx1,ky1,kz1, ...
#   "planar"      = B: kx0..kx60, ky0..ky60, kz0..kz60
KK_ORDER = "interleaved"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _f32(a):
    return np.ascontiguousarray(np.asarray(a, dtype="<f4"))


def _sections(raw_npz, tag):
    """npz 字节 → [(id, payload)]。改段内布局就改这里。"""
    z = np.load(io.BytesIO(raw_npz))
    n = int(z["tau"].shape[0])
    ktb = struct.pack("<I", n) + _f32(z["tau"]).tobytes() + _f32(z["dL_mm"]).tobytes()
    ptb = struct.pack("<I", n) + _f32(z["p_tip"]).tobytes() + _f32(z["R_tip"]).tobytes()
    ang = np.degrees(np.arccos(np.clip(z["R_tip"][:, 2, 2], -1.0, 1.0)))
    meta = dict(tag=tag, n=n,
                tau_max=float(z["tau"].max()),
                bend_deg=[round(float(ang.min()), 2), round(float(ang.max()), 2)],
                dtype="float32", order="C", layout="n|tau(n,6)|dL(n,6)")
    out = [("KTB1", ktb), ("PTB1", ptb)]
    # ── 曲率段（可选，源表有 kS/kK 时加）──
    # KSB1 = u32 n | u32 m | kS(n,m)   f32   弧长网格 [m]
    # KKB1 = u32 n | u32 m | kK(n,m,3) f32   曲率矢量 [1/m]
    if "kK" in z.files and "kS" in z.files:
        m = int(z["kK"].shape[1])
        out.append(("KSB1", struct.pack("<II", n, m) +
                    _f32(z["kS"]).tobytes()))
        _kk = z["kK"]
        if KK_ORDER == "planar":
            _kk = np.transpose(_kk, (0, 2, 1))     # (n,m,3)->(n,3,m)：三轴分块
        out.append(("KKB1", struct.pack("<II", n, m) +
                    _f32(_kk).tobytes()))
        meta["curvature"] = dict(section=["KSB1", "KKB1"], nodes=m,
                                 kk_order=KK_ORDER,
                                 s_range=[round(float(z["kS"].min()), 6),
                                          round(float(z["kS"].max()), 6)],
                                 kappa_norm_max=round(float(
                                     np.linalg.norm(z["kK"], axis=2).max()), 4))
    out.append(("META", json.dumps(meta, ensure_ascii=False).encode("utf-8")))
    return out


def build(taus, prefix, out):
    blobs = []
    for t in taus:
        p = os.path.join(ROOT, f"{prefix}{t:g}.npz")
        if not os.path.exists(p):
            sys.exit(f"缺表 {p}")
        raw = open(p, "rb").read()
        blobs += _sections(raw, f"{prefix}{t:g}")
        print(f"  + {os.path.basename(p)}  {os.path.getsize(p)/1e6:.2f} MB  "
              f"→ KTB1/PTB1 (n={np.load(io.BytesIO(raw))['tau'].shape[0]})")

    n = len(blobs)
    base = 16 + 12 * n
    idx, off, payload = b"", base, b""
    for sid, b in blobs:
        assert len(sid) == 4, f"段 id 必须 4 字节: {sid!r}"
        idx += sid.encode("ascii") + struct.pack("<II", len(b), off)
        off += len(b)
        payload += b
    with open(out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, n))
        f.write(idx)
        f.write(payload)
    print(f"[td] {out}  {os.path.getsize(out)/1e6:.2f} MB  "
          f"version={VERSION} sections={n} header={base}B")
    return 0


def dump(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        ver, n = struct.unpack("<II", f.read(8))
        entries = [struct.unpack("<4sII", f.read(12)) for _ in range(n)]
        hdr = 16 + 12 * n
        print(f"magic={magic!r}  version={ver}  sections={n}  header={hdr}B  "
              f"file={os.path.getsize(path)/1e6:.2f} MB")
        if magic != MAGIC:
            print(f"⚠ magic 不是 {MAGIC!r}")
        for sid, ln, off in entries:
            f.seek(off)
            data = f.read(ln)
            extra = ""
            if sid in (b"KTB1", b"PTB1"):
                (nn,) = struct.unpack("<I", data[:4])
                want = 4 + nn * 12 * 4 if sid == b"KTB1" else 4 + nn * 12 * 4
                extra = f"  n={nn}"
                if len(data) != want:
                    extra += f"  ⚠ 段长 {len(data)} != 预期 {want}"
            elif sid == b"META":
                extra = "  " + data.decode("utf-8", "replace")[:110]
            print(f"  {sid.decode('ascii', 'replace'):4s} len={ln:9d} off={off:9d}{extra}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, nargs="+", default=[20.0])
    ap.add_argument("--prefix", default="vc_table_")
    ap.add_argument("--out", default="tdcr_model.tdcrmodel")
    ap.add_argument("--dump", default=None, help="只解析已有包")
    ap.add_argument("--kk-order", choices=("interleaved", "planar"),
                    default="interleaved",
                    help="KKB1 排列: interleaved(A, kx0,ky0,kz0,…) / planar(B, kx0..kx60,ky0..ky60,…)")
    a = ap.parse_args()
    global KK_ORDER
    KK_ORDER = a.kk_order
    if a.dump:
        return dump(a.dump)
    return build(a.tau, a.prefix, a.out)


if __name__ == "__main__":
    sys.exit(main())
