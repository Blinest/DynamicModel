#!/bin/sh
# 一键：建表（缺则建）→ 打包 .tdcrmodel
# 用法: ./tdcr_build/make_tdcrmodel.sh [tau...] [--n 900]
set -e
cd "$(dirname "$0")/.."
TAU="${1:-20 40}"
OUT="tdcr_model_$(date +%Y%m%d).tdcrmodel"
exec python -u tdcr_build/tdcr_pack.py make --tau $TAU --n 900 --out "$OUT"
