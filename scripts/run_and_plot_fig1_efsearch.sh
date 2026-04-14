#!/usr/bin/env bash
# ============================================================
# 固定索引参数，改变 efSearch，生成 recall-延迟 曲线
# 用法:
#   bash scripts/run_and_plot_fig1_efsearch.sh
#
# 可通过环境变量覆盖默认值，例如:
#   N=500000 DATASET=sift1M EFS_LIST="16 32 64 128 256" bash scripts/run_and_plot_fig1_efsearch.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/acorn_exp_common.sh"

# ---------- 固定索引参数 ----------
N="${N:-1000000}"
GAMMA="${GAMMA:-12}"
DATASET="${DATASET:-custom}"
M="${M:-32}"
M_BETA="${M_BETA:-32}"
GROWTH_INTERVAL="${GROWTH_INTERVAL:-4}"
REDUCED_SYNC="${REDUCED_SYNC:-1}"
RESET_INDEX="${RESET_INDEX:-0}"

# ---------- 变化的 efSearch 列表 ----------
EFS_LIST="${EFS_LIST:-16 24 32 48 64 80 100 128 160 200 256}"

FIGURE_DIR="$LOG_ROOT/fig1_efsearch_recall_latency"

build_test_acorn
mkdir -p "$FIGURE_DIR"

if [[ "$RESET_INDEX" == "1" ]]; then
    remove_cached_indices "$DATASET" "$N" "$GAMMA" "$M" "$M_BETA"
fi

# ---------- 运行实验 ----------
for efs in $EFS_LIST; do
    log_file="$FIGURE_DIR/efsearch_${efs}.log"
    if [[ -f "$log_file" ]]; then
        echo "[skip] $log_file already exists (delete it to re-run)"
        continue
    fi
    run_case \
        "fig1_efsearch_recall_latency" \
        "single_thread" \
        "$log_file" \
        "1" \
        "$N" \
        "$GAMMA" \
        "$DATASET" \
        "$M" \
        "$M_BETA" \
        "1" \
        "$GROWTH_INTERVAL" \
        "$efs" \
        "$REDUCED_SYNC" \
        "1"
done

# ---------- 解析日志 -> CSV ----------
CSV_FILE="$FIGURE_DIR/summary.csv"
parse_logs_to_csv "$FIGURE_DIR" "$CSV_FILE"
echo "CSV written to: $CSV_FILE"

# ---------- 画图 ----------
resolve_python
"$PYTHON_BIN" "$SCRIPT_DIR/plot_recall_latency.py" \
    --csv "$CSV_FILE" \
    --out "$FIGURE_DIR/recall_latency.png"

echo "Plot saved to: $FIGURE_DIR/recall_latency.png"
