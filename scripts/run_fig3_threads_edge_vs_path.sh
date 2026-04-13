#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./acorn_exp_common.sh
source "$SCRIPT_DIR/acorn_exp_common.sh"

N="${N:-1000000}"
GAMMA="${GAMMA:-12}"
DATASET="${DATASET:-custom}"
M="${M:-32}"
M_BETA="${M_BETA:-32}"
REDUCED_SYNC="${REDUCED_SYNC:-1}"
GROWTH_INTERVAL="${GROWTH_INTERVAL:-4}"
EFSEARCH="${EFSEARCH:-100}"
RESET_INDEX="${RESET_INDEX:-0}"
THREAD_LIST="${THREAD_LIST:-1 2 4 8 16}"

FIGURE_DIR="$LOG_ROOT/fig3_threads_edge_vs_path"

build_test_acorn
mkdir -p "$FIGURE_DIR"

if [[ "$RESET_INDEX" == "1" ]]; then
    remove_cached_indices "$DATASET" "$N" "$GAMMA" "$M" "$M_BETA"
fi

for threads in $THREAD_LIST; do
    edge_log="$FIGURE_DIR/edgewise_${threads}t.log"
    run_case \
        "fig3_threads_edge_vs_path" \
        "edgewise" \
        "$edge_log" \
        "$threads" \
        "$N" \
        "$GAMMA" \
        "$DATASET" \
        "$M" \
        "$M_BETA" \
        "1" \
        "$GROWTH_INTERVAL" \
        "$EFSEARCH" \
        "$REDUCED_SYNC" \
        "$threads"

    path_log="$FIGURE_DIR/pathwise_${threads}t.log"
    run_case \
        "fig3_threads_edge_vs_path" \
        "pathwise" \
        "$path_log" \
        "$threads" \
        "$N" \
        "$GAMMA" \
        "$DATASET" \
        "$M" \
        "$M_BETA" \
        "$threads" \
        "$GROWTH_INTERVAL" \
        "$EFSEARCH" \
        "$REDUCED_SYNC" \
        "1"
done

parse_logs_to_csv "$FIGURE_DIR" "$FIGURE_DIR/summary.csv"
echo "fig3 logs: $FIGURE_DIR"
echo "fig3 csv : $FIGURE_DIR/summary.csv"
