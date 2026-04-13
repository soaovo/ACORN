#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./acorn_exp_common.sh
source "$SCRIPT_DIR/acorn_exp_common.sh"

N="${N:-1000000}"
GAMMA="${GAMMA:-12}"
DATASET="${DATASET:-custom}"
M="${M:-32}"
REDUCED_SYNC="${REDUCED_SYNC:-1}"
GROWTH_INTERVAL="${GROWTH_INTERVAL:-4}"
EFSEARCH="${EFSEARCH:-100}"
THREAD_LIST="${THREAD_LIST:-4 8}"
MBETA_LIST="${MBETA_LIST:-16 24 32 40 48 64 96}"
FORCE_REBUILD_EVERY_CASE="${FORCE_REBUILD_EVERY_CASE:-0}"

FIGURE_DIR="$LOG_ROOT/fig2_edge_vs_path_rebuild"

build_test_acorn
mkdir -p "$FIGURE_DIR"

for m_beta in $MBETA_LIST; do
    remove_cached_indices "$DATASET" "$N" "$GAMMA" "$M" "$m_beta"

    for threads in $THREAD_LIST; do
        edge_log="$FIGURE_DIR/mbeta_${m_beta}_edgewise_${threads}t.log"
        run_case \
            "fig2_edge_vs_path_rebuild" \
            "edgewise" \
            "$edge_log" \
            "$threads" \
            "$N" \
            "$GAMMA" \
            "$DATASET" \
            "$M" \
            "$m_beta" \
            "1" \
            "$GROWTH_INTERVAL" \
            "$EFSEARCH" \
            "$REDUCED_SYNC" \
            "$threads"

        if [[ "$FORCE_REBUILD_EVERY_CASE" == "1" ]]; then
            remove_cached_indices "$DATASET" "$N" "$GAMMA" "$M" "$m_beta"
        fi

        path_log="$FIGURE_DIR/mbeta_${m_beta}_pathwise_${threads}t.log"
        run_case \
            "fig2_edge_vs_path_rebuild" \
            "pathwise" \
            "$path_log" \
            "$threads" \
            "$N" \
            "$GAMMA" \
            "$DATASET" \
            "$M" \
            "$m_beta" \
            "$threads" \
            "$GROWTH_INTERVAL" \
            "$EFSEARCH" \
            "$REDUCED_SYNC" \
            "1"
    done
done

parse_logs_to_csv "$FIGURE_DIR" "$FIGURE_DIR/summary.csv"
echo "fig2 logs: $FIGURE_DIR"
echo "fig2 csv : $FIGURE_DIR/summary.csv"
