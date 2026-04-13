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
GROWTH_INTERVAL="${GROWTH_INTERVAL:-4}"
REDUCED_SYNC="${REDUCED_SYNC:-1}"
RESET_INDEX="${RESET_INDEX:-0}"
EFS_LIST="${EFS_LIST:-16 24 32 48 64 80 100 128 160 200 256}"

FIGURE_DIR="$LOG_ROOT/fig1_efsearch_single_thread"

build_test_acorn
mkdir -p "$FIGURE_DIR"

if [[ "$RESET_INDEX" == "1" ]]; then
    remove_cached_indices "$DATASET" "$N" "$GAMMA" "$M" "$M_BETA"
fi

for efs in $EFS_LIST; do
    log_file="$FIGURE_DIR/efsearch_${efs}.log"
    run_case \
        "fig1_efsearch_single_thread" \
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

parse_logs_to_csv "$FIGURE_DIR" "$FIGURE_DIR/summary.csv"
echo "fig1 logs: $FIGURE_DIR"
echo "fig1 csv : $FIGURE_DIR/summary.csv"
