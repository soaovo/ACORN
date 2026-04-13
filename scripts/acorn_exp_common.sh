#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/acorn_experiments}"
BUILD_JOBS="${BUILD_JOBS:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TEST_ACORN_BIN=""

resolve_python() {
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
        return 0
    fi
    echo "Cannot find python3 or python in PATH." >&2
    exit 1
}

resolve_test_acorn_bin() {
    local candidates=(
        "$BUILD_DIR/demos/test_acorn"
        "$BUILD_DIR/demos/test_acorn.exe"
        "$BUILD_DIR/test_acorn"
        "$BUILD_DIR/test_acorn.exe"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            TEST_ACORN_BIN="$candidate"
            return 0
        fi
    done
    return 1
}

build_test_acorn() {
    if resolve_test_acorn_bin; then
        return 0
    fi

    mkdir -p "$BUILD_DIR"

    cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DBUILD_TESTING=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_BUILD_TYPE=Release

    if [[ -n "$BUILD_JOBS" ]]; then
        cmake --build "$BUILD_DIR" --target test_acorn -j "$BUILD_JOBS"
    else
        cmake --build "$BUILD_DIR" --target test_acorn -j
    fi

    resolve_test_acorn_bin || {
        echo "Built the project but could not find the test_acorn binary." >&2
        exit 1
    }
}

millions_from_n() {
    local n="$1"
    echo $((n / 1000000))
}

remove_cached_indices() {
    local dataset="$1"
    local n="$2"
    local gamma="$3"
    local m="$4"
    local m_beta="$5"
    local n_millions

    n_millions=$(millions_from_n "$n")

    if [[ "$dataset" == "sift1M" || "$dataset" == "sift1B" ]]; then
        rm -f \
            "$REPO_ROOT/tmp/base_${n_millions}m_nc=${gamma}_assignment=rand_alpha=0.json" \
            "$REPO_ROOT/tmp/hybrid_${n_millions}m_nc=${gamma}_assignment=rand_alpha=0.json" \
            "$REPO_ROOT/tmp/hybrid_gamma1_${n_millions}m_nc=${gamma}_assignment=rand_alpha=0.json"
        return 0
    fi

    mkdir -p "$REPO_ROOT/tmp/$dataset"
    rm -f \
        "$REPO_ROOT/tmp/$dataset/base_M=${m}_efc=40.json" \
        "$REPO_ROOT/tmp/$dataset/hybrid_M=${m}_efc40_Mb=${m_beta}_gamma=${gamma}.json" \
        "$REPO_ROOT/tmp/$dataset/hybrid_M=${m}_efc40_Mb=${m_beta}_gamma=1.json"
}

run_case() {
    local figure="$1"
    local mode="$2"
    local log_file="$3"
    local omp_threads="$4"
    local n="$5"
    local gamma="$6"
    local dataset="$7"
    local m="$8"
    local m_beta="$9"
    local pathwise_width="${10}"
    local growth_interval="${11}"
    local efsearch="${12}"
    local reduced_sync="${13}"
    local edgewise_nt="${14}"

    mkdir -p "$(dirname "$log_file")"

    (
        cd "$REPO_ROOT"
        export OMP_NUM_THREADS="$omp_threads"
        export OMP_DYNAMIC=FALSE

        echo "EXP_FIGURE=$figure"
        echo "EXP_MODE=$mode"
        echo "EXP_LOG_FILE=$log_file"
        echo "EXP_OMP_NUM_THREADS=$omp_threads"
        echo "EXP_N=$n"
        echo "EXP_GAMMA=$gamma"
        echo "EXP_DATASET=$dataset"
        echo "EXP_M=$m"
        echo "EXP_M_BETA=$m_beta"
        echo "EXP_PATHWISE_WIDTH=$pathwise_width"
        echo "EXP_GROWTH_INTERVAL=$growth_interval"
        echo "EXP_EFSEARCH=$efsearch"
        echo "EXP_REDUCED_SYNC=$reduced_sync"
        echo "EXP_EDGEWISE_NT=$edgewise_nt"
        echo "EXP_CMD=$TEST_ACORN_BIN $n $gamma $dataset $m $m_beta $pathwise_width $growth_interval $efsearch $reduced_sync $edgewise_nt"

        "$TEST_ACORN_BIN" \
            "$n" \
            "$gamma" \
            "$dataset" \
            "$m" \
            "$m_beta" \
            "$pathwise_width" \
            "$growth_interval" \
            "$efsearch" \
            "$reduced_sync" \
            "$edgewise_nt"
    ) | tee "$log_file"
}

parse_logs_to_csv() {
    local log_dir="$1"
    local output_csv="$2"

    resolve_python
    "$PYTHON_BIN" "$SCRIPT_DIR/parse_acorn_logs.py" \
        --log-dir "$log_dir" \
        --csv "$output_csv"
}
