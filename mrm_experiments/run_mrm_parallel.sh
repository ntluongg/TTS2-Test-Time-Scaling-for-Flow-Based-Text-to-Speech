#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/mrm_experiments"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
mkdir -p "$LOG_DIR"

EXP_NAME="${EXP_NAME:-F5TTS_v1_Base}"
CKPT_STEP="${CKPT_STEP:-1250000}"
TESTSET="${TESTSET:-seedtts_test_zh}"
NFE_STEP="${NFE_STEP:-32}"
SEARCH_N="${SEARCH_N:-8}"
SEEDS="${SEEDS:-2 3}"
MAX_UTTERANCES="${MAX_UTTERANCES:-50}"
RESULTS_TAG="${RESULTS_TAG:-b200_parallel}"

GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
PORT_A="${PORT_A:-8000}"
PORT_B="${PORT_B:-8001}"

FIX_T1="${FIX_T1:-0.5}"
SWEEP_T2_VALUES="${SWEEP_T2_VALUES:-0.65 0.75 0.85 0.90}"

FIX_T2="${FIX_T2:-0.8}"
SWEEP_T1_VALUES="${SWEEP_T1_VALUES:-0.20 0.35 0.50 0.65}"

PIDS=()

cleanup() {
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait || true
}

trap cleanup EXIT

wait_for_server() {
    local port="$1"
    local tries="${2:-60}"

    for ((i = 1; i <= tries; i++)); do
        if curl -fsS "http://127.0.0.1:${port}/docs" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done

    echo "Reward server on port ${port} did not become ready" >&2
    return 1
}

start_reward_server() {
    local gpu="$1"
    local port="$2"
    local log_file="$3"

    echo "Starting reward server on GPU ${gpu}, port ${port}"
    CUDA_VISIBLE_DEVICES="$gpu" \
    REWARD_DEVICE="cuda" \
    python -m uvicorn mrm_experiments.reward:app --host 127.0.0.1 --port "$port" \
        >"$log_file" 2>&1 &

    local pid=$!
    PIDS+=("$pid")
    wait_for_server "$port"
}

run_eval() {
    local gpu="$1"
    local port="$2"
    local seed="$3"
    local t1="$4"
    local t2="$5"
    local log_file="$6"

    local cmd=(
        python "$SCRIPT_DIR/eval_infer_batch_mrm_search.py"
        -s "$seed"
        -n "$EXP_NAME"
        -c "$CKPT_STEP"
        -t "$TESTSET"
        -nfe "$NFE_STEP"
        --search-type mrm
        --search-n "$SEARCH_N"
        --mrm-t1 "$t1"
        --mrm-t2 "$t2"
        --reward-url "http://127.0.0.1:${port}/infer"
        --results-tag "$RESULTS_TAG"
    )

    if [[ "$MAX_UTTERANCES" -gt 0 ]]; then
        cmd+=(--max-utterances "$MAX_UTTERANCES")
    fi

    echo "GPU ${gpu} | seed=${seed} | t1=${t1} | t2=${t2}"
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" | tee "$log_file"
}

run_sweep() {
    local gpu="$1"
    local port="$2"
    local mode="$3"
    local fixed="$4"
    shift 4
    local values=("$@")

    for value in "${values[@]}"; do
        for seed in $SEEDS; do
            local t1
            local t2

            if [[ "$mode" == "fix_t1" ]]; then
                t1="$fixed"
                t2="$value"
            else
                t1="$value"
                t2="$fixed"
            fi

            local stamp="seed${seed}_t1${t1}_t2${t2}"
            local log_file="$LOG_DIR/gpu${gpu}_${stamp}.log"
            run_eval "$gpu" "$port" "$seed" "$t1" "$t2" "$log_file"
        done
    done
}

echo "Logs: $LOG_DIR"
echo "Sweep A: fix t1=${FIX_T1}, move t2 over: ${SWEEP_T2_VALUES}"
echo "Sweep B: fix t2=${FIX_T2}, move t1 over: ${SWEEP_T1_VALUES}"

start_reward_server "$GPU_A" "$PORT_A" "$LOG_DIR/reward_gpu${GPU_A}.log"
start_reward_server "$GPU_B" "$PORT_B" "$LOG_DIR/reward_gpu${GPU_B}.log"

read -r -a sweep_t2_array <<<"$SWEEP_T2_VALUES"
read -r -a sweep_t1_array <<<"$SWEEP_T1_VALUES"

run_sweep "$GPU_A" "$PORT_A" "fix_t1" "$FIX_T1" "${sweep_t2_array[@]}" &
worker_a=$!
PIDS+=("$worker_a")

run_sweep "$GPU_B" "$PORT_B" "fix_t2" "$FIX_T2" "${sweep_t1_array[@]}" &
worker_b=$!
PIDS+=("$worker_b")

wait "$worker_a" "$worker_b"

echo "All MRM sweeps completed."
