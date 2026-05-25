#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/mrm_experiments"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs_fixed_t2}"
mkdir -p "$LOG_DIR"

EXP_NAME="${EXP_NAME:-F5TTS_v1_Base}"
CKPT_STEP="${CKPT_STEP:-1250000}"
TESTSET="${TESTSET:-seedtts_test_zh}"
NFE_STEP="${NFE_STEP:-32}"
SEARCH_N="${SEARCH_N:-8}"
SEEDS="${SEEDS:-2 3}"
MAX_UTTERANCES="${MAX_UTTERANCES:-50}"
RESULTS_TAG="${RESULTS_TAG:-fixed_t2_fanout}"

EXPERIMENT_GPU="${EXPERIMENT_GPU:-0}"
REWARD_GPU="${REWARD_GPU:-1}"
REWARD_BASE_PORT="${REWARD_BASE_PORT:-8000}"
REWARD_INSTANCES="${REWARD_INSTANCES:-4}"
EXPERIMENT_WORKERS="${EXPERIMENT_WORKERS:-4}"

FIX_T2="${FIX_T2:-0.8}"
SWEEP_T1_VALUES="${SWEEP_T1_VALUES:-0.20 0.35 0.50 0.65}"

PIDS=()
REWARD_PIDS=()
WORKER_PIDS=()
TASKS=()

cleanup() {
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait || true
}

trap cleanup EXIT

wait_for_server() {
    local port="$1"
    local tries="${2:-90}"

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
    local index="$1"
    local port="$2"
    local log_file="$3"

    echo "Starting reward server ${index} on GPU ${REWARD_GPU}, port ${port}"
    CUDA_VISIBLE_DEVICES="$REWARD_GPU" \
    REWARD_DEVICE="cuda" \
    python -m uvicorn mrm_experiments.reward:app --host 127.0.0.1 --port "$port" \
        >"$log_file" 2>&1 &

    local pid=$!
    PIDS+=("$pid")
    REWARD_PIDS+=("$pid")
    wait_for_server "$port"
}

make_tasks() {
    read -r -a sweep_t1_array <<<"$SWEEP_T1_VALUES"
    for t1 in "${sweep_t1_array[@]}"; do
        for seed in $SEEDS; do
            TASKS+=("${seed}|${t1}|${FIX_T2}")
        done
    done
}

run_one() {
    local worker_id="$1"
    local reward_port="$2"
    local task="$3"

    IFS="|" read -r seed t1 t2 <<<"$task"
    local stamp="worker${worker_id}_seed${seed}_t1${t1}_t2${t2}"
    local log_file="$LOG_DIR/${stamp}.log"

    echo "Worker ${worker_id} | seed=${seed} | t1=${t1} | t2=${t2} | reward_port=${reward_port}"

    CUDA_VISIBLE_DEVICES="$EXPERIMENT_GPU" \
    python "$SCRIPT_DIR/eval_infer_batch_mrm_search.py" \
        -s "$seed" \
        -n "$EXP_NAME" \
        -c "$CKPT_STEP" \
        -t "$TESTSET" \
        -nfe "$NFE_STEP" \
        --search-type mrm \
        --search-n "$SEARCH_N" \
        --mrm-t1 "$t1" \
        --mrm-t2 "$t2" \
        --reward-url "http://127.0.0.1:${reward_port}/infer" \
        --results-tag "$RESULTS_TAG" \
        --max-utterances "$MAX_UTTERANCES" \
        | tee "$log_file"
}

echo "Logs: $LOG_DIR"
echo "Fixed t2=${FIX_T2}, sweeping t1 over: ${SWEEP_T1_VALUES}"
echo "Experiment GPU: ${EXPERIMENT_GPU}"
echo "Reward GPU: ${REWARD_GPU}"
echo "Reward instances: ${REWARD_INSTANCES}"
echo "Experiment workers: ${EXPERIMENT_WORKERS}"

for ((i = 0; i < REWARD_INSTANCES; i++)); do
    port=$((REWARD_BASE_PORT + i))
    start_reward_server "$i" "$port" "$LOG_DIR/reward_${i}.log"
done

make_tasks

if [[ "${#TASKS[@]}" -eq 0 ]]; then
    echo "No tasks to run." >&2
    exit 1
fi

active_jobs=0
task_index=0

while [[ "$task_index" -lt "${#TASKS[@]}" ]]; do
    while [[ "$active_jobs" -lt "$EXPERIMENT_WORKERS" && "$task_index" -lt "${#TASKS[@]}" ]]; do
        worker_id="$active_jobs"
        reward_port=$((REWARD_BASE_PORT + (worker_id % REWARD_INSTANCES)))
        run_one "$worker_id" "$reward_port" "${TASKS[$task_index]}" &
        pid=$!
        PIDS+=("$pid")
        WORKER_PIDS+=("$pid")
        active_jobs=$((active_jobs + 1))
        task_index=$((task_index + 1))
    done

    wait -n "${WORKER_PIDS[@]}"
    active_jobs=$((active_jobs - 1))
done

while [[ "$active_jobs" -gt 0 ]]; do
    wait -n "${WORKER_PIDS[@]}"
    active_jobs=$((active_jobs - 1))
done

for pid in "${REWARD_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
done

echo "Fixed t2 sweep completed."
