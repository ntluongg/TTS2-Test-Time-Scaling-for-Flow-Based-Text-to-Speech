#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXP_NAME="${EXP_NAME:-F5TTS_v1_Base}"
CKPT_STEP="${CKPT_STEP:-1250000}"
TESTSET="${TESTSET:-ls_pc_test_clean}"
MAX_UTTERANCES="${MAX_UTTERANCES:-1}"
SEEDS_STR="${SEEDS:-42}"

T1="${T1:-0.5}"
T2="${T2:-0.8}"
MRM_NFE_STEP="${MRM_NFE_STEP:-32}"

ACCEL_CONFIG="${ACCEL_CONFIG:-}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"

BON_N_LIST_STR="${BON_N_LIST:-2 4 8 16 32 64}"
F5_NFE_LIST_STR="${F5_NFE_LIST:-4 8 16 32 64 128 256 512 1024 2048}"
MRM_N_LIST_STR="${MRM_N_LIST:-2 4}"
BON_GROUP_SIZE="${BON_GROUP_SIZE:-3}"

BON_PORT_BASE="${BON_PORT_BASE:-8001}"
MRM_PORT_BASE="${MRM_PORT_BASE:-8001}"

LOG_DIR="${LOG_DIR:-logs_scale_search}"
mkdir -p "$LOG_DIR"

read -r -a SEEDS_ARR <<<"$SEEDS_STR"
read -r -a BON_N_LIST_ARR <<<"$BON_N_LIST_STR"
read -r -a F5_NFE_LIST_ARR <<<"$F5_NFE_LIST_STR"
read -r -a MRM_N_LIST_ARR <<<"$MRM_N_LIST_STR"

ACCEL_CMD=(accelerate launch)
if [[ -n "$ACCEL_CONFIG" ]]; then
  ACCEL_CMD+=(--config_file "$ACCEL_CONFIG")
fi

run_custom_search_job() {
  local seed="$1"
  local search_type="$2"
  local search_n="$3"
  local nfe_step="$4"
  local port="$5"
  local tag="$6"

  local log_file="${LOG_DIR}/${tag}.log"

  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
  "${ACCEL_CMD[@]}" src/f5_tts/eval//eval_infer_batch_mrm_search.py \
    -s "$seed" \
    -n "$EXP_NAME" \
    -c "$CKPT_STEP" \
    -t "$TESTSET" \
    -nfe "$nfe_step" \
    --search-type "$search_type" \
    --search-n "$search_n" \
    --mrm-t1 "$T1" \
    --mrm-t2 "$T2" \
    --reward-url "http://127.0.0.1:${port}/infer" \
    --max-utterances "$MAX_UTTERANCES" \
    --results-tag "$tag" \
    >"$log_file" 2>&1
}

run_original_f5_job() {
  local seed="$1"
  local nfe_step="$2"
  local tag="$3"

  local log_file="${LOG_DIR}/${tag}.log"

  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
  "${ACCEL_CMD[@]}" src/f5_tts/eval/eval_infer_batch.py \
    -s "$seed" \
    -n "$EXP_NAME" \
    -c "$CKPT_STEP" \
    -t "$TESTSET" \
    -nfe "$nfe_step" \
    >"$log_file" 2>&1
}

run_original_f5_parallel() {
  local pids=()

  echo "===== Original F5 sweep: nfe = ${F5_NFE_LIST_STR} ====="

  for nfe in "${F5_NFE_LIST_ARR[@]}"; do
    echo "===== Original F5 launch: nfe=${nfe} ====="
    for seed in "${SEEDS_ARR[@]}"; do
      tag="f5_original__seed-${seed}__nfe-${nfe}"
      run_original_f5_job "$seed" "$nfe" "$tag" &
      pids+=("$!")
    done
  done

  wait "${pids[@]}"
  echo "===== Original F5 sweep finished ====="
}

run_bon_grouped_parallel() {
  local total="${#BON_N_LIST_ARR[@]}"
  local group_start=0
  local group_id=1

  echo "===== BoN sweep: N = ${BON_N_LIST_STR} ====="

  while (( group_start < total )); do
    local pids=()
    local group_end=$((group_start + BON_GROUP_SIZE))
    if (( group_end > total )); then
      group_end=$total
    fi

    echo "===== BoN group ${group_id}: indexes ${group_start}..$((group_end - 1)) ====="

    for ((i = group_start; i < group_end; i++)); do
      local n="${BON_N_LIST_ARR[$i]}"
      local port=$((BON_PORT_BASE + i))

      for seed in "${SEEDS_ARR[@]}"; do
        local tag="bon__seed-${seed}__N-${n}__nfe-${MRM_NFE_STEP}__port-${port}"
        run_custom_search_job "$seed" "bon" "$n" "$MRM_NFE_STEP" "$port" "$tag" &
        pids+=("$!")
      done
    done

    wait "${pids[@]}"
    echo "===== BoN group ${group_id} finished ====="

    group_start=$group_end
    group_id=$((group_id + 1))
  done
}

run_mrm_parallel() {
  local pids=()

  echo "===== MRM sweep: N = ${MRM_N_LIST_STR} ====="

  for i in "${!MRM_N_LIST_ARR[@]}"; do
    local n="${MRM_N_LIST_ARR[$i]}"
    local port=$((MRM_PORT_BASE + i))

    echo "===== MRM launch: N=${n} ====="
    for seed in "${SEEDS_ARR[@]}"; do
      local tag="mrm__seed-${seed}__N-${n}__nfe-${MRM_NFE_STEP}__t1-${T1}__t2-${T2}__port-${port}"
      run_custom_search_job "$seed" "mrm" "$n" "$MRM_NFE_STEP" "$port" "$tag" &
      pids+=("$!")
    done
  done

  wait "${pids[@]}"
  echo "===== MRM sweep finished ====="
}

run_original_f5_parallel
run_bon_grouped_parallel
run_mrm_parallel

echo "===== all experiments finished ====="
