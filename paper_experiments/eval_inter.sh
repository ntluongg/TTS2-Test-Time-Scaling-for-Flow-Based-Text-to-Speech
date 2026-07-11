#!/bin/bash

# Function to run all three test types for a given seed in parallel
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
run_seed_parallel() {
    local seed=$1

    echo "Starting seed $seed with parallel test types..."

    # Run all three test types for this seed in parallel
    accelerate launch src/f5_tts/eval/eval_infer_batch.py -s $seed -n "F5TTS_v1_Base" -t "seedtts_test_zh" -nfe 32 &
    pid1=$!

    # accelerate launch --config_file /home/nhatth2/.cache/huggingface/accelerate/luong_config.yml src/f5_tts/eval/eval_infer_batch.py -s $seed -n "F5TTS_v1_Base" -t "seedtts_test_en" -nfe 32 &
    # pid2=$!

    # accelerate launch --config_file /home/nhatth2/.cache/huggingface/accelerate/luong_config.yml src/f5_tts/eval/eval_infer_batch.py -s $seed -n "F5TTS_v1_Base" -t "ls_pc_test_clean" -nfe 32 &
    # pid3=$!

    # Wait for all three parallel jobs for this seed to complete
    wait $pid1 $pid2 $pid3

    echo "Seed $seed completed"
    echo "------------------------"
}

# Run seeds sequentially (0 to 15)
for seed in {2..6}; do
    run_seed_parallel $seed
done
# run_seed_parallel 0

echo "All seeds completed!"