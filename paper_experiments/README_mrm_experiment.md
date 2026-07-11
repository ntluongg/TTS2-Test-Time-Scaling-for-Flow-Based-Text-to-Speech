# MRM search experiment

This folder contains the experimental MRM search runner only for server-side sweeps.

## File naming

- Original repo eval code:
  - `src/f5_tts/eval/eval_infer_batch.py`
- Experimental MRM eval code:
  - `paper_experiments/eval_infer_batch_mrm_search.py`

That separation is intentional so it is always obvious which entrypoint is the modified one.

## What `t1` and `t2` mean

Inside `MRMSearch`:

- `t1`: switch from pure denoise to SMC
- `t2`: switch from SMC to final BoN-like refinement

So there are two main sweeps:

- fixed `t2`, move `t1`
- fixed `t1`, move `t2`

## Output folder naming

Each run now writes to:

```text
results/mrm_search/exp-...__ckpt-...__set-...__seed-...__ode-...__nfe-...__mel-...__search-...__N-...__t1-...__t2-...__cfg-...__speed-...__tag-...
```

Example:

```text
results/mrm_search/exp-F5TTS_v1_Base__ckpt-1250000__set-seedtts_test_zh__seed-2__ode-euler__nfe-32__mel-vocos__search-mrm__N-8__t1-0.35__t2-0.8__cfg-2__speed-1__tag-fixed_t2_fanout
```

This makes it easy to see:

- which model/checkpoint
- which dataset
- which seed
- which search type
- which `t1` / `t2`
- which extra tag grouped the run

## Scripts

- Fixed `t2`, move `t1`:
  - `paper_experiments/run_mrm_fixed_t2_fanout.sh`
- Fixed `t1`, move `t2`:
  - `paper_experiments/run_mrm_fixed_t1_fanout.sh`

Both scripts assume by default:

- `GPU0` runs both experiment workers and reward-server instances.
- Only 1 reward instance and 1 experiment worker is launched to be accessible for reviewers with limited hardware.
- Only 1 sample is tested (`MAX_UTTERANCES=1`) by default to quickly verify the code works.

## How the multi-instance setup works

The reward code is served with FastAPI + Uvicorn from:

- `paper_experiments/reward.py`

You can run one instance like this:

```bash
CUDA_VISIBLE_DEVICES=1 \
REWARD_DEVICE=cuda \
python -m uvicorn paper_experiments.reward:app --host 127.0.0.1 --port 8000
```

You can run multiple instances on the same GPU by changing only the port:

```bash
CUDA_VISIBLE_DEVICES=1 REWARD_DEVICE=cuda python -m uvicorn paper_experiments.reward:app --host 127.0.0.1 --port 8000
CUDA_VISIBLE_DEVICES=1 REWARD_DEVICE=cuda python -m uvicorn paper_experiments.reward:app --host 127.0.0.1 --port 8001
CUDA_VISIBLE_DEVICES=1 REWARD_DEVICE=cuda python -m uvicorn paper_experiments.reward:app --host 127.0.0.1 --port 8002
CUDA_VISIBLE_DEVICES=1 REWARD_DEVICE=cuda python -m uvicorn paper_experiments.reward:app --host 127.0.0.1 --port 8003
```

Each server is the same model stack, just listening on a different port.

Then experiment workers on `GPU0` call:

- `http://127.0.0.1:8000/infer`
- `http://127.0.0.1:8001/infer`
- `http://127.0.0.1:8002/infer`
- `http://127.0.0.1:8003/infer`

This spreads reward work across multiple processes on the reward GPU.

## Easiest way to run

The provided scripts already start the reward instances for you.

### Fixed `t2` first

```bash
bash paper_experiments/run_mrm_fixed_t2_fanout.sh
```

By default, this will run on a single GPU (`GPU0`) with 1 worker, 1 reward instance, and test only 1 sample. 

**For reviewers with multiple GPUs or wanting a full run:**
You can scale up the evaluation by specifying environment variables:

```bash
EXPERIMENT_WORKERS=4 \
REWARD_INSTANCES=4 \
EXPERIMENT_GPU=0 \
REWARD_GPU=1 \
FIX_T2=0.8 \
SWEEP_T1_VALUES="0.20 0.35 0.50 0.65 0.75" \
SEEDS="2 3 4" \
MAX_UTTERANCES=100 \
RESULTS_TAG=fixed_t2_first \
bash paper_experiments/run_mrm_fixed_t2_fanout.sh
```

This means:

- `t2` stays at `0.8`
- `t1` is swept over the listed values
- 4 experiment workers run on `GPU0`
- 4 reward instances run on `GPU1`

### Fixed `t1` later

```bash
bash paper_experiments/run_mrm_fixed_t1_fanout.sh
```

Similarly, to scale up:

```bash
EXPERIMENT_WORKERS=4 \
REWARD_INSTANCES=4 \
EXPERIMENT_GPU=0 \
REWARD_GPU=1 \
FIX_T1=0.5 \
SWEEP_T2_VALUES="0.65 0.75 0.80 0.85 0.90" \
SEEDS="2 3 4" \
MAX_UTTERANCES=100 \
RESULTS_TAG=fixed_t1_second \
bash paper_experiments/run_mrm_fixed_t1_fanout.sh
```

This means:

- `t1` stays at `0.5`
- `t2` is swept over the listed values

## Manual single-run command

If you want to launch one experiment by hand:

```bash
CUDA_VISIBLE_DEVICES=0 \
python paper_experiments/eval_infer_batch_mrm_search.py \
  -s 2 \
  -n F5TTS_v1_Base \
  -c 1250000 \
  -t seedtts_test_zh \
  -nfe 32 \
  --search-type mrm \
  --search-n 8 \
  --mrm-t1 0.35 \
  --mrm-t2 0.8 \
  --reward-url http://127.0.0.1:8000/infer \
  --max-utterances 100 \
  --results-tag manual_debug
```

## Notes

- The reward path was optimized to keep more work on GPU.
- The `sim-o` reference embedding is cached to reduce repeated work.
- The MRM eval runner can also accept reward URLs pointing to different ports if you want to orchestrate servers manually.
