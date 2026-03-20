---
description: RunPod deployment context for A100 PCIe 80GB pods
globs:
  - "setup_runpod.sh"
  - "start_session.sh"
  - "Makefile"
  - "*.py"
alwaysApply: true
---

# RunPod Deployment Context

## Environment
- **GPU**: NVIDIA A100 PCIe 80GB VRAM
- **System**: 125 GB RAM, 16 vCPU
- **Network volume**: `/workspace` (persistent across restarts)
- **Container disk**: local NVMe (wiped on pod stop)

## First-Time Setup
```bash
cd /workspace/MLAI2026COLUMBIA
chmod +x setup_runpod.sh && ./setup_runpod.sh
```

## Every Session
```bash
source /workspace/MLAI2026COLUMBIA/start_session.sh
```
This activates the conda env, resolves the model path, and exports `RESOLVED_MODEL_PATH`.

## Model Path
Scripts default to `/home/LLM/Llama-2-7b-chat-hf`. On RunPod, override with:
```
--model_path $RESOLVED_MODEL_PATH
```
Or use the Makefile which handles this automatically.

## Storage Layout
```
/workspace/                          # network volume (persistent)
├── MLAI2026COLUMBIA/                # this repo
├── models/                          # HuggingFace model cache
├── envs/iris/                       # conda environment
├── .cache/huggingface/              # HF cache (symlinked from ~/.cache)
└── results_backup/                  # backed-up experiment outputs
```

## Key Commands (via Makefile)

### Fast evaluation (start here)
- `make smoke-test` — 5 behaviors x 100 steps x 4 methods (~30 min). Quick sanity check.
- `make quick-eval` — 15 behaviors x 200 steps x 4 methods (~3 h). Publishable ASR estimate.
- `make smoke-test METHODS=A,B` — run only a subset of robust methods
- `make quick-eval METHODS=C` — run only method C

### Full attacks
- `make attack-single BEHAVIOR_ID=1` — run one base GCG attack
- `make attack-all` — run all 50 base GCG behaviors
- `make attack-ours` — run I-GCG with initialized suffixes
- `make robust-a BEHAVIOR_ID=1` — robust GCG variant A (single behavior, full 500 steps)
- `make robust-b BEHAVIOR_ID=1` — robust GCG variant B
- `make robust-c BEHAVIOR_ID=1` — robust GCG variant C (expensive)
- `make robust-d BEHAVIOR_ID=1` — robust GCG variant D

### Utilities
- `make status` — GPU/disk dashboard
- `make backup-results` — copy results to network volume
- `make help` — list all targets

## Compute Estimates (A100 PCIe)

### Fast eval (scripts/fast_robust_eval.py)
- Smoke test: ~30 min total (A ~5m, B ~4m, C ~12m, D ~6m)
- Quick eval: ~3.2 h total (A ~35m, B ~25m, C ~90m, D ~40m)

### Full runs
- Base GCG (1000 steps): ~25-42 min/behavior, ~21-35h total (50 behaviors)
- I-GCG (4000 steps): ~100-167 min/behavior, ~3.5-6 days total
- Robust GCG full (500 steps x 15 behaviors): 5-9h per method
- Robust GCG C is 8-15x slower than base GCG
- Many behaviors converge early (100-400 steps), real time is often 30-50% less

## Important Notes
- Only 1 GPU available; `run_multiple_attack_our_target.py` device_list should be `[0]`
- Use `tmux` or `screen` for long runs (SSH disconnect protection)
- Always `make backup-results` before stopping the pod
- Fast eval results go to `output/fast_eval/<tier>/<timestamp>/`
