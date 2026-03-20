# =============================================================================
# Makefile — common tasks for MLAI2026COLUMBIA on RunPod
#
# All targets assume:
#   - source start_session.sh has been run (sets RESOLVED_MODEL_PATH)
#   - Network volume mounted at /workspace
# =============================================================================

SHELL := /bin/bash
NETWORK_VOL ?= /workspace
REPO_DIR ?= $(NETWORK_VOL)/MLAI2026COLUMBIA
RESULTS_BACKUP ?= $(NETWORK_VOL)/results_backup

# Auto-resolve model path; override with: make attack-single MODEL_PATH=/path/to/model
MODEL_PATH ?= $(RESOLVED_MODEL_PATH)

# Defaults
BEHAVIOR_ID ?= 1
DEVICE ?= 0
DEFENSE ?= no_defense
CONFIG ?= behaviors_config.json
OURS_CONFIG ?= behaviors_ours_config.json
INIT_CONFIG ?= behaviors_ours_config_init.json

METHODS ?= A,B,C,D
CYBER_CONFIG ?= data/cyber_behaviors.json

.PHONY: help status attack-single attack-all attack-ours robust-a robust-b robust-c robust-d \
        generate-config backup-results clean-output tmux-run smoke-test quick-eval

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Status ──────────────────────────────────────────────────────────────────

status: ## GPU utilization, VRAM, disk usage
	@echo "=== GPU ==="
	@nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
		--format=csv,noheader 2>/dev/null || echo "No GPU found"
	@echo ""
	@echo "=== Disk ==="
	@df -h $(NETWORK_VOL) 2>/dev/null || echo "Network volume not mounted"
	@echo ""
	@echo "=== Running Python processes ==="
	@ps aux | grep '[p]ython.*attack\|[p]ython.*robust_gcg\|[p]ython.*smooth' | head -5 || echo "None"

# ── Base GCG attack ────────────────────────────────────────────────────────

attack-single: ## Run single behavior: make attack-single BEHAVIOR_ID=1
	python attack_llm_core_base.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--id $(BEHAVIOR_ID) \
		--defense $(DEFENSE) \
		--behaviors_config $(CONFIG)

attack-all: ## Run all 50 behaviors serially (adapts multi-GPU script for 1 GPU)
	python run_multiple_attack_our_target.py \
		--defense $(DEFENSE) \
		--behaviors_config $(CONFIG)

# ── I-GCG (our target) ─────────────────────────────────────────────────────

attack-ours-init: ## Step 1: Generate initial suffixes with I-GCG
	python attack_llm_core_best_update_our_target.py \
		--model_path $(MODEL_PATH) \
		--behaviors_config $(OURS_CONFIG)

generate-config: ## Step 2: Build config with initialized suffixes
	python generate_our_config.py

attack-ours: ## Step 3: Run I-GCG with initialized config
	python run_multiple_attack_our_target.py \
		--behaviors_config $(INIT_CONFIG)

# ── Robust GCG variants ────────────────────────────────────────────────────

robust-a: ## Robust GCG A: suffix char perturbation
	python scripts/robust_gcg_A_suffix_charperturb.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--id $(BEHAVIOR_ID)

robust-b: ## Robust GCG B: token perturbation
	python scripts/robust_gcg_B_token_perturb.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--id $(BEHAVIOR_ID)

robust-c: ## Robust GCG C: generation eval (expensive)
	python scripts/robust_gcg_C_generation_eval.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--id $(BEHAVIOR_ID)

robust-d: ## Robust GCG D: inert buffer
	python scripts/robust_gcg_D_inert_buffer.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--id $(BEHAVIOR_ID)

# ── SmoothLLM defense evaluation ───────────────────────────────────────────

attack-with-smooth: ## Run attack + SmoothLLM defense eval
	python attack_llm_core_base.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--id $(BEHAVIOR_ID) \
		--defense smooth_llm \
		--behaviors_config $(CONFIG)

# ── Fast evaluation (smoke / quick) ─────────────────────────────────────────

smoke-test: ## Smoke test: 5 behaviors x 100 steps (~30 min on A100)
	python scripts/fast_robust_eval.py \
		--tier smoke \
		--methods $(METHODS) \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--behaviors_config $(CYBER_CONFIG)

quick-eval: ## Quick eval: 15 behaviors x 200 steps (~3 h on A100)
	python scripts/fast_robust_eval.py \
		--tier quick \
		--methods $(METHODS) \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--behaviors_config $(CYBER_CONFIG)

# ── Analysis ────────────────────────────────────────────────────────────────

analyze: ## Run result analysis and generate plots
	python scripts/analyze_results.py

# ── Utilities ───────────────────────────────────────────────────────────────

backup-results: ## Copy output/ to network volume
	@mkdir -p $(RESULTS_BACKUP)
	rsync -av --progress output/ $(RESULTS_BACKUP)/
	rsync -av --progress Our_GCG_target_len_20/ $(RESULTS_BACKUP)/Our_GCG_target_len_20/ 2>/dev/null || true
	@echo "Results backed up to $(RESULTS_BACKUP)"

clean-output: ## Remove local output (keeps network backup)
	@echo "This will delete output/ and Our_GCG_target_len_20/ locally."
	@read -p "Continue? [y/N] " ans && [ "$${ans}" = "y" ] && \
		rm -rf output/ Our_GCG_target_len_20/ && echo "Cleaned." || echo "Aborted."

tmux-run: ## Start a tmux session and run attack-all inside it
	tmux new-session -d -s gcg_attack "cd $(REPO_DIR) && source start_session.sh && make attack-all; exec bash"
	@echo "Attack running in tmux session 'gcg_attack'. Attach with: tmux attach -t gcg_attack"

vram: ## Show live VRAM usage
	@watch -n 1 nvidia-smi
