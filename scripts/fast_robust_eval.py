#!/usr/bin/env python3
"""Fast robust GCG evaluation — smoke test or quick eval across methods A-D.

Loads the model **once** and runs multiple behaviors x methods sequentially,
aggregating ASR into a summary table.

Tiers
-----
smoke : 5 behaviors, 100 steps, reduced batch/perturbation — ~20-40 min on A100
quick : 15 behaviors, 200 steps, full batch, reduced perturbation — ~2-4 hours

Usage
-----
    # Smoke test, all methods:
    python scripts/fast_robust_eval.py --tier smoke --model_path /path/to/llama2

    # Quick eval, methods A and B only:
    python scripts/fast_robust_eval.py --tier quick --methods A,B

    # Override device or config:
    python scripts/fast_robust_eval.py --tier smoke --device 0 --behaviors_config data/cyber_behaviors.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from llm_attacks.minimal_gcg.opt_utils import load_model_and_tokenizer
from robust_gcg.attack_harness import run_attack_with_model

# ---------------------------------------------------------------------------
# Method registry — maps short name to (select_candidate, extra_init) pairs.
# Imports are deferred to avoid loading unused modules.
# ---------------------------------------------------------------------------

def _load_method(name: str):
    if name == "A":
        from scripts.robust_gcg_A_suffix_charperturb import select_candidate
        return select_candidate, None
    elif name == "B":
        from scripts.robust_gcg_B_token_perturb import select_candidate, extra_init
        return select_candidate, extra_init
    elif name == "C":
        from scripts.robust_gcg_C_generation_eval import select_candidate
        return select_candidate, None
    elif name == "D":
        from scripts.robust_gcg_D_inert_buffer import select_candidate, extra_init
        return select_candidate, extra_init
    else:
        raise ValueError(f"Unknown method '{name}'. Choose from A, B, C, D.")


# ---------------------------------------------------------------------------
# Tier presets
# ---------------------------------------------------------------------------

SMOKE_IDS = [1, 4, 7, 10, 13]

TIER_PRESETS: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "num_steps": 100,
        "batch_size": 128,
        "top_k": 128,
        "robust_topk": 8,
        "n_pert_samples": 2,
        "warm_start_steps": 25,
    },
    "quick": {
        "num_steps": 200,
        "batch_size": 256,
        "top_k": 256,
        "robust_topk": 8,
        "n_pert_samples": 3,
        "warm_start_steps": 30,
    },
}


# ---------------------------------------------------------------------------
# Wilson score confidence interval
# ---------------------------------------------------------------------------

def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p_hat = successes / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(results: List[Dict[str, Any]]):
    method_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        method_groups.setdefault(r["method"], []).append(r)

    header = f"{'Method':<22} {'ASR':>5} {'95% CI':>14} {'n':>3} {'MeanSteps':>10} {'MeanTime':>10}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for method, runs in sorted(method_groups.items()):
        n = len(runs)
        successes = sum(1 for r in runs if r["converged"])
        asr = successes / n if n else 0
        lo, hi = _wilson_ci(successes, n)
        mean_steps = np.mean([r["total_steps"] for r in runs])
        mean_time = np.mean([r["total_wall_time"] for r in runs])
        print(
            f"{method:<22} {asr:>5.1%} [{lo:.2f}, {hi:.2f}]  {n:>3d} "
            f"{mean_steps:>10.1f} {mean_time:>9.1f}s"
        )
    print(sep)

    total_time = sum(r["total_wall_time"] for r in results)
    print(f"Total wall time: {total_time:.0f}s ({total_time / 60:.1f} min)\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fast robust GCG evaluation across methods A-D",
    )
    parser.add_argument(
        "--tier", type=str, required=True, choices=["smoke", "quick"],
        help="smoke (~30 min) or quick (~3 h) on A100 PCIe",
    )
    parser.add_argument(
        "--methods", type=str, default="A,B,C,D",
        help="Comma-separated method list, e.g. 'A,B' (default: A,B,C,D)",
    )
    parser.add_argument("--model_path", type=str, default="/home/LLM/Llama-2-7b-chat-hf")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--template_name", type=str, default="llama-2")
    parser.add_argument(
        "--behaviors_config", type=str, default="data/cyber_behaviors.json",
    )
    parser.add_argument("--seed", type=int, default=235711)
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Root output directory (default: output/fast_eval/<tier>/<timestamp>)",
    )
    parser.add_argument(
        "--smoothllm", action="store_true",
        help="Run SmoothLLM sweep after each behavior (slow; off by default)",
    )
    cli = parser.parse_args()

    methods = [m.strip().upper() for m in cli.methods.split(",")]
    preset = TIER_PRESETS[cli.tier]

    with open(cli.behaviors_config) as f:
        all_behaviors = json.load(f)
    max_id = len(all_behaviors)

    if cli.tier == "smoke":
        behavior_ids = [bid for bid in SMOKE_IDS if bid <= max_id]
    else:
        behavior_ids = list(range(1, max_id + 1))

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = cli.output_dir or f"output/fast_eval/{cli.tier}/{ts}"
    Path(output_root).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  Fast Robust GCG — tier={cli.tier}")
    print(f"  Methods: {methods}")
    print(f"  Behaviors: {behavior_ids} ({len(behavior_ids)} total)")
    print(f"  Steps: {preset['num_steps']}  batch: {preset['batch_size']}  "
          f"topk: {preset['top_k']}")
    print(f"  Robust topk: {preset['robust_topk']}  "
          f"pert samples: {preset['n_pert_samples']}  "
          f"warm start: {preset['warm_start_steps']}")
    print(f"  Output: {output_root}")
    print("=" * 60)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli.device)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nLoading model…")
    t_load = time.time()
    model, tokenizer = load_model_and_tokenizer(
        cli.model_path, low_cpu_mem_usage=True, use_cache=False, device=device,
    )
    print(f"Model loaded in {time.time() - t_load:.1f}s\n")

    all_results: List[Dict[str, Any]] = []
    total_runs = len(behavior_ids) * len(methods)
    run_idx = 0

    for bid in behavior_ids:
        for method_name in methods:
            run_idx += 1
            label = f"[{run_idx}/{total_runs}] method={method_name} id={bid}"
            print(f"\n{'─' * 60}")
            print(f"  {label}")
            print(f"{'─' * 60}")

            select_fn, extra_init_fn = _load_method(method_name)

            use_scaffold = method_name == "D"

            params: Dict[str, Any] = {
                **preset,
                "model_path": cli.model_path,
                "device": cli.device,
                "template_name": cli.template_name,
                "id": bid,
                "behaviors_config": cli.behaviors_config,
                "use_scaffold": use_scaffold,
                "seed": cli.seed,
                "output_path": f"{output_root}/{method_name}/id_{bid}",
            }

            full_method_name = {
                "A": "A_suffix_charperturb",
                "B": "B_token_perturb",
                "C": "C_generation_eval",
                "D": "D_inert_buffer",
            }[method_name]

            try:
                summary = run_attack_with_model(
                    method_name=full_method_name,
                    select_candidate=select_fn,
                    model=model,
                    tokenizer=tokenizer,
                    params=params,
                    extra_init=extra_init_fn,
                    skip_smoothllm=not cli.smoothllm,
                    skip_plots=True,
                )
            except Exception as e:
                print(f"  ERROR on {label}: {e}")
                summary = {
                    "behavior_id": bid,
                    "total_steps": 0,
                    "converged": False,
                    "final_clean_asr": 0.0,
                    "final_suffix": "",
                    "smoothllm_sweep": {},
                    "total_wall_time": 0.0,
                    "error": str(e),
                }

            summary["method"] = full_method_name
            all_results.append(summary)

            _print_summary(all_results)

    report_path = Path(output_root) / "fast_eval_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Full report saved to {report_path}")

    _print_summary(all_results)


if __name__ == "__main__":
    main()
