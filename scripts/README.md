# Scripts

## Method Scripts (single behavior, single method)

Run one robust GCG method on one behavior. Used as building blocks by the
experiment orchestrators and directly via Makefile targets.

| Script | Method | Makefile target |
|--------|--------|-----------------|
| `robust_gcg_A_suffix_charperturb.py` | A: suffix character perturbation | `robust-a` |
| `robust_gcg_B_token_perturb.py` | B: token neighborhood perturbation | `robust-b` |
| `robust_gcg_C_generation_eval.py` | C: full-prompt perturbation + generation eval | `robust-c` |
| `robust_gcg_D_inert_buffer.py` | D: inert buffer / scaffold perturbation | `robust-d` |
| `robust_gcg_E_kmerge.py` | E: I-GCG k-merge candidate selection | (none) |
| `robust_gcg_F_slot_kmerge.py` | F: SlotGCG interleaved tokens + k-merge | `robust-f` |

All method scripts share CLI arguments via `robust_gcg.attack_harness.build_parser()`:
`--model_path`, `--device`, `--id`, `--behaviors_config`, `--num_steps`,
`--batch_size`, `--top_k`, `--seed`, `--output_path`, etc.

Each script defines a `select_candidate()` function and an optional `extra_init()`
hook, then calls `run_attack()` from the shared harness.

## Experiment Scripts (multi-behavior orchestrators)

Load the model once and run multiple methods / behaviors / seeds, producing
structured JSON reports.

| Script | Purpose | Makefile target | Est. runtime |
|--------|---------|-----------------|--------------|
| `fast_robust_eval.py` | Tiered sweep of methods A-D/F | `smoke-test`, `quick-eval` | 30m to 3h |
| `improved_gcg_experiment.py` | Baseline vs multiflip vs D vs E | `experiment-improved` | 10h |
| `thorough_method_D_eval.py` | Deep method D eval, multi-seed | `thorough-D` | 8.5h |
| `transfer_experiment.py` | B1 suffix transfer experiment | `transfer-experiment` | 7.5h |
| `slotgcg_experiment.py` | Method F + verification + SmoothLLM | `slotgcg-experiment` | 5h |
| `target_ablation_experiment.py` | Verification-gap ablation F-A to F-D | `target-ablation` | 4h |
| `fc_scaled_experiment.py` | F-C (ARCA target update) on 30 behaviors | `fc-scaled` | 10h |

All experiment scripts support `--dry_run` (5 steps per condition) for validation.

## Analysis

| Script | Purpose | Makefile target |
|--------|---------|-----------------|
| `analyze_results.py` | Aggregate runs into comparison CSV and plots | `analyze` |
