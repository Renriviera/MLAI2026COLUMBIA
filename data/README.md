# Behavior Datasets

All datasets are JSON arrays where each element defines one adversarial behavior
for GCG optimization. Scripts index behaviors by `args.id - 1` (1-based CLI,
0-based array).

## JSON Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int | yes | 1-based behavior identifier |
| `behaviour_raw` | string | no | Original AdvBench/HarmBench phrasing |
| `behaviour` | string | yes | Prompt sent to the model (may be code-framed) |
| `behaviour_scaffolded` | string | no | `behaviour` wrapped in a code-scaffold with hash-comment buffer (used by methods D/F and `--use_scaffold`) |
| `target` | string | yes | Expected assistant prefix that GCG optimizes toward |
| `source` | string | no | Origin tag (`"local"`, `"advbench"`, etc.) |
| `adv_init_suffix` | string | yes | Initial adversarial suffix tokens (e.g. `"! ! ! ..."` or a pre-optimized string) |
| `adv_init_suffix_baseline` | string | no | Naive `"! ! ! ..."` baseline when `adv_init_suffix` is pre-optimized |
| `scaffold_buffer_char_start` | int | no | Start character index of the hash-comment buffer in `behaviour_scaffolded` |
| `scaffold_buffer_char_end` | int | no | End character index of the hash-comment buffer |
| `step` | int | yes | Number of GCG optimization steps (overridable from CLI) |
| `batch_size` | int | yes | Candidate batch size per step |
| `top_k` | int | yes | Top-K candidates for gradient-based sampling |

## Dataset Variants

### `cyber_behaviors.json` (v1)

- **15 behaviors**, cyber-security domain
- **Target model:** Qwen-2-7B-Instruct
- **Target style:** Compliant prefix -- `"Sure, my output is harmful. Here is ..."`
- **Suffix init:** Naive `"! ! ! ..."` (20 tokens)
- **Steps:** 500, batch 256, top_k 256
- **Scaffold:** Yes (`behaviour_scaffolded` + buffer char ranges)

### `cyber_behaviors_v2.json` (v2)

- **5 behaviors**, expanded cyber domain
- **Target model:** Qwen-2-7B-Instruct
- **Target style:** Code-continuation -- `"Here is a Python script...\n\n```python\nimport..."`
- **Suffix init:** Naive `"! ! ! ..."` (40 tokens, longer than v1)
- **Steps:** 1000, batch 256, top_k 256
- **Scaffold:** Yes

### `cyber_behaviors_v2_all15.json`

- **15 behaviors**, union of v1 and v2 sets
- **Target model:** Qwen-2-7B-Instruct
- **Target style:** v2 code-continuation targets
- **Suffix init:** Pre-optimized from B1 transfer experiment (long gibberish tokens); also carries `adv_init_suffix_baseline` for cold-start comparison
- **Steps:** 300, batch 128, top_k 128 (fewer steps because suffix is warm-started)
- **Scaffold:** Yes

### `cyber_behaviors_v2_all30.json`

- **30 behaviors**, BIDs 1-15 from `v2_all15` + 15 new from AdvBench `harmful_behaviors.csv`
- **Target model:** Qwen-2-7B-Instruct
- **Target style:** v2 code-continuation targets
- **Suffix init:** BIDs 1-15 carry pre-optimized suffixes from B1 transfer; BIDs 16-30 use naive `"! ! ! ..."` (40 tokens)
- **Steps:** 500, batch 256, top_k 256 (overridden from all15 defaults)
- **Scaffold:** Yes
- **New domains (BIDs 16-30):** software exploitation, virus distribution, infrastructure attack, webcam hijacking, DDoS, hospital hacking, bank hacking, financial fraud, self-driving car hijacking, drone control, email hacking, AV evasion, voting manipulation, social engineering, spyware

### `cyber_behaviors_v2_vicuna.json`

- **3 behaviors**, subset for Vicuna baseline comparison
- **Target model:** Vicuna-7B-v1.5
- **Target style:** v2 code-continuation targets
- **Suffix init:** Naive `"! ! ! ..."` (40 tokens)
- **Steps:** 200, batch 256, top_k 256
- **Scaffold:** No (`behaviour_scaffolded` absent)

## Building New Datasets

```bash
python data/build_cyber_dataset.py
```

This script:
1. Merges behavior sources (AdvBench, HarmBench, local)
2. Filters for cyber-security keywords
3. Optionally frames behaviors as code tasks ("Write a Python script that...")
4. Generates `target` strings in the chosen style (v1 compliant prefix or v2 code-continuation)
5. Wraps each behavior in the code scaffold via `robust_gcg.scaffold.build_scaffold`
6. Writes the result to `data/cyber_behaviors.json`

To create a custom dataset, edit the behavior list in `build_cyber_dataset.py` or
provide a CSV with `goal` and `target` columns matching the AdvBench format.
