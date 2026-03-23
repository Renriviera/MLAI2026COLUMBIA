# llm_attacks

## Active code (`minimal_gcg/`)

The modules used by all current experiments:

- **`opt_utils.py`** -- GCG math: `token_gradients`, `sample_control`, `get_filtered_cands`, `get_logits`, `target_loss`, `load_model_and_tokenizer`
- **`string_utils.py`** -- `SuffixManager` for chat-template-aware prompt construction and slice computation
- **`slot_utils.py`** -- SlotGCG utilities: attention-based slot allocation (VSS), embedding interleaving

## Upstream reference (`base/`, `gcg/`)

The original LLM-Attacks / I-GCG multi-worker framework (`AttackPrompt`, `PromptManager`, `MultiPromptAttack`, `ModelWorker`). Not used by active experiments but retained for reference. See docstrings at the top of each module.
