#!/usr/bin/env python3
"""Solution C — Full generation + refusal check (binary signal).

Robust candidate selection: for each top-K candidate (ranked by clean loss),
apply M character perturbations to the **full perturbable prompt**, re-tokenise,
generate, and check for refusal.  Select the candidate with the highest
empirical survival rate (tie-break: clean loss).

This is the most faithful SmoothLLM simulation but also the most expensive.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from robust_gcg.attack_harness import run_attack
from robust_gcg.perturbation import apply_perturbation
from robust_gcg.eval_utils import TEST_PREFIXES, _is_jailbroken


def _generate_short(model, tokenizer, prompt_ids, max_new_tokens=16):
    """Greedy decode a short continuation (enough to check refusal)."""
    ids = prompt_ids.unsqueeze(0).to(model.device)
    attn = torch.ones_like(ids)
    gen_config = copy.deepcopy(model.generation_config)
    gen_config.max_new_tokens = max_new_tokens
    out = model.generate(
        ids, attention_mask=attn, generation_config=gen_config,
        pad_token_id=tokenizer.pad_token_id,
    )[0]
    return tokenizer.decode(out[prompt_ids.shape[0]:], skip_special_tokens=True).strip()


def select_candidate(
    new_adv_suffix,
    clean_losses,
    input_ids,
    suffix_manager,
    model,
    tokenizer,
    step,
    args,
    **_extra,
):
    K = min(args.robust_topk, len(new_adv_suffix))
    M = args.n_pert_samples
    topk_indices = clean_losses.topk(K, largest=False).indices

    device = model.device
    user_prompt = suffix_manager.instruction

    best_score = -1.0
    best_loss = float("inf")
    best_suffix = new_adv_suffix[topk_indices[0]]

    for idx in topk_indices:
        candidate = new_adv_suffix[idx]
        survived = 0

        # Build the full prompt string once
        full_ids = suffix_manager.get_input_ids(adv_string=candidate).to(device)
        full_prompt_text = tokenizer.decode(
            full_ids[: suffix_manager._assistant_role_slice.stop],
            skip_special_tokens=True,
        )
        perturbable = f"{user_prompt} {candidate}"

        for _ in range(M):
            pert_perturbable = apply_perturbation(perturbable, args.pert_type, args.pert_pct)
            pert_full = full_prompt_text.replace(perturbable, pert_perturbable)
            pert_ids = tokenizer(pert_full, return_tensors="pt").input_ids.squeeze(0).to(device)

            gen_str = _generate_short(model, tokenizer, pert_ids, max_new_tokens=16)
            if _is_jailbroken(gen_str):
                survived += 1

        score = survived / M
        c_loss = clean_losses[idx].item()
        if score > best_score or (score == best_score and c_loss < best_loss):
            best_score = score
            best_loss = c_loss
            best_suffix = candidate

    # robust_loss = 1 - survival_rate (lower is better, for consistency)
    return best_suffix, 1.0 - best_score


if __name__ == "__main__":
    run_attack(
        method_name="C_generation_eval",
        select_candidate=select_candidate,
    )
