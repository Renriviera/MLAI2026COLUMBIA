#!/usr/bin/env python3
"""Solution E -- I-GCG K-merge candidate selection.

Instead of picking the single lowest-loss candidate, this method:
  1. Takes the top-K candidates by clean loss.
  2. Iteratively merges their changed token positions into a cumulative
     suffix (each merge step incorporates one more candidate).
  3. Re-evaluates the K merged suffixes and returns the best.

This explores multiple token-position changes per step while remaining
within the coordinate-gradient framework.  It is a clean-loss method
(no perturbation robustness), ported from the standalone I-GCG script
``attack_llm_core_best_update_our_target.py``.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from llm_attacks.minimal_gcg.opt_utils import get_logits, target_loss
from robust_gcg.attack_harness import run_attack


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
    """K-merge selection: merge top-K candidates and pick the best."""
    K = min(getattr(args, "kmerge_k", 7), len(new_adv_suffix))

    _, sorted_indices = torch.sort(clean_losses, descending=False)
    top_k_indices = sorted_indices[:K]

    current_suffix = suffix_manager.adv_string if hasattr(suffix_manager, "adv_string") else ""
    if not current_suffix:
        current_suffix = tokenizer.decode(
            input_ids[suffix_manager._control_slice], skip_special_tokens=True
        )

    base_ids = tokenizer(current_suffix, add_special_tokens=False).input_ids
    merged_ids = copy.copy(base_ids)

    all_merged_suffixes = []
    for k_idx in range(K):
        cand_idx = top_k_indices[k_idx].item()
        cand_str = new_adv_suffix[cand_idx]
        cand_ids = tokenizer(cand_str, add_special_tokens=False).input_ids

        n = min(len(base_ids), len(cand_ids))
        for pos in range(n):
            if base_ids[pos] != cand_ids[pos]:
                merged_ids[pos] = cand_ids[pos]

        all_merged_suffixes.append(
            tokenizer.decode(merged_ids, skip_special_tokens=True)
        )

    logits, ids = get_logits(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        control_slice=suffix_manager._control_slice,
        test_controls=all_merged_suffixes,
        return_ids=True,
        batch_size=512,
    )
    losses = target_loss(logits, ids, suffix_manager._target_slice)

    best_idx = losses.argmin().item()
    best_suffix = all_merged_suffixes[best_idx]
    best_loss = losses[best_idx].item()

    return best_suffix, best_loss


def extra_parser_setup(parser):
    parser.add_argument(
        "--kmerge_k", type=int, default=7,
        help="Number of top candidates to merge (K in I-GCG).",
    )


if __name__ == "__main__":
    run_attack(
        method_name="E_kmerge",
        select_candidate=select_candidate,
        extra_parser_setup=extra_parser_setup,
    )
