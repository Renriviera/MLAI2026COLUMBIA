#!/usr/bin/env python3
"""Solution A — Suffix-only character perturbation with pad/truncate.

Robust candidate selection: for each top-K candidate (ranked by clean loss),
apply M random character perturbations to the *suffix string only*,
re-tokenise, pad/truncate to the original control_slice length, compute
target_loss, and average.  Select the candidate with lowest mean robust loss.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from robust_gcg.perturbation import apply_perturbation
from robust_gcg.attack_harness import run_attack
from llm_attacks.minimal_gcg.opt_utils import target_loss as compute_target_loss


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

    control_slice = suffix_manager._control_slice
    target_slice = suffix_manager._target_slice
    loss_slice = suffix_manager._loss_slice
    control_len = control_slice.stop - control_slice.start
    device = model.device
    crit = nn.CrossEntropyLoss()

    best_robust_loss = float("inf")
    best_suffix = new_adv_suffix[topk_indices[0]]

    for idx in topk_indices:
        candidate = new_adv_suffix[idx]
        losses = []

        for _ in range(M):
            perturbed = apply_perturbation(candidate, args.pert_type, args.pert_pct)
            new_toks = tokenizer(perturbed, add_special_tokens=False).input_ids

            # Pad / truncate to control_len
            if len(new_toks) > control_len:
                new_toks = new_toks[:control_len]
            elif len(new_toks) < control_len:
                orig_toks = input_ids[control_slice].tolist()
                new_toks = new_toks + orig_toks[len(new_toks):]

            pert_ids = input_ids.clone()
            pert_ids[control_slice] = torch.tensor(
                new_toks, device=device, dtype=input_ids.dtype
            )

            logits = model(pert_ids.unsqueeze(0)).logits
            loss_val = crit(logits[0, loss_slice, :], pert_ids[target_slice]).item()
            losses.append(loss_val)

        mean_loss = sum(losses) / len(losses)
        if mean_loss < best_robust_loss:
            best_robust_loss = mean_loss
            best_suffix = candidate

    return best_suffix, best_robust_loss


if __name__ == "__main__":
    run_attack(
        method_name="A_suffix_charperturb",
        select_candidate=select_candidate,
    )
