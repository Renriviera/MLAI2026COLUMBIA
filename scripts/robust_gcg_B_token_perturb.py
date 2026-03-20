#!/usr/bin/env python3
"""Solution B — Token-level perturbation with precomputed neighbourhoods.

Robust candidate selection: perturb directly in token space (no re-tokenisation).
For each top-K candidate, randomly replace pct% of suffix token positions with
a draw from that token's *perturbation neighbourhood* N(t).  Fully batchable.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from robust_gcg.attack_harness import run_attack
from robust_gcg.token_robustness import (
    compute_token_neighborhoods,
    compute_token_robustness,
    get_robust_token_mask,
)
from llm_attacks.minimal_gcg.opt_utils import target_loss as compute_target_loss


# ---- extra init: load / compute neighbourhoods ----------------------------

def extra_init(args, model, tokenizer, suffix_manager):
    if args.token_neighborhoods_path and Path(args.token_neighborhoods_path).exists():
        neighborhoods = torch.load(args.token_neighborhoods_path, weights_only=False)
    else:
        print("  [B] Computing token neighbourhoods (one-time)…")
        neighborhoods = compute_token_neighborhoods(
            tokenizer,
            pert_type=args.pert_type,
            pert_pct=args.pert_pct,
            n_samples=50,
        )

    # Optional: filter fragile tokens
    not_allowed_extra = None
    if args.token_robustness_path and Path(args.token_robustness_path).exists():
        robustness = torch.load(args.token_robustness_path, weights_only=False)
        not_allowed_extra = get_robust_token_mask(robustness, args.robustness_threshold)

    return {"neighborhoods": neighborhoods, "not_allowed_extra": not_allowed_extra}


# ---- candidate selector ---------------------------------------------------

def select_candidate(
    new_adv_suffix,
    clean_losses,
    input_ids,
    suffix_manager,
    model,
    tokenizer,
    step,
    args,
    neighborhoods=None,
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
    pert_frac = args.pert_pct / 100.0

    # Build all perturbed input_ids for a single batched forward pass
    all_pert_ids = []
    candidate_indices = []  # which candidate each row belongs to

    for rank, idx in enumerate(topk_indices):
        candidate = new_adv_suffix[idx]
        suffix_toks = tokenizer(candidate, add_special_tokens=False).input_ids[:control_len]
        if len(suffix_toks) < control_len:
            orig = input_ids[control_slice].tolist()
            suffix_toks = suffix_toks + orig[len(suffix_toks):]

        for _ in range(M):
            pert_toks = list(suffix_toks)
            n_perturb = max(1, int(control_len * pert_frac))
            positions = random.sample(range(control_len), min(n_perturb, control_len))
            for pos in positions:
                tid = pert_toks[pos]
                neighbors = neighborhoods.get(tid, [tid]) if neighborhoods else [tid]
                pert_toks[pos] = random.choice(neighbors) if neighbors else tid

            row = input_ids.clone()
            row[control_slice] = torch.tensor(pert_toks, device=device, dtype=input_ids.dtype)
            all_pert_ids.append(row)
            candidate_indices.append(rank)

    # Batched forward pass
    batch = torch.stack(all_pert_ids)
    crit = nn.CrossEntropyLoss(reduction="none")
    logits = model(batch).logits
    # Compute per-row loss
    loss_s = slice(loss_slice.start, loss_slice.stop)
    tgt_s = slice(target_slice.start, target_slice.stop)
    per_row_loss = crit(
        logits[:, loss_s, :].transpose(1, 2), batch[:, tgt_s]
    ).mean(dim=-1)

    # Aggregate per candidate
    best_robust_loss = float("inf")
    best_suffix = new_adv_suffix[topk_indices[0]]
    for rank, idx in enumerate(topk_indices):
        mask = [i for i, c in enumerate(candidate_indices) if c == rank]
        mean_loss = per_row_loss[mask].mean().item()
        if mean_loss < best_robust_loss:
            best_robust_loss = mean_loss
            best_suffix = new_adv_suffix[idx]

    return best_suffix, best_robust_loss


if __name__ == "__main__":
    run_attack(
        method_name="B_token_perturb",
        select_candidate=select_candidate,
        extra_init=extra_init,
    )
