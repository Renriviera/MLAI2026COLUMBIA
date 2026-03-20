#!/usr/bin/env python3
"""Solution D — Inert buffer zone length regulation.

Robust candidate selection: perturb the **full perturbable prompt** at the
character level, re-tokenise, then adjust a pre-placed inert buffer zone
(hash-comment block from the scaffold) so that the total token count matches
the original.  This keeps all pre-computed slices valid.

Requires ``--use_scaffold`` and a dataset built with scaffold metadata.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from robust_gcg.attack_harness import run_attack
from robust_gcg.perturbation import apply_perturbation


# ---- helpers --------------------------------------------------------------

def _find_subseq(seq: list[int], subseq: list[int]) -> int:
    """Return the start index of *subseq* in *seq*, or -1 if not found."""
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i : i + m] == subseq:
            return i
    return -1


# ---- extra init -----------------------------------------------------------

def extra_parser_setup(parser):
    parser.add_argument(
        "--buffer_token_id",
        type=int,
        default=None,
        help="Token ID to use for buffer padding (default: '#' token).",
    )


def extra_init(args, model, tokenizer, suffix_manager):
    # We need the scaffold buffer boundaries (in char space) and the
    # assistant-role token subsequence to relocate slices after perturbation.
    with open(args.behaviors_config) as f:
        all_beh = json.load(f)
    beh = all_beh[args.id - 1]

    buf_char_start = beh.get("scaffold_buffer_char_start", 0)
    buf_char_end = beh.get("scaffold_buffer_char_end", 0)

    # Tokenise the clean prompt once to get the buffer slice in token space
    clean_ids = suffix_manager.get_input_ids(adv_string=beh["adv_init_suffix"])
    original_len = len(clean_ids)

    # Detect the assistant-role marker tokens (e.g. "[/INST]")
    asst_slice = suffix_manager._assistant_role_slice
    asst_marker = clean_ids[asst_slice.start : asst_slice.stop].tolist()

    # Find the buffer region in token space by tokenising just the scaffolded prompt
    scaffolded = beh.get("behaviour_scaffolded", "")
    scaffold_toks = tokenizer(scaffolded, add_special_tokens=False).input_ids
    # We identify the buffer as the tokens corresponding to the hash-comment lines
    buf_str = scaffolded[buf_char_start:buf_char_end]
    buf_toks = tokenizer(buf_str, add_special_tokens=False).input_ids

    # Determine the buffer token to use for padding
    buffer_token = args.buffer_token_id
    if buffer_token is None:
        hash_toks = tokenizer("#", add_special_tokens=False).input_ids
        buffer_token = hash_toks[0] if hash_toks else scaffold_toks[0]

    return {
        "original_len": original_len,
        "asst_marker": asst_marker,
        "buffer_token": buffer_token,
        "buffer_toks": buf_toks,
        "buf_token_count": len(buf_toks),
    }


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
    original_len=None,
    asst_marker=None,
    buffer_token=None,
    buf_token_count=None,
    **_extra,
):
    if original_len is None:
        # Fallback: use current input_ids length
        original_len = len(input_ids)

    K = min(args.robust_topk, len(new_adv_suffix))
    M = args.n_pert_samples
    topk_indices = clean_losses.topk(K, largest=False).indices

    control_slice = suffix_manager._control_slice
    target_slice = suffix_manager._target_slice
    loss_slice = suffix_manager._loss_slice
    device = model.device
    crit = nn.CrossEntropyLoss()

    user_prompt = suffix_manager.instruction

    best_robust_loss = float("inf")
    best_suffix = new_adv_suffix[topk_indices[0]]

    for idx in topk_indices:
        candidate = new_adv_suffix[idx]
        losses = []

        # Build clean full prompt text for this candidate
        cand_ids = suffix_manager.get_input_ids(adv_string=candidate).to(device)
        full_prompt_text = tokenizer.decode(
            cand_ids[: suffix_manager._assistant_role_slice.stop],
            skip_special_tokens=True,
        )
        perturbable = f"{user_prompt} {candidate}"

        for _ in range(M):
            pert_perturbable = apply_perturbation(perturbable, args.pert_type, args.pert_pct)
            pert_full = full_prompt_text.replace(perturbable, pert_perturbable)

            pert_ids = tokenizer(pert_full, return_tensors="pt").input_ids.squeeze(0)
            delta = len(pert_ids) - original_len

            # Adjust buffer to match original length
            ids_list = pert_ids.tolist()
            if delta > 0 and buf_token_count and buf_token_count > delta:
                # Prompt grew → remove tokens from buffer region
                # Locate the assistant marker to find the buffer (before it)
                marker_pos = _find_subseq(ids_list, asst_marker) if asst_marker else -1
                if marker_pos > 0:
                    # Remove 'delta' tokens just before the marker area
                    remove_start = max(0, marker_pos - delta)
                    ids_list = ids_list[:remove_start] + ids_list[remove_start + delta:]
                else:
                    ids_list = ids_list[:original_len]
            elif delta < 0:
                # Prompt shrank → insert buffer tokens
                n_insert = -delta
                marker_pos = _find_subseq(ids_list, asst_marker) if asst_marker else -1
                insert_pos = marker_pos if marker_pos > 0 else len(ids_list) // 2
                ids_list = (
                    ids_list[:insert_pos]
                    + [buffer_token] * n_insert
                    + ids_list[insert_pos:]
                )
            # Final safety: hard truncate/pad to original_len
            if len(ids_list) > original_len:
                ids_list = ids_list[:original_len]
            elif len(ids_list) < original_len:
                ids_list = ids_list + [buffer_token] * (original_len - len(ids_list))

            adjusted = torch.tensor(ids_list, device=device, dtype=input_ids.dtype)

            logits = model(adjusted.unsqueeze(0)).logits
            loss_val = crit(logits[0, loss_slice, :], adjusted[target_slice]).item()
            losses.append(loss_val)

        mean_loss = sum(losses) / max(len(losses), 1)
        if mean_loss < best_robust_loss:
            best_robust_loss = mean_loss
            best_suffix = candidate

    return best_suffix, best_robust_loss


if __name__ == "__main__":
    run_attack(
        method_name="D_inert_buffer",
        select_candidate=select_candidate,
        extra_parser_setup=extra_parser_setup,
        extra_init=extra_init,
    )
