#!/usr/bin/env python3
"""Solution D — Inert buffer zone length regulation.

Robust candidate selection: perturb the **full perturbable prompt** at the
character level, but keep the scaffold's inert buffer zone untouched.
Re-tokenise the perturbed regions, then adjust the buffer token count so
the total sequence length matches the original.  Special tokens, assistant-
role markers, and target tokens are preserved exactly in token space.

Requires ``--use_scaffold`` and a dataset built with scaffold metadata.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from robust_gcg.attack_harness import run_attack
from robust_gcg.perturbation import apply_perturbation


# ---- helpers --------------------------------------------------------------

def _pad_or_truncate(toks: list[int], target_len: int, pad_id: int) -> list[int]:
    """Hard pad / truncate *toks* to exactly *target_len*."""
    if len(toks) > target_len:
        return toks[:target_len]
    if len(toks) < target_len:
        return toks + [pad_id] * (target_len - len(toks))
    return toks


# ---- extra init -----------------------------------------------------------

def extra_parser_setup(parser):
    parser.add_argument(
        "--buffer_token_id",
        type=int,
        default=None,
        help="Token ID to use for buffer padding (default: '#' token).",
    )


def extra_init(args, model, tokenizer, suffix_manager):
    with open(args.behaviors_config) as f:
        all_beh = json.load(f)
    beh = all_beh[args.id - 1]

    buf_char_start = beh.get("scaffold_buffer_char_start", 0)
    buf_char_end = beh.get("scaffold_buffer_char_end", 0)

    clean_ids = suffix_manager.get_input_ids(adv_string=beh["adv_init_suffix"])
    original_len = len(clean_ids)

    buffer_token = args.buffer_token_id
    if buffer_token is None:
        hash_toks = tokenizer("#", add_special_tokens=False).input_ids
        buffer_token = hash_toks[0] if hash_toks else tokenizer.eos_token_id

    return {
        "original_len": original_len,
        "buf_char_start": buf_char_start,
        "buf_char_end": buf_char_end,
        "buffer_token": buffer_token,
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
    buf_char_start=0,
    buf_char_end=0,
    buffer_token=None,
    **_extra,
):
    if original_len is None:
        original_len = len(input_ids)

    if original_len != len(input_ids):
        if not getattr(select_candidate, "_len_warned", False):
            warnings.warn(
                f"[D] original_len ({original_len}) != len(input_ids) "
                f"({len(input_ids)}); using len(input_ids). "
                f"(This warning is shown once per run.)",
                stacklevel=2,
            )
            select_candidate._len_warned = True
        original_len = len(input_ids)

    if buffer_token is None:
        buffer_token = tokenizer.eos_token_id

    K = min(args.robust_topk, len(new_adv_suffix))
    M = args.n_pert_samples
    topk_indices = clean_losses.topk(K, largest=False).indices

    goal_slice = suffix_manager._goal_slice
    control_slice = suffix_manager._control_slice
    asst_slice = suffix_manager._assistant_role_slice
    target_slice = suffix_manager._target_slice
    loss_slice = suffix_manager._loss_slice
    device = model.device
    crit = nn.CrossEntropyLoss()

    user_prompt = suffix_manager.instruction

    # Immutable token regions (preserved exactly for every perturbation)
    prefix_toks = input_ids[: goal_slice.start].tolist()
    asst_toks = input_ids[asst_slice].tolist()
    tgt_toks = input_ids[target_slice].tolist()
    orig_content_len = control_slice.stop - goal_slice.start

    best_robust_loss = float("inf")
    best_suffix = new_adv_suffix[topk_indices[0]]

    for idx in topk_indices:
        candidate = new_adv_suffix[idx]
        losses = []

        perturbable = f"{user_prompt} {candidate}"
        pre_buf_text = perturbable[:buf_char_start]
        post_buf_text = perturbable[buf_char_end:]

        for _ in range(M):
            pert_pre = apply_perturbation(pre_buf_text, args.pert_type, args.pert_pct)
            pert_post = apply_perturbation(post_buf_text, args.pert_type, args.pert_pct)

            pre_toks = tokenizer(pert_pre, add_special_tokens=False).input_ids
            post_toks = tokenizer(pert_post, add_special_tokens=False).input_ids

            needed_buf = orig_content_len - len(pre_toks) - len(post_toks)

            if needed_buf > 0:
                content_toks = pre_toks + [buffer_token] * needed_buf + post_toks
            else:
                trim = min(-needed_buf, len(pre_toks))
                content_toks = pre_toks[: len(pre_toks) - trim] + post_toks

            content_toks = _pad_or_truncate(content_toks, orig_content_len, buffer_token)

            pert_ids_list = prefix_toks + content_toks + asst_toks + tgt_toks
            pert_ids_list = _pad_or_truncate(pert_ids_list, original_len, buffer_token)

            adjusted = torch.tensor(pert_ids_list, device=device, dtype=input_ids.dtype)

            try:
                logits = model(adjusted.unsqueeze(0)).logits
                loss_val = crit(
                    logits[0, loss_slice, :], adjusted[target_slice]
                ).item()
            except RuntimeError as exc:
                warnings.warn(
                    f"[D] Loss computation failed at step {step}, "
                    f"candidate {idx}: {exc}",
                    stacklevel=2,
                )
                continue

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
