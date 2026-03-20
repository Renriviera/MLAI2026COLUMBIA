"""Precompute per-token robustness scores and perturbation neighbourhoods.

For each token *t* in the vocabulary we estimate:

* **Robustness** R(t): the probability that decoding *t* to a string,
  applying a random character perturbation, and re-tokenising yields exactly
  *t* again.
* **Neighbourhood** N(t): the multiset of alternative tokens that the
  perturb → re-tokenise pipeline can produce.

These are cached to disk so they only need to be computed once per
(model, pert_type, pert_pct) combination.
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from .perturbation import apply_perturbation


def _model_tag(tokenizer) -> str:
    name = getattr(tokenizer, "name_or_path", "unknown")
    return hashlib.md5(name.encode()).hexdigest()[:8]


def compute_token_robustness(
    tokenizer,
    pert_type: str = "RandomSwapPerturbation",
    pert_pct: float = 10,
    n_samples: int = 100,
    cache_dir: str = "output/robust_eval/cache",
) -> Dict[int, float]:
    """Return ``{token_id: robustness_score}`` for every token.

    *robustness_score* ∈ [0, 1] is the fraction of perturbation samples
    where the token survives re-tokenisation intact.
    """
    tag = _model_tag(tokenizer)
    cache_path = Path(cache_dir) / f"token_robustness_{tag}_{pert_type}_{int(pert_pct)}.pt"
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    vocab_size = tokenizer.vocab_size
    robustness: Dict[int, float] = {}

    for tid in tqdm(range(vocab_size), desc="token robustness"):
        decoded = tokenizer.decode([tid])
        if len(decoded) == 0:
            robustness[tid] = 1.0
            continue
        survived = 0
        for _ in range(n_samples):
            perturbed = apply_perturbation(decoded, pert_type, pert_pct)
            re_ids = tokenizer(perturbed, add_special_tokens=False).input_ids
            if len(re_ids) == 1 and re_ids[0] == tid:
                survived += 1
        robustness[tid] = survived / n_samples

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(robustness, cache_path)
    return robustness


def compute_token_neighborhoods(
    tokenizer,
    pert_type: str = "RandomSwapPerturbation",
    pert_pct: float = 10,
    n_samples: int = 100,
    cache_dir: str = "output/robust_eval/cache",
) -> Dict[int, List[int]]:
    """Return ``{token_id: [list of neighbour token_ids]}`` for the vocabulary.

    Each neighbour list contains the *unique* token IDs that appeared as the
    first token after perturb → re-tokenise (excluding the original token).
    """
    tag = _model_tag(tokenizer)
    cache_path = (
        Path(cache_dir) / f"token_neighborhoods_{tag}_{pert_type}_{int(pert_pct)}.pt"
    )
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    vocab_size = tokenizer.vocab_size
    neighborhoods: Dict[int, List[int]] = {}

    for tid in tqdm(range(vocab_size), desc="token neighborhoods"):
        decoded = tokenizer.decode([tid])
        if len(decoded) == 0:
            neighborhoods[tid] = []
            continue
        counter: Counter[int] = Counter()
        for _ in range(n_samples):
            perturbed = apply_perturbation(decoded, pert_type, pert_pct)
            re_ids = tokenizer(perturbed, add_special_tokens=False).input_ids
            if re_ids:
                counter[re_ids[0]] += 1
        # Keep neighbours that appeared at least once, excluding self
        neighbors = [t for t in counter if t != tid]
        # Also include self so sampling can "keep" the token
        neighborhoods[tid] = [tid] + neighbors

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(neighborhoods, cache_path)
    return neighborhoods


def get_robust_token_mask(
    robustness: Dict[int, float],
    threshold: float = 0.3,
) -> torch.Tensor:
    """Return a 1-D tensor of token IDs whose robustness is *below* threshold.

    These fragile tokens can be added to ``not_allowed_tokens`` to prevent
    GCG from choosing them in the suffix.
    """
    fragile = [tid for tid, score in robustness.items() if score < threshold]
    return torch.tensor(fragile, dtype=torch.long)
