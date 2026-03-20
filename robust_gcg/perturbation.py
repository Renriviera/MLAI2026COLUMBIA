"""Re-implementation of SmoothLLM perturbation primitives.

These match the semantics of smooth-llm/lib/perturbations.py exactly so that
each attack script can apply perturbations without importing the full SmoothLLM
defense object.
"""

import random
import string

_ALPHABET = string.printable


def random_swap_perturbation(text: str, pct: float) -> str:
    """Swap *pct*% of characters with random printable ASCII."""
    chars = list(text)
    n_swap = int(len(chars) * pct / 100)
    indices = random.sample(range(len(chars)), min(n_swap, len(chars)))
    for i in indices:
        chars[i] = random.choice(_ALPHABET)
    return "".join(chars)


def random_patch_perturbation(text: str, pct: float) -> str:
    """Replace a contiguous block of *pct*% characters."""
    chars = list(text)
    width = int(len(chars) * pct / 100)
    if width == 0 or len(chars) == 0:
        return text
    max_start = len(chars) - width
    start = random.randint(0, max(max_start, 0))
    patch = [random.choice(_ALPHABET) for _ in range(width)]
    chars[start : start + width] = patch
    return "".join(chars)


def random_insert_perturbation(text: str, pct: float) -> str:
    """Insert random characters at *pct*% of positions."""
    chars = list(text)
    n_insert = int(len(chars) * pct / 100)
    indices = random.sample(range(len(chars)), min(n_insert, len(chars)))
    for i in sorted(indices, reverse=True):
        chars.insert(i, random.choice(_ALPHABET))
    return "".join(chars)


_DISPATCH = {
    "RandomSwapPerturbation": random_swap_perturbation,
    "RandomPatchPerturbation": random_patch_perturbation,
    "RandomInsertPerturbation": random_insert_perturbation,
    "swap": random_swap_perturbation,
    "patch": random_patch_perturbation,
    "insert": random_insert_perturbation,
}


def apply_perturbation(text: str, pert_type: str, pct: float) -> str:
    """Dispatch to the correct perturbation function by name."""
    fn = _DISPATCH.get(pert_type)
    if fn is None:
        raise ValueError(
            f"Unknown perturbation type {pert_type!r}. "
            f"Choose from {list(_DISPATCH.keys())}"
        )
    return fn(text, pct)
