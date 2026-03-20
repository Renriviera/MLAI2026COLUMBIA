# Lightweight top-level imports (no torch dependency)
from .perturbation import (
    random_swap_perturbation,
    random_patch_perturbation,
    random_insert_perturbation,
    apply_perturbation,
)
from .scaffold import build_scaffold, SCAFFOLD_TYPES


def __getattr__(name):
    """Lazy-load heavy (torch-dependent) submodules on first access."""
    if name in (
        "compute_token_robustness",
        "compute_token_neighborhoods",
        "get_robust_token_mask",
    ):
        from . import token_robustness
        return getattr(token_robustness, name)
    if name in ("RobustEvaluator", "ExperimentLogger", "plot_run_results"):
        from . import eval_utils
        return getattr(eval_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
