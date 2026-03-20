#!/usr/bin/env python3
"""Offline post-hoc aggregation and comparison plots.

Run after any combination of the 4 attack scripts has finished:

    python scripts/analyze_results.py --scan_all
    python scripts/analyze_results.py --results_dirs dir1 dir2 ...

Produces comparison plots + CSV table in ``output/robust_eval/comparison/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

_REPO = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = _REPO / "output" / "robust_eval"
COMPARISON_DIR = OUTPUT_ROOT / "comparison"

PERT_TYPES = ["RandomSwapPerturbation", "RandomPatchPerturbation", "RandomInsertPerturbation"]
PERT_TYPE_SHORT = {"RandomSwapPerturbation": "Swap", "RandomPatchPerturbation": "Patch", "RandomInsertPerturbation": "Insert"}
PERT_PCTS = [10, 15, 20]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_run_dirs() -> list[Path]:
    """Scan ``output/robust_eval/`` for any directories containing summaries."""
    dirs = []
    for method_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not method_dir.is_dir() or method_dir.name in ("cache", "comparison"):
            continue
        for ts_dir in sorted(method_dir.iterdir()):
            if ts_dir.is_dir() and list(ts_dir.glob("summary_*.json")):
                dirs.append(ts_dir)
    return dirs


def load_summaries(run_dir: Path) -> list[dict]:
    summaries = []
    for sf in sorted(run_dir.glob("summary_*.json")):
        with open(sf) as f:
            s = json.load(f)
        s["_run_dir"] = str(run_dir)
        summaries.append(s)
    return summaries


def load_logs(run_dir: Path) -> dict[int, list[dict]]:
    logs = {}
    for lf in sorted(run_dir.glob("log_*.json")):
        bid = int(lf.stem.replace("log_", ""))
        with open(lf) as f:
            logs[bid] = json.load(f)
    return logs


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def build_table(all_summaries: list[dict]) -> list[dict]:
    """Build a flat comparison table with one row per (method, behaviour)."""
    rows = []
    for s in all_summaries:
        method = s.get("method", "?")
        bid = s.get("behavior_id", "?")
        sweep = s.get("smoothllm_sweep", {})

        row = {
            "method": method,
            "behavior_id": bid,
            "clean_asr": s.get("final_clean_asr", 0),
            "steps": s.get("total_steps", 0),
            "wall_time": s.get("total_wall_time", 0),
        }
        for pt in PERT_TYPES:
            for pp in PERT_PCTS:
                key = f"smooth_{pt}_{pp}"
                entry = sweep.get(key, {})
                row[f"{PERT_TYPE_SHORT[pt].lower()}_{pp}"] = 1.0 if entry.get("jailbroken") else 0.0
        rows.append(row)
    return rows


def aggregate_by_method(rows: list[dict]) -> list[dict]:
    """Average metrics across behaviours for each method."""
    method_buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        method_buckets[r["method"]].append(r)

    agg = []
    for method, rs in sorted(method_buckets.items()):
        n = len(rs)
        entry = {"method": method, "n_behaviors": n}
        for col in ["clean_asr", "steps", "wall_time"] + [
            f"{PERT_TYPE_SHORT[pt].lower()}_{pp}" for pt in PERT_TYPES for pp in PERT_PCTS
        ]:
            vals = [r.get(col, 0) for r in rs]
            entry[col] = float(np.mean(vals))
        if entry["steps"] > 0:
            entry["sec_per_step"] = entry["wall_time"] / entry["steps"]
        else:
            entry["sec_per_step"] = 0.0
        agg.append(entry)
    return agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_asr_by_method(agg: list[dict], out: Path):
    methods = [a["method"] for a in agg]
    clean = [a["clean_asr"] for a in agg]
    swap10 = [a.get("swap_10", 0) for a in agg]
    swap15 = [a.get("swap_15", 0) for a in agg]

    x = np.arange(len(methods))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, clean, w, label="Clean ASR")
    ax.bar(x, swap10, w, label="Swap 10%")
    ax.bar(x + w, swap15, w, label="Swap 15%")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("ASR")
    ax.set_title("ASR by Method")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "asr_by_method.png", dpi=150)
    plt.close(fig)


def plot_asr_vs_pertpct(agg: list[dict], out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, pt in zip(axes, PERT_TYPES):
        short = PERT_TYPE_SHORT[pt].lower()
        for a in agg:
            vals = [a.get(f"{short}_{pp}", 0) for pp in PERT_PCTS]
            ax.plot(PERT_PCTS, vals, "o-", label=a["method"])
        ax.set_xlabel("Perturbation %")
        ax.set_ylabel("ASR")
        ax.set_title(PERT_TYPE_SHORT[pt])
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
    fig.suptitle("ASR vs Perturbation Percentage")
    fig.tight_layout()
    fig.savefig(out / "asr_vs_pertpct.png", dpi=150)
    plt.close(fig)


def plot_cost_vs_asr(agg: list[dict], out: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for a in agg:
        ax.scatter(a.get("sec_per_step", 0), a.get("swap_10", 0), s=100)
        ax.annotate(a["method"], (a.get("sec_per_step", 0), a.get("swap_10", 0)),
                     fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("Seconds per GCG step")
    ax.set_ylabel("ASR (Swap 10%)")
    ax.set_title("Efficiency Frontier: Cost vs Robust ASR")
    fig.tight_layout()
    fig.savefig(out / "cost_vs_asr.png", dpi=150)
    plt.close(fig)


def plot_loss_curves_overlay(all_logs: dict[str, dict[int, list[dict]]], out: Path):
    """Overlay loss curves from all methods for each shared behaviour."""
    # Collect all behaviour IDs that appear in at least two methods
    bid_sets = [set(logs.keys()) for logs in all_logs.values()]
    if not bid_sets:
        return
    shared_bids = sorted(set.intersection(*bid_sets)) if len(bid_sets) > 1 else sorted(bid_sets[0])
    if not shared_bids:
        shared_bids = sorted(bid_sets[0])

    for bid in shared_bids[:5]:  # limit to 5 behaviours
        fig, ax = plt.subplots(figsize=(10, 5))
        for method, logs in all_logs.items():
            if bid not in logs:
                continue
            entries = logs[bid]
            steps = [e["step"] for e in entries]
            clean = [float(e.get("clean_loss", 0)) for e in entries]
            ax.plot(steps, clean, label=method, alpha=0.7)
        ax.set_xlabel("Step")
        ax.set_ylabel("Clean Loss")
        ax.set_title(f"Loss Curves — Behaviour #{bid}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"loss_curves_overlay_bid{bid}.png", dpi=150)
        plt.close(fig)


def plot_per_behavior_heatmap(rows: list[dict], out: Path):
    methods = sorted(set(r["method"] for r in rows))
    bids = sorted(set(r["behavior_id"] for r in rows))
    if not methods or not bids:
        return

    data = np.full((len(bids), len(methods)), np.nan)
    for r in rows:
        bi = bids.index(r["behavior_id"])
        mi = methods.index(r["method"])
        data[bi, mi] = r.get("swap_10", 0)

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.5), max(4, len(bids) * 0.4)))
    cmap = LinearSegmentedColormap.from_list("rg", ["#2ecc71", "#e74c3c"])
    im = ax.imshow(data, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(bids)))
    ax.set_yticklabels([f"#{b}" for b in bids], fontsize=8)
    ax.set_xlabel("Method")
    ax.set_ylabel("Behaviour")
    ax.set_title("SmoothLLM ASR (Swap 10%) — per behaviour × method")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="ASR")
    fig.tight_layout()
    fig.savefig(out / "per_behavior_heatmap.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(agg: list[dict], out: Path):
    if not agg:
        return
    path = out / "comparison_table.csv"
    cols = list(agg[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in agg:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse robust GCG results")
    parser.add_argument("--results_dirs", nargs="*", default=None, help="Explicit run directories")
    parser.add_argument("--scan_all", action="store_true", help="Scan output/robust_eval/ for all runs")
    parser.add_argument("--output", default=str(COMPARISON_DIR))
    args = parser.parse_args()

    if args.scan_all:
        run_dirs = discover_run_dirs()
    elif args.results_dirs:
        run_dirs = [Path(d) for d in args.results_dirs]
    else:
        print("Provide --scan_all or --results_dirs")
        sys.exit(1)

    if not run_dirs:
        print("No run directories found.")
        sys.exit(0)

    print(f"Found {len(run_dirs)} run directories:")
    for d in run_dirs:
        print(f"  {d}")

    # Load everything
    all_summaries: list[dict] = []
    all_logs: dict[str, dict[int, list[dict]]] = {}

    for rd in run_dirs:
        summaries = load_summaries(rd)
        all_summaries.extend(summaries)
        method = summaries[0]["method"] if summaries else rd.parent.name
        logs = load_logs(rd)
        if method in all_logs:
            all_logs[method].update(logs)
        else:
            all_logs[method] = logs

    # Build table
    rows = build_table(all_summaries)
    agg = aggregate_by_method(rows)

    # Print summary
    print("\n=== Aggregated Results ===")
    for a in agg:
        print(f"  {a['method']:30s}  clean={a['clean_asr']:.2f}  "
              f"swap10={a.get('swap_10',0):.2f}  swap15={a.get('swap_15',0):.2f}  "
              f"swap20={a.get('swap_20',0):.2f}  "
              f"sec/step={a.get('sec_per_step',0):.1f}")

    # Output
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    write_csv(agg, out)
    plot_asr_by_method(agg, out)
    plot_asr_vs_pertpct(agg, out)
    plot_cost_vs_asr(agg, out)
    plot_loss_curves_overlay(all_logs, out)
    plot_per_behavior_heatmap(rows, out)

    print(f"\nAll comparison outputs saved to {out}")


if __name__ == "__main__":
    main()
