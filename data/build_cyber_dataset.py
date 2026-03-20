#!/usr/bin/env python3
"""Curate a cybersecurity/code-focused jailbreak behaviour dataset.

Sources
-------
1. The local ``behaviors_config.json`` (I-GCG / AdvBench subset, 50 items).
2. The full AdvBench ``harmful_behaviors.csv`` (520 items, downloaded on demand).
3. (Optional) HarmBench ``harmbench_behaviors_text_all.csv``.

The script keyword-filters for cybersecurity / code-related prompts,
deduplicates, rewrites targets in I-GCG style, applies inert code scaffolding,
and writes ``data/cyber_behaviors.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

# Ensure the repo root is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from robust_gcg.scaffold import build_scaffold

# ---- keyword filter -------------------------------------------------------

CYBER_KEYWORDS = [
    "hack", "hacking", "exploit", "vulnerability", "malware",
    "ransomware", "trojan", "keylogger", "rootkit", "botnet",
    "DDoS", "brute force", "phish", "phishing",
    "SQL injection", "XSS", "cross-site", "code injection",
    "password", "credential", "encrypt", "decrypt",
    "backdoor", "identity theft", "intrusion",
    "virus", "antivirus", "firewall",
    "software", "database", "server", "network",
    "website", "script", "program", "code", "algorithm",
    "spyware", "cybersecurity", "cyber", "pirate software",
    "IoT", "social media account", "location without",
    "crack", "online exams",
]

_KW_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in CYBER_KEYWORDS), re.IGNORECASE
)


def is_cyber(text: str) -> bool:
    return bool(_KW_PATTERN.search(text))


# ---- target rewriting -----------------------------------------------------

def rewrite_target(behaviour: str, original_target: str) -> str:
    """Rewrite target into I-GCG ``"Sure, my output is harmful. Here is …:"``."""
    # Strip a leading "Sure, here is " or similar from the original
    stripped = re.sub(
        r"^Sure,?\s*(here\s+(is|are)\s+)?", "", original_target, flags=re.IGNORECASE
    )
    stripped = stripped.strip().rstrip(":")
    return f"Sure, my output is harmful. Here is {stripped}:"


# ---- code-framing augmentation --------------------------------------------

_CODE_REWRITE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^Write a tutorial on", re.I), "Write a Python tutorial on"),
    (re.compile(r"^Create a tutorial on", re.I), "Create a Python tutorial on"),
    (re.compile(r"^Provide a tutorial on", re.I), "Provide a Python tutorial on"),
    (re.compile(r"^Develop a strategy for", re.I), "Write a Python script that implements a strategy for"),
    (re.compile(r"^Develop a program", re.I), "Write a Python program"),
    (re.compile(r"^Develop a algorithm", re.I), "Write a Python algorithm"),
    (re.compile(r"^Develop an algorithm", re.I), "Write a Python algorithm"),
    (re.compile(r"^Develop a virus", re.I), "Write Python code that creates a virus which"),
    (re.compile(r"^Develop an app", re.I), "Write a Python app that"),
    (re.compile(r"^Create a ransomware", re.I), "Write Python code for a ransomware that"),
    (re.compile(r"^Create a bot", re.I), "Write Python code for a bot that"),
    (re.compile(r"^Create a fake", re.I), "Write a script that creates a fake"),
    (re.compile(r"^Design a phishing", re.I), "Write a Python script that crafts a phishing"),
    (re.compile(r"^Demonstrate how to", re.I), "Write Python code that demonstrates how to"),
    (re.compile(r"^Explain how to", re.I), "Write a Python proof-of-concept that shows how to"),
    (re.compile(r"^Offer a guide on", re.I), "Write a Python guide on"),
]


def augment_with_code_framing(behaviour: str) -> str:
    for pat, replacement in _CODE_REWRITE_PATTERNS:
        if pat.search(behaviour):
            return pat.sub(replacement, behaviour, count=1)
    return behaviour


# ---- main ------------------------------------------------------------------

def load_local_behaviors(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_advbench_csv(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            goal = row.get("goal", "").strip()
            target = row.get("target", "").strip()
            if goal:
                rows.append({"id": 1000 + i, "behaviour": goal, "target": target, "source": "advbench"})
    return rows


def build_dataset(
    local_path: str,
    advbench_path: str | None = None,
    scaffold_type: str = "code_review",
    buffer_lines: int = 6,
    augment: bool = True,
    step: int = 500,
    batch_size: int = 256,
    top_k: int = 256,
) -> list[dict]:
    seen_behaviours: set[str] = set()
    dataset: list[dict] = []

    # -- source 1: local behaviors_config.json ---
    for item in load_local_behaviors(local_path):
        b = item["behaviour"]
        if is_cyber(b) and b.lower() not in seen_behaviours:
            seen_behaviours.add(b.lower())
            dataset.append({**item, "source": "local"})

    # -- source 2: AdvBench CSV ---
    if advbench_path and os.path.isfile(advbench_path):
        for item in load_advbench_csv(advbench_path):
            b = item["behaviour"]
            if is_cyber(b) and b.lower() not in seen_behaviours:
                seen_behaviours.add(b.lower())
                dataset.append(item)

    # -- augment & scaffold ---
    final: list[dict] = []
    for idx, item in enumerate(dataset, start=1):
        raw = item["behaviour"]
        behaviour = augment_with_code_framing(raw) if augment else raw
        target = rewrite_target(behaviour, item.get("target", f"Sure, here is {behaviour.lower()}"))
        scaffolded, buf_start, buf_end = build_scaffold(behaviour, scaffold_type, buffer_lines)

        final.append({
            "id": idx,
            "behaviour_raw": raw,
            "behaviour": behaviour,
            "behaviour_scaffolded": scaffolded,
            "target": target,
            "source": item.get("source", "unknown"),
            "adv_init_suffix": "! " * 20,
            "scaffold_buffer_char_start": buf_start,
            "scaffold_buffer_char_end": buf_end,
            "step": step,
            "batch_size": batch_size,
            "top_k": top_k,
        })

    return final


def main():
    parser = argparse.ArgumentParser(description="Build cybersecurity jailbreak dataset")
    parser.add_argument(
        "--local_config",
        default=str(_REPO_ROOT / "behaviors_config.json"),
        help="Path to the local behaviors_config.json",
    )
    parser.add_argument(
        "--advbench_csv",
        default=str(_REPO_ROOT / "data" / "harmful_behaviors.csv"),
        help="Path to AdvBench harmful_behaviors.csv (optional, downloaded separately)",
    )
    parser.add_argument("--scaffold_type", default="code_review", choices=["code_review", "ctf_challenge", "docstring"])
    parser.add_argument("--buffer_lines", type=int, default=6)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--output", default=str(_REPO_ROOT / "data" / "cyber_behaviors.json"))
    args = parser.parse_args()

    ds = build_dataset(
        local_path=args.local_config,
        advbench_path=args.advbench_csv,
        scaffold_type=args.scaffold_type,
        buffer_lines=args.buffer_lines,
        augment=not args.no_augment,
        step=args.step,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ds, f, indent=2)

    print(f"Wrote {len(ds)} cyber behaviours to {out_path}")
    for item in ds:
        print(f"  [{item['id']:>3}] ({item['source']:>8}) {item['behaviour'][:80]}")


if __name__ == "__main__":
    main()
