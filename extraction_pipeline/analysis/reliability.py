"""
Reliability analysis: do multiple ACE extraction runs produce the same number
of sentences per comment?

This script analyzes the reliability of previously collected ACE extraction runs.
It compares two runs stored under:
    reliability_runs/run_1/181/<comment_index>.json
    reliability_runs/run_2/181/<comment_index>.json

Usage (from the extraction_pipeline/ directory):
    uv run analysis/reliability.py

NOTE: This script assumes the data (in reliability_runs) has already been collected!
If so, you can run this at any time to perform the analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "reliability_runs"

ARTICLE_ID = "181"
COMMENT_INDICES = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def load_sentence_count(run: str, article_id: str, comment_index: int) -> int:
    path = RUNS_DIR / run / article_id / f"{comment_index}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data["ace_sentences"])


def load_sentences(run: str, article_id: str, comment_index: int) -> list[str]:
    path = RUNS_DIR / run / article_id / f"{comment_index}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["ace_sentences"]


def main() -> None:
    # Check that the data is present, so we can perform the analysis immediately.
    runs = sorted(r.name for r in RUNS_DIR.iterdir() if r.is_dir())
    if len(runs) < 2:
        print(f"Need at least 2 runs in {RUNS_DIR}, found: {runs}")
        return

    run1, run2 = runs[0], runs[1]
    print(f"Comparing  {run1}  vs  {run2}  for article {ARTICLE_ID}")
    print(f"Comments: {COMMENT_INDICES}\n")
    print("NOTE: Data already collected, running analysis only.\n")

    # --- Per-comment counts and diffs ---
    counts: dict[int, tuple[int, int]] = {}
    diffs: list[int] = []

    header = f"{'Comment':>9} {'Run1':>6} {'Run2':>6} {'Diff':>6}"
    print(header)
    print("-" * len(header))

    for idx in COMMENT_INDICES:
        n1 = load_sentence_count(run1, ARTICLE_ID, idx)
        n2 = load_sentence_count(run2, ARTICLE_ID, idx)
        d = n2 - n1
        counts[idx] = (n1, n2)
        diffs.append(d)
        marker = "  <-- outlier" if abs(d) > 1 else ""
        print(f"{idx:>9} {n1:>6} {n2:>6} {d:>+6}{marker}")

    # --- Summary statistics on the diffs ---
    print()
    print("=== Diff statistics (run2 - run1) ===")
    print(f"  All diffs   : {diffs}")
    print(f"  Min / Max   : {min(diffs)} / {max(diffs)}")
    print(f"  Mean        : {mean(diffs):.2f}")
    if len(diffs) > 1:
        print(f"  Std dev     : {stdev(diffs):.2f}")

    all_same = len(set(diffs)) == 1
    print()
    if all_same:
        print(f"RESULT: All comments differ by exactly {diffs[0]} sentence(s). "
              "The model is consistent with a fixed offset.")
    else:
        unique_diffs = sorted(set(diffs))
        print(f"RESULT: Diffs are NOT uniform — {len(unique_diffs)} distinct values: {unique_diffs}.")

        # Check if most diffs cluster tightly (within ±1 of the median)
        sorted_d = sorted(diffs)
        median = sorted_d[len(sorted_d) // 2]
        outliers = [COMMENT_INDICES[i] for i, d in enumerate(diffs) if abs(d - median) > 1]
        if outliers:
            print(f"  Comments with unusual drift (|diff - median| > 1): {outliers}")
        else:
            print(f"  All diffs are within ±1 of the median ({median:+d}), "
                  "suggesting minor stochastic variation only.")

    # --- Sentence-level overlap per comment ---
    print()
    print("=== Sentence-level overlap (exact string match) ===")
    overlap_header = f"{'Comment':>9} {'#Run1':>6} {'#Run2':>6} {'Common':>7} {'Jaccard':>8}"
    print(overlap_header)
    print("-" * len(overlap_header))

    jaccard_scores: list[float] = []
    for idx in COMMENT_INDICES:
        s1 = set(load_sentences(run1, ARTICLE_ID, idx))
        s2 = set(load_sentences(run2, ARTICLE_ID, idx))
        common = s1 & s2
        union = s1 | s2
        jaccard = len(common) / len(union) if union else 1.0
        jaccard_scores.append(jaccard)
        print(f"{idx:>9} {len(s1):>6} {len(s2):>6} {len(common):>7} {jaccard:>8.3f}")

    print()
    print(f"  Mean Jaccard similarity : {mean(jaccard_scores):.3f}")
    if len(jaccard_scores) > 1:
        print(f"  Std dev (Jaccard)       : {stdev(jaccard_scores):.3f}")
    print()

    low_overlap = [COMMENT_INDICES[i] for i, j in enumerate(jaccard_scores) if j < 0.5]
    if low_overlap:
        print(f"  Low-overlap comments (Jaccard < 0.5): {low_overlap}")
    else:
        print("  All comments have >50% sentence overlap — content is broadly stable.")


if __name__ == "__main__":
    main()
