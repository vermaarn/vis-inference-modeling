from __future__ import annotations

"""
Run the full NYT comment analysis pipeline for one or more article_ids.

This script orchestrates the extraction pipeline step scripts:

1. 1_extract_ace_comments.py
2. 2_classify_ace_sentences.py
3. 3_dependency_classification.py
4. 4_combine_dataframe.py
5. 5_combine_similar_nodes.py
6. 10_visualize_graph.py

Usage (from extraction_pipeline/):

    # Single article
    python run_article_pipeline.py --article-id 181

    # Multiple articles
    python run_article_pipeline.py --article-ids 181 202 305

It assumes:
- Raw article comments live at ../data/comment_data/{article_id}.json
- OPENAI_API_KEY is available via your environment / .env (as used by the step scripts)
"""

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run_step(cmd: list[str]) -> None:
    """Run a single pipeline step, streaming output, and fail fast on error."""
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)
    subprocess.run([sys.executable, *cmd], cwd=SCRIPT_DIR, check=True)


def run_pipeline_for_article(article_id: str) -> None:
    """
    Run the full 6-step pipeline for a single article_id.
    """
    # 1) Extract ACE comments
    run_step(
        [
            "1_extract_ace_comments.py",
            "--article_id",
            str(article_id),
        ]
    )

    # 2) Classify ACE sentences
    run_step(
        [
            "2_classify_ace_sentences.py",
            "--article-id",
            str(article_id),
        ]
    )

    # 3) Dependency classification
    run_step(
        [
            "3_dependency_classification.py",
            "--article-id",
            str(article_id),
        ]
    )

    # 4) Combine per-comment data
    run_step(
        [
            "4_combine_dataframe.py",
            "--article-id",
            str(article_id),
        ]
    )

    # 5) Combine similar nodes (reduced graphs)
    combined_path = SCRIPT_DIR / "combined_data" / f"{article_id}.json"
    run_step(
        [
            "5_combine_similar_nodes.py",
            "--input",
            str(combined_path),
        ]
    )

    # 6) Visualize graph from reduced data
    reduced_path = SCRIPT_DIR / "reduced_data" / f"{article_id}.json"
    run_step(
        [
            "10_visualize_graph.py",
            "--input",
            str(reduced_path),
        ]
    )

    print("\nPipeline completed successfully.")
    print(f"- Reduced data JSON: {reduced_path}")
    print(f"- HTML visualizations: graph_visualizations/{article_id}/comment_*.html")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full NYT comment analysis pipeline for one or more article_ids."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--article-id",
        help="Single article ID to process (e.g. 181).",
    )
    group.add_argument(
        "--article-ids",
        nargs="+",
        help="List of article IDs to process (e.g. 181 202 305).",
    )
    args = parser.parse_args()

    article_ids = args.article_ids if args.article_ids else [args.article_id]

    for i, article_id in enumerate(article_ids, 1):
        print(f"\n{'#' * 80}")
        print(f"# Processing article {article_id} ({i}/{len(article_ids)})")
        print(f"{'#' * 80}")
        run_pipeline_for_article(article_id)


if __name__ == "__main__":
    main()

