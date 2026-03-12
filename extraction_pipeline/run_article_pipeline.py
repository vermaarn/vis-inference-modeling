from __future__ import annotations

"""
Run the full NYT comment analysis pipeline for a single article_id.

This script simply orchestrates the existing step scripts documented in PIPELINE.md:

1. 1_extract_ace_comments.py
2. 2_classify_ace_sentences.py
3. 3_cluster_themes.py
4. 4_dependency_classification.py
5. 5_combine_dataframe.py
6. 6_combine_similar_nodes.py
7. 10_visualize_graph.py

Usage (from inside extraction_pipeline_v4/):

    python run_article_pipeline.py --article-id 181

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
    Run the full 7‑step pipeline for a single article_id.

    Steps match the "End-to-end order for one article" in PIPELINE.md.
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

    # 3) Cluster themes
    classifications_path = (
        SCRIPT_DIR
        / "ace_classifications"
        / f"ace_sentence_classifications_{article_id}.json"
    )
    clusters_path = (
        SCRIPT_DIR
        / "ace_clusters"
        / f"ace_sentence_theme_clusters_{article_id}.json"
    )
    run_step(
        [
            "3_cluster_themes.py",
            "--input",
            str(classifications_path),
            "--output",
            str(clusters_path),
        ]
    )

    # 4) Dependency classification
    run_step(
        [
            "4_dependency_classification.py",
            "--article-id",
            str(article_id),
        ]
    )

    # 5) Combine per-comment data
    run_step(
        [
            "5_combine_dataframe.py",
            "--article-id",
            str(article_id),
        ]
    )

    # 6) Combine similar nodes (reduced graphs)
    combined_path = SCRIPT_DIR / "combined_data" / f"{article_id}.json"
    run_step(
        [
            "6_combine_similar_nodes.py",
            "--input",
            str(combined_path),
        ]
    )

    # 7) Visualize graph from reduced data
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
        description="Run the full NYT comment analysis pipeline for a single article_id."
    )
    parser.add_argument(
        "--article-id",
        required=True,
        help="Article ID to process (e.g. 181).",
    )
    args = parser.parse_args()

    run_pipeline_for_article(args.article_id)


if __name__ == "__main__":
    main()

