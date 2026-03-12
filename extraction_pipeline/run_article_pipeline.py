from __future__ import annotations

"""
Run the full NYT comment analysis pipeline for one or more article_ids.

This script orchestrates the extraction pipeline step scripts:

1. 1_extract_ace_comments.py
2. 2_classify_ace_sentences.py
3. 3_dependency_classification.py
4. 4_combine_dataframe.py
5. 10_visualize_graph.py

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

TEST_TIMEOUT = 5  # seconds to wait before killing a step in test mode


def run_step(cmd: list[str], *, test: bool = False) -> None:
    """Run a single pipeline step, streaming output, and fail fast on error.

    When *test* is True the step is killed after TEST_TIMEOUT seconds and the
    timeout is treated as success (the point is just to verify the script starts).
    """
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    if test:
        print(f"  [test mode – will cancel after {TEST_TIMEOUT}s]")
    print("=" * 80)

    try:
        subprocess.run(
            [sys.executable, *cmd],
            cwd=SCRIPT_DIR,
            check=True,
            timeout=TEST_TIMEOUT if test else None,
        )
    except subprocess.TimeoutExpired:
        print(f"  ⏱  Timed out after {TEST_TIMEOUT}s (expected in test mode)")
    except subprocess.CalledProcessError:
        raise


def run_pipeline_for_article(
    article_id: str,
    *,
    test: bool = False,
    comment_index: int | None = None,
) -> None:
    """
    Run the full 5-step pipeline for a single article_id.

    When *test* is True each step is killed after TEST_TIMEOUT seconds so you
    can quickly verify every script at least starts up correctly.
    """
    # 1) Extract ACE comments
    cmd_1 = [
        "1_extract_ace_comments.py",
        "--article-id",
        str(article_id),
    ]
    if comment_index is not None:
        cmd_1 += ["--comment-index", str(comment_index)]
    run_step(cmd_1, test=test)

    # 2) Classify ACE sentences
    cmd_2 = [
        "2_classify_ace_sentences.py",
        "--article-id",
        str(article_id),
    ]
    if comment_index is not None:
        cmd_2 += ["--comment-index", str(comment_index)]
    run_step(cmd_2, test=test)

    # 3) Dependency classification
    cmd_3 = [
        "3_dependency_classification.py",
        "--article-id",
        str(article_id),
    ]
    if comment_index is not None:
        cmd_3 += ["--comment-index", str(comment_index)]
    run_step(cmd_3, test=test)

    # 4) Combine per-comment data
    cmd_4 = [
        "4_combine_dataframe.py",
        "--article-id",
        str(article_id),
    ]
    if comment_index is not None:
        cmd_4 += ["--comment-index", str(comment_index)]
    run_step(cmd_4, test=test)

    # 5) Visualize graph from combined data
    combined_path = SCRIPT_DIR / "combined_data" / f"{article_id}.json"
    run_step(
        [
            "10_visualize_graph.py",
            "--input",
            str(combined_path),
        ],
        test=test,
    )

    if test:
        print("\nTest run finished – all steps started successfully.")
    else:
        print("\nPipeline completed successfully.")
    print(f"- Combined data JSON: {combined_path}")
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
    parser.add_argument(
        "--comment-index",
        type=int,
        default=None,
        help=(
            "Optional 1-based comment index. If set, each step in the pipeline is "
            "restricted to this single comment for the selected article(s)."
        ),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: run each step but kill it after 5 seconds.",
    )
    args = parser.parse_args()

    article_ids = args.article_ids if args.article_ids else [args.article_id]

    if args.test:
        print("*** TEST MODE – each step will be cancelled after 5 seconds ***")

    for i, article_id in enumerate(article_ids, 1):
        print(f"\n{'#' * 80}")
        print(f"# Processing article {article_id} ({i}/{len(article_ids)})")
        print(f"{'#' * 80}")
        run_pipeline_for_article(
            article_id,
            test=args.test,
            comment_index=args.comment_index,
        )


if __name__ == "__main__":
    main()

