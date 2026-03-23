"""
Combine per-comment ACE data into a single JSON list per article.

For each comment in a given article, merges:
  - ace_comments/{article_id}/{comment_index}.json (raw comment + ACE sentences + source mappings)
  - ace_dependency_graphs/{article_id}/{comment_index}.json (dependency graph)
  - ace_classifications/ace_sentence_classifications_{article_id}.json (per-sentence tags)

Output: combined_data/{article_id}.json — a JSON list with one object per comment.

Usage:
    python 4_combine_dataframe.py --article-id 181
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR


def load_classifications(
    data_dir: Path, article_id: str
) -> tuple[dict[int, dict[str, str]], str]:
    """Load classifications and return ({comment_id: {sentence: tag}}, image_description)."""
    path = data_dir / "ace_classifications" / f"ace_sentence_classifications_{article_id}.json"
    if not path.exists():
        print(f"Warning: classifications file not found at {path}")
        return {}, ""

    with open(path) as f:
        raw = json.load(f)

    grouped: dict[int, dict[str, str]] = defaultdict(dict)
    image_description = ""
    for entry in raw:
        tag = entry.get("comment_tag", "")
        if isinstance(tag, list):
            tag = tag[0] if tag else "unknown"
        if not tag:
            tag = "unknown"
        grouped[entry["comment_id"]][entry["original_comment"]] = tag
        if not image_description:
            image_description = entry.get("image_description", "")
    return dict(grouped), image_description


def discover_comment_indices(
    data_dir: Path,
    article_id: str,
    comment_index: int | None = None,
) -> list[int]:
    """Find all comment indices available in the ace_comments folder.

    If comment_index is provided, only that index is returned (if present).
    """
    comments_dir = data_dir / "ace_comments" / article_id
    if not comments_dir.is_dir():
        return []

    if comment_index is not None:
        target = comments_dir / f"{comment_index}.json"
        if target.exists():
            return [comment_index]
        print(
            f"No ace_comments file found for article {article_id}, comment_index {comment_index}"
        )
        return []

    return sorted(
        int(p.stem) for p in comments_dir.glob("*.json") if p.stem.isdigit()
    )


def combine_comment(
    data_dir: Path,
    article_id: str,
    comment_index: int,
    classifications_by_comment: dict[int, dict[str, str]],
    image_description: str = "",
) -> dict | None:
    """Build a single combined object for one comment."""
    comment_path = data_dir / "ace_comments" / article_id / f"{comment_index}.json"
    graph_path = data_dir / "ace_dependency_graphs" / article_id / f"{comment_index}.json"

    if not comment_path.exists():
        print(f"  Skipping comment {comment_index}: ace_comments file missing")
        return None

    with open(comment_path) as f:
        comment_data = json.load(f)

    dependency_graph = None
    if graph_path.exists():
        with open(graph_path) as f:
            graph_data = json.load(f)
        dependency_graph = graph_data.get("dependency_graph")
    else:
        print(f"  Warning: dependency graph missing for comment {comment_index}")

    tag_lookup = classifications_by_comment.get(comment_index, {})
    if dependency_graph is not None:
        for node in dependency_graph:
            node["comment_tag"] = tag_lookup.get(node["sentence"], "unknown")

    return {
        "article_id": article_id,
        "comment_index": comment_index,
        "image_description": image_description,
        "raw_comment": comment_data.get("raw_comment"),
        "ace_sentences": comment_data.get("ace_sentences", []),
        "source_mappings": comment_data.get("source_mappings", []),
        "dependency_graph": dependency_graph,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine per-comment ACE data into a single JSON list."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "Root data directory containing ace_comments/, ace_dependency_graphs/, "
            f"ace_classifications/ (default: {DEFAULT_DATA_DIR})."
        ),
    )
    parser.add_argument(
        "--article-id",
        required=True,
        help="Article ID to combine.",
    )
    parser.add_argument(
        "--comment-index",
        type=int,
        default=None,
        help=(
            "Optional 1-based comment index. If set, only this comment is combined "
            "for the given article."
        ),
    )

    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    article_id = args.article_id

    classifications, image_description = load_classifications(data_dir, article_id)
    indices = discover_comment_indices(
        data_dir,
        article_id,
        comment_index=args.comment_index,
    )

    if not indices:
        print(f"No comments found for article {article_id}")
        return

    print(f"Combining {len(indices)} comments for article {article_id} ...")

    combined = []
    for idx in indices:
        result = combine_comment(data_dir, article_id, idx, classifications, image_description)
        if result is not None:
            combined.append(result)

    out_dir = data_dir / "combined_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{article_id}.json"

    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(combined)} combined comments to {out_path}")


if __name__ == "__main__":
    main()
