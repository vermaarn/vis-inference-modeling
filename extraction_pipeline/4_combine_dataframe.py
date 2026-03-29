"""
Combine per-comment ACE data into a single JSON list per article.

For each comment in a given article, merges:
  - ace_comments/{article_id}/{comment_index}.json
        → raw_comment, ace_sentences, source_mappings, order
  - ace_classifications/ace_sentence_classifications_{article_id}.json
        → comment_tag, reasoning, image_description  (per ACE sentence)
  - ace_dependency_graphs/{article_id}/{comment_index}.json
        → dependency_graph  (nodes with id, sentence, depends_on[{id, edge_type, justification}])

Output: combined_data/{article_id}.json — a JSON list with one object per comment.

Usage:
    python 4_combine_dataframe.py --article-id 181
    python 4_combine_dataframe.py --all
    python 4_combine_dataframe.py --all --skip-existing
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
) -> tuple[dict[int, dict[str, dict[str, str]]], str]:
    """Load classifications and return ({comment_id: {sentence: {tag, reasoning}}}, image_description)."""
    path = data_dir / "ace_classifications" / f"ace_sentence_classifications_{article_id}.json"
    if not path.exists():
        print(f"  Warning: classifications file not found at {path}")
        return {}, ""

    with open(path) as f:
        raw = json.load(f)

    grouped: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    image_description = ""
    for entry in raw:
        tag = entry.get("comment_tag", "")
        if isinstance(tag, list):
            tag = tag[0] if tag else "unknown"
        if not tag:
            tag = "unknown"
        reasoning = entry.get("reasoning", "")
        grouped[entry["comment_id"]][entry["original_comment"]] = {
            "comment_tag": tag,
            "reasoning": reasoning,
        }
        if not image_description:
            image_description = entry.get("image_description", "")
    return dict(grouped), image_description


def discover_article_ids(data_dir: Path) -> list[str]:
    """Find all article IDs that have a subdirectory in ace_comments/."""
    comments_dir = data_dir / "ace_comments"
    if not comments_dir.is_dir():
        return []
    return sorted(
        (d.name for d in comments_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda x: int(x),
    )


def discover_comment_indices(
    data_dir: Path,
    article_id: str,
    comment_index: int | None = None,
) -> list[int]:
    """Find all comment indices available in the ace_comments folder."""
    comments_dir = data_dir / "ace_comments" / article_id
    if not comments_dir.is_dir():
        return []

    if comment_index is not None:
        target = comments_dir / f"{comment_index}.json"
        if target.exists():
            return [comment_index]
        print(
            f"  No ace_comments file found for article {article_id}, comment_index {comment_index}"
        )
        return []

    return sorted(
        int(p.stem) for p in comments_dir.glob("*.json") if p.stem.isdigit()
    )


def combine_comment(
    data_dir: Path,
    article_id: str,
    comment_index: int,
    classifications_by_comment: dict[int, dict[str, dict[str, str]]],
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

    classification_lookup = classifications_by_comment.get(comment_index, {})
    if dependency_graph is not None:
        clean_graph = []
        for node in dependency_graph:
            if not isinstance(node, dict) or "sentence" not in node:
                continue
            info = classification_lookup.get(node["sentence"], {})
            node["comment_tag"] = info.get("comment_tag", "unknown")
            node["reasoning"] = info.get("reasoning", "")
            clean_graph.append(node)
        dependency_graph = clean_graph

    return {
        "article_id": article_id,
        "comment_index": comment_index,
        "image_description": image_description,
        "raw_comment": comment_data.get("raw_comment"),
        "ace_sentences": comment_data.get("ace_sentences", []),
        "source_mappings": comment_data.get("source_mappings", {}),
        "order": comment_data.get("order", {}),
        "dependency_graph": dependency_graph,
    }


def combine_article(
    data_dir: Path,
    article_id: str,
    comment_index: int | None = None,
) -> list[dict]:
    """Combine all comments for a single article, returning the list of combined objects."""
    classifications, image_description = load_classifications(data_dir, article_id)
    indices = discover_comment_indices(data_dir, article_id, comment_index=comment_index)

    if not indices:
        print(f"  No comments found for article {article_id}")
        return []

    combined = []
    for idx in indices:
        result = combine_comment(data_dir, article_id, idx, classifications, image_description)
        if result is not None:
            combined.append(result)
    return combined


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
        type=str,
        default=None,
        help="Article ID to combine. If omitted, use --all to process every article.",
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
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all article IDs found in ace_comments/.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip articles that already have an output file in combined_data/.",
    )

    args = parser.parse_args()

    if not args.all and not args.article_id:
        parser.error("You must specify either --article-id or --all.")

    data_dir = args.data_dir.resolve()
    out_dir = data_dir / "combined_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        article_ids = discover_article_ids(data_dir)
        print(f"Found {len(article_ids)} article(s) in ace_comments/.")
    else:
        article_ids = [args.article_id]

    total_written = 0
    for article_id in article_ids:
        out_path = out_dir / f"{article_id}.json"

        if args.skip_existing and out_path.exists():
            print(f"Skipping article {article_id} (already exists at {out_path})")
            continue

        print(f"Combining article {article_id} ...")
        combined = combine_article(
            data_dir, article_id, comment_index=args.comment_index,
        )

        if not combined:
            continue

        with open(out_path, "w") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

        print(f"  Wrote {len(combined)} combined comments to {out_path}")
        total_written += 1

    print(f"\nDone. Wrote combined data for {total_written} article(s).")


if __name__ == "__main__":
    main()
