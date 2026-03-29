"""
Combine per-comment ACE data into a single JSON list per article.

For each comment in a given article, merges outputs from:

  1) 1_extract_ace_comments.py — ace_comments/{article_id}/{comment_index}.json
        → raw_comment, ace_sentences, source_mappings, order

  2) 2_classify_ace_sentences.py — ace_classifications/ace_sentence_classifications_{article_id}.json
        (or ace_sentence_classifications.json at data-dir root, filtered by article_id)
        → sentence_classifications[] (full step-2 rows), comment_tag/reasoning merged into graph nodes

  3) 3_dependency_classification.py — ace_dependency_graphs/{article_id}/{comment_index}.json
        → dependency_graph (nodes with id, sentence, depends_on[{id, edge_type, justification}])

Article IDs and comment indices are the union of all three trees, so nothing present in only
classifications or only graphs is dropped. Step-1 fields may be null/empty if only 2+3 exist.

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


def _normalize_comment_id(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def load_classifications(
    data_dir: Path, article_id: str
) -> tuple[
    dict[int, dict[str, dict[str, str]]],
    str,
    dict[int, list[dict]],
]:
    """Load step-2 file(s): lookup by sentence, global image_description, rows per comment_id."""
    per_article = (
        data_dir / "ace_classifications" / f"ace_sentence_classifications_{article_id}.json"
    )
    monolithic = data_dir / "ace_sentence_classifications.json"
    path = per_article if per_article.exists() else monolithic
    if not per_article.exists() and not monolithic.exists():
        print(f"  Warning: classifications not found ({per_article} or {monolithic})")
        return {}, "", {}

    with open(path) as f:
        raw = json.load(f)

    grouped: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    rows_by_comment: dict[int, list[dict]] = defaultdict(list)
    image_description = ""

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        row_article = str(entry.get("article_id", "")).strip()
        if path == monolithic and row_article != str(article_id).strip():
            continue
        cid = _normalize_comment_id(entry.get("comment_id"))
        if cid is None:
            continue
        tag = entry.get("comment_tag", "")
        if isinstance(tag, list):
            tag = tag[0] if tag else "unknown"
        if not tag:
            tag = "unknown"
        reasoning = entry.get("reasoning", "")
        oc = entry.get("original_comment", "")
        grouped[cid][oc] = {"comment_tag": tag, "reasoning": reasoning}
        rows_by_comment[cid].append(
            {
                "article_id": entry.get("article_id", article_id),
                "comment_id": cid,
                "original_comment": oc,
                "reasoning": reasoning,
                "comment_tag": tag,
                "image_description": entry.get("image_description", ""),
            }
        )
        if not image_description:
            image_description = entry.get("image_description", "")

    return dict(grouped), image_description, dict(rows_by_comment)


def discover_article_ids(data_dir: Path) -> list[str]:
    """Union of article IDs from ace_comments/, ace_dependency_graphs/, ace_classifications/."""
    ids: set[int] = set()
    for sub in ("ace_comments", "ace_dependency_graphs"):
        root = data_dir / sub
        if root.is_dir():
            ids.update(
                int(d.name) for d in root.iterdir() if d.is_dir() and d.name.isdigit()
            )
    cls_root = data_dir / "ace_classifications"
    if cls_root.is_dir():
        for p in cls_root.glob("ace_sentence_classifications_*.json"):
            stem = p.stem.replace("ace_sentence_classifications_", "")
            if stem.isdigit():
                ids.add(int(stem))
    mono = data_dir / "ace_sentence_classifications.json"
    if mono.exists():
        try:
            with open(mono) as f:
                blob = json.load(f)
            for entry in blob:
                if not isinstance(entry, dict):
                    continue
                aid = str(entry.get("article_id", "")).strip()
                if aid.isdigit():
                    ids.add(int(aid))
        except (json.JSONDecodeError, OSError):
            print(f"  Warning: could not read article ids from {mono}")
    return sorted(str(i) for i in ids)


def discover_comment_indices(
    data_dir: Path,
    article_id: str,
    comment_index: int | None = None,
    extra_indices: frozenset[int] | None = None,
) -> list[int]:
    """Union of comment indices from ace_comments, dependency graphs, and optional extras."""
    found: set[int] = set()
    if extra_indices:
        found.update(extra_indices)

    comments_dir = data_dir / "ace_comments" / article_id
    if comments_dir.is_dir():
        found.update(int(p.stem) for p in comments_dir.glob("*.json") if p.stem.isdigit())

    graphs_dir = data_dir / "ace_dependency_graphs" / article_id
    if graphs_dir.is_dir():
        found.update(int(p.stem) for p in graphs_dir.glob("*.json") if p.stem.isdigit())

    if comment_index is not None:
        if comment_index in found:
            return [comment_index]
        print(
            f"  No data found for article {article_id}, comment_index {comment_index} "
            f"(checked ace_comments, ace_dependency_graphs, classifications)"
        )
        return []

    return sorted(found)


def combine_comment(
    data_dir: Path,
    article_id: str,
    comment_index: int,
    classifications_by_comment: dict[int, dict[str, dict[str, str]]],
    image_description: str = "",
    sentence_classification_rows: list[dict] | None = None,
) -> dict | None:
    """Build a single combined object for one comment."""
    comment_path = data_dir / "ace_comments" / article_id / f"{comment_index}.json"
    graph_path = data_dir / "ace_dependency_graphs" / article_id / f"{comment_index}.json"

    comment_data: dict = {}
    if comment_path.exists():
        with open(comment_path) as f:
            comment_data = json.load(f)
    elif graph_path.exists() or sentence_classification_rows:
        comment_data = {
            "raw_comment": None,
            "ace_sentences": [],
            "source_mappings": {},
            "order": {},
        }
        print(
            f"  Warning: ace_comments missing for article {article_id} comment {comment_index}; "
            "using graph/classifications only where available"
        )
    else:
        print(f"  Skipping comment {comment_index}: no ace_comments, graph, or classifications")
        return None

    dependency_graph = None
    if graph_path.exists():
        with open(graph_path) as f:
            graph_data = json.load(f)
        dependency_graph = graph_data.get("dependency_graph")
    else:
        print(f"  Warning: dependency graph missing for comment {comment_index}")

    classification_lookup = classifications_by_comment.get(comment_index, {})
    rows = sentence_classification_rows if sentence_classification_rows is not None else []

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

    ace_sentences = list(comment_data.get("ace_sentences") or [])
    if not ace_sentences and dependency_graph:
        ace_sentences = [
            n["sentence"]
            for n in sorted(dependency_graph, key=lambda x: x.get("id", 0))
            if isinstance(n, dict) and "sentence" in n
        ]
    if not ace_sentences and rows:
        ace_sentences = [r.get("original_comment", "") for r in rows if r.get("original_comment")]

    return {
        "article_id": article_id,
        "comment_index": comment_index,
        "image_description": image_description,
        "raw_comment": comment_data.get("raw_comment"),
        "ace_sentences": ace_sentences,
        "source_mappings": comment_data.get("source_mappings", {}),
        "order": comment_data.get("order", {}),
        "sentence_classifications": rows,
        "dependency_graph": dependency_graph,
    }


def combine_article(
    data_dir: Path,
    article_id: str,
    comment_index: int | None = None,
) -> list[dict]:
    """Combine all comments for a single article, returning the list of combined objects."""
    classifications, image_description, rows_by_comment = load_classifications(
        data_dir, article_id
    )
    extra = frozenset(rows_by_comment.keys())
    indices = discover_comment_indices(
        data_dir,
        article_id,
        comment_index=comment_index,
        extra_indices=extra,
    )

    if not indices:
        print(f"  No comments found for article {article_id}")
        return []

    combined = []
    for idx in indices:
        result = combine_comment(
            data_dir,
            article_id,
            idx,
            classifications,
            image_description,
            sentence_classification_rows=rows_by_comment.get(idx, []),
        )
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
            f"ace_classifications/ (and optionally ace_sentence_classifications.json; "
            f"default: {DEFAULT_DATA_DIR})."
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
        help=(
            "Process all article IDs found across ace_comments/, ace_dependency_graphs/, "
            "ace_classifications/, and ace_sentence_classifications.json (if present)."
        ),
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
        print(f"Found {len(article_ids)} article(s) across pipeline outputs.")
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
