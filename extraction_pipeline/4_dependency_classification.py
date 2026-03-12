"""
Build dependency graphs (DAGs) over ACE sentences in each comment.

Reads ACE comment JSONs from ace_comments/{article_id}/, sends each comment's
ace_sentences to an LLM with the causal_direction prompt, and writes one JSON
per comment to ace_dependency_graphs/{article_id}/{comment_index}.json with
article_id, comment_index, and dependency_graph (nodes with id, sentence, depends_on).

Usage:
  python 4_dependency_classification.py --article-id 181
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import argparse

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ACE_COMMENTS_DIR = SCRIPT_DIR / "ace_comments"
DEFAULT_PROMPTS_DIR = SCRIPT_DIR / "prompts"
DEFAULT_PROMPT_FILE = DEFAULT_PROMPTS_DIR / "dependency_classification.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "ace_dependency_graphs"


def _ensure_client(api_key: str | None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY in .env or pass --api-key."
        )
    return OpenAI(api_key=key)


def load_comment_files(
    ace_comments_dir: Path,
    article_id: str,
) -> List[Path]:
    """
    Return paths to all comment JSON files for the given article_id, sorted by
    comment index (numeric stem).
    """
    article_dir = ace_comments_dir / article_id
    if not article_dir.is_dir():
        return []
    paths = list(article_dir.glob("*.json"))
    # Sort by numeric stem so 1, 2, ..., 10, 11, ...
    def key(p: Path) -> int:
        try:
            return int(p.stem)
        except ValueError:
            return -1
    return sorted(paths, key=key)


def load_prompt(prompt_path: Path) -> str:
    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def build_dependency_graph(
    client: OpenAI,
    prompt_text: str,
    comment_data: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    """
    Send one comment (article_id, comment_index, ace_sentences) to the model
    and return the full response object (article_id, comment_index, dependency_graph).
    """
    input_payload = {
        "article_id": comment_data.get("article_id", ""),
        "comment_index": comment_data.get("comment_index", 0),
        "ace_sentences": comment_data.get("ace_sentences") or [],
    }
    user_content = prompt_text + "\n\nInput:\n" + json.dumps(
        input_payload, indent=2, ensure_ascii=False
    )

    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You build dependency graphs over ACE sentences. Respond only with valid JSON in the required shape.",
            },
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    elapsed = time.perf_counter() - start_time

    raw = response.choices[0].message.content
    try:
        out = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON: {e}\nRaw: {raw[:500]}...")

    dependency_graph = out.get("dependency_graph")
    if not isinstance(dependency_graph, list):
        raise ValueError(
            f"Model response missing or invalid 'dependency_graph' list: {list(out.keys())}"
        )

    # Normalize: ensure article_id and comment_index match input
    out["article_id"] = comment_data.get("article_id", out.get("article_id"))
    out["comment_index"] = comment_data.get("comment_index", out.get("comment_index"))

    print(
        f"  Comment {out['comment_index']}: {len(dependency_graph)} nodes, "
        f"API call took {elapsed:.2f}s"
    )
    return out


def run_dependency_classification(
    ace_comments_dir: Path,
    prompt_path: Path,
    output_dir: Path,
    article_id: str,
    api_key: str | None = None,
    model: str = "gpt-5.2",
) -> List[Dict[str, Any]]:
    """
    Load all comments for article_id, build a dependency graph for each, and
    write one JSON per comment to output_dir/{article_id}/{comment_index}.json.
    """
    comment_paths = load_comment_files(ace_comments_dir, article_id)
    if not comment_paths:
        print(f"No comment files found in {ace_comments_dir / article_id}")
        return []

    print(
        f"Building dependency graphs for article_id={article_id} "
        f"({len(comment_paths)} comments)"
    )

    prompt_text = load_prompt(prompt_path)
    client = _ensure_client(api_key)

    results: List[Dict[str, Any]] = []
    out_article_dir = output_dir / article_id
    out_article_dir.mkdir(parents=True, exist_ok=True)

    for json_path in comment_paths:
        try:
            with json_path.open("r", encoding="utf-8") as f:
                comment_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Skipping {json_path}: {e}")
            continue

        ace_sentences = comment_data.get("ace_sentences") or []
        if not ace_sentences:
            print(f"Skipping {json_path}: no ace_sentences")
            continue

        comment_index = comment_data.get("comment_index")
        if comment_index is None:
            comment_index = int(json_path.stem) if json_path.stem.isdigit() else 0
        comment_data["comment_index"] = comment_index
        if "article_id" not in comment_data:
            comment_data["article_id"] = article_id

        try:
            out = build_dependency_graph(client, prompt_text, comment_data, model)
        except (ValueError, Exception) as e:
            print(f"  Error on comment {comment_index}: {e}")
            continue

        results.append(out)
        out_path = out_article_dir / f"{comment_index}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  Wrote {out_path}")

    print(f"Done. Wrote {len(results)} dependency graphs to {out_article_dir}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dependency graphs (DAGs) over ACE sentences per comment."
    )
    parser.add_argument(
        "--ace-comments-dir",
        type=Path,
        default=DEFAULT_ACE_COMMENTS_DIR,
        help=f"Directory containing per-article ACE comment JSONs (default: {DEFAULT_ACE_COMMENTS_DIR})",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help=f"Path to dependency classification prompt .txt (default: {DEFAULT_PROMPT_FILE})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for dependency graph JSONs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="OpenAI model (default: gpt-5.2)",
    )
    parser.add_argument(
        "--article-id",
        type=str,
        required=True,
        help="Article ID: only process comments under ace_comments/<article_id>/",
    )
    args = parser.parse_args()

    run_dependency_classification(
        ace_comments_dir=args.ace_comments_dir,
        prompt_path=args.prompt_file,
        output_dir=args.output_dir,
        article_id=args.article_id,
        api_key=args.api_key,
        model=args.model,
    )


if __name__ == "__main__":
    main()
