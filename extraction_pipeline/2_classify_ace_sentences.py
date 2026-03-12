"""
Classify ACE sentences from extracted ACE comments into semantic categories.

Reads ACE comment JSONs produced by 1_extract_ace_comments.py (under ace_comments/),
loads the classification prompt from prompts/classify_ace_sentences.txt, sends all
ACE sentences to an LLM in one or more batches, and writes a JSON file with
article_id, comment_id, original_comment, and comment_tag per sentence.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ACE_COMMENTS_DIR = SCRIPT_DIR / "ace_comments"
DEFAULT_PROMPTS_DIR = SCRIPT_DIR / "prompts"
DEFAULT_PROMPT_FILE = DEFAULT_PROMPTS_DIR / "classify_ace_sentences.txt"
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "ace_sentence_classifications.json"
DEFAULT_INTERMEDIATE_DIR = SCRIPT_DIR / "ace_sentence_classifications_batches"
DEFAULT_ACE_CLASSIFICATIONS_DIR = SCRIPT_DIR / "ace_classifications"

# Max sentences per API call to stay within context limits
BATCH_SIZE = 500


def _ensure_client(api_key: str | None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY in .env or pass --api-key."
        )
    return OpenAI(api_key=key)


def load_ace_comment_items(
    ace_comments_dir: Path,
    article_id: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Load all ACE comment JSON files under ace_comments_dir and return a flat list
    of items: { article_id, comment_id, original_comment } for each ACE sentence.
    If article_id is provided, only load sentences for that article_id.
    """
    items: List[Dict[str, Any]] = []
    if not ace_comments_dir.is_dir():
        return items

    for article_dir in sorted(ace_comments_dir.iterdir()):
        if not article_dir.is_dir():
            continue
        if article_id is not None and article_dir.name != str(article_id):
            continue
        current_article_id = article_dir.name
        for json_path in sorted(article_dir.glob("*.json")):
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"Skipping {json_path}: {e}")
                continue
            comment_id = data.get("comment_index")
            if comment_id is None:
                comment_id = int(json_path.stem) if json_path.stem.isdigit() else 0
            ace_sentences = data.get("ace_sentences") or []
            for sent in ace_sentences:
                s = (sent if isinstance(sent, str) else str(sent)).strip()
                if not s:
                    continue
                items.append({
                    "article_id": str(current_article_id),
                    "comment_id": comment_id,
                    "original_comment": s,
                })
    return items


def load_prompt(prompt_path: Path) -> str:
    """Load the classification prompt from a text file."""
    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def classify_batch(
    client: OpenAI,
    prompt_text: str,
    batch: List[Dict[str, Any]],
    model: str,
) -> List[Dict[str, Any]]:
    """Send one batch of items to the model and return classifications."""
    user_content = (
        prompt_text
        + "\n\n---\n\nInput sentences to classify (use these exact fields in your response):\n"
        + json.dumps(batch, indent=2, ensure_ascii=False)
    )
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You classify ACE sentences into the given categories. Respond only with valid JSON."},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    elapsed = time.perf_counter() - start_time
    print(
        f"API call for batch of {len(batch)} sentences took {elapsed:.2f} seconds / ({elapsed/60:.2f} minutes)."
    )
    raw = response.choices[0].message.content
    try:
        out = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON: {e}\nRaw: {raw[:500]}...")
    classifications = out.get("classifications")
    if not isinstance(classifications, list):
        raise ValueError(f"Model response missing 'classifications' list: {list(out.keys())}")
    return classifications


def run_classification(
    ace_comments_dir: Path,
    prompt_path: Path,
    output_path: Path,
    api_key: str | None = None,
    model: str = "gpt-5.2",
    batch_size: int = BATCH_SIZE,
    intermediate_dir: Path | None = DEFAULT_INTERMEDIATE_DIR,
    article_id: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Load ACE items, run classification in batches, merge results, and save JSON.
    Returns the full list of { article_id, comment_id, original_comment, comment_tag }.
    If article_id is provided, only classify sentences for that article_id.
    """
    items = load_ace_comment_items(ace_comments_dir, article_id=article_id)
    if not items:
        print("No ACE sentences found. Nothing to classify.")
        return []

    if article_id is not None:
        print(
            f"Loaded {len(items)} ACE sentences from {ace_comments_dir} for article_id={article_id}"
        )
    else:
        print(f"Loaded {len(items)} ACE sentences from {ace_comments_dir}")

    prompt_text = load_prompt(prompt_path)
    client = _ensure_client(api_key)

    all_classifications: List[Dict[str, Any]] = []

    if intermediate_dir is not None:
        intermediate_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_index = i // batch_size
        batch_num = batch_index + 1
        print(f"Classifying batch {batch_num} ({len(batch)} sentences)...")
        classifications = classify_batch(client, prompt_text, batch, model)

        batch_rows: List[Dict[str, Any]] = []
        for c in classifications:
            if not isinstance(c, dict):
                continue
            row = {
                "article_id": c.get("article_id", ""),
                "comment_id": c.get("comment_id", 0),
                "original_comment": c.get("original_comment", ""),
                "comment_tag": c.get("comment_tag", ""),
            }
            all_classifications.append(row)
            batch_rows.append(row)

        if intermediate_dir is not None:
            batch_path = intermediate_dir / f"batch_{batch_num:04d}.json"
            with batch_path.open("w", encoding="utf-8") as f:
                json.dump(batch_rows, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(batch_rows)} classifications to {batch_path}")

    # Decide where to save the final JSON.
    # If we're classifying a single article and the caller is using the default
    # output path, write into ace_classifications/ with the article_id in the name.
    if article_id is not None and output_path == DEFAULT_OUTPUT_JSON:
        final_output_dir = DEFAULT_ACE_CLASSIFICATIONS_DIR
        final_output_dir.mkdir(parents=True, exist_ok=True)
        final_output_path = final_output_dir / f"ace_sentence_classifications_{article_id}.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_output_path = output_path

    with final_output_path.open("w", encoding="utf-8") as f:
        json.dump(all_classifications, f, indent=2, ensure_ascii=False)
    print(f"Saved combined {len(all_classifications)} classifications to {final_output_path}")
    return all_classifications


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify ACE sentences from ace_comments/ into semantic categories."
    )
    parser.add_argument(
        "--ace-comments-dir",
        type=Path,
        default=DEFAULT_ACE_COMMENTS_DIR,
        help=f"Directory containing per-article ACE JSONs (default: {DEFAULT_ACE_COMMENTS_DIR})",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help=f"Path to classification prompt .txt (default: {DEFAULT_PROMPT_FILE})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=DEFAULT_INTERMEDIATE_DIR,
        help=(
            "Directory to store intermediate per-batch classification JSON files "
            f"(default: {DEFAULT_INTERMEDIATE_DIR})"
        ),
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
        help="OpenAI model for classification (default: gpt-5.2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Max sentences per API call (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--article-id",
        type=str,
        default=None,
        help="If set, only classify ACE sentences for this article_id subdirectory.",
    )
    args = parser.parse_args()

    run_classification(
        ace_comments_dir=args.ace_comments_dir,
        prompt_path=args.prompt_file,
        output_path=args.output,
        api_key=args.api_key,
        model=args.model,
        batch_size=args.batch_size,
        intermediate_dir=args.intermediate_dir,
        article_id=args.article_id,
    )


if __name__ == "__main__":
    main()
