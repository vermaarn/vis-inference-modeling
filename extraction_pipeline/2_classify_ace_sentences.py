"""
Classify ACE sentences from extracted ACE comments into semantic categories.

Reads ACE comment JSONs produced by 1_extract_ace_comments.py (under ace_comments/),
loads the classification prompt from prompts/classify_ace_sentences.txt, sends all
ACE sentences to an LLM in one or more batches, and writes a JSON file with
article_id, comment_id, original_comment, and comment_tag per sentence.
"""

from __future__ import annotations

import base64
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
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "images"

# Max sentences per API call to stay within context limits
BATCH_SIZE = 500

IMAGE_ANALYSIS_PROMPT = """Analyze this data visualization image. Provide a concise, structured description covering:

1. **Chart type**: Identify the chart type (e.g., stacked area chart, faceted bar chart, line chart, scatter plot, etc.).
2. **Layout techniques**: Explain whether the chart uses faceting, stacking, small multiples, or other layout techniques.
3. **Data variables**: List all data variables shown in the visualization. For each variable, provide:
   - The data variable name
   - The type of variable (temporal, quantitative, categorical/nominal, ordinal)
   - The visual encoding used (e.g., position on x-axis, position on y-axis, color/hue, area/size, faceting, label, shape)

Present the variables in a table with columns: Data Variable | Type of Variable | Visual Encoding.

Keep the explanation concise and structured. Do not speculate beyond what is visible in the image."""


def _ensure_client(api_key: str | None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY in .env or pass --api-key."
        )
    return OpenAI(api_key=key)


def _encode_image_base64(image_path: Path) -> str:
    """Read an image file and return its base64-encoded string."""
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_visualization(
    client: OpenAI,
    article_id: str,
    images_dir: Path = DEFAULT_IMAGES_DIR,
    model: str = "gpt-5.2",
) -> str:
    """
    Send the visualization image for *article_id* to the vision model and
    return a structured text description (chart type, variables, encodings).
    Returns an empty string if the image file is not found.
    """
    image_path = images_dir / f"{article_id}.png"
    if not image_path.is_file():
        print(f"Warning: visualization image not found at {image_path}")
        return ""

    b64 = _encode_image_base64(image_path)
    print(f"Describing visualization image for article {article_id} …")
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_ANALYSIS_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        temperature=0.1,
        max_completion_tokens=1024,
    )
    elapsed = time.perf_counter() - start
    description = response.choices[0].message.content.strip()
    print(
        f"Image description for article {article_id} obtained in {elapsed:.2f}s "
        f"({len(description)} chars)."
    )
    return description


def load_ace_comment_items(
    ace_comments_dir: Path,
    article_id: str | None = None,
    article_id_folder: Path | None = None,
) -> List[Dict[str, Any]]:
    """
    Load all ACE comment JSON files and return a flat list of items:
    { article_id, comment_id, original_comment } for each ACE sentence.

    If article_id_folder is set, load only from that directory (single-article mode).
    Else if article_id is set, load only from ace_comments_dir / article_id.
    Otherwise load from all subdirs of ace_comments_dir.
    """
    items: List[Dict[str, Any]] = []

    if article_id_folder is not None:
        # Single folder specified (e.g. via --article-id /path/to/folder)
        if not article_id_folder.is_dir():
            return items
        dirs_to_scan = [(article_id_folder, article_id_folder.name)]
    else:
        if not ace_comments_dir.is_dir():
            return items
        if article_id is not None:
            # Restrict to ace_comments_dir / article_id
            subdir = ace_comments_dir / str(article_id)
            if not subdir.is_dir():
                return items
            dirs_to_scan = [(subdir, str(article_id))]
        else:
            dirs_to_scan = [
                (d, d.name) for d in sorted(ace_comments_dir.iterdir()) if d.is_dir()
            ]

    for article_dir, current_article_id in dirs_to_scan:
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


def _build_image_info_section(
    descriptions: Dict[str, str],
    article_ids: set[str],
) -> str:
    """
    Format image descriptions for the article_ids present in a batch into a
    text block suitable for injecting into the classification prompt.
    """
    sections: List[str] = []
    for aid in sorted(article_ids):
        desc = descriptions.get(aid, "")
        if desc:
            sections.append(f"[Article {aid}]\n{desc}")
    if not sections:
        return "(No visualization description available.)"
    return "\n\n".join(sections)


def run_classification(
    ace_comments_dir: Path,
    prompt_path: Path,
    output_path: Path,
    api_key: str | None = None,
    model: str = "gpt-5.2",
    batch_size: int = BATCH_SIZE,
    intermediate_dir: Path | None = DEFAULT_INTERMEDIATE_DIR,
    article_id: str | None = None,
    article_id_folder: Path | None = None,
    images_dir: Path = DEFAULT_IMAGES_DIR,
) -> List[Dict[str, Any]]:
    """
    Load ACE items, run classification in batches, merge results, and save JSON.
    Returns the full list of { article_id, comment_id, original_comment, comment_tag }.
    If article_id_folder is set, load only from that directory. Else if article_id
    is set, load only from ace_comments_dir / article_id.
    """
    items = load_ace_comment_items(
        ace_comments_dir,
        article_id=article_id,
        article_id_folder=article_id_folder,
    )
    if not items:
        print("No ACE sentences found. Nothing to classify.")
        return []

    effective_article_id = article_id
    if article_id_folder is not None:
        effective_article_id = article_id_folder.name
        print(
            f"Loaded {len(items)} ACE sentences from {article_id_folder} (article_id={effective_article_id})"
        )
    elif article_id is not None:
        print(
            f"Loaded {len(items)} ACE sentences from {ace_comments_dir} for article_id={article_id}"
        )
    else:
        print(f"Loaded {len(items)} ACE sentences from {ace_comments_dir}")

    prompt_template = load_prompt(prompt_path)
    client = _ensure_client(api_key)

    # --- Gather image descriptions for all article_ids present in the data ---
    unique_article_ids = {item["article_id"] for item in items}
    image_descriptions: Dict[str, str] = {}
    for aid in sorted(unique_article_ids):
        desc = describe_visualization(client, aid, images_dir=images_dir, model=model)
        image_descriptions[aid] = desc

    all_classifications: List[Dict[str, Any]] = []

    if intermediate_dir is not None:
        intermediate_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_index = i // batch_size
        batch_num = batch_index + 1

        batch_article_ids = {item["article_id"] for item in batch}
        image_info = _build_image_info_section(image_descriptions, batch_article_ids)
        prompt_text = prompt_template.replace("<IMAGE_INFORMATION>", image_info)

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
    if effective_article_id is not None and output_path == DEFAULT_OUTPUT_JSON:
        final_output_dir = DEFAULT_ACE_CLASSIFICATIONS_DIR
        final_output_dir.mkdir(parents=True, exist_ok=True)
        final_output_path = final_output_dir / f"ace_sentence_classifications_{effective_article_id}.json"
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
        help=(
            "If set, only classify ACE sentences for this article. Can be an article_id "
            "(then reads from <ace-comments-dir>/<article-id>/) or a path to a folder "
            "containing the ACE comment JSONs (folder name is used as article_id for output)."
        ),
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help=f"Directory containing visualization PNGs named {{article_id}}.png (default: {DEFAULT_IMAGES_DIR})",
    )
    args = parser.parse_args()

    # If --article-id looks like a path to an existing directory, use it as the ace_comments folder
    article_id = args.article_id
    article_id_folder = None
    if article_id is not None:
        as_path = Path(article_id)
        is_path_like = os.path.isabs(article_id) or os.path.sep in article_id or (os.path.altsep and os.path.altsep in article_id)
        if is_path_like and as_path.is_dir():
            article_id_folder = as_path.resolve()
            article_id = None  # article_id derived from folder name inside run_classification
        else:
            article_id_folder = None

    run_classification(
        ace_comments_dir=args.ace_comments_dir,
        prompt_path=args.prompt_file,
        output_path=args.output,
        api_key=args.api_key,
        model=args.model,
        batch_size=args.batch_size,
        intermediate_dir=args.intermediate_dir,
        article_id=article_id,
        article_id_folder=article_id_folder,
        images_dir=args.images_dir,
    )


if __name__ == "__main__":
    main()
