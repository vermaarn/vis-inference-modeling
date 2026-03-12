"""
Utility script to run only the ACE sentence extraction stage for NYT article
comments, saving results into the ``ace_comments`` folder.

Contains inlined ACE conversion logic (no dependency on extraction_pipeline
module). Does not run visualization tagging or clustering.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import argparse
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Resolve the default comment data directory relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTICLES_DATA_DIR = PROJECT_ROOT / "data" / "comment_data"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS_DIR = SCRIPT_DIR / "prompts"
DEFAULT_ACE_PROMPT_FILE = DEFAULT_PROMPTS_DIR / "ace_comment_to_sentences.txt"


# --- ACE extraction (inlined from explorer/public/extraction_pipeline/extraction_pipeline.py) ---


def read_json_file(json_path: str) -> List[Dict[str, Any]]:
    """Read and parse JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


ACE_FEW_SHOT_COMMENT = """Comment:
1. The rate of wind and solar have been becoming higher since 2000, but fossil fuels occupies more than half of materials. In fact, the situation is seen as a problem, so the governments of various countries are promoting renewable energy. I had an opportunity to think about a solution to improve the situations when I was a junior high school student.
2. I wonder why the electric power generation in the world decreased in around 2008 and 2021. I think that the Lehman shock in 2008 and the spreading of coronavirus relate to the result. Indeed, the economy was in a slump because of them, so demand of electric power generation decreased.
3. If the rate of fossil fuels remains steady, the global warming and the air pollution will worsen. So, the precious natural environment in Akita, where I live, could be negatively affected. For instance, the growth of Akita cedars may deteriorate.
4. Catchy headline : What is the Solution to Increase the Rate of Clean Energy?
"""

ACE_FEW_SHOT_OUTPUT = """{
  "ace_sentences": [
    "The rate of wind energy and solar energy has increased since 2000.",
    "However, fossil fuels occupy more than half of energy materials.",
    "This situation is a problem.",
    "The governments of many countries promote renewable energy.",
    "When I was a junior high school student, I had an opportunity.",
    "At that time, I thought about a solution to improve this situation.",
    "The electric power generation in the world decreased around 2008.",
    "The electric power generation in the world also decreased around 2021.",
    "The Lehman Shock happened in 2008.",
    "The coronavirus spread in 2021.",
    "The Lehman Shock caused an economic slump.",
    "The spread of coronavirus caused an economic slump.",
    "An economic slump reduces the demand for electric power.",
    "The decrease in electric power generation relates to these events.",
    "The rate of fossil fuels may remain steady.",
    "If the rate of fossil fuels remains steady, global warming will worsen.",
    "If the rate of fossil fuels remains steady, air pollution will worsen.",
    "Akita has a precious natural environment.",
    "I live in Akita.",
    "If global warming worsens, the natural environment in Akita will be negatively affected.",
    "Akita cedars grow in Akita.",
    "If global warming worsens, the growth of Akita cedars may deteriorate.",
    "The headline is:",
    "What is the solution to increase the rate of clean energy?"
  ],
  "source_mappings": {
    "The rate of wind energy and solar energy has increased since 2000.": ["The rate of wind and solar have been becoming higher since 2000"],
    "However, fossil fuels occupy more than half of energy materials.": ["but fossil fuels occupies more than half of materials"],
    "This situation is a problem.": ["the situation is seen as a problem"],
    "The governments of many countries promote renewable energy.": ["the governments of various countries are promoting renewable energy"],
    "When I was a junior high school student, I had an opportunity.": ["I had an opportunity", "when I was a junior high school student"],
    "At that time, I thought about a solution to improve this situation.": ["to think about a solution to improve the situations"],
    "The electric power generation in the world decreased around 2008.": ["the electric power generation in the world decreased in around", "2008"],
    "The electric power generation in the world also decreased around 2021.": ["the electric power generation in the world decreased in around", "2021"],
    "The Lehman Shock happened in 2008.": ["the Lehman shock in 2008"],
    "The coronavirus spread in 2021.": ["the spreading of coronavirus"m "2008"],
    "The Lehman Shock caused an economic slump.": ["the Lehman shock", "the economy was in a slump because of them"],
    "The spread of coronavirus caused an economic slump.": ["the economy was in a slump because of them"],
    "An economic slump reduces the demand for electric power.": ["demand of electric power generation decreased"],
    "The decrease in electric power generation relates to these events.": ["I wonder why the electric power generation in the world decreased in around 2008 and 2021", "I think that the Lehman shock in 2008 and the spreading of coronavirus relate to the result"],
    "The rate of fossil fuels may remain steady.": ["If the rate of fossil fuels remains steady"],
    "If the rate of fossil fuels remains steady, global warming will worsen.": ["the global warming", "will worsen"],
    "If the rate of fossil fuels remains steady, air pollution will worsen.": ["the air pollution will worsen"],
    "Akita has a precious natural environment.": ["the precious natural environment in Akita"],
    "I live in Akita.": ["where I live"],
    "If global warming worsens, the natural environment in Akita will be negatively affected.": ["the precious natural environment in Akita, where I live, could be negatively affected"],
    "Akita cedars grow in Akita.": ["the growth of Akita cedars"],
    "If global warming worsens, the growth of Akita cedars may deteriorate.": ["the growth of Akita cedars may deteriorate"],
    "The headline is:": ["Catchy headline"],
    "What is the solution to increase the rate of clean energy?": ["What is the Solution to Increase the Rate of Clean Energy?"]
  }
}
"""


def load_ace_prompt_template(prompt_path: Path = DEFAULT_ACE_PROMPT_FILE) -> str:
    """Load the ACE conversion prompt template from a text file."""
    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read()


def _ensure_client(api_key: str | None) -> OpenAI:
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
            "in a .env file or pass as argument."
        )
    return OpenAI(api_key=api_key)


def generate_ace_for_comment(
    comment_text: str,
    client: OpenAI,
    model: str = "gpt-5.2",
) -> Dict[str, List[str]]:
    """
    Convert a free-form article comment into Attempto Controlled English (ACE) sentences.

    Returns a dict with ``ace_sentences`` and ``source_mappings`` lists.
    """
    if not comment_text.strip():
        return {"ace_sentences": [], "source_mappings": {}}

    system_content = (
        "You are an expert in Attempto Controlled English (ACE). "
        "Given an article comment, you rewrite it as a sequence of simple ACE sentences. "
        "Avoid rhetorical flourishes, and express each idea as its own clear sentence."
    )

    # Load prompt template from disk and substitute placeholders.
    template = load_ace_prompt_template()
    user_instructions = (
        template.replace("<ACE_FEW_SHOT_COMMENT>", ACE_FEW_SHOT_COMMENT.strip())
        .replace("<ACE_FEW_SHOT_OUTPUT>", ACE_FEW_SHOT_OUTPUT.strip())
        .replace("<COMMENT_TEXT>", comment_text.strip())
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_instructions},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    content = response.choices[0].message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse ACE JSON for comment: {exc}\nRaw: {content}")

    ace_sentences = parsed.get("ace_sentences") or []
    source_mappings = parsed.get("source_mappings") or {}
    cleaned_sentences = [str(s).strip() for s in ace_sentences if str(s).strip()]
    cleaned_mappings = {
        str(k).strip(): [str(v).strip() for v in vals]
        for k, vals in source_mappings.items()
    }
    return {"ace_sentences": cleaned_sentences, "source_mappings": cleaned_mappings}


def extract_ace_for_article(
    article_id: str,
    articles_data_dir: str = str(DEFAULT_ARTICLES_DATA_DIR),
    ace_comments_base_dir: str = "ace_comments",
    api_key: str | None = None,
    model: str = "gpt-5.2",
) -> Dict[str, Any]:
    """
    Run ACE extraction (stage 1) for a single article's comments.

    For each top-level comment in the article JSON:
      - Convert the raw comment text to ACE sentences.
      - Save the result under:
            {ace_comments_base_dir}/{article_id}/{comment_index}.json
    """
    json_path = os.path.join(articles_data_dir, f"{article_id}.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    print(f"Reading {json_path}...")
    comments_data: List[Dict[str, Any]] = read_json_file(json_path)

    client = _ensure_client(api_key)

    ace_article_dir = os.path.join(ace_comments_base_dir, str(article_id))
    os.makedirs(ace_article_dir, exist_ok=True)

    processed_comments: List[Dict[str, Any]] = []

    for idx, comment in enumerate(comments_data, start=1):
        raw_comment = str(comment.get("comment info", "") or "").strip()
        if not raw_comment:
            print(f"Skipping empty comment {idx} for article {article_id}.")
            continue

        print(f"Processing comment {idx} for article {article_id} (ACE conversion only)...")
        ace_result = generate_ace_for_comment(
            raw_comment,
            client=client,
            model=model,
        )

        ace_output: Dict[str, Any] = {
            "article_id": article_id,
            "comment_index": idx,
            "raw_comment": raw_comment,
            "ace_sentences": ace_result["ace_sentences"],
            "source_mappings": ace_result["source_mappings"],
        }

        ace_output_path = os.path.join(ace_article_dir, f"{idx}.json")
        with open(ace_output_path, "w", encoding="utf-8") as ace_f:
            json.dump(ace_output, ace_f, indent=2, ensure_ascii=False)

        processed_comments.append(ace_output)

    print(f"✓ Finished ACE extraction for article {article_id}")

    return {
        "article_id": article_id,
        "source_file": json_path,
        "ace_comments_dir": ace_article_dir,
        "comments": processed_comments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ACE sentences for NYT article comments and save them to the "
            "ace_comments folder (stage 1 only, no visualization tagging)."
        )
    )
    parser.add_argument(
        "--article_id",
        type=str,
        help="Article ID (e.g., '38'). If omitted, use --all to process every JSON in the data directory.",
    )
    parser.add_argument(
        "--articles-data-dir",
        type=str,
        default=str(DEFAULT_ARTICLES_DATA_DIR),
        help="Directory containing article JSON files",
    )
    parser.add_argument(
        "--ace-comments-dir",
        type=str,
        default="ace_comments",
        help="Base directory where per-article ACE JSON files are written (default: ace_comments)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="OpenAI model to use for ACE extraction (default: gpt-5.2)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all article JSON files found in --articles-data-dir",
    )

    args = parser.parse_args()

    if not args.all and not args.article_id:
        parser.error("You must specify either --article_id or --all.")

    results: List[Dict[str, Any]] = []

    if args.all:
        data_path = Path(args.articles_data_dir)
        article_ids = [p.stem for p in data_path.glob("*.json")]
        print(f"Found {len(article_ids)} articles to process for ACE extraction.")

        for article_id in sorted(article_ids):
            try:
                result = extract_ace_for_article(
                    article_id=article_id,
                    articles_data_dir=args.articles_data_dir,
                    ace_comments_base_dir=args.ace_comments_dir,
                    api_key=args.api_key,
                    model=args.model,
                )
                results.append(result)
            except Exception as e:
                print(f"✗ Error extracting ACE for article {article_id}: {e}")
    else:
        result = extract_ace_for_article(
            article_id=args.article_id,
            articles_data_dir=args.articles_data_dir,
            ace_comments_base_dir=args.ace_comments_dir,
            api_key=args.api_key,
            model=args.model,
        )
        results.append(result)


if __name__ == "__main__":
    main()

