"""
Build dependency graphs (DAGs) with typed edges over ACE sentences in each comment.

Reads ACE comment JSONs from ace_comments/{article_id}/, sends each comment's
ace_sentences to an LLM with the dependency_classification prompt, and writes one
JSON per comment to ace_dependency_graphs/{article_id}/{comment_index}.json with
article_id, comment_index, and dependency_graph (nodes with id, sentence,
depends_on list of {id, edge_type}).

Edge types: Causal, Elaboration, Conditional, Evaluative, Questioning,
Contrastive, Narrative/Referential, Uncategorized.

Usage:
  python 3_dependency_classification.py --article-id 181
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

VALID_EDGE_TYPES = frozenset([
    "Causal",
    "Elaboration",
    "Conditional",
    "Evaluative",
    "Questioning",
    "Contrastive",
    "Narrative/Referential",
    "Uncategorized",
])

# ---------------------------------------------------------------------------
# Few-shot examples injected into the prompt at <DEPENDENCY_EDGE_EXAMPLES>
# ---------------------------------------------------------------------------

_EDGE_EXAMPLES: List[Dict[str, Any]] = [
    {
        "input": {
            "article_id": "181",
            "comment_index": 1,
            "ace_sentences": [
                "The rate of wind energy and solar energy has increased since 2000.",
                "However, fossil fuels occupy more than half of energy materials.",
                "This situation is a problem.",
                "The governments of various countries promote renewable energy.",
                "When I was a junior high school student, I had an opportunity.",
                "At that time, I thought about a solution to improve this situation.",
                "The electric power generation in the world decreased around 2008.",
                "The electric power generation in the world also decreased around 2021.",
                "The Lehman Shock happened in 2008.",
                "The coronavirus spread in 2021.",
                "The Lehman Shock caused an economic slump.",
                "The spread of coronavirus caused an economic slump.",
                "An economic slump reduces the demand for electric power generation.",
                "The demand for electric power generation decreased.",
                "The decrease in electric power generation relates to these events.",
                "The rate of fossil fuels may remain steady.",
                "If the rate of fossil fuels remains steady then global warming will worsen.",
                "If the rate of fossil fuels remains steady then air pollution will worsen.",
                "Akita has a precious natural environment.",
                "I live in Akita.",
                "If global warming worsens then the natural environment in Akita could be negatively affected.",
                "Akita cedars grow in Akita.",
                "If global warming worsens then the growth of Akita cedars may deteriorate.",
                "The headline is: What is the solution to increase the rate of clean energy?",
            ],
            "order": {
                "The rate of wind energy and solar energy has increased since 2000.": 1,
                "However, fossil fuels occupy more than half of energy materials.": 2,
                "This situation is a problem.": 3,
                "The governments of various countries promote renewable energy.": 4,
                "When I was a junior high school student, I had an opportunity.": 5,
                "At that time, I thought about a solution to improve this situation.": 6,
                "The electric power generation in the world decreased around 2008.": 7,
                "The electric power generation in the world also decreased around 2021.": 8,
                "The Lehman Shock happened in 2008.": 9,
                "The coronavirus spread in 2021.": 10,
                "The Lehman Shock caused an economic slump.": 11,
                "The spread of coronavirus caused an economic slump.": 12,
                "An economic slump reduces the demand for electric power generation.": 13,
                "The demand for electric power generation decreased.": 14,
                "The decrease in electric power generation relates to these events.": 15,
                "The rate of fossil fuels may remain steady.": 16,
                "If the rate of fossil fuels remains steady then global warming will worsen.": 17,
                "If the rate of fossil fuels remains steady then air pollution will worsen.": 18,
                "Akita has a precious natural environment.": 19,
                "I live in Akita.": 20,
                "If global warming worsens then the natural environment in Akita could be negatively affected.": 21,
                "Akita cedars grow in Akita.": 22,
                "If global warming worsens then the growth of Akita cedars may deteriorate.": 23,
                "The headline is: What is the solution to increase the rate of clean energy?": 24,
            },
        },
        "output": {
            "article_id": "181",
            "comment_index": 1,
            "dependency_graph": [
                {"id": 0, "sentence": "The rate of wind energy and solar energy has increased since 2000.", "depends_on": []},
                {"id": 1, "sentence": "However, fossil fuels occupy more than half of energy materials.", "depends_on": []},
                {"id": 2, "sentence": "This situation is a problem.", "depends_on": [
                    {"id": 1, "edge_type": "Evaluative"},
                ]},
                {"id": 3, "sentence": "The governments of various countries promote renewable energy.", "depends_on": [
                    {"id": 0, "edge_type": "Causal"},
                ]},
                {"id": 4, "sentence": "When I was a junior high school student, I had an opportunity.", "depends_on": []},
                {"id": 5, "sentence": "At that time, I thought about a solution to improve this situation.", "depends_on": [
                    {"id": 2, "edge_type": "Narrative/Referential"},
                    {"id": 4, "edge_type": "Narrative/Referential"},
                ]},
                {"id": 6, "sentence": "The electric power generation in the world decreased around 2008.", "depends_on": []},
                {"id": 7, "sentence": "The electric power generation in the world also decreased around 2021.", "depends_on": []},
                {"id": 8, "sentence": "The Lehman Shock happened in 2008.", "depends_on": [
                    {"id": 6, "edge_type": "Causal"},
                ]},
                {"id": 9, "sentence": "The coronavirus spread in 2021.", "depends_on": [
                    {"id": 7, "edge_type": "Causal"},
                ]},
                {"id": 10, "sentence": "The Lehman Shock caused an economic slump.", "depends_on": [
                    {"id": 8, "edge_type": "Causal"},
                ]},
                {"id": 11, "sentence": "The spread of coronavirus caused an economic slump.", "depends_on": [
                    {"id": 9, "edge_type": "Causal"},
                ]},
                {"id": 12, "sentence": "An economic slump reduces the demand for electric power generation.", "depends_on": [
                    {"id": 10, "edge_type": "Causal"},
                    {"id": 11, "edge_type": "Causal"},
                ]},
                {"id": 13, "sentence": "The demand for electric power generation decreased.", "depends_on": [
                    {"id": 12, "edge_type": "Causal"},
                ]},
                {"id": 14, "sentence": "The decrease in electric power generation relates to these events.", "depends_on": [
                    {"id": 6, "edge_type": "Narrative/Referential"},
                    {"id": 7, "edge_type": "Narrative/Referential"},
                    {"id": 13, "edge_type": "Causal"},
                ]},
                {"id": 15, "sentence": "The rate of fossil fuels may remain steady.", "depends_on": [
                    {"id": 1, "edge_type": "Elaboration"},
                ]},
                {"id": 16, "sentence": "If the rate of fossil fuels remains steady then global warming will worsen.", "depends_on": [
                    {"id": 15, "edge_type": "Conditional"},
                ]},
                {"id": 17, "sentence": "If the rate of fossil fuels remains steady then air pollution will worsen.", "depends_on": [
                    {"id": 15, "edge_type": "Conditional"},
                ]},
                {"id": 18, "sentence": "Akita has a precious natural environment.", "depends_on": []},
                {"id": 19, "sentence": "I live in Akita.", "depends_on": []},
                {"id": 20, "sentence": "If global warming worsens then the natural environment in Akita could be negatively affected.", "depends_on": [
                    {"id": 16, "edge_type": "Conditional"},
                    {"id": 18, "edge_type": "Elaboration"},
                    {"id": 19, "edge_type": "Narrative/Referential"},
                ]},
                {"id": 21, "sentence": "Akita cedars grow in Akita.", "depends_on": [
                    {"id": 19, "edge_type": "Narrative/Referential"},
                ]},
                {"id": 22, "sentence": "If global warming worsens then the growth of Akita cedars may deteriorate.", "depends_on": [
                    {"id": 16, "edge_type": "Conditional"},
                    {"id": 21, "edge_type": "Elaboration"},
                ]},
                {"id": 23, "sentence": "The headline is: What is the solution to increase the rate of clean energy?", "depends_on": []},
            ],
        },
    },
    {
        "input": {
            "article_id": "181",
            "comment_index": 4,
            "ace_sentences": [
                "Electricity generation has increased over time.",
                "The use of clean energy sources in the world has grown during the last 20 years.",
                "The use of fossil fuels has been consistently higher than the use of clean energy sources.",
                "Coal is a popular energy source.",
                "Gas is a popular energy source.",
                "Wind power has experienced much growth recently.",
                "Solar power has experienced much growth recently.",
                "New ways to generate electricity have proliferated.",
                "The use of gas has steadily increased.",
                "The use of coal has steadily increased.",
                "I wonder why the use of gas and coal has steadily increased.",
                "The accessibility of resources may explain the increase.",
                "The global population has grown.",
                "I wonder how much global population growth affects how electricity is generated.",
                "Statistics about electricity generation vary from country to country.",
                "I wonder how greatly these statistics vary from country to country.",
                "I canvassed for a local democratic campaign.",
                "During the canvassing, I noticed a fact.",
                "The fact is that clean energy was a very important issue.",
                "Everyone is impacted by the negative effects of climate change.",
                "The expenditure of fossil fuels contributes to global warming.",
                "It has been proven that the expenditure of fossil fuels greatly contributes to global warming.",
                "The headline is: Good Ole' Coal: It is electric!",
            ],
            "order": {
                "Electricity generation has increased over time.": 1,
                "The use of clean energy sources in the world has grown during the last 20 years.": 2,
                "The use of fossil fuels has been consistently higher than the use of clean energy sources.": 3,
                "Coal is a popular energy source.": 4,
                "Gas is a popular energy source.": 5,
                "Wind power has experienced much growth recently.": 6,
                "Solar power has experienced much growth recently.": 7,
                "New ways to generate electricity have proliferated.": 8,
                "The use of gas has steadily increased.": 9,
                "The use of coal has steadily increased.": 10,
                "I wonder why the use of gas and coal has steadily increased.": 11,
                "The accessibility of resources may explain the increase.": 12,
                "The global population has grown.": 13,
                "I wonder how much global population growth affects how electricity is generated.": 14,
                "Statistics about electricity generation vary from country to country.": 15,
                "I wonder how greatly these statistics vary from country to country.": 16,
                "I canvassed for a local democratic campaign.": 17,
                "During the canvassing, I noticed a fact.": 18,
                "The fact is that clean energy was a very important issue.": 19,
                "Everyone is impacted by the negative effects of climate change.": 20,
                "The expenditure of fossil fuels contributes to global warming.": 21,
                "It has been proven that the expenditure of fossil fuels greatly contributes to global warming.": 22,
                "The headline is: Good Ole' Coal: It is electric!": 23,
            },
        },
        "output": {
            "article_id": "181",
            "comment_index": 4,
            "dependency_graph": [
                {"id": 0, "sentence": "Electricity generation has increased over time.", "depends_on": []},
                {"id": 1, "sentence": "The use of clean energy sources in the world has grown during the last 20 years.", "depends_on": []},
                {"id": 2, "sentence": "The use of fossil fuels has been consistently higher than the use of clean energy sources.", "depends_on": [
                    {"id": 1, "edge_type": "Contrastive"},
                ]},
                {"id": 3, "sentence": "Coal is a popular energy source.", "depends_on": []},
                {"id": 4, "sentence": "Gas is a popular energy source.", "depends_on": []},
                {"id": 5, "sentence": "Wind power has experienced much growth recently.", "depends_on": [
                    {"id": 1, "edge_type": "Elaboration"},
                ]},
                {"id": 6, "sentence": "Solar power has experienced much growth recently.", "depends_on": [
                    {"id": 1, "edge_type": "Elaboration"},
                ]},
                {"id": 7, "sentence": "New ways to generate electricity have proliferated.", "depends_on": [
                    {"id": 5, "edge_type": "Elaboration"},
                    {"id": 6, "edge_type": "Elaboration"},
                ]},
                {"id": 8, "sentence": "The use of gas has steadily increased.", "depends_on": [
                    {"id": 4, "edge_type": "Elaboration"},
                ]},
                {"id": 9, "sentence": "The use of coal has steadily increased.", "depends_on": [
                    {"id": 3, "edge_type": "Elaboration"},
                ]},
                {"id": 10, "sentence": "I wonder why the use of gas and coal has steadily increased.", "depends_on": [
                    {"id": 7, "edge_type": "Contrastive"},
                    {"id": 8, "edge_type": "Questioning"},
                    {"id": 9, "edge_type": "Questioning"},
                ]},
                {"id": 11, "sentence": "The accessibility of resources may explain the increase.", "depends_on": [
                    {"id": 10, "edge_type": "Causal"},
                ]},
                {"id": 12, "sentence": "The global population has grown.", "depends_on": []},
                {"id": 13, "sentence": "I wonder how much global population growth affects how electricity is generated.", "depends_on": [
                    {"id": 0, "edge_type": "Questioning"},
                    {"id": 12, "edge_type": "Questioning"},
                ]},
                {"id": 14, "sentence": "Statistics about electricity generation vary from country to country.", "depends_on": []},
                {"id": 15, "sentence": "I wonder how greatly these statistics vary from country to country.", "depends_on": [
                    {"id": 14, "edge_type": "Questioning"},
                ]},
                {"id": 16, "sentence": "I canvassed for a local democratic campaign.", "depends_on": []},
                {"id": 17, "sentence": "During the canvassing, I noticed a fact.", "depends_on": [
                    {"id": 16, "edge_type": "Narrative/Referential"},
                ]},
                {"id": 18, "sentence": "The fact is that clean energy was a very important issue.", "depends_on": [
                    {"id": 17, "edge_type": "Elaboration"},
                ]},
                {"id": 19, "sentence": "Everyone is impacted by the negative effects of climate change.", "depends_on": []},
                {"id": 20, "sentence": "The expenditure of fossil fuels contributes to global warming.", "depends_on": [
                    {"id": 2, "edge_type": "Causal"},
                    {"id": 19, "edge_type": "Causal"},
                ]},
                {"id": 21, "sentence": "It has been proven that the expenditure of fossil fuels greatly contributes to global warming.", "depends_on": [
                    {"id": 20, "edge_type": "Elaboration"},
                ]},
                {"id": 22, "sentence": "The headline is: Good Ole' Coal: It is electric!", "depends_on": []},
            ],
        },
    },
]


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
    comment_index: int | None = None,
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
    paths = sorted(paths, key=key)

    if comment_index is not None:
        target = str(comment_index)
        paths = [p for p in paths if p.stem == target]

    return paths


def _format_edge_examples() -> str:
    """Render _EDGE_EXAMPLES as Markdown suitable for injection into the prompt."""
    sections: List[str] = []
    for i, ex in enumerate(_EDGE_EXAMPLES, 1):
        inp = json.dumps(ex["input"], indent=2, ensure_ascii=False)
        out = json.dumps(ex["output"], indent=2, ensure_ascii=False)
        sections.append(
            f"### Example {i}\n\n"
            f"**Input:**\n```json\n{inp}\n```\n\n"
            f"**Output:**\n```json\n{out}\n```"
        )
    return "\n\n".join(sections)


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
        "order": comment_data.get("order") or {},
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
                "content": "You build dependency graphs with typed edges over ACE sentences. Respond only with valid JSON in the required shape.",
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
    comment_index: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Load all comments for article_id, build a dependency graph for each, and
    write one JSON per comment to output_dir/{article_id}/{comment_index}.json.
    """
    comment_paths = load_comment_files(ace_comments_dir, article_id, comment_index=comment_index)
    if not comment_paths:
        print(f"No comment files found in {ace_comments_dir / article_id}")
        return []

    print(
        f"Building dependency graphs for article_id={article_id} "
        f"({len(comment_paths)} comments)"
    )

    prompt_text = load_prompt(prompt_path)
    prompt_text = prompt_text.replace(
        "<DEPENDENCY_EDGE_EXAMPLES>", _format_edge_examples()
    )
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
    parser.add_argument(
        "--comment-index",
        type=int,
        default=None,
        help=(
            "Optional 1-based comment index. If set, only this comment is processed "
            "for the given article."
        ),
    )
    args = parser.parse_args()

    run_dependency_classification(
        ace_comments_dir=args.ace_comments_dir,
        prompt_path=args.prompt_file,
        output_dir=args.output_dir,
        article_id=args.article_id,
        api_key=args.api_key,
        model=args.model,
        comment_index=args.comment_index,
    )


if __name__ == "__main__":
    main()
