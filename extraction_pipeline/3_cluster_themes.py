"""
Cluster ACE sentence classifications into common statements (themes) per category.

Reads a JSON file of ACE sentence classifications (e.g., ace_classifications/ace_sentence_classifications_181.json),
groups sentences by comment_tag, and for each category asks an LLM to identify common propositions and assign
sentences to those common statements.

Outputs a JSON file that, for each category, lists the common statements and the sentence objects that belong to
each statement.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

NUM_PASSES = 3
DEFAULT_BATCH_SIZE = 500
MAX_EXAMPLES_PER_TOPIC = 3

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_PROMPTS_DIR = SCRIPT_DIR / "prompts"
DEFAULT_PROMPT_FILE = DEFAULT_PROMPTS_DIR / "cluster_ace_themes.txt"
DEFAULT_ACE_CLASSIFICATIONS_DIR = SCRIPT_DIR / "ace_classifications"
DEFAULT_ACE_CLUSTERS_DIR = SCRIPT_DIR / "ace_clusters"

# Default to article 181 as requested, but allow override via CLI.
DEFAULT_ARTICLE_ID = "181"
DEFAULT_INPUT_JSON = (
    DEFAULT_ACE_CLASSIFICATIONS_DIR
    / f"ace_sentence_classifications_{DEFAULT_ARTICLE_ID}.json"
)
DEFAULT_OUTPUT_JSON = (
    DEFAULT_ACE_CLUSTERS_DIR / f"ace_sentence_theme_clusters_{DEFAULT_ARTICLE_ID}.json"
)


def _ensure_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY in .env."
        )
    return OpenAI(api_key=key)


def load_classifications(path: Path) -> List[Dict[str, Any]]:
    """Load ACE sentence classifications from JSON."""
    if not path.is_file():
        raise FileNotFoundError(f"Input classifications JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"Expected top-level list in classifications JSON, got {type(data)}"
        )
    return data


def load_prompt(prompt_path: Path) -> str:
    """Load the clustering prompt from a text file."""
    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def group_by_category(
    rows: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Group classification rows by their comment_tag."""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        tag = row.get("comment_tag") or ""
        if not tag:
            # Skip rows without a category; they should not normally exist.
            continue
        buckets.setdefault(tag, []).append(row)
    return buckets


def _build_memory(
    common_statements: List[Dict[str, Any]], max_examples: int = MAX_EXAMPLES_PER_TOPIC
) -> List[Dict[str, Any]]:
    """Build memory: for each topic, id, statement, and up to max_examples example sentences."""
    memory: List[Dict[str, Any]] = []
    for st in common_statements:
        sents = st.get("sentences") or []
        examples = [
            s.get("original_comment", "") or ""
            for s in sents[:max_examples]
            if s.get("original_comment")
        ]
        memory.append({
            "id": st.get("id", ""),
            "statement": st.get("statement", ""),
            "example_sentences": examples,
        })
    return memory


def _merge_batch_results(
    comment_tag: str,
    batches: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Merge common_statements from multiple batch responses by id; combine sentences."""
    by_id: Dict[str, Dict[str, Any]] = {}
    for batch_cs in batches:
        for st in batch_cs:
            sid = st.get("id") or ""
            sents = list(st.get("sentences") or [])
            if sid in by_id:
                by_id[sid]["sentences"].extend(sents)
            else:
                by_id[sid] = {
                    "id": sid,
                    "statement": st.get("statement", ""),
                    "sentences": sents,
                }
    return list(by_id.values())


def _common_statements_to_list(common_statements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize to list of { id, statement, sentences } for merging."""
    return [
        {
            "id": st.get("id", ""),
            "statement": st.get("statement", ""),
            "sentences": list(st.get("sentences") or []),
        }
        for st in common_statements
    ]


def _merge_very_related_topics(
    client: OpenAI,
    model: str,
    comment_tag: str,
    common_statements: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Ask the LLM which topics are very related; merge only those. Returns updated list.
    """
    if len(common_statements) <= 1:
        return common_statements

    summary = [
        {"id": st.get("id", ""), "statement": st.get("statement", "")}
        for st in common_statements
    ]
    user_content = (
        f"You are given the current list of topics for category {comment_tag!r}. "
        "Only merge two topics if they are VERY closely related (same proposition). "
        "When in doubt, do not merge.\n\n"
        "Current topics (id, statement):\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n\nRespond with a JSON object with a single key \"merge_pairs\": "
        "a list of pairs of topic ids to merge, e.g. {\"merge_pairs\": [[\"S1\", \"S2\"]]}. "
        "Use empty list if nothing should be merged. No other text."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You output only valid JSON."},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = response.choices[0].message.content
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return common_statements
    pairs = out.get("merge_pairs") or []
    if not pairs:
        return common_statements

    by_id = {st["id"]: dict(st) for st in common_statements}
    merged_ids: set = set()
    for a, b in pairs:
        a, b = str(a).strip(), str(b).strip()
        if a not in by_id or b not in by_id or a == b:
            continue
        if a in merged_ids or b in merged_ids:
            continue
        combined = by_id[a]["sentences"] + by_id[b]["sentences"]
        by_id[a]["sentences"] = combined
        by_id[a]["statement"] = by_id[a]["statement"] or by_id[b]["statement"]
        del by_id[b]
        merged_ids.add(b)
    return list(by_id.values())


def cluster_category_pass1(
    client: OpenAI,
    model: str,
    base_prompt: str,
    comment_tag: str,
    items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    First pass: cluster all items in one call. Returns category dict with common_statements.
    """
    user_content = (
        base_prompt
        + "\n\n---\n\n"
        + f"comment_tag for this batch:\n{json.dumps(comment_tag, ensure_ascii=False)}\n\n"
        + "Input sentences for this category (JSON array):\n"
        + json.dumps(items, indent=2, ensure_ascii=False)
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You analyze ACE sentences within a single category and "
                    "cluster them into common propositions. Respond only with valid JSON."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    raw = response.choices[0].message.content
    try:
        out = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model returned invalid JSON for category {comment_tag!r}: {e}\n"
            f"Raw (truncated): {raw[:500]}..."
        )
    if not isinstance(out, dict):
        raise ValueError(
            f"Expected a JSON object for category {comment_tag!r}, got {type(out)}"
        )
    if out.get("comment_tag") != comment_tag:
        raise ValueError(
            f"Model response has mismatched comment_tag for {comment_tag!r}: "
            f"{out.get('comment_tag')!r}"
        )
    if "common_statements" not in out or not isinstance(out["common_statements"], list):
        raise ValueError(
            f"Model response for {comment_tag!r} missing 'common_statements' list."
        )
    return out


def cluster_batch_with_memory(
    client: OpenAI,
    model: str,
    base_prompt: str,
    comment_tag: str,
    memory: List[Dict[str, Any]],
    batch_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Cluster one batch of items using existing topics (memory). Prefer assigning to
    existing topic ids when a sentence fits; create new topics only when needed.
    Returns common_statements for this batch only.
    """
    memory_block = (
        "Existing topics (prefer assigning sentences to these by id when they fit; "
        "only create new topics when the sentence does not fit any):\n"
        + json.dumps(memory, indent=2, ensure_ascii=False)
        + "\n\n"
    )
    user_content = (
        base_prompt
        + "\n\n---\n\n"
        + f"comment_tag for this batch:\n{json.dumps(comment_tag, ensure_ascii=False)}\n\n"
        + memory_block
        + "Input sentences for this batch (assign to existing topic id or create new; "
        "preserve article_id, comment_id, original_comment, comment_tag):\n"
        + json.dumps(batch_items, indent=2, ensure_ascii=False)
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You assign ACE sentences to existing topics by id when they fit, "
                    "or create new topics. Respond only with valid JSON. "
                    "Use the same output structure: comment_tag and common_statements "
                    "with id, statement, sentences."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    raw = response.choices[0].message.content
    try:
        out = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model returned invalid JSON for batch (category {comment_tag!r}): {e}\n"
            f"Raw (truncated): {raw[:500]}..."
        )
    if not isinstance(out, dict):
        raise ValueError(f"Expected JSON object for batch, got {type(out)}")
    cs = out.get("common_statements")
    if not isinstance(cs, list):
        raise ValueError("Batch response missing common_statements list.")
    return cs


def run_clustering(
    input_path: Path,
    prompt_path: Path,
    output_path: Path,
    model: str = "gpt-5.2",
    num_passes: int = NUM_PASSES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Load ACE classifications, run multi-pass clustering (with memory and shuffle) per
    category, and write a single JSON file summarizing the themes.
    """
    rows = load_classifications(input_path)
    if not rows:
        print(f"No classifications found in {input_path}. Nothing to cluster.")
        return {}

    article_ids = {str(r.get("article_id", "")).strip() for r in rows if r.get("article_id")}
    article_id = article_ids.pop() if len(article_ids) == 1 else None

    buckets = group_by_category(rows)
    if not buckets:
        print("No categories found in classifications. Nothing to cluster.")
        return {}

    print(
        f"Loaded {len(rows)} classified sentences from {input_path} "
        f"across {len(buckets)} categories. num_passes={num_passes}, batch_size={batch_size}."
    )

    prompt_text = load_prompt(prompt_path)
    client = _ensure_client()
    rng = random.Random(shuffle_seed)
    category_clusters: List[Dict[str, Any]] = []

    for idx, (comment_tag, items) in enumerate(sorted(buckets.items()), start=1):
        print(
            f"[{idx}/{len(buckets)}] Clustering {len(items)} sentences "
            f"for category: {comment_tag!r} ({num_passes} passes)"
        )
        # Pass 1: full category in one call
        pass1 = cluster_category_pass1(
            client=client,
            model=model,
            base_prompt=prompt_text,
            comment_tag=comment_tag,
            items=items,
        )
        common_statements = _common_statements_to_list(pass1["common_statements"])
        memory = _build_memory(common_statements)

        for pass_num in range(2, num_passes + 1):
            # Shuffle sentence order before this pass (so batches see different order)
            shuffled = list(items)
            rng.shuffle(shuffled)

            batch_results: List[List[Dict[str, Any]]] = []
            for i in range(0, len(shuffled), batch_size):
                batch = shuffled[i : i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(shuffled) + batch_size - 1) // batch_size
                print(
                    f"  Pass {pass_num} batch {batch_num}/{total_batches} "
                    f"({len(batch)} sentences)"
                )
                cs = cluster_batch_with_memory(
                    client=client,
                    model=model,
                    base_prompt=prompt_text,
                    comment_tag=comment_tag,
                    memory=memory,
                    batch_items=batch,
                )
                batch_results.append(cs)

            # Merge all batch results by topic id
            common_statements = _merge_batch_results(comment_tag, batch_results)
            # Combine only very related topics
            common_statements = _merge_very_related_topics(
                client=client,
                model=model,
                comment_tag=comment_tag,
                common_statements=common_statements,
            )
            memory = _build_memory(common_statements)

        clustered: Dict[str, Any] = {
            "comment_tag": comment_tag,
            "common_statements": common_statements,
        }
        category_clusters.append(clustered)

    result: Dict[str, Any] = {
        "article_id": article_id,
        "input_path": str(input_path),
        "categories": category_clusters,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved clustered themes to {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster ACE sentence classifications into common statements (themes) "
            "per category."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_INPUT_JSON,
        help=(
            "Path to ACE sentence classifications JSON "
            f"(default: {DEFAULT_INPUT_JSON})"
        ),
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help=(
            "Path to clustering prompt .txt "
            f"(default: {DEFAULT_PROMPT_FILE})"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=(
            "Output JSON path for clustered themes "
            f"(default: {DEFAULT_OUTPUT_JSON})"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="OpenAI model for clustering (default: gpt-5.2)",
    )
    parser.add_argument(
        "--num-passes",
        type=int,
        default=NUM_PASSES,
        help=f"Number of LLM passes per category (default: {NUM_PASSES})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for passes 2+ (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Random seed for shuffling sentence order between passes (default: None)",
    )

    args = parser.parse_args()

    run_clustering(
        input_path=args.input,
        prompt_path=args.prompt_file,
        output_path=args.output,
        model=args.model,
        num_passes=args.num_passes,
        batch_size=args.batch_size,
        shuffle_seed=args.shuffle_seed,
    )


if __name__ == "__main__":
    main()

