"""
Classify ACE sentences as visual vs non-visual observations.

Reads ACE comment JSONs produced by 1_extract_ace_comments.py (under ace_comments/),
loads the visualization image for each article, sends all ACE sentences to an LLM
in batches, and writes a JSON file tagging each sentence as either
"Visual observation" or "Non-visual observation".
"""

from __future__ import annotations

import base64
import csv
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
DEFAULT_PROMPT_FILE = DEFAULT_PROMPTS_DIR / "classify_visual_observations.txt"
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "ace_visual_classifications.json"
DEFAULT_INTERMEDIATE_DIR = SCRIPT_DIR / "ace_visual_classifications_batches"
DEFAULT_VISUAL_CLASSIFICATIONS_DIR = SCRIPT_DIR / "ace_visual_classifications"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "images"

BATCH_SIZE = 100

# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

_CLASSIFICATION_EXAMPLES: List[Dict[str, Any]] = [
    {
        "input": [
            {"article_id": "183", "comment_id": 1, "original_comment": "The author asks about gender-neutrality."},
            {"article_id": "183", "comment_id": 1, "original_comment": "Young women reach out to the family."},
        ],
        "output": {
            "classifications": [
                {"article_id": "183", "comment_id": 1, "original_comment": "The author asks about gender-neutrality.", "reasoning": "Step 4: This is a meta-commentary about what the author does, not verifiable from the chart alone.", "comment_tag": "Non-visual observation"},
                {"article_id": "183", "comment_id": 1, "original_comment": "Young women reach out to the family.", "reasoning": "Step 3: Describes a pattern visible in the chart data about young women's contact behavior.", "comment_tag": "Visual observation"},
            ]
        },
    },
    {
        "input": [
            {"article_id": "183", "comment_id": 4, "original_comment": "Young women reach out to their parents more often than young men."},
            {"article_id": "183", "comment_id": 4, "original_comment": "Less than 25 percent of young men visit their parents in person at least once a day or a few times a week."},
            {"article_id": "183", "comment_id": 4, "original_comment": "Text messages and phone calls are more accessible ways of connection than in-person visits."},
            {"article_id": "183", "comment_id": 4, "original_comment": "Traditional gender stereotypes may affect how often a young person reaches out to the person's parents."},
            {"article_id": "183", "comment_id": 4, "original_comment": "Society typically encourages young men to be independent."},
            {"article_id": "183", "comment_id": 4, "original_comment": "I have a very close relationship with my parents."},
            {"article_id": "183", "comment_id": 4, "original_comment": "Some people think that daughters are favorites for a reason."},
        ],
        "output": {
            "classifications": [
                {"article_id": "183", "comment_id": 4, "original_comment": "Young women reach out to their parents more often than young men.", "reasoning": "Step 3: Describes a comparative pattern between two groups visible in the chart.", "comment_tag": "Visual observation"},
                {"article_id": "183", "comment_id": 4, "original_comment": "Less than 25 percent of young men visit their parents in person at least once a day or a few times a week.", "reasoning": "Step 2: Cites a specific data value ('less than 25 percent') read from the chart.", "comment_tag": "Visual observation"},
                {"article_id": "183", "comment_id": 4, "original_comment": "Text messages and phone calls are more accessible ways of connection than in-person visits.", "reasoning": "Step 4: General world knowledge about communication accessibility, not verifiable from the chart.", "comment_tag": "Non-visual observation"},
                {"article_id": "183", "comment_id": 4, "original_comment": "Traditional gender stereotypes may affect how often a young person reaches out to the person's parents.", "reasoning": "Step 4: Proposes a causal explanation (stereotypes) for the chart pattern; goes beyond what the chart shows.", "comment_tag": "Non-visual observation"},
                {"article_id": "183", "comment_id": 4, "original_comment": "Society typically encourages young men to be independent.", "reasoning": "Step 4: Standalone claim about societal norms, not shown in the chart.", "comment_tag": "Non-visual observation"},
                {"article_id": "183", "comment_id": 4, "original_comment": "I have a very close relationship with my parents.", "reasoning": "Step 4: Personal anecdote about the commenter's own life, not in the chart.", "comment_tag": "Non-visual observation"},
                {"article_id": "183", "comment_id": 4, "original_comment": "Some people think that daughters are favorites for a reason.", "reasoning": "Step 4: Normative/evaluative stance about societal beliefs, not verifiable from the chart.", "comment_tag": "Non-visual observation"},
            ]
        },
    },
    {
        "input": [
            {"article_id": "181", "comment_id": 7, "original_comment": "The graph uses a stacked area chart."},
            {"article_id": "181", "comment_id": 7, "original_comment": "Coal generated over 5000 TWh in China by 2020."},
            {"article_id": "181", "comment_id": 7, "original_comment": "China's electricity generation increased sharply after 2000."},
            {"article_id": "181", "comment_id": 7, "original_comment": "China's electricity generation increased because of rapid industrialization."},
            {"article_id": "181", "comment_id": 7, "original_comment": "If coal use continues at this rate, emissions will keep rising."},
            {"article_id": "181", "comment_id": 7, "original_comment": "This trend is concerning."},
        ],
        "output": {
            "classifications": [
                {"article_id": "181", "comment_id": 7, "original_comment": "The graph uses a stacked area chart.", "reasoning": "Step 1: Names the chart type, a visual element.", "comment_tag": "Visual observation"},
                {"article_id": "181", "comment_id": 7, "original_comment": "Coal generated over 5000 TWh in China by 2020.", "reasoning": "Step 2: Cites a specific data value ('over 5000 TWh') read from the chart.", "comment_tag": "Visual observation"},
                {"article_id": "181", "comment_id": 7, "original_comment": "China's electricity generation increased sharply after 2000.", "reasoning": "Step 3: Describes a trend visible in the chart.", "comment_tag": "Visual observation"},
                {"article_id": "181", "comment_id": 7, "original_comment": "China's electricity generation increased because of rapid industrialization.", "reasoning": "Step 4: Proposes a causal explanation ('because of rapid industrialization') that goes beyond what the chart shows.", "comment_tag": "Non-visual observation"},
                {"article_id": "181", "comment_id": 7, "original_comment": "If coal use continues at this rate, emissions will keep rising.", "reasoning": "Step 4: Future-oriented prediction not shown in the chart.", "comment_tag": "Non-visual observation"},
                {"article_id": "181", "comment_id": 7, "original_comment": "This trend is concerning.", "reasoning": "Step 4: Emotional/evaluative reaction, not verifiable from the chart.", "comment_tag": "Non-visual observation"},
            ]
        },
    },
]


def _format_classification_examples() -> str:
    """Render _CLASSIFICATION_EXAMPLES as Markdown for injection into the prompt."""
    sections: List[str] = []
    for i, ex in enumerate(_CLASSIFICATION_EXAMPLES, 1):
        inp = json.dumps(ex["input"], indent=2, ensure_ascii=False)
        out = json.dumps(ex["output"], indent=2, ensure_ascii=False)
        sections.append(
            f"### Example {i}\n\n"
            f"**Input:**\n```json\n{inp}\n```\n\n"
            f"**Output:**\n```json\n{out}\n```"
        )
    return "## Few-shot examples\n\n" + "\n\n".join(sections)


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
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_visualization(
    client: OpenAI,
    article_id: str,
    images_dir: Path = DEFAULT_IMAGES_DIR,
    model: str = "gpt-5.4",
) -> str:
    """
    Send the visualization image for *article_id* to the vision model and
    return a structured text description. Returns empty string if not found.
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
    comment_index: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Load all ACE comment JSON files and return a flat list of items:
    { article_id, comment_id, original_comment } for each ACE sentence.
    """
    items: List[Dict[str, Any]] = []

    if article_id_folder is not None:
        if not article_id_folder.is_dir():
            return items
        dirs_to_scan = [(article_id_folder, article_id_folder.name)]
    else:
        if not ace_comments_dir.is_dir():
            return items
        if article_id is not None:
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
            if comment_index is not None and int(comment_id) != int(comment_index):
                continue
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


def _build_comment_grouped_batches(
    items: List[Dict[str, Any]],
    max_batch_size: int,
) -> List[List[Dict[str, Any]]]:
    """
    Group items by (article_id, comment_id) and pack whole comments into
    batches that respect *max_batch_size* while keeping each comment's
    sentences together.
    """
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    group_order: List[tuple] = []
    for item in items:
        key = (item["article_id"], item["comment_id"])
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(item)

    batches: List[List[Dict[str, Any]]] = []
    current_batch: List[Dict[str, Any]] = []

    for key in group_order:
        comment_items = groups[key]
        if len(comment_items) > max_batch_size:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
            batches.append(comment_items)
            continue
        if current_batch and len(current_batch) + len(comment_items) > max_batch_size:
            batches.append(current_batch)
            current_batch = []
        current_batch.extend(comment_items)

    if current_batch:
        batches.append(current_batch)

    return batches


def load_prompt(prompt_path: Path) -> str:
    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def classify_batch(
    client: OpenAI,
    prompt_text: str,
    batch: List[Dict[str, Any]],
    model: str,
) -> List[Dict[str, Any]]:
    """Send one batch of items to the model and return visual/non-visual classifications."""
    user_content = (
        prompt_text
        + "\n\n---\n\nInput sentences to classify (use these exact fields in your response):\n"
        + json.dumps(batch, indent=2, ensure_ascii=False)
    )
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert at determining whether statements about data "
                    "visualizations are visual observations (describing what is visible "
                    "in the chart) or non-visual observations (going beyond the chart). "
                    "Follow the decision procedure in the user prompt exactly. "
                    "Always produce reasoning before selecting a category. "
                    "Respond with valid JSON only."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    elapsed = time.perf_counter() - start_time
    print(
        f"API call for batch of {len(batch)} sentences took {elapsed:.2f}s "
        f"({elapsed / 60:.2f} min)."
    )
    raw = response.choices[0].message.content
    try:
        out = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON: {e}\nRaw: {raw[:500]}...")
    classifications = out.get("classifications")
    if not isinstance(classifications, list):
        raise ValueError(
            f"Model response missing 'classifications' list: {list(out.keys())}"
        )
    return classifications


def _build_image_info_section(
    descriptions: Dict[str, str],
    article_ids: set[str],
) -> str:
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
    model: str = "gpt-5.4-mini",
    batch_size: int = BATCH_SIZE,
    intermediate_dir: Path | None = DEFAULT_INTERMEDIATE_DIR,
    article_id: str | None = None,
    article_id_folder: Path | None = None,
    images_dir: Path = DEFAULT_IMAGES_DIR,
    comment_index: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Load ACE items, run visual/non-visual classification in batches, and save JSON.
    """
    items = load_ace_comment_items(
        ace_comments_dir,
        article_id=article_id,
        article_id_folder=article_id_folder,
        comment_index=comment_index,
    )
    if not items:
        print("No ACE sentences found. Nothing to classify.")
        return []

    effective_article_id = article_id
    if article_id_folder is not None:
        effective_article_id = article_id_folder.name
        print(
            f"Loaded {len(items)} ACE sentences from {article_id_folder} "
            f"(article_id={effective_article_id})"
        )
    elif article_id is not None:
        print(
            f"Loaded {len(items)} ACE sentences from {ace_comments_dir} "
            f"for article_id={article_id}"
        )
    else:
        print(f"Loaded {len(items)} ACE sentences from {ace_comments_dir}")

    prompt_template = load_prompt(prompt_path)
    client = _ensure_client(api_key)

    unique_article_ids = {item["article_id"] for item in items}
    image_descriptions: Dict[str, str] = {}
    for aid in sorted(unique_article_ids):
        desc = describe_visualization(client, aid, images_dir=images_dir, model=model)
        image_descriptions[aid] = desc

    all_classifications: List[Dict[str, Any]] = []

    if intermediate_dir is not None:
        intermediate_dir.mkdir(parents=True, exist_ok=True)

    batches = _build_comment_grouped_batches(items, batch_size)
    print(
        f"Split {len(items)} sentences into {len(batches)} comment-grouped "
        f"batch(es) (max {batch_size} sentences each)."
    )

    for batch_index, batch in enumerate(batches):
        batch_num = batch_index + 1

        batch_article_ids = {item["article_id"] for item in batch}
        image_info = _build_image_info_section(image_descriptions, batch_article_ids)
        prompt_text = prompt_template.replace("<IMAGE_INFORMATION>", image_info)
        prompt_text = prompt_text.replace(
            "<CLASSIFICATION_EXAMPLES>", _format_classification_examples()
        )

        print(f"Classifying batch {batch_num} ({len(batch)} sentences)...")
        classifications = classify_batch(client, prompt_text, batch, model)

        batch_rows: List[Dict[str, Any]] = []
        for c in classifications:
            if not isinstance(c, dict):
                continue
            tag = c.get("comment_tag", "")
            if isinstance(tag, list):
                tag = tag[0] if tag else "Non-visual observation"
            if tag not in ("Visual observation", "Non-visual observation"):
                tag = "Non-visual observation"
            row_article_id = c.get("article_id", "")
            row = {
                "article_id": row_article_id,
                "comment_id": c.get("comment_id", 0),
                "original_comment": c.get("original_comment", ""),
                "reasoning": c.get("reasoning", ""),
                "comment_tag": tag,
                "image_description": image_descriptions.get(row_article_id, ""),
            }
            all_classifications.append(row)
            batch_rows.append(row)

        if intermediate_dir is not None:
            batch_path = intermediate_dir / f"batch_{batch_num:04d}.json"
            with batch_path.open("w", encoding="utf-8") as f:
                json.dump(batch_rows, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(batch_rows)} classifications to {batch_path}")

    if effective_article_id is not None and output_path == DEFAULT_OUTPUT_JSON:
        final_output_dir = DEFAULT_VISUAL_CLASSIFICATIONS_DIR
        final_output_dir.mkdir(parents=True, exist_ok=True)
        final_output_path = (
            final_output_dir
            / f"ace_visual_classifications_{effective_article_id}.json"
        )
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_output_path = output_path

    with final_output_path.open("w", encoding="utf-8") as f:
        json.dump(all_classifications, f, indent=2, ensure_ascii=False)
    print(
        f"Saved combined {len(all_classifications)} classifications "
        f"to {final_output_path}"
    )

    # Print summary
    visual_count = sum(
        1 for c in all_classifications if c["comment_tag"] == "Visual observation"
    )
    non_visual_count = len(all_classifications) - visual_count
    print(f"\n--- Summary ---")
    print(f"Total sentences:        {len(all_classifications)}")
    print(f"Visual observations:    {visual_count} ({100 * visual_count / len(all_classifications):.1f}%)")
    print(f"Non-visual observations: {non_visual_count} ({100 * non_visual_count / len(all_classifications):.1f}%)")

    return all_classifications


DEFAULT_VISUAL_CSV = SCRIPT_DIR / "visual_observations.csv"
DEFAULT_PROPOSED_LABELS_JSON = SCRIPT_DIR / "proposed_visual_labels.json"
DEFAULT_LABEL_ITERATIONS_DIR = SCRIPT_DIR / "label_iterations"

# ---------------------------------------------------------------------------
# Seed categories for seeded label refinement (--seed-categories)
# ---------------------------------------------------------------------------

SEED_VISUAL_CATEGORIES: List[Dict[str, Any]] = [
    {
        "label": "Retrieve & Identify (Read the Data)",
        "description": (
            "Baseline extraction of what is explicitly shown on the chart without "
            "performing math or finding patterns. Includes identifying text or "
            "marks, finding a single specific data point or extremum, and "
            "identifying the topic or scope of the visualization. Equivalent to "
            "VLAT 'Retrieve Value', HOLF 'Retrieve Value'."
        ),
        "decision_criteria": (
            "The observer simply locates and extracts information that is "
            "explicitly written or encoded on the chart: a label, a single value, "
            "an axis title, a chart type, a mark identity, or the topic/scope. "
            "No computation, comparison, or pattern recognition is required."
        ),
        "examples": [
            "This says 20%.",
            "The x-axis is labeled in years.",
            "The graph uses a stacked area chart.",
            "Coal generated over 5000 TWh in China by 2020.",
            "The chart is about electricity generation by fuel type.",
        ],
    },
    {
        "label": "Compare & Compute (Read Between the Data)",
        "description": (
            "The observer holds at least two pieces of information to find a "
            "difference, sum, ratio, ranking, or cross-panel comparison. Includes "
            "two-point comparisons (A vs B), cross-panel comparisons, ranking and "
            "ordering, and understanding denominators or baselines to compute a "
            "relative value. Equivalent to VLAT 'Make Comparisons', HOLF "
            "'Determine Range'."
        ),
        "decision_criteria": (
            "The sentence relates two or more data points to find a delta, "
            "hierarchy, or relative measure. It cites a comparison between "
            "specific values, a ranking, a ratio, or a cross-panel contrast. "
            "The key signal is that two or more pieces of chart information are "
            "being related to each other."
        ),
        "examples": [
            "Women are 20% more likely to use text or video-chat compared to men.",
            "Less than 25% of young men visit their parents vs. over 40% of young women.",
            "The top 0.1% have a benefit that is up to 7 times the average.",
            "In-person visits are the least common form of communication.",
            "China generates far more electricity from coal than from any other source.",
        ],
    },
    {
        "label": "Infer & Characterize (Read Beyond the Data)",
        "description": (
            "The observer synthesizes all or many data points into a single "
            "pattern, trend, correlation, or functional form. Includes trends "
            "and trajectories, cross-variable relationships and geographic "
            "clustering, statistical significance or residuals, and density or "
            "concentration patterns. Equivalent to VLAT 'Find Correlations/"
            "Trends', CALVI 'Find Anomalies/Clusters'."
        ),
        "decision_criteria": (
            "The sentence describes a direction, shape, overall pattern, "
            "correlation, clustering, or distributional property across the "
            "data as a whole — not individual points. The observer is looking "
            "at the behavior of the data rather than reading off specific values "
            "or comparing a small number of points."
        ),
        "examples": [
            "Over time, there has been an overall steady increase in clean gigawatt hours.",
            "China's graph shows a very steep increase in electricity generation.",
            "Admissions are heavily skewed in favor of high-income legacy applicants.",
            "There appears to be a positive correlation between income and acceptance rate.",
            "Abortion is most restricted in the south.",
        ],
    },
]

PROPOSE_LABELS_SYSTEM_PROMPT = (
    "You are an expert in data visualization literacy and discourse analysis. "
    "You are given a large collection of reader observations about data "
    "visualizations, all of which have already been classified as 'visual "
    "observations' — they describe something visible in or directly derivable "
    "from a chart. Your job is to propose a taxonomy of fine-grained "
    "classification labels that could meaningfully subdivide these visual "
    "observations. Respond with valid JSON only."
)

REFINE_LABELS_SYSTEM_PROMPT = (
    "You are an expert in data visualization literacy and discourse analysis. "
    "You have an existing set of classification labels for visual observations "
    "about data charts. You will be shown a new batch of observations and must "
    "evaluate whether the current labels adequately cover them. If not, you "
    "must refine the taxonomy — splitting, merging, adding, or redefining "
    "labels as needed. Respond with valid JSON only."
)


def _format_obs_block(observations: List[Dict[str, Any]]) -> str:
    lines = []
    for o in observations:
        lines.append(
            f"- [article {o['article_id']}, comment {o['comment_id']}] "
            f"{o['original_comment']}"
        )
    return "\n".join(lines)


def _format_labels_block(labels: List[Dict[str, Any]]) -> str:
    parts = []
    for i, lbl in enumerate(labels, 1):
        part = (
            f"{i}. **{lbl.get('label', '???')}**\n"
            f"   Description: {lbl.get('description', '')}\n"
            f"   Decision criteria: {lbl.get('decision_criteria', '')}"
        )
        examples = lbl.get("examples", [])
        if examples:
            part += "\n   Examples:\n" + "\n".join(f"     - {ex}" for ex in examples[:3])
        parts.append(part)
    return "\n\n".join(parts)


def _build_initial_proposal_prompt(
    observations: List[Dict[str, Any]],
    num_labels: int,
) -> str:
    obs_block = _format_obs_block(observations)
    return f"""Below are {len(observations)} visual observations about data visualizations. Every one of them describes something visible in or directly derivable from a chart (chart elements, data values, comparisons, trends, patterns, etc.).

Your task: propose exactly {num_labels} classification labels that together cover ALL of the observations below. Each label should be:
1. Mutually exclusive (a given observation should fit exactly one label)
2. Collectively exhaustive (every observation below should fit at least one label)
3. Meaningful for analysis — labels should capture qualitatively different TYPES of visual observation (e.g., identifying chart type vs. reading a data value vs. describing a trend)

For each proposed label, provide:
- "label": a short, descriptive name (e.g., "Chart element identification")
- "description": 1-2 sentences defining what observations belong in this category
- "decision_criteria": a brief rule for deciding when an observation belongs here
- "examples": 3-5 example observations from the list below that fit this label

Respond with a JSON object:
{{
  "num_labels": {num_labels},
  "labels": [
    {{
      "label": "...",
      "description": "...",
      "decision_criteria": "...",
      "examples": ["...", "..."]
    }}
  ],
  "coverage_notes": "<brief notes on any observations that were hard to place or edge cases>"
}}

---

Visual observations ({len(observations)} total):

{obs_block}"""


def _build_refinement_prompt(
    observations: List[Dict[str, Any]],
    current_labels: List[Dict[str, Any]],
    iteration: int,
) -> str:
    obs_block = _format_obs_block(observations)
    labels_block = _format_labels_block(current_labels)
    return f"""## Current taxonomy ({len(current_labels)} labels)

{labels_block}

---

## New batch of visual observations (iteration {iteration}, {len(observations)} observations)

{obs_block}

---

## Your task

You have been shown {len(observations)} NEW visual observations that were randomly sampled from a larger corpus. Evaluate whether the current {len(current_labels)} labels above adequately classify ALL of these observations.

For EACH observation, mentally assign it to the best-fitting current label. Then answer:

1. **Are there observations that don't fit well into any existing label?** If so, new labels may be needed.
2. **Are there labels that are too broad and lump together meaningfully different observations?** If so, they should be split into finer-grained labels.
3. **Are there labels that overlap too much or that no observations fall into?** If so, they should be merged or removed.
4. **Are any label definitions or decision criteria ambiguous given these new observations?** If so, refine the wording.

Based on this analysis, output an UPDATED taxonomy. You may:
- Keep labels unchanged (if they work well)
- Split a label into 2+ finer-grained labels
- Merge labels that overlap
- Add entirely new labels
- Refine descriptions / decision criteria / examples

The updated taxonomy does NOT need to have the same number of labels as before — use as many as needed for a clean, mutually exclusive, collectively exhaustive taxonomy.

Respond with a JSON object:
{{
  "analysis": "<2-3 sentences summarizing what changed and why>",
  "changes_made": ["<list of specific changes, e.g. 'Split X into X1 and X2', 'Merged A and B into C', 'Added new label D'>"],
  "num_labels": <int>,
  "labels": [
    {{
      "label": "...",
      "description": "...",
      "decision_criteria": "...",
      "examples": ["...", "..."]
    }}
  ],
  "coverage_notes": "<brief notes on edge cases or remaining ambiguities>"
}}"""


def load_visual_observations(
    classifications_path: Path,
    per_article_dir: Path | None = None,
) -> List[Dict[str, Any]]:
    """
    Load visual observations from the combined JSON or per-article directory.
    Returns only rows with comment_tag == "Visual observation".
    """
    all_rows: List[Dict[str, Any]] = []

    if per_article_dir is not None and per_article_dir.is_dir():
        for json_path in sorted(per_article_dir.glob("*.json")):
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(data, list):
                all_rows.extend(data)
    elif classifications_path.is_file():
        with classifications_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            all_rows = data

    return [r for r in all_rows if r.get("comment_tag") == "Visual observation"]


def export_visual_observations_csv(
    observations: List[Dict[str, Any]],
    csv_path: Path,
) -> None:
    """Write visual observations to a CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["article_id", "comment_id", "original_comment", "reasoning", "image_description"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for obs in observations:
            writer.writerow(obs)
    print(f"Exported {len(observations)} visual observations to {csv_path}")


def _print_labels(labels: List[Dict[str, Any]], header: str = "LABELS") -> None:
    print(f"\n{'='*60}")
    print(f"{header} ({len(labels)})")
    print(f"{'='*60}")
    for i, lbl in enumerate(labels, 1):
        print(f"\n{i}. {lbl.get('label', '???')}")
        print(f"   {lbl.get('description', '')}")
        print(f"   Criteria: {lbl.get('decision_criteria', '')}")
        examples = lbl.get("examples", [])
        if examples:
            print(f"   Examples:")
            for ex in examples[:3]:
                print(f"     - {ex}")


def propose_labels(
    observations: List[Dict[str, Any]],
    num_labels: int,
    api_key: str | None = None,
    model: str = "gpt-5.4-mini",
    output_path: Path = DEFAULT_PROPOSED_LABELS_JSON,
    sample_size: int = 500,
    num_iterations: int = 25,
    iterations_dir: Path = DEFAULT_LABEL_ITERATIONS_DIR,
    seed_labels: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Iteratively refine a taxonomy for visual observations.

    If *seed_labels* is provided, skip the from-scratch proposal and start
    every iteration as a refinement round using the seeds as the initial
    taxonomy.  Otherwise, round 1 asks the LLM to propose *num_labels*
    initial labels and subsequent rounds refine.

    Rounds 2–*num_iterations* (or 1–*num_iterations* when seeded): sample a
    fresh *sample_size* observations, present the current labels, and ask the
    LLM whether the categories still fit or need to be split / merged / added
    / refined.  The updated label set is saved after every round.
    """
    import random

    client = _ensure_client(api_key)
    iterations_dir.mkdir(parents=True, exist_ok=True)

    effective_sample = min(sample_size, len(observations))

    if seed_labels:
        current_labels: List[Dict[str, Any]] = [dict(l) for l in seed_labels]
        print(f"\nSeeded with {len(current_labels)} initial categories:")
        _print_labels(current_labels, header="SEED CATEGORIES")
    else:
        current_labels: List[Dict[str, Any]] = []

    for iteration in range(1, num_iterations + 1):
        sample = random.sample(observations, effective_sample)

        print(f"\n{'#'*60}")
        print(f"# ITERATION {iteration}/{num_iterations}  "
              f"({effective_sample} sampled observations)")
        print(f"{'#'*60}")

        if iteration == 1 and not seed_labels:
            user_prompt = _build_initial_proposal_prompt(sample, num_labels)
            system_prompt = PROPOSE_LABELS_SYSTEM_PROMPT
        else:
            user_prompt = _build_refinement_prompt(sample, current_labels, iteration)
            system_prompt = REFINE_LABELS_SYSTEM_PROMPT

        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_completion_tokens=4096,
        )
        elapsed = time.perf_counter() - start
        print(f"API call took {elapsed:.2f}s ({elapsed / 60:.2f} min).")

        raw = response.choices[0].message.content
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"WARNING: invalid JSON on iteration {iteration}, keeping "
                  f"previous labels. Error: {e}")
            result = {"labels": current_labels, "parse_error": str(e)}

        new_labels = result.get("labels", current_labels)
        if not new_labels:
            print(f"WARNING: empty labels on iteration {iteration}, keeping previous.")
            new_labels = current_labels

        if iteration > 1:
            analysis = result.get("analysis", "")
            changes = result.get("changes_made", [])
            if analysis:
                print(f"\nAnalysis: {analysis}")
            if changes:
                print(f"Changes made:")
                for ch in changes:
                    print(f"  - {ch}")

        current_labels = new_labels

        _print_labels(current_labels,
                       header=f"LABELS AFTER ITERATION {iteration}")

        coverage = result.get("coverage_notes", "")
        if coverage:
            print(f"\nCoverage notes: {coverage}")

        iter_result = {
            "iteration": iteration,
            "num_labels": len(current_labels),
            "labels": current_labels,
            "analysis": result.get("analysis", ""),
            "changes_made": result.get("changes_made", []),
            "coverage_notes": coverage,
            "_meta": {
                "total_visual_observations": len(observations),
                "sample_size": effective_sample,
                "model": model,
            },
        }
        iter_path = iterations_dir / f"iteration_{iteration:02d}.json"
        with iter_path.open("w", encoding="utf-8") as f:
            json.dump(iter_result, f, indent=2, ensure_ascii=False)
        print(f"Saved iteration {iteration} to {iter_path}")

    final_result = {
        "num_labels": len(current_labels),
        "labels": current_labels,
        "coverage_notes": result.get("coverage_notes", ""),
        "_meta": {
            "total_visual_observations": len(observations),
            "sample_size": effective_sample,
            "num_iterations": num_iterations,
            "model": model,
            "num_labels_initial": len(seed_labels) if seed_labels else num_labels,
            "num_labels_final": len(current_labels),
            "seeded": bool(seed_labels),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"FINAL TAXONOMY after {num_iterations} iterations")
    print(f"{'='*60}")
    if seed_labels:
        print(f"Seeded with {len(seed_labels)} categories, ended with {len(current_labels)}.")
    else:
        print(f"Started with {num_labels} requested labels, ended with {len(current_labels)}.")
    print(f"Saved to {output_path}")
    _print_labels(current_labels, header="FINAL LABELS")

    return final_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Classify ACE sentences as visual observations (about the chart) "
            "vs non-visual observations."
        )
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
        default="gpt-5.4-mini",
        help="OpenAI model for classification (default: gpt-5.4-mini)",
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
            "(reads from <ace-comments-dir>/<article-id>/) or a path to a folder."
        ),
    )
    parser.add_argument(
        "--comment-index",
        type=int,
        default=None,
        help=(
            "Optional 1-based comment index. If set, only ACE sentences from this "
            "comment are classified for the selected article."
        ),
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help=f"Directory containing visualization PNGs named {{article_id}}.png (default: {DEFAULT_IMAGES_DIR})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all article folders found in --ace-comments-dir.",
    )

    propose_group = parser.add_argument_group("label proposal mode")
    propose_group.add_argument(
        "--propose-labels",
        action="store_true",
        help=(
            "Instead of classifying, load existing visual classification results, "
            "export visual observations to a CSV, and iteratively refine a "
            "taxonomy of classification labels over multiple rounds."
        ),
    )
    propose_group.add_argument(
        "--num-labels",
        type=int,
        default=5,
        help="Number of classification labels to propose in the initial round (default: 5).",
    )
    propose_group.add_argument(
        "--num-iterations",
        type=int,
        default=25,
        help="Number of iterative refinement rounds (default: 25).",
    )
    propose_group.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of randomly sampled observations per iteration (default: 500).",
    )
    propose_group.add_argument(
        "--classifications-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=(
            "Path to the combined visual-classifications JSON to read from "
            f"(default: {DEFAULT_OUTPUT_JSON}). Used by --propose-labels."
        ),
    )
    propose_group.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_VISUAL_CSV,
        help=f"CSV output path for visual observations (default: {DEFAULT_VISUAL_CSV}).",
    )
    propose_group.add_argument(
        "--labels-output",
        type=Path,
        default=DEFAULT_PROPOSED_LABELS_JSON,
        help=f"JSON output path for final proposed labels (default: {DEFAULT_PROPOSED_LABELS_JSON}).",
    )
    propose_group.add_argument(
        "--iterations-dir",
        type=Path,
        default=DEFAULT_LABEL_ITERATIONS_DIR,
        help=f"Directory to save per-iteration label snapshots (default: {DEFAULT_LABEL_ITERATIONS_DIR}).",
    )
    propose_group.add_argument(
        "--seed-categories",
        action="store_true",
        help=(
            "Seed the iterative refinement with three reading-level categories "
            "(Retrieve & Identify, Compare & Compute, Infer & Characterize) "
            "instead of asking the model to propose labels from scratch. "
            "All iterations become refinement rounds."
        ),
    )
    args = parser.parse_args()

    if args.propose_labels:
        visual_obs = load_visual_observations(
            classifications_path=args.classifications_json,
            per_article_dir=DEFAULT_VISUAL_CLASSIFICATIONS_DIR,
        )
        if not visual_obs:
            parser.error(
                "No visual observations found. Run classification first "
                "(--all or --article-id), then use --propose-labels."
            )
        export_visual_observations_csv(visual_obs, args.csv_output)
        propose_labels(
            observations=visual_obs,
            num_labels=args.num_labels,
            api_key=args.api_key,
            model=args.model,
            output_path=args.labels_output,
            sample_size=args.sample_size,
            num_iterations=args.num_iterations,
            iterations_dir=args.iterations_dir,
            seed_labels=SEED_VISUAL_CATEGORIES if args.seed_categories else None,
        )
        return

    if not args.all and not args.article_id:
        parser.error("You must specify either --article-id, --all, or --propose-labels.")

    if args.all and args.article_id:
        parser.error("--all and --article-id are mutually exclusive.")

    article_id = args.article_id
    article_id_folder = None
    if article_id is not None:
        as_path = Path(article_id)
        is_path_like = (
            os.path.isabs(article_id)
            or os.path.sep in article_id
            or (os.path.altsep and os.path.altsep in article_id)
        )
        if is_path_like and as_path.is_dir():
            article_id_folder = as_path.resolve()
            article_id = None
        else:
            article_id_folder = None

    if args.comment_index is not None and article_id is None and article_id_folder is None:
        parser.error("--comment-index requires --article-id (single-article mode).")

    if args.all:
        ace_dir = args.ace_comments_dir
        if not ace_dir.is_dir():
            parser.error(f"ACE comments directory not found: {ace_dir}")
        article_dirs = sorted(d.name for d in ace_dir.iterdir() if d.is_dir())
        print(f"Found {len(article_dirs)} article(s) to process: {article_dirs}")

        all_results: List[Dict[str, Any]] = []
        for aid in article_dirs:
            print(f"\n{'='*60}")
            print(f"Processing article {aid}...")
            print(f"{'='*60}")
            try:
                results = run_classification(
                    ace_comments_dir=args.ace_comments_dir,
                    prompt_path=args.prompt_file,
                    output_path=args.output,
                    api_key=args.api_key,
                    model=args.model,
                    batch_size=args.batch_size,
                    intermediate_dir=args.intermediate_dir,
                    article_id=aid,
                    article_id_folder=None,
                    images_dir=args.images_dir,
                    comment_index=args.comment_index,
                )
                all_results.extend(results)
            except Exception as e:
                print(f"Error processing article {aid}: {e}")

        combined_path = args.output
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        with combined_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        visual_count = sum(
            1 for c in all_results if c["comment_tag"] == "Visual observation"
        )
        non_visual_count = len(all_results) - visual_count
        print(f"\n{'='*60}")
        print(f"ALL ARTICLES COMBINED")
        print(f"{'='*60}")
        print(f"Total sentences:         {len(all_results)}")
        print(f"Visual observations:     {visual_count} ({100 * visual_count / max(len(all_results), 1):.1f}%)")
        print(f"Non-visual observations: {non_visual_count} ({100 * non_visual_count / max(len(all_results), 1):.1f}%)")
        print(f"Saved combined results to {combined_path}")
    else:
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
            comment_index=args.comment_index,
        )


if __name__ == "__main__":
    main()
