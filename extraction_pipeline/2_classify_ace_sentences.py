"""
Classify ACE sentences from extracted ACE comments into semantic categories.

Reads ACE comment JSONs produced by 1_extract_ace_comments.py (under ace_comments/),
loads the classification prompt from prompts/classify_ace_sentences.txt, sends all
ACE sentences to an LLM in one or more batches, and writes a JSON file with
article_id, comment_id, original_comment, and comment_tag per sentence.

Each sentence receives exactly one category tag.
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

# Max sentences per API call — kept small to preserve per-item attention
BATCH_SIZE = 200

# ---------------------------------------------------------------------------
# Few-shot examples injected into the prompt at <CLASSIFICATION_EXAMPLES>
# ---------------------------------------------------------------------------

_CLASSIFICATION_EXAMPLES: List[Dict[str, Any]] = [
    # Example 1 — article 183 (parent contact), short comment
    {
        "input": [
            {"article_id": "183", "comment_id": 1, "original_comment": "The author asks about gender-neutrality."},
            {"article_id": "183", "comment_id": 1, "original_comment": "Young women reach out to the family."},
        ],
        "output": {
            "classifications": [
                {"article_id": "183", "comment_id": 1, "original_comment": "The author asks about gender-neutrality.", "reasoning": "Step 5: Reports that the author poses a question, indicating curiosity about information not shown.", "comment_tag": "Curiosity"},
                {"article_id": "183", "comment_id": 1, "original_comment": "Young women reach out to the family.", "reasoning": "Step 2c: Synthesizes a pattern from the chart data about young women's behavior overall, without citing a specific number.", "comment_tag": "Visual Observation: Cross-point Pattern Recognition"},
            ]
        },
    },
    # Example 2 — article 183, long comment spanning many categories
    {
        "input": [
            {"article_id": "183", "comment_id": 4, "original_comment": "Young women reach out to their parents more often than young men."},
            {"article_id": "183", "comment_id": 4, "original_comment": "Less than 25 percent of young men visit their parents in person at least once a day or a few times a week."},
            {"article_id": "183", "comment_id": 4, "original_comment": "Less than 25 percent of young women visit their parents in person at least once a day or a few times a week."},
            {"article_id": "183", "comment_id": 4, "original_comment": "Text messages and phone calls are more accessible ways of connection than in-person visits."},
            {"article_id": "183", "comment_id": 4, "original_comment": "Text messages and phone calls show a greater divide than in-person visits between young men and young women in interaction with their parents."},
            {"article_id": "183", "comment_id": 4, "original_comment": "Traditional gender stereotypes may affect how often a young person reaches out to the person's parents."},
            {"article_id": "183", "comment_id": 4, "original_comment": "Society typically encourages young men to be independent."},
            {"article_id": "183", "comment_id": 4, "original_comment": "Society expects women to lead family-centric lives."},
            {"article_id": "183", "comment_id": 4, "original_comment": "This phenomenon may have deeper roots."},
            {"article_id": "183", "comment_id": 4, "original_comment": "These deeper roots may include biological instincts."},
            {"article_id": "183", "comment_id": 4, "original_comment": "Biological instincts may be unique to each gender."},
            {"article_id": "183", "comment_id": 4, "original_comment": "I have a very close relationship with my parents."},
            {"article_id": "183", "comment_id": 4, "original_comment": "I am an only daughter."},
            {"article_id": "183", "comment_id": 4, "original_comment": "People frequently mention that I reach out to these adults more than other people do."},
            {"article_id": "183", "comment_id": 4, "original_comment": "Some people think that daughters are favorites for a reason."},
        ],
        "output": {
            "classifications": [
                {"article_id": "183", "comment_id": 4, "original_comment": "Young women reach out to their parents more often than young men.", "reasoning": "Step 2c: Compares two groups ('more often') across the chart without citing a specific number.", "comment_tag": "Visual Observation: Cross-point Pattern Recognition"},
                {"article_id": "183", "comment_id": 4, "original_comment": "Less than 25 percent of young men visit their parents in person at least once a day or a few times a week.", "reasoning": "Step 2b: Extracts a specific data value ('less than 25 percent') for one group from the chart.", "comment_tag": "Visual Observation: Data Point Extraction"},
                {"article_id": "183", "comment_id": 4, "original_comment": "Less than 25 percent of young women visit their parents in person at least once a day or a few times a week.", "reasoning": "Step 2b: Extracts a specific data value ('less than 25 percent') for a different group.", "comment_tag": "Visual Observation: Data Point Extraction"},
                {"article_id": "183", "comment_id": 4, "original_comment": "Text messages and phone calls are more accessible ways of connection than in-person visits.", "reasoning": "Step 7: General world knowledge about communication accessibility, not derived from the chart.", "comment_tag": "Prior Knowledge: Background"},
                {"article_id": "183", "comment_id": 4, "original_comment": "Text messages and phone calls show a greater divide than in-person visits between young men and young women in interaction with their parents.", "reasoning": "Step 2c: Compares multiple categories and groups without citing a specific number.", "comment_tag": "Visual Observation: Cross-point Pattern Recognition"},
                {"article_id": "183", "comment_id": 4, "original_comment": "Traditional gender stereotypes may affect how often a young person reaches out to the person's parents.", "reasoning": "Step 6: 'May affect' links a background cause (gender stereotypes) to the chart observation (contact frequency). This is causal reasoning.", "comment_tag": "Inference: Explanatory"},
                {"article_id": "183", "comment_id": 4, "original_comment": "Society typically encourages young men to be independent.", "reasoning": "Step 7: General claim about societal norms, stated as standalone fact without linking to chart data.", "comment_tag": "Prior Knowledge: Background"},
                {"article_id": "183", "comment_id": 4, "original_comment": "Society expects women to lead family-centric lives.", "reasoning": "Step 7: Standalone societal claim not directly tied to any chart observation.", "comment_tag": "Prior Knowledge: Background"},
                {"article_id": "183", "comment_id": 4, "original_comment": "This phenomenon may have deeper roots.", "reasoning": "Step 6: 'May have deeper roots' proposes a causal explanation for the observed chart pattern.", "comment_tag": "Inference: Explanatory"},
                {"article_id": "183", "comment_id": 4, "original_comment": "These deeper roots may include biological instincts.", "reasoning": "Step 6: Extends causal reasoning by proposing a specific mechanism ('biological instincts').", "comment_tag": "Inference: Explanatory"},
                {"article_id": "183", "comment_id": 4, "original_comment": "Biological instincts may be unique to each gender.", "reasoning": "Step 7: States a general claim about biology without linking it causally to chart data.", "comment_tag": "Prior Knowledge: Background"},
                {"article_id": "183", "comment_id": 4, "original_comment": "I have a very close relationship with my parents.", "reasoning": "Step 3: Personal self-reference about the commenter's own life.", "comment_tag": "Prior Knowledge: Personal / Episodic"},
                {"article_id": "183", "comment_id": 4, "original_comment": "I am an only daughter.", "reasoning": "Step 3: Personal self-reference.", "comment_tag": "Prior Knowledge: Personal / Episodic"},
                {"article_id": "183", "comment_id": 4, "original_comment": "People frequently mention that I reach out to these adults more than other people do.", "reasoning": "Step 3: Recounts a personal experience ('People mention that I...').", "comment_tag": "Prior Knowledge: Personal / Episodic"},
                {"article_id": "183", "comment_id": 4, "original_comment": "Some people think that daughters are favorites for a reason.", "reasoning": "Step 4b: Expresses a general evaluative stance ('favorites for a reason') without prescribing action.", "comment_tag": "Evaluative: Reactive"},
            ]
        },
    },
    # Example 3 — article 181 (electricity generation), contrastive pairs
    # Demonstrates Visual Observation subtypes vs Explanatory vs Predictive vs Evaluative
    {
        "input": [
            {"article_id": "181", "comment_id": 7, "original_comment": "The graph uses a stacked area chart."},
            {"article_id": "181", "comment_id": 7, "original_comment": "The y-axis is counted in gigawatt hours."},
            {"article_id": "181", "comment_id": 7, "original_comment": "Coal generated over 5000 TWh in China by 2020."},
            {"article_id": "181", "comment_id": 7, "original_comment": "China's electricity generation increased sharply after 2000."},
            {"article_id": "181", "comment_id": 7, "original_comment": "China's electricity generation increased because of rapid industrialization."},
            {"article_id": "181", "comment_id": 7, "original_comment": "If coal use continues at this rate, emissions will keep rising."},
            {"article_id": "181", "comment_id": 7, "original_comment": "This trend is concerning."},
        ],
        "output": {
            "classifications": [
                {"article_id": "181", "comment_id": 7, "original_comment": "The graph uses a stacked area chart.", "reasoning": "Step 2a: Names the chart type, a visual structural property.", "comment_tag": "Visual Observation: Chart Structure & Text"},
                {"article_id": "181", "comment_id": 7, "original_comment": "The y-axis is counted in gigawatt hours.", "reasoning": "Step 2a: Reads the axis label/unit text on the chart without extracting a data value.", "comment_tag": "Visual Observation: Chart Structure & Text"},
                {"article_id": "181", "comment_id": 7, "original_comment": "Coal generated over 5000 TWh in China by 2020.", "reasoning": "Step 2b: Extracts a specific data value ('over 5000 TWh') for one variable at one time point.", "comment_tag": "Visual Observation: Data Point Extraction"},
                {"article_id": "181", "comment_id": 7, "original_comment": "China's electricity generation increased sharply after 2000.", "reasoning": "Step 2c: Traces a trend ('increased sharply') across multiple time points rather than reporting a single value.", "comment_tag": "Visual Observation: Cross-point Pattern Recognition"},
                {"article_id": "181", "comment_id": 7, "original_comment": "China's electricity generation increased because of rapid industrialization.", "reasoning": "Step 6: 'Because of rapid industrialization' proposes a cause for the chart pattern. Although it mentions a trend, the primary function is causal explanation.", "comment_tag": "Inference: Explanatory"},
                {"article_id": "181", "comment_id": 7, "original_comment": "If coal use continues at this rate, emissions will keep rising.", "reasoning": "Step 8: Future-oriented hypothetical reasoning ('if... will').", "comment_tag": "Inference: Predictive / Hypothetical"},
                {"article_id": "181", "comment_id": 7, "original_comment": "This trend is concerning.", "reasoning": "Step 4b: Expresses an emotional reaction ('concerning') to the data without prescribing action.", "comment_tag": "Evaluative: Reactive"},
            ]
        },
    },
    # Example 4 — article 173 (admissions), contrastive: Background vs Explanatory vs Curiosity
    {
        "input": [
            {"article_id": "173", "comment_id": 3, "original_comment": "Legacy admissions have been controversial in higher education."},
            {"article_id": "173", "comment_id": 3, "original_comment": "The skew toward high-income applicants may be driven by legacy admission policies."},
            {"article_id": "173", "comment_id": 3, "original_comment": "I wonder whether this data accounts for financial aid recipients."},
        ],
        "output": {
            "classifications": [
                {"article_id": "173", "comment_id": 3, "original_comment": "Legacy admissions have been controversial in higher education.", "reasoning": "Step 7: Standalone fact about the world not directly shown in the chart, with no causal link to chart data.", "comment_tag": "Prior Knowledge: Background"},
                {"article_id": "173", "comment_id": 3, "original_comment": "The skew toward high-income applicants may be driven by legacy admission policies.", "reasoning": "Step 6: Links a chart observation ('skew toward high-income') to a proposed cause ('legacy admission policies'). 'May be driven by' signals causal reasoning.", "comment_tag": "Inference: Explanatory"},
                {"article_id": "173", "comment_id": 3, "original_comment": "I wonder whether this data accounts for financial aid recipients.", "reasoning": "Step 5: 'I wonder whether' is an explicit expression of uncertainty and desire for information.", "comment_tag": "Curiosity"},
            ]
        },
    },
    # Example 5 — article 92 (air pollution), demonstrating Background vs Explanatory vs Curiosity vs Prescriptive
    {
        "input": [
            {"article_id": "92", "comment_id": 141, "original_comment": "Carbon dioxide damages the atmosphere."},
            {"article_id": "92", "comment_id": 141, "original_comment": "This could be because there is a high use in vehicles and motors."},
            {"article_id": "92", "comment_id": 141, "original_comment": "I wonder what are other ways that air pollution has caused death for many Americans."},
            {"article_id": "92", "comment_id": 141, "original_comment": "People should start using more recyclable materials and resources."},
        ],
        "output": {
            "classifications": [
                {"article_id": "92", "comment_id": 141, "original_comment": "Carbon dioxide damages the atmosphere.", "reasoning": "Step 7: Standalone factual claim about environmental science, not derived from or linked to the chart data.", "comment_tag": "Prior Knowledge: Background"},
                {"article_id": "92", "comment_id": 141, "original_comment": "This could be because there is a high use in vehicles and motors.", "reasoning": "Step 6: 'Could be because' proposes a causal explanation for a chart observation, linking it to vehicle/motor use.", "comment_tag": "Inference: Explanatory"},
                {"article_id": "92", "comment_id": 141, "original_comment": "I wonder what are other ways that air pollution has caused death for many Americans.", "reasoning": "Step 5: 'I wonder what' signals an explicit information need and desire for additional explanation.", "comment_tag": "Curiosity"},
                {"article_id": "92", "comment_id": 141, "original_comment": "People should start using more recyclable materials and resources.", "reasoning": "Step 4a: 'Should start using' contains a deontic operator prescribing action.", "comment_tag": "Evaluative: Prescriptive"},
            ]
        },
    },
    # Example 6 — article 95 (COVID vaccines), demonstrating Predictive vs Prescriptive vs Personal vs Background
    {
        "input": [
            {"article_id": "95", "comment_id": 154, "original_comment": "If restrictions are lifted in Texas, then results similar to graph C may occur in Texas."},
            {"article_id": "95", "comment_id": 154, "original_comment": "We should do our best to stay on a model that is reflected by graph A and graph B."},
            {"article_id": "95", "comment_id": 154, "original_comment": "I am a student."},
            {"article_id": "95", "comment_id": 154, "original_comment": "More than half a million Americans died from the pandemic."},
        ],
        "output": {
            "classifications": [
                {"article_id": "95", "comment_id": 154, "original_comment": "If restrictions are lifted in Texas, then results similar to graph C may occur in Texas.", "reasoning": "Step 8: Hypothetical conditional ('if… then… may occur') projects a future outcome beyond the chart data.", "comment_tag": "Inference: Predictive / Hypothetical"},
                {"article_id": "95", "comment_id": 154, "original_comment": "We should do our best to stay on a model that is reflected by graph A and graph B.", "reasoning": "Step 4a: 'We should' prescribes a course of action.", "comment_tag": "Evaluative: Prescriptive"},
                {"article_id": "95", "comment_id": 154, "original_comment": "I am a student.", "reasoning": "Step 3: Personal self-reference about the commenter's own identity.", "comment_tag": "Prior Knowledge: Personal / Episodic"},
                {"article_id": "95", "comment_id": 154, "original_comment": "More than half a million Americans died from the pandemic.", "reasoning": "Step 7: Standalone factual claim about pandemic deaths, not directly visible in the chart data.", "comment_tag": "Prior Knowledge: Background"},
            ]
        },
    },
    # Example 7 — article 116 (CO2 emissions), demonstrating Meta vs Curiosity vs Background vs Predictive
    {
        "input": [
            {"article_id": "116", "comment_id": 167, "original_comment": "A catchy headline for the graph would be 'Developing Countries Lead the Charge for Carbon Emissions!'."},
            {"article_id": "116", "comment_id": 167, "original_comment": "I wonder what causes this big difference between the small ratio and the big ratio of countries."},
            {"article_id": "116", "comment_id": 167, "original_comment": "The rich countries make up much less of the world's population than the less rich countries."},
            {"article_id": "116", "comment_id": 167, "original_comment": "If we do not see a drastic change in greenhouse gas emissions then humanity will face these dangers in around 20 years."},
        ],
        "output": {
            "classifications": [
                {"article_id": "116", "comment_id": 167, "original_comment": "A catchy headline for the graph would be 'Developing Countries Lead the Charge for Carbon Emissions!'.", "reasoning": "Step 1: A user-generated headline about the graph artifact, not text read from the chart itself.", "comment_tag": "Meta / Paratext"},
                {"article_id": "116", "comment_id": 167, "original_comment": "I wonder what causes this big difference between the small ratio and the big ratio of countries.", "reasoning": "Step 5: 'I wonder what causes' expresses curiosity and desire for explanation.", "comment_tag": "Curiosity"},
                {"article_id": "116", "comment_id": 167, "original_comment": "The rich countries make up much less of the world's population than the less rich countries.", "reasoning": "Step 7: Standalone factual claim about global demographics, not derived from the chart.", "comment_tag": "Prior Knowledge: Background"},
                {"article_id": "116", "comment_id": 167, "original_comment": "If we do not see a drastic change in greenhouse gas emissions then humanity will face these dangers in around 20 years.", "reasoning": "Step 8: Future-oriented hypothetical reasoning ('if… then… will face') projecting beyond the chart data.", "comment_tag": "Inference: Predictive / Hypothetical"},
            ]
        },
    },
    # Example 8 — diverse Personal/episodic subtypes and Reactive evaluation
    {
        "input": [
            {"article_id": "128", "comment_id": 202, "original_comment": "I am an immigrant from Venezuela."},
            {"article_id": "128", "comment_id": 202, "original_comment": "It is sad to see the refugees leave."},
            {"article_id": "128", "comment_id": 202, "original_comment": "I wonder where all the refugees end up going to."},
            {"article_id": "128", "comment_id": 202, "original_comment": "The refugees are going to need food."},
        ],
        "output": {
            "classifications": [
                {"article_id": "128", "comment_id": 202, "original_comment": "I am an immigrant from Venezuela.", "reasoning": "Step 3: Personal self-reference about the commenter's own identity and immigration background.", "comment_tag": "Prior Knowledge: Personal / Episodic"},
                {"article_id": "128", "comment_id": 202, "original_comment": "It is sad to see the refugees leave.", "reasoning": "Step 4b: 'It is sad' expresses an emotional reaction without prescribing action.", "comment_tag": "Evaluative: Reactive"},
                {"article_id": "128", "comment_id": 202, "original_comment": "I wonder where all the refugees end up going to.", "reasoning": "Step 5: 'I wonder where' expresses explicit curiosity about information not shown.", "comment_tag": "Curiosity"},
                {"article_id": "128", "comment_id": 202, "original_comment": "The refugees are going to need food.", "reasoning": "Step 8: 'Are going to need' is a future-oriented prediction about consequences not shown in the chart.", "comment_tag": "Inference: Predictive / Hypothetical"},
            ]
        },
    },
    # Example 9 — Evaluative: Prescriptive vs Reactive contrastive pairs
    {
        "input": [
            {"article_id": "173", "comment_id": 12, "original_comment": "It is shocking to see how much quarantine affected people's everyday life."},
            {"article_id": "173", "comment_id": 12, "original_comment": "The government needs to do more to address income inequality."},
            {"article_id": "173", "comment_id": 12, "original_comment": "I am worried for those who do not make a large salary."},
            {"article_id": "173", "comment_id": 12, "original_comment": "It seems unfair to accept higher income applicants more than lower income applicants."},
        ],
        "output": {
            "classifications": [
                {"article_id": "173", "comment_id": 12, "original_comment": "It is shocking to see how much quarantine affected people's everyday life.", "reasoning": "Step 4b: 'It is shocking' expresses an emotional reaction to the data without prescribing what should change.", "comment_tag": "Evaluative: Reactive"},
                {"article_id": "173", "comment_id": 12, "original_comment": "The government needs to do more to address income inequality.", "reasoning": "Step 4a: 'Needs to do more' is a deontic operator prescribing governmental action.", "comment_tag": "Evaluative: Prescriptive"},
                {"article_id": "173", "comment_id": 12, "original_comment": "I am worried for those who do not make a large salary.", "reasoning": "Step 4b: 'I am worried' reports the reader's emotional state in response to the data without recommending a course of action.", "comment_tag": "Evaluative: Reactive"},
                {"article_id": "173", "comment_id": 12, "original_comment": "It seems unfair to accept higher income applicants more than lower income applicants.", "reasoning": "Step 4a: 'Unfair' is a moral judgment that implicitly prescribes that the practice should change.", "comment_tag": "Evaluative: Prescriptive"},
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
    """Read an image file and return its base64-encoded string."""
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
    comment_index: int | None = None,
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
    sentences together for discourse context.

    If a single comment exceeds *max_batch_size*, it gets its own batch.
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
            {
                "role": "system",
                "content": (
                    "You are an expert classifier of Attempto Controlled English "
                    "(ACE) sentences from reader comments about data visualizations. "
                    "Follow the decision procedure and disambiguation rules in the "
                    "user prompt exactly. Always produce reasoning before selecting "
                    "a category. Respond with valid JSON only."
                ),
            },
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
    model: str = "gpt-5.4-mini",
    batch_size: int = BATCH_SIZE,
    intermediate_dir: Path | None = DEFAULT_INTERMEDIATE_DIR,
    article_id: str | None = None,
    article_id_folder: Path | None = None,
    images_dir: Path = DEFAULT_IMAGES_DIR,
    comment_index: int | None = None,
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
        comment_index=comment_index,
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
        desc = describe_visualization(client, aid, images_dir=images_dir)
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
                tag = tag[0] if tag else "unknown"
            if not tag:
                tag = "unknown"
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
            "(then reads from <ace-comments-dir>/<article-id>/) or a path to a folder "
            "containing the ACE comment JSONs (folder name is used as article_id for output)."
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

    if args.comment_index is not None and article_id is None and article_id_folder is None:
        parser.error("--comment-index currently requires --article-id (single-article mode).")

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
