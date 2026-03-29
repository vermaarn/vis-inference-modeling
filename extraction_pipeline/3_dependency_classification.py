"""
Build dependency graphs (DAGs) with typed edges over ACE sentences in each comment.

Reads ACE comment JSONs from ace_comments/{article_id}/, batches comments by
total ACE sentence count (like 2_classify_ace_sentences.py), sends each batch to
an LLM with the dependency_classification prompt, and writes one JSON per comment
to ace_dependency_graphs/{article_id}/{comment_index}.json with article_id,
comment_index, and dependency_graph (nodes with id, sentence, depends_on list
of {id, edge_type}).

Edge types: Inferential, Elaborative, Evaluative, Contrastive, Uncategorized.

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
DEFAULT_INTERMEDIATE_DIR = SCRIPT_DIR / "ace_dependency_graphs_batches"

# Max ACE sentences per API call across one or more whole comments (same idea as
# 2_classify_ace_sentences.py — pack whole comments, never split one across calls).
BATCH_SIZE = 200

VALID_EDGE_TYPES = frozenset([
    "Inferential",
    "Elaborative",
    "Evaluative",
    "Contrastive",
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
                    {"id": 1, "edge_type": "Evaluative", "justification": "(1) Edge exists because 'this situation' in B directly references the fossil-fuel dominance described in A. (2) Evaluative because B renders a subjective value judgment ('a problem') about A's factual content, rather than adding detail (Elaborative), reasoning about causes (Inferential), or introducing a counterpoint (Contrastive)."},
                ]},
                {"id": 3, "sentence": "The governments of various countries promote renewable energy.", "depends_on": [
                    {"id": 0, "edge_type": "Inferential", "justification": "(1) Edge exists because the source text uses 'so' to connect A (renewables increasing) to B (governments promoting them). (2) Inferential because B is presented as a causal consequence of A — A is the reason governments act — not a refinement of A (Elaborative) or a subjective stance (Evaluative)."},
                ]},
                {"id": 4, "sentence": "When I was a junior high school student, I had an opportunity.", "depends_on": []},
                {"id": 5, "sentence": "At that time, I thought about a solution to improve this situation.", "depends_on": [
                    {"id": 2, "edge_type": "Elaborative", "justification": "(1) Edge exists because 'this situation' in B anaphorically references the problem described in A. (2) Elaborative because B continues the same topic through anaphoric reference rather than reasoning beyond A (Inferential) or judging it (Evaluative)."},
                    {"id": 4, "edge_type": "Elaborative", "justification": "(1) Edge exists because 'at that time' anchors B to the temporal frame introduced in A ('when I was a junior high school student'). (2) Elaborative because B continues a personal narrative sequence from A rather than introducing a cause, judgment, or counterpoint."},
                ]},
                {"id": 6, "sentence": "The electric power generation in the world decreased around 2008.", "depends_on": []},
                {"id": 7, "sentence": "The electric power generation in the world also decreased around 2021.", "depends_on": []},
                {"id": 8, "sentence": "The Lehman Shock happened in 2008.", "depends_on": [
                    {"id": 6, "edge_type": "Inferential", "justification": "(1) Edge exists because the viewer introduces the Lehman Shock to explain the 2008 decrease in A — B depends on A as the observation being explained. (2) Inferential because B provides a causal explanation for A, introducing new propositional content (an external event) rather than refining A's detail (Elaborative)."},
                ]},
                {"id": 9, "sentence": "The coronavirus spread in 2021.", "depends_on": [
                    {"id": 7, "edge_type": "Inferential", "justification": "(1) Edge exists because the viewer introduces the coronavirus to explain the 2021 decrease in A — B depends on A as the observation being explained. (2) Inferential because B provides a causal explanation for A, introducing a new external event rather than elaborating on A's content."},
                ]},
                {"id": 10, "sentence": "The Lehman Shock caused an economic slump.", "depends_on": [
                    {"id": 8, "edge_type": "Inferential", "justification": "(1) Edge exists because B explicitly states a consequence of the event in A using the word 'caused.' (2) Inferential because B introduces a new effect (economic slump) that follows from A, constituting a causal chain rather than a detail or judgment."},
                ]},
                {"id": 11, "sentence": "The spread of coronavirus caused an economic slump.", "depends_on": [
                    {"id": 9, "edge_type": "Inferential", "justification": "(1) Edge exists because B explicitly states a consequence of the event in A using the word 'caused.' (2) Inferential because B introduces a new effect (economic slump) that follows from A, constituting a causal chain."},
                ]},
                {"id": 12, "sentence": "An economic slump reduces the demand for electric power generation.", "depends_on": [
                    {"id": 10, "edge_type": "Inferential", "justification": "(1) Edge exists because B states the downstream effect of the economic slump described in A. (2) Inferential because B reasons beyond A by introducing a new mechanism (slump → reduced demand), not merely refining A (Elaborative)."},
                    {"id": 11, "edge_type": "Inferential", "justification": "(1) Edge exists because B states the downstream effect of the economic slump described in A. (2) Inferential because B reasons beyond A by introducing the same causal mechanism from a parallel premise."},
                ]},
                {"id": 13, "sentence": "The demand for electric power generation decreased.", "depends_on": [
                    {"id": 12, "edge_type": "Inferential", "justification": "(1) Edge exists because B is the realized outcome of the mechanism described in A (reduced demand → observed decrease). (2) Inferential because B follows as a causal consequence of A rather than restating or refining it."},
                ]},
                {"id": 14, "sentence": "The decrease in electric power generation relates to these events.", "depends_on": [
                    {"id": 6, "edge_type": "Elaborative", "justification": "(1) Edge exists because 'these events' anaphorically refers back to the 2008 decrease mentioned in A. (2) Elaborative because B ties A into a summary through referential continuity rather than introducing a cause or judgment."},
                    {"id": 7, "edge_type": "Elaborative", "justification": "(1) Edge exists because 'these events' anaphorically refers back to the 2021 decrease mentioned in A. (2) Elaborative because B ties A into the same summary through shared reference rather than reasoning beyond it."},
                    {"id": 13, "edge_type": "Inferential", "justification": "(1) Edge exists because 'relates to' links the decrease (B) to the demand reduction explained in A. (2) Inferential because B attributes the decrease to the causal chain in A, going beyond mere restatement."},
                ]},
                {"id": 15, "sentence": "The rate of fossil fuels may remain steady.", "depends_on": [
                    {"id": 1, "edge_type": "Elaborative", "justification": "(1) Edge exists because B discusses the future trajectory of the same fossil-fuel dominance described in A. (2) Elaborative because B stays within A's topic (fossil-fuel rates) and adds a speculative detail, rather than reasoning about causes (Inferential) or expressing a judgment (Evaluative)."},
                ]},
                {"id": 16, "sentence": "If the rate of fossil fuels remains steady then global warming will worsen.", "depends_on": [
                    {"id": 15, "edge_type": "Inferential", "justification": "(1) Edge exists because 'if…then' explicitly takes A's premise and states a predicted consequence. (2) Inferential because B introduces a new prediction contingent on A, not a refinement (Elaborative) or a subjective reaction (Evaluative)."},
                ]},
                {"id": 17, "sentence": "If the rate of fossil fuels remains steady then air pollution will worsen.", "depends_on": [
                    {"id": 15, "edge_type": "Inferential", "justification": "(1) Edge exists because 'if…then' explicitly takes A's premise and states a second predicted consequence. (2) Inferential because B introduces a new prediction (air pollution) contingent on A, parallel to node 16."},
                ]},
                {"id": 18, "sentence": "Akita has a precious natural environment.", "depends_on": []},
                {"id": 19, "sentence": "I live in Akita.", "depends_on": []},
                {"id": 20, "sentence": "If global warming worsens then the natural environment in Akita could be negatively affected.", "depends_on": [
                    {"id": 16, "edge_type": "Inferential", "justification": "(1) Edge exists because B chains a further 'if…then' consequence from the global warming prediction in A. (2) Inferential because B predicts a new outcome (local environmental harm) contingent on A's conditional, going beyond A's content."},
                    {"id": 18, "edge_type": "Elaborative", "justification": "(1) Edge exists because B references the same 'natural environment' in Akita that A describes. (2) Elaborative because B specifies which environment is at stake by drawing on A's detail, not reasoning about causes or expressing a stance."},
                    {"id": 19, "edge_type": "Elaborative", "justification": "(1) Edge exists because 'Akita' in B connects to the personal location introduced in A ('I live in Akita'). (2) Elaborative because the connection is through a shared referent (the place name) that anchors B to A's context, not through causation or judgment."},
                ]},
                {"id": 21, "sentence": "Akita cedars grow in Akita.", "depends_on": [
                    {"id": 19, "edge_type": "Elaborative", "justification": "(1) Edge exists because the shared place name 'Akita' connects B to A's personal location statement. (2) Elaborative because B adds a factual detail about Akita that extends A's context, not a cause, judgment, or counterpoint."},
                ]},
                {"id": 22, "sentence": "If global warming worsens then the growth of Akita cedars may deteriorate.", "depends_on": [
                    {"id": 16, "edge_type": "Inferential", "justification": "(1) Edge exists because B chains another conditional consequence from the global warming prediction in A. (2) Inferential because B introduces a specific predicted outcome (cedar growth deteriorating) contingent on A."},
                    {"id": 21, "edge_type": "Elaborative", "justification": "(1) Edge exists because B specifies the impact on the species (Akita cedars) introduced in A. (2) Elaborative because B contextualizes the prediction by drawing on A's detail about cedars, not introducing a new cause or stance."},
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
                    {"id": 1, "edge_type": "Contrastive", "justification": "(1) Edge exists because B directly references 'clean energy sources' from A and compares fossil fuels against them. (2) Contrastive because 'consistently higher' sets up a tension with A's growth claim — the two observations pull in different directions — rather than refining A (Elaborative) or reasoning from it (Inferential)."},
                ]},
                {"id": 3, "sentence": "Coal is a popular energy source.", "depends_on": []},
                {"id": 4, "sentence": "Gas is a popular energy source.", "depends_on": []},
                {"id": 5, "sentence": "Wind power has experienced much growth recently.", "depends_on": [
                    {"id": 1, "edge_type": "Elaborative", "justification": "(1) Edge exists because wind power is an instance of the 'clean energy sources' mentioned in A. (2) Elaborative because B specifies a particular clean source (general → specific), staying within A's informational territory rather than introducing a cause or judgment."},
                ]},
                {"id": 6, "sentence": "Solar power has experienced much growth recently.", "depends_on": [
                    {"id": 1, "edge_type": "Elaborative", "justification": "(1) Edge exists because solar power is another instance of the 'clean energy sources' in A. (2) Elaborative because B specifies a second clean source (general → specific), parallel to wind power, without introducing new reasoning."},
                ]},
                {"id": 7, "sentence": "New ways to generate electricity have proliferated.", "depends_on": [
                    {"id": 5, "edge_type": "Elaborative", "justification": "(1) Edge exists because B generalizes from the specific wind growth in A to a broader claim about new methods. (2) Elaborative because B zooms out from A's specific detail to a general summary (specific → general), not a cause or stance."},
                    {"id": 6, "edge_type": "Elaborative", "justification": "(1) Edge exists because B generalizes from the specific solar growth in A to the same broader claim. (2) Elaborative because this is a specific-to-general refinement, not a causal, evaluative, or contrastive move."},
                ]},
                {"id": 8, "sentence": "The use of gas has steadily increased.", "depends_on": [
                    {"id": 4, "edge_type": "Elaborative", "justification": "(1) Edge exists because B adds a trend (steady increase) to the general claim about gas's popularity in A. (2) Elaborative because B adds quantitative detail within A's topic rather than reasoning about why gas is popular (Inferential) or judging it (Evaluative)."},
                ]},
                {"id": 9, "sentence": "The use of coal has steadily increased.", "depends_on": [
                    {"id": 3, "edge_type": "Elaborative", "justification": "(1) Edge exists because B adds a trend (steady increase) to the general claim about coal's popularity in A. (2) Elaborative because B refines A with trend detail, not a cause or judgment."},
                ]},
                {"id": 10, "sentence": "I wonder why the use of gas and coal has steadily increased.", "depends_on": [
                    {"id": 7, "edge_type": "Contrastive", "justification": "(1) Edge exists because B's wondering about fossil-fuel growth is prompted by the tension with A's claim that new clean methods have proliferated. (2) Contrastive because the coexistence of proliferating clean methods (A) and rising fossil fuels (B) creates a surprising tension, not a same-direction refinement (Elaborative)."},
                    {"id": 8, "edge_type": "Evaluative", "justification": "(1) Edge exists because 'I wonder why' directly questions the gas increase described in A. (2) Evaluative because B expresses the viewer's subjective curiosity about A, not a factual refinement (Elaborative) or a causal explanation (Inferential)."},
                    {"id": 9, "edge_type": "Evaluative", "justification": "(1) Edge exists because 'I wonder why' directly questions the coal increase described in A. (2) Evaluative because B is a curiosity response to A's factual content, not an elaboration or inference."},
                ]},
                {"id": 11, "sentence": "The accessibility of resources may explain the increase.", "depends_on": [
                    {"id": 10, "edge_type": "Inferential", "justification": "(1) Edge exists because 'may explain' offers an answer to the question posed in A. (2) Inferential because B introduces a causal explanation (resource accessibility) that reasons beyond A, rather than refining A (Elaborative) or expressing a further stance (Evaluative)."},
                ]},
                {"id": 12, "sentence": "The global population has grown.", "depends_on": []},
                {"id": 13, "sentence": "I wonder how much global population growth affects how electricity is generated.", "depends_on": [
                    {"id": 0, "edge_type": "Evaluative", "justification": "(1) Edge exists because B's wonder about electricity generation is triggered by the increase observed in A. (2) Evaluative because 'I wonder how much' is a subjective curiosity response to A's factual observation, not a factual refinement or causal claim."},
                    {"id": 12, "edge_type": "Evaluative", "justification": "(1) Edge exists because B directly questions the impact of the population growth stated in A. (2) Evaluative because B registers curiosity about A rather than explaining it (Inferential) or adding detail to it (Elaborative)."},
                ]},
                {"id": 14, "sentence": "Statistics about electricity generation vary from country to country.", "depends_on": []},
                {"id": 15, "sentence": "I wonder how greatly these statistics vary from country to country.", "depends_on": [
                    {"id": 14, "edge_type": "Evaluative", "justification": "(1) Edge exists because 'these statistics' in B references the variation stated in A. (2) Evaluative because 'I wonder how greatly' expresses curiosity about A's claim rather than adding factual detail (Elaborative) or reasoning about causes (Inferential)."},
                ]},
                {"id": 16, "sentence": "I canvassed for a local democratic campaign.", "depends_on": []},
                {"id": 17, "sentence": "During the canvassing, I noticed a fact.", "depends_on": [
                    {"id": 16, "edge_type": "Elaborative", "justification": "(1) Edge exists because 'during the canvassing' situates B within the same episode introduced in A. (2) Elaborative because B continues A's personal narrative sequence rather than reasoning from it (Inferential) or judging it (Evaluative)."},
                ]},
                {"id": 18, "sentence": "The fact is that clean energy was a very important issue.", "depends_on": [
                    {"id": 17, "edge_type": "Elaborative", "justification": "(1) Edge exists because B spells out the content of the vague 'fact' mentioned in A. (2) Elaborative because B specifies A's reference (filling in detail), not introducing a cause, judgment, or counterpoint."},
                ]},
                {"id": 19, "sentence": "Everyone is impacted by the negative effects of climate change.", "depends_on": []},
                {"id": 20, "sentence": "The expenditure of fossil fuels contributes to global warming.", "depends_on": [
                    {"id": 2, "edge_type": "Inferential", "justification": "(1) Edge exists because B states a causal mechanism linking the high fossil-fuel use from A to global warming. (2) Inferential because B introduces a new propositional claim (fossil fuels → warming) that reasons beyond A's observation, not merely restating it (Elaborative)."},
                    {"id": 19, "edge_type": "Inferential", "justification": "(1) Edge exists because B provides the specific mechanism (fossil fuels → warming) behind the broad climate impact in A. (2) Inferential because B explains the cause of A's claim rather than refining it (Elaborative) or reacting to it (Evaluative)."},
                ]},
                {"id": 21, "sentence": "It has been proven that the expenditure of fossil fuels greatly contributes to global warming.", "depends_on": [
                    {"id": 20, "edge_type": "Elaborative", "justification": "(1) Edge exists because B restates A's claim with added emphasis ('it has been proven,' 'greatly'). (2) Elaborative because B strengthens and intensifies the same assertion from A without introducing a new cause, judgment, or counterpoint."},
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
        wrapped_in = {"comments": [ex["input"]]}
        wrapped_out = {"results": [ex["output"]]}
        inp = json.dumps(wrapped_in, indent=2, ensure_ascii=False)
        out = json.dumps(wrapped_out, indent=2, ensure_ascii=False)
        sections.append(
            f"### Example {i}\n\n"
            f"**Input:**\n```json\n{inp}\n```\n\n"
            f"**Output:**\n```json\n{out}\n```"
        )
    return "\n\n".join(sections)


def load_prompt(prompt_path: Path) -> str:
    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def _comment_payload(comment_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "article_id": comment_data.get("article_id", ""),
        "comment_index": comment_data.get("comment_index", 0),
        "ace_sentences": comment_data.get("ace_sentences") or [],
        "order": comment_data.get("order") or {},
    }


def _batch_comments_by_sentence_count(
    comments: List[Dict[str, Any]],
    max_sentences: int,
) -> List[List[Dict[str, Any]]]:
    """
    Pack whole comments into batches so the total ACE sentence count per batch
    does not exceed max_sentences. Mirrors 2_classify_ace_sentences.py: never
    split a single comment across batches; oversized comments get their own batch.
    """
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_count = 0

    for c in comments:
        n = len(c.get("ace_sentences") or [])
        if n == 0:
            continue
        if n > max_sentences:
            if current:
                batches.append(current)
                current = []
                current_count = 0
            batches.append([c])
            continue
        if current and current_count + n > max_sentences:
            batches.append(current)
            current = []
            current_count = 0
        current.append(c)
        current_count += n

    if current:
        batches.append(current)

    return batches


def build_dependency_graph_batch(
    client: OpenAI,
    prompt_text: str,
    comment_batch: List[Dict[str, Any]],
    model: str,
) -> List[Dict[str, Any]]:
    """
    Send one or more comments to the model and return a list of response objects
    (article_id, comment_index, dependency_graph), same order as comment_batch.
    """
    if not comment_batch:
        return []

    input_payload = {"comments": [_comment_payload(c) for c in comment_batch]}
    user_content = prompt_text + "\n\nInput:\n" + json.dumps(
        input_payload, indent=2, ensure_ascii=False
    )

    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You build dependency graphs with typed edges over ACE sentences. "
                    "Respond only with valid JSON: an object with a `results` array, "
                    "one graph per input comment, in order."
                ),
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

    results = out.get("results")
    if not isinstance(results, list):
        raise ValueError(
            f"Model response missing or invalid 'results' list: {list(out.keys())}"
        )
    if len(results) != len(comment_batch):
        raise ValueError(
            f"Expected {len(comment_batch)} result(s), got {len(results)}"
        )

    normalized: List[Dict[str, Any]] = []
    for i, item in enumerate(results):
        if not isinstance(item, dict):
            raise ValueError(f"results[{i}] is not an object")
        cd = comment_batch[i]
        dependency_graph = item.get("dependency_graph")
        if not isinstance(dependency_graph, list):
            raise ValueError(
                f"results[{i}] missing or invalid 'dependency_graph' list: {list(item.keys())}"
            )
        item["article_id"] = cd.get("article_id", item.get("article_id"))
        item["comment_index"] = cd.get("comment_index", item.get("comment_index"))
        normalized.append(item)

    total_nodes = sum(len(r.get("dependency_graph") or []) for r in normalized)
    print(
        f"  Batch ({len(comment_batch)} comment(s), {total_nodes} nodes total): "
        f"API call took {elapsed:.2f}s"
    )
    return normalized


def run_dependency_classification(
    ace_comments_dir: Path,
    prompt_path: Path,
    output_dir: Path,
    article_id: str,
    api_key: str | None = None,
    model: str = "gpt-5.2",
    comment_index: int | None = None,
    batch_size: int = BATCH_SIZE,
    intermediate_dir: Path | None = DEFAULT_INTERMEDIATE_DIR,
) -> List[Dict[str, Any]]:
    """
    Load all comments for article_id, build dependency graphs in batches (by ACE
    sentence count), and write one JSON per comment to
    output_dir/{article_id}/{comment_index}.json.
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

    prepared: List[Dict[str, Any]] = []
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

        ci = comment_data.get("comment_index")
        if ci is None:
            ci = int(json_path.stem) if json_path.stem.isdigit() else 0
        comment_data["comment_index"] = ci
        if "article_id" not in comment_data:
            comment_data["article_id"] = article_id
        prepared.append(comment_data)

    if not prepared:
        print("No comments with ace_sentences to process.")
        return []

    batches = _batch_comments_by_sentence_count(prepared, batch_size)
    total_sentences = sum(len(c.get("ace_sentences") or []) for c in prepared)
    print(
        f"Split {len(prepared)} comment(s) ({total_sentences} ACE sentences) into "
        f"{len(batches)} batch(es) (max {batch_size} sentences per batch)."
    )

    if intermediate_dir is not None:
        intermediate_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    out_article_dir = output_dir / article_id
    out_article_dir.mkdir(parents=True, exist_ok=True)

    for batch_index, batch in enumerate(batches):
        batch_num = batch_index + 1
        print(f"Dependency batch {batch_num}/{len(batches)} ({len(batch)} comment(s))...")
        try:
            outs = build_dependency_graph_batch(client, prompt_text, batch, model)
        except (ValueError, Exception) as e:
            cis = [c.get("comment_index") for c in batch]
            print(f"  Error on batch {batch_num} (comments {cis}): {e}")
            continue

        batch_for_disk: List[Dict[str, Any]] = []
        for out in outs:
            results.append(out)
            ci = out.get("comment_index")
            out_path = out_article_dir / f"{ci}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"  Wrote {out_path}")
            batch_for_disk.append(out)

        if intermediate_dir is not None:
            batch_path = intermediate_dir / f"{article_id}_batch_{batch_num:04d}.json"
            with batch_path.open("w", encoding="utf-8") as f:
                json.dump(batch_for_disk, f, indent=2, ensure_ascii=False)
            print(f"  Saved intermediate batch to {batch_path}")

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
        default="gpt-5.4-mini",
        help="OpenAI model (default: gpt-5.4-mini)",
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=(
            "Max total ACE sentences per API call across one or more whole comments "
            f"(default: {BATCH_SIZE})"
        ),
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=DEFAULT_INTERMEDIATE_DIR,
        help=(
            "Directory for per-batch JSON snapshots "
            f"(default: {DEFAULT_INTERMEDIATE_DIR})"
        ),
    )
    parser.add_argument(
        "--no-intermediate",
        action="store_true",
        help="Do not write per-batch intermediate JSON files.",
    )
    args = parser.parse_args()

    intermediate_dir: Path | None = None if args.no_intermediate else args.intermediate_dir

    run_dependency_classification(
        ace_comments_dir=args.ace_comments_dir,
        prompt_path=args.prompt_file,
        output_dir=args.output_dir,
        article_id=args.article_id,
        api_key=args.api_key,
        model=args.model,
        comment_index=args.comment_index,
        batch_size=args.batch_size,
        intermediate_dir=intermediate_dir,
    )


if __name__ == "__main__":
    main()
