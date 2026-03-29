"""
Per-article stacked bar chart of sentence subcategory proportions (fine-grained 1–10 order),
excluding Meta / Paratext and Uncategorizable — same logic as 1_across_plot.ipynb.

Writes HTML chart and CSVs under analysis/figures/ by default.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import altair as alt
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_PIPELINE = SCRIPT_DIR.parent
CLASSIFICATIONS_DIR = REPO_PIPELINE / "ace_classifications"
FIGURES_DIR = SCRIPT_DIR / "figures"

DEFAULT_HTML = FIGURES_DIR / "per_article_subcategory_stacked_bars.html"
DEFAULT_PDF = FIGURES_DIR / "per_article_subcategory_stacked_bars.pdf"
DEFAULT_SUBCAT_COUNTS_CSV = FIGURES_DIR / "per_article_subcategory_stacked_bars_subcat_counts.csv"
DEFAULT_FINE_PROPS_CSV = FIGURES_DIR / "per_article_subcategory_stacked_bars_fine_props.csv"
DEFAULT_SUMMARY_CSV = FIGURES_DIR / "per_article_subcategory_stacked_bars_summary.csv"

_TAG_CLEANUP = {
    "L3: Trend and pattern analysis": "Visual Observation: Pattern Recognition",
    "L1: Elemental and encoded properties": "Visual Observation: Chart Elements",
    "L2: Statistical concepts and relations": "Visual Observation: Data Extraction",
    "VO1: Chart Structure, Layout & Text": "Visual Observation: Chart Elements",
    "VO2: Data Point Reading": "Visual Observation: Data Extraction",
    "VO3: Comparisons, Trends & Patterns": "Visual Observation: Pattern Recognition",
    "Background knowledge": "Prior Knowledge: General",
    "Personal/episodic retrieval": "Prior Knowledge: Personal",
    "Prior Knowledge: Personal /Episodic": "Prior Knowledge: Personal",
    "Evaluative / affective judgment": "Evaluative: Reactive",
    "Explanatory inference": "Inference: Causal",
    "Predictive / counterfactual inference": "Inference: Hypothetical",
    "Information need / curiosity": "Curiosity",
    "Meta /Paratext": "Meta / Paratext",
    "Meta / paratext": "Meta / Paratext",
    "Visual Observation: Chart Structure & Text": "Visual Observation: Chart Elements",
    "Visual Observation: Data Point Extraction": "Visual Observation: Data Extraction",
    "Visual Observation: Cross-point Pattern Recognition": "Visual Observation: Pattern Recognition",
    "Prior Knowledge: Background": "Prior Knowledge: General",
    "Prior Knowledge: Personal / Episodic": "Prior Knowledge: Personal",
    "Inference: Explanatory": "Inference: Causal",
    "Inference: Predictive / Hypothetical": "Inference: Hypothetical",
}
EXCLUDED = {"Meta / Paratext", "Uncategorizable"}

FINEGRAIN_CATS = [
    "Visual Observation: Chart Elements",
    "Visual Observation: Data Extraction",
    "Visual Observation: Pattern Recognition",
    "Curiosity",
    "Inference: Causal",
    "Inference: Hypothetical",
    "Prior Knowledge: General",
    "Prior Knowledge: Personal",
    "Evaluative: Reactive",
    "Evaluative: Prescriptive",
]

FINEGRAIN_COLORS = [
    "#5C4033",
    "#7A5A44",
    "#9A7B5F",
    "#6F8572",
    "#4F7A5A",
    "#8FBF9A",
    "#4F6FA3",
    "#7FA9D6",
    "#E3D18A",
    "#C2A14D",
]

# Prefixed labels so ascending alphabetical order = stack bottom → top (a bottom, j top).
FINEGRAIN_PLOT_LABELS = [
    "a. Visual Observation: Chart Elements",
    "b. Visual Observation: Data Extraction",
    "c. Visual Observation: Pattern Recognition",
    "d. Curiosity",
    "e. Inference: Causal",
    "f. Inference: Hypothetical",
    "g. Prior Knowledge: General",
    "h. Prior Knowledge: Personal",
    "i. Evaluative: Reactive",
    "j. Evaluative: Prescriptive",
]
SUBCAT_TO_PLOT_LABEL = dict(zip(FINEGRAIN_CATS, FINEGRAIN_PLOT_LABELS))

SUBCAT_TO_GROUP = {
    "Visual Observation: Chart Elements": "Visual Observation",
    "Visual Observation: Data Extraction": "Visual Observation",
    "Visual Observation: Pattern Recognition": "Visual Observation",
    "Prior Knowledge: General": "Prior Knowledge",
    "Prior Knowledge: Personal": "Prior Knowledge",
    "Evaluative: Prescriptive": "Evaluative Response",
    "Evaluative: Reactive": "Evaluative Response",
    "Inference: Causal": "Inference",
    "Inference: Hypothetical": "Inference",
    "Curiosity": "Curiosity",
}

VO_SUBCATS = [
    "Visual Observation: Chart Elements",
    "Visual Observation: Data Extraction",
    "Visual Observation: Pattern Recognition",
]


def load_sentence_classifications_df() -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(CLASSIFICATIONS_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows.extend(data)
        else:
            rows.append(data)
    if not rows:
        return pd.DataFrame(columns=["article_id", "comment_id", "comment_tag"])
    df_raw = pd.DataFrame(rows)
    df_raw["article_id"] = df_raw["article_id"].astype(str)
    df_raw["comment_id"] = df_raw["comment_id"].astype(int)
    df_raw["comment_tag"] = df_raw["comment_tag"].replace(_TAG_CLEANUP)
    return df_raw[~df_raw["comment_tag"].isin(EXCLUDED)].copy()


def build_subcat_counts_and_chart(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, alt.Chart]:
    df_fine = df[df["comment_tag"].isin(FINEGRAIN_CATS)].copy()
    df_fine["subcat"] = df_fine["comment_tag"]
    df_fine["group"] = df_fine["subcat"].map(SUBCAT_TO_GROUP)

    fine_counts = df_fine.groupby(["article_id", "subcat"], observed=True).size().unstack(fill_value=0)
    fine_props = fine_counts.div(fine_counts.sum(axis=1), axis=0)

    for c in FINEGRAIN_CATS:
        if c not in fine_props.columns:
            fine_props[c] = 0.0
    fine_props = fine_props[FINEGRAIN_CATS]

    summary = fine_props[FINEGRAIN_CATS].describe().T

    fine_props_out = fine_props.copy()
    fine_props_out["Beyond-plot"] = 1 - fine_props_out[VO_SUBCATS].sum(axis=1)
    article_order = fine_props_out["Beyond-plot"].sort_values(ascending=False).index.tolist()

    subcat_counts = (
        df_fine.groupby(["article_id", "group", "subcat"], observed=True)
        .size()
        .reset_index(name="n")
    )
    subcat_counts["group_total"] = subcat_counts.groupby(["article_id", "group"], observed=True)[
        "n"
    ].transform("sum")
    subcat_counts["subcat_prop"] = subcat_counts["n"] / subcat_counts["group_total"]
    subcat_counts["subcat_total"] = subcat_counts.groupby(["article_id"], observed=True)["n"].transform("sum")
    subcat_counts["top_prop"] = subcat_counts["n"] / subcat_counts["subcat_total"]
    subcat_counts["subcat_plot"] = subcat_counts["subcat"].map(SUBCAT_TO_PLOT_LABEL)

    n_all_sentences = len(df)
    global_by_subcat = (
        df_fine.groupby(["group", "subcat"], observed=True)
        .size()
        .reset_index(name="n")
        .sort_values(["group", "subcat"])
    )
    global_total_fine = int(global_by_subcat["n"].sum())
    global_by_subcat["subcat_total"] = global_total_fine
    global_by_subcat["top_prop"] = (
        global_by_subcat["n"] / global_by_subcat["subcat_total"] if global_total_fine else 0.0
    )
    global_by_subcat["group_total"] = global_by_subcat.groupby("group", observed=True)["n"].transform("sum")
    global_by_subcat["subcat_prop"] = global_by_subcat["n"] / global_by_subcat["group_total"]
    global_by_subcat["subcat_plot"] = global_by_subcat["subcat"].map(SUBCAT_TO_PLOT_LABEL)
    global_by_subcat["article_id"] = "All sentences"
    global_by_subcat["prop_of_all_sentences"] = (
        global_by_subcat["n"] / n_all_sentences if n_all_sentences else 0.0
    )

    color_enc = alt.Color(
        "subcat_plot:N",
        sort=FINEGRAIN_PLOT_LABELS,
        scale=alt.Scale(domain=FINEGRAIN_PLOT_LABELS, range=FINEGRAIN_COLORS),
        title="Subcategory",
        legend=alt.Legend(orient="right", title="Subcategory"),
    )

    chart_articles = (
        alt.Chart(
            subcat_counts,
            title=(
                "Sentence subcategory proportions by article\n"
                "(Stacked order 1–10: VO → Curiosity → Inference → Prior Knowledge → Evaluative)"
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("article_id:N", sort=article_order, title="Article"),
            y=alt.Y("top_prop:Q", stack="normalize", title="Proportion of sentences"),
            color=color_enc,
            order=alt.Order("subcat_plot:N", sort="ascending"),
            tooltip=[
                "article_id",
                "group",
                "subcat",
                alt.Tooltip("top_prop:Q", format=".1%", title="Article prop (of all)"),
                alt.Tooltip("subcat_prop:Q", format=".1%", title="Subcat prop (of group)"),
                "n",
            ],
        )
        .properties(width=800, height=350)
    )

    chart_all = (
        alt.Chart(
            global_by_subcat,
            title="Across all sentences",
        )
        .mark_bar()
        .encode(
            x=alt.X("article_id:N", title=None),
            y=alt.Y("top_prop:Q", stack="normalize", title="Proportion of sentences"),
            color=color_enc,
            order=alt.Order("subcat_plot:N", sort="ascending"),
            tooltip=[
                alt.Tooltip("subcat:O", title="Subcategory"),
                "group",
                alt.Tooltip("top_prop:Q", format=".1%", title="Prop (of fine-subcat sentences)"),
                alt.Tooltip("prop_of_all_sentences:Q", format=".1%", title="Prop (of all sentences)"),
                alt.Tooltip("subcat_prop:Q", format=".1%", title="Subcat prop (of group)"),
                "n",
            ],
        )
        .properties(width=72, height=350)
    )

    chart = alt.hconcat(chart_all, chart_articles, spacing=16).resolve_scale(color="shared", y="shared")

    return subcat_counts, fine_props_out, summary, chart


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--html",
        type=Path,
        default=DEFAULT_HTML,
        help=f"Output Altair HTML (default: {DEFAULT_HTML})",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help=f"Output Altair PDF (default: {DEFAULT_PDF})",
    )
    parser.add_argument(
        "--subcat-counts-csv",
        type=Path,
        default=DEFAULT_SUBCAT_COUNTS_CSV,
        help="CSV for the plotting table (article, group, subcat, counts, proportions)",
    )
    parser.add_argument(
        "--fine-props-csv",
        type=Path,
        default=DEFAULT_FINE_PROPS_CSV,
        help="CSV: per-article proportion columns + Beyond-plot",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="CSV: describe().T over articles per subcategory",
    )
    parser.add_argument(
        "--no-vegafusion",
        action="store_true",
        help="Do not enable vegafusion (use if unsupported in your environment)",
    )
    args = parser.parse_args()

    if not CLASSIFICATIONS_DIR.is_dir():
        print(f"Missing directory: {CLASSIFICATIONS_DIR}", file=sys.stderr)
        return 1

    if not args.no_vegafusion:
        try:
            alt.data_transformers.enable("vegafusion")
        except Exception:
            pass

    df = load_sentence_classifications_df()
    if df.empty:
        print(f"No classification rows under {CLASSIFICATIONS_DIR}", file=sys.stderr)
        return 1

    subcat_counts, fine_props_out, summary, chart = build_subcat_counts_and_chart(df)

    args.html.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(args.html))
    chart.save(str(args.pdf))
    subcat_counts.to_csv(args.subcat_counts_csv, index=False)
    fine_props_out.to_csv(args.fine_props_csv)
    summary.round(6).to_csv(args.summary_csv)

    n_fine = len(df[df["comment_tag"].isin(FINEGRAIN_CATS)])
    print(
        f"Wrote {args.html}\n"
        f"       {args.subcat_counts_csv} ({len(subcat_counts)} rows)\n"
        f"       {args.fine_props_csv} ({len(fine_props_out)} articles)\n"
        f"       {args.summary_csv}"
    )
    print(f"Sentences (excl. Meta/Uncategorizable): {len(df):,}; in fine subcats: {n_fine:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
