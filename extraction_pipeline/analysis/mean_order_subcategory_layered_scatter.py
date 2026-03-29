"""
Layered Altair scatter: x = fine subcategories (canonical 1–10 order), y = mean sentence order.

Data is built from ``combined_data/{article_id}.json`` only: for each comment object, each row in
``sentence_classifications`` supplies ``comment_tag``; ``order`` maps ``original_comment`` (ACE
sentence text) to its 1-based position in that comment. One dataframe row per classified sentence
with a matching ``order`` entry.

Sentence-level rows are written to CSV (one row per ACE statement with category and order).

Small jittered dots = mean order per (article, subcategory), aggregating every matching sentence
across all comments in that article. Colored like `per_article_subcategory_stacked_bars.py`
(FINEGRAIN_COLORS).
Large dots = grand mean order per subcategory (black).
X-axis subcategories are ordered left → right by ascending grand mean order (ties broken by name).

Same tag cleanup / exclusions as `1_across_plot_order.ipynb` and the stacked bars script.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

from per_article_subcategory_stacked_bars import (
    EXCLUDED,
    FINEGRAIN_CATS,
    FINEGRAIN_COLORS,
    _TAG_CLEANUP,
)

warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_PIPELINE = SCRIPT_DIR.parent
COMBINED_DATA_DIR = REPO_PIPELINE / "combined_data"
FIGURES_DIR = SCRIPT_DIR / "figures"
DEFAULT_HTML = FIGURES_DIR / "mean_order_subcategory_layered_scatter.html"
DEFAULT_SENTENCE_CSV = FIGURES_DIR / "mean_order_subcategory_sentences.csv"


def load_sentence_order_df_from_combined() -> pd.DataFrame:
    """One row per (article, comment, ACE sentence) with order and comment_tag from combined_data."""
    rows: list[dict] = []
    for path in sorted(COMBINED_DATA_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for comment in data:
            if not isinstance(comment, dict):
                continue
            article_id = str(comment.get("article_id", "")).strip()
            ci_raw = comment.get("comment_index")
            if ci_raw is None:
                continue
            try:
                comment_index = int(ci_raw)
            except (TypeError, ValueError):
                continue
            order_map = comment.get("order")
            if not isinstance(order_map, dict):
                order_map = {}
            for cls_row in comment.get("sentence_classifications") or []:
                if not isinstance(cls_row, dict):
                    continue
                ace_sentence = cls_row.get("original_comment", "")
                if not ace_sentence or ace_sentence not in order_map:
                    continue
                tag = cls_row.get("comment_tag", "")
                if isinstance(tag, list):
                    tag = tag[0] if tag else ""
                if not tag:
                    tag = "unknown"
                pos = order_map[ace_sentence]
                try:
                    order_num = float(pos)
                except (TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "article_id": article_id,
                        "comment_index": comment_index,
                        "ace_sentence": ace_sentence,
                        "comment_tag": tag,
                        "order": order_num,
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=["article_id", "comment_index", "ace_sentence", "comment_tag", "order"]
        )
    return pd.DataFrame(rows)


def build_merged_df() -> pd.DataFrame:
    df_merged = load_sentence_order_df_from_combined()
    df_merged["comment_tag"] = df_merged["comment_tag"].replace(_TAG_CLEANUP)
    df_merged = df_merged[~df_merged["comment_tag"].isin(EXCLUDED)].copy()
    df_merged = df_merged[df_merged["comment_tag"].isin(FINEGRAIN_CATS)].copy()
    return df_merged


def build_chart(
    df_merged: pd.DataFrame,
    *,
    seed: int,
    width: int,
    height: int,
    jitter_half_width: float,
) -> alt.LayerChart:
    per_article_subcat = (
        df_merged.groupby(["article_id", "comment_tag"], observed=True)["order"]
        .mean()
        .reset_index()
    )

    grand_subcat_means = (
        df_merged.groupby("comment_tag", observed=True)["order"]
        .mean()
        .reset_index()
        .rename(columns={"order": "grand_mean_order"})
    )

    ordered_with_mean = grand_subcat_means.sort_values(
        ["grand_mean_order", "comment_tag"], kind="stable"
    )
    seen = set(ordered_with_mean["comment_tag"].astype(str))
    tail = [c for c in FINEGRAIN_CATS if c not in seen]
    cat_order = ordered_with_mean["comment_tag"].astype(str).tolist() + tail

    subcat_to_color = dict(zip(FINEGRAIN_CATS, FINEGRAIN_COLORS, strict=True))
    color_range = [subcat_to_color[c] for c in cat_order]

    rng = np.random.default_rng(seed)
    cat_to_x = {cat: i for i, cat in enumerate(cat_order)}
    per_article_subcat["jitter"] = (
        per_article_subcat["comment_tag"].map(cat_to_x).astype(float)
        + rng.uniform(-jitter_half_width, jitter_half_width, size=len(per_article_subcat))
    )

    color_scale = alt.Scale(domain=cat_order, range=color_range)

    large_dots = (
        alt.Chart(grand_subcat_means)
        .mark_circle(size=220, color="black")
        .encode(
            x=alt.X(
                "comment_tag:N",
                title=None,
                sort=cat_order,
                axis=alt.Axis(labels=False, ticks=False, grid=False),
            ),
            y=alt.Y(
                "grand_mean_order:Q",
                title="Mean order (all comments)",
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip("comment_tag:N"),
                alt.Tooltip("grand_mean_order:Q", format=".2f"),
            ],
        )
    )

    small_dots = (
        alt.Chart(per_article_subcat)
        .mark_circle(size=55, opacity=0.45)
        .encode(
            x=alt.X(
                "jitter:Q",
                title=None,
                axis=alt.Axis(labels=False, ticks=False, domain=False, grid=False),
                scale=alt.Scale(domain=[-0.5, len(cat_order) - 0.5]),
            ),
            y=alt.Y(
                "order:Q",
                title="Mean order in article",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "comment_tag:N",
                scale=color_scale,
                sort=cat_order,
                legend=alt.Legend(title="Subcategory"),
            ),
            tooltip=[
                alt.Tooltip("comment_tag:N"),
                alt.Tooltip("article_id:N"),
                alt.Tooltip("order:Q", format=".2f", title="mean_order"),
            ],
        )
    )

    return (
        alt.layer(small_dots, large_dots)
        .properties(
            title="Mean sentence order: grand subcategory means and per-article distributions",
            width=width,
            height=height,
        )
        .configure_axis(labelFontSize=13, titleFontSize=15)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--html",
        type=Path,
        default=DEFAULT_HTML,
        help=f"Output Altair HTML (default: {DEFAULT_HTML})",
    )
    parser.add_argument(
        "--sentence-csv",
        type=Path,
        default=DEFAULT_SENTENCE_CSV,
        help=(
            "Write one row per ACE sentence (article_id, comment_index, ace_sentence, "
            f"comment_tag, order) after fine-tag filters (default: {DEFAULT_SENTENCE_CSV})"
        ),
    )
    parser.add_argument(
        "--no-sentence-csv",
        action="store_true",
        help="Do not write the sentence-level CSV",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for x jitter")
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--height", type=int, default=430)
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.18,
        help="Half-width of uniform jitter on category index (matches notebook ±0.18)",
    )
    parser.add_argument(
        "--no-vegafusion",
        action="store_true",
        help="Do not enable vegafusion",
    )
    args = parser.parse_args()

    if not COMBINED_DATA_DIR.is_dir():
        print(f"Missing directory: {COMBINED_DATA_DIR}", file=sys.stderr)
        return 1

    if not args.no_vegafusion:
        try:
            alt.data_transformers.enable("vegafusion")
        except Exception:
            pass

    df_merged = build_merged_df()
    if df_merged.empty:
        print("No rows after merge / filter; nothing to plot.", file=sys.stderr)
        return 1

    chart = build_chart(
        df_merged,
        seed=args.seed,
        width=args.width,
        height=args.height,
        jitter_half_width=args.jitter,
    )
    args.html.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(args.html))
    if not args.no_sentence_csv:
        args.sentence_csv.parent.mkdir(parents=True, exist_ok=True)
        df_merged.to_csv(args.sentence_csv, index=False, encoding="utf-8")
        print(f"Wrote {args.sentence_csv}")
    n_comments = df_merged.groupby(["article_id", "comment_index"], observed=True).ngroups
    print(
        f"Wrote {args.html} ({len(df_merged):,} sentence rows, "
        f"{df_merged['article_id'].nunique()} articles, {n_comments} comments)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
