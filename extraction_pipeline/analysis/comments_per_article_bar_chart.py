"""
Bar chart: number of comments per article from combined_data/*.json, sorted by n_comments (ascending).

Each file is a JSON list of combined comment objects (see 4_combine_dataframe.py).
Writes interactive HTML under analysis/figures/ by default.

Interactivity:
  - Top (overview): drag a horizontal interval to choose which articles appear in the detail pane.
  - Bottom (detail): scroll (wheel) or drag to pan/zoom both axes on the current subset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_PIPELINE = SCRIPT_DIR.parent
COMBINED_DATA_DIR = REPO_PIPELINE / "combined_data"
DEFAULT_HTML = SCRIPT_DIR / "figures" / "comments_per_article_bar_chart.html"
BAR_COLOR = "#72b7b2"


def load_combined_dataframe(combined_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(combined_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for obj in data:
            if not isinstance(obj, dict):
                continue
            aid = obj.get("article_id")
            if aid is None:
                aid = path.stem
            rows.append(
                {
                    "article_id": str(aid),
                    "comment_index": obj.get("comment_index"),
                }
            )
    return pd.DataFrame(rows)


def build_chart(comments_per_article: pd.DataFrame) -> alt.VConcatChart:
    order = comments_per_article["article_id"].tolist()
    x = alt.X(
        "article_id:N",
        sort=order,
        title="Article ID",
        axis=alt.Axis(labelAngle=-90, labelLimit=200),
    )
    y = alt.Y("n_comments:Q", title="Number of Comments")
    tooltip = [
        alt.Tooltip("article_id:N", title="Article"),
        alt.Tooltip("n_comments:Q", title="Comments"),
    ]

    brush = alt.selection_interval(encodings=["x"], empty=False)
    pan_zoom = alt.selection_interval(bind="scales")

    overview = (
        alt.Chart(comments_per_article)
        .mark_bar(color=BAR_COLOR)
        .encode(x=x, y=y, tooltip=tooltip)
        .properties(
            width=1000,
            height=140,
            title="Overview — drag horizontally to select a section",
        )
        .add_params(brush)
    )

    detail = (
        alt.Chart(comments_per_article)
        .mark_bar(color=BAR_COLOR)
        .encode(x=x, y=y, tooltip=tooltip)
        .transform_filter(brush)
        .properties(
            width=1000,
            height=340,
            title="Detail — scroll or drag to pan/zoom",
        )
        .add_params(pan_zoom)
    )

    return (
        alt.vconcat(overview, detail, spacing=12)
        .resolve_scale(x="independent", y="independent")
        .properties(
            title=alt.TitleParams(
                text="Number of Comments per Article (sorted)",
                anchor="start",
            )
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combined-data-dir",
        type=Path,
        default=COMBINED_DATA_DIR,
        help=f"Directory of per-article combined JSON lists (default: {COMBINED_DATA_DIR})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_HTML,
        help=f"Output HTML path (default: {DEFAULT_HTML})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to save comments_per_article summary CSV",
    )
    args = parser.parse_args()
    combined_dir = args.combined_data_dir.resolve()

    if not combined_dir.is_dir():
        print(f"Missing directory: {combined_dir}", file=sys.stderr)
        return 1

    df = load_combined_dataframe(combined_dir)
    if df.empty:
        print(f"No comment rows loaded from {combined_dir}/*.json", file=sys.stderr)
        return 1

    comments_per_article = (
        df.groupby("article_id", observed=True)["comment_index"]
        .count()
        .reset_index(name="n_comments")
        .sort_values("n_comments")
    )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        comments_per_article.to_csv(args.csv, index=False)

    chart = build_chart(comments_per_article)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(args.output))
    print(
        f"Wrote {args.output} ({len(comments_per_article)} articles, {len(df)} comments)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
