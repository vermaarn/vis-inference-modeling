"""
Histogram of word counts per raw_comment across all JSON files under ace_comments/,
plus a histogram of unique comment_tag subcategories per comment from ace_classifications/.
Both charts are written side by side (horizontal concat) to one HTML file.

Each ace_comments file is one comment; words are counted by splitting on whitespace after stripping.
Word-count chart: Altair bar mark with 10-word-wide bins on word_count (0–9, 10–19, …).
  Includes a slider: minimum unique subcategories (from classifications); comments below
  that threshold are excluded from this histogram only.
Subcategory chart: per (article_id, comment_id), count of distinct comment_tag values.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd

ACE_COMMENTS_DIR = Path(__file__).resolve().parent.parent / "ace_comments"
CLASSIFICATIONS_DIR = Path(__file__).resolve().parent.parent / "ace_classifications"
DEFAULT_HTML = Path(__file__).resolve().parent / "figures/comment_word_count_histogram.html"
BIN_WIDTH = 10
WORD_CHART_WIDTH = 520
SUBCAT_CHART_WIDTH = 400
CHART_HEIGHT = 400
SUBCAT_BAR_COLOR = "#4c78a8"


def word_count(text: str | None) -> int:
    if not text:
        return 0
    s = str(text).strip()
    if not s:
        return 0
    return len(s.split())


def load_comment_rows() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(ACE_COMMENTS_DIR.glob("**/*.json")):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("raw_comment", "")
        rows.append(
            {
                "article_id": str(data.get("article_id", path.parent.name)),
                "comment_index": data.get("comment_index"),
                "rel_path": str(path.relative_to(ACE_COMMENTS_DIR)),
                "word_count": word_count(raw),
            }
        )
    return rows


def load_classifications_df() -> pd.DataFrame:
    """Rows from ace_classifications/*.json with article_id, comment_id, comment_tag."""
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
    df = pd.DataFrame(rows)
    if "comment_tag" not in df.columns:
        raise KeyError("ace_classifications data has no comment_tag column.")
    df = df[["article_id", "comment_id", "comment_tag"]].copy()
    df["article_id"] = df["article_id"].astype(str)
    df["comment_id"] = df["comment_id"].astype(int)
    return df


def merge_word_counts_with_subcat_counts(
    df: pd.DataFrame, df_cls: pd.DataFrame
) -> pd.DataFrame:
    """Per-comment word_count plus n_subcategories (0 if no classification rows)."""
    n_sub = (
        df_cls.dropna(subset=["comment_tag"])
        .groupby(["article_id", "comment_id"], observed=True)["comment_tag"]
        .nunique()
        .reset_index(name="n_subcategories")
    )
    out = df.copy()
    out["comment_id"] = out["comment_index"].astype(int)
    out = out.merge(n_sub, on=["article_id", "comment_id"], how="left")
    out["n_subcategories"] = out["n_subcategories"].fillna(0).astype(int)
    return out


def build_word_count_chart(merged: pd.DataFrame) -> alt.Chart:
    """Histogram of word_count; slider filters to comments with n_subcategories >= N (0 = no filter)."""
    max_sub = max(1, int(merged["n_subcategories"].max()))
    min_subcategories = alt.param(
        value=0,
        bind=alt.binding_range(
            min=0,
            max=max_sub,
            step=1,
            name="Min unique subcategories: ",
        ),
    )
    return (
        alt.Chart(merged)
        .transform_filter(alt.datum.n_subcategories >= min_subcategories)
        .mark_bar()
        .encode(
            x=alt.X(
                "word_count:Q",
                bin=alt.Bin(step=BIN_WIDTH, nice=False),
                title="Word count (bin)",
                axis=alt.Axis(labelAngle=-35),
            ),
            y=alt.Y("count():Q", title="Number of comments"),
            tooltip=[
                alt.Tooltip("count()", title="Comments in bin"),
            ],
        )
        .properties(
            title=f"Word count per raw_comment ({BIN_WIDTH}-word bins)",
            width=WORD_CHART_WIDTH,
            height=CHART_HEIGHT,
        )
        .add_params(min_subcategories)
    )


def build_subcategories_per_comment_chart(df_cls: pd.DataFrame) -> alt.Chart:
    """Histogram: number of distinct comment_tag values per (article_id, comment_id)."""
    comment_keys = ["article_id", "comment_id"]
    subcat_per_comment = (
        df_cls.dropna(subset=["comment_tag"])
        .groupby(comment_keys, observed=True)["comment_tag"]
        .nunique()
        .reset_index(name="n_subcategories")
    )
    n_comments = len(subcat_per_comment)
    return (
        alt.Chart(subcat_per_comment)
        .mark_bar(color=SUBCAT_BAR_COLOR)
        .encode(
            x=alt.X(
                "n_subcategories:O",
                title="Number of unique subcategories per comment",
            ),
            y=alt.Y("count():Q", title="Number of comments"),
            tooltip=[
                "n_subcategories",
                alt.Tooltip("count()", title="Comments"),
            ],
        )
        .properties(
            width=SUBCAT_CHART_WIDTH,
            height=CHART_HEIGHT,
            title=f"Unique subcategories per comment ({n_comments} comments)",
        )
    )


def build_chart(merged: pd.DataFrame, df_cls: pd.DataFrame) -> alt.HConcatChart:
    word_chart = build_word_count_chart(merged)
    subcat_chart = build_subcategories_per_comment_chart(df_cls)
    return alt.hconcat(word_chart, subcat_chart).resolve_scale(y="independent")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_HTML,
        help=f"Write interactive HTML (default: {DEFAULT_HTML})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to save per-comment word counts as CSV",
    )
    args = parser.parse_args()

    if not ACE_COMMENTS_DIR.is_dir():
        print(f"Missing directory: {ACE_COMMENTS_DIR}", file=sys.stderr)
        return 1
    if not CLASSIFICATIONS_DIR.is_dir():
        print(f"Missing directory: {CLASSIFICATIONS_DIR}", file=sys.stderr)
        return 1

    rows = load_comment_rows()
    if not rows:
        print(f"No JSON files under {ACE_COMMENTS_DIR}", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    df_cls = load_classifications_df()
    if df_cls.empty:
        print(f"No classification rows under {CLASSIFICATIONS_DIR}", file=sys.stderr)
        return 1

    if args.csv is not None:
        df.to_csv(args.csv, index=False)

    merged = merge_word_counts_with_subcat_counts(df, df_cls)
    chart = build_chart(merged, df_cls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(args.output))
    n_sub = (
        df_cls.dropna(subset=["comment_tag"])
        .groupby(["article_id", "comment_id"], observed=True)["comment_tag"]
        .nunique()
    )
    print(
        f"Wrote {args.output} ({len(df)} ace_comments; "
        f"{n_sub.shape[0]} comments in classifications)"
    )
    print("Word-count panel: slider sets minimum unique subcategories (left chart only).")
    wc = df["word_count"]
    print(
        f"word_count: min={wc.min()} max={wc.max()} "
        f"mean={wc.mean():.1f} median={wc.median():.1f}"
    )
    print(
        f"unique subcategories per comment: min={n_sub.min()} max={n_sub.max()} "
        f"mean={n_sub.mean():.2f} median={n_sub.median():.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
