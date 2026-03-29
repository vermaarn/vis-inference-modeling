"""
Build per-comment subcategory distribution vectors *within* each article,
run UMAP + HDBSCAN clustering, and emit one JSON file per article into
comment_cluster_data/{article_id}.json.

Each JSON has the same shape as cluster_data.json but at the comment level:
  {
    "article_id": "...",
    "subcategories": [...],
    "comments": [
      {"comment_id": 1, "x": ..., "y": ..., "cluster": ..., "vector": {...},
       "n_sentences": ..., "text_preview": "..."},
      ...
    ]
  }
"""
from __future__ import annotations

import json
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import umap

# ── Paths ──────────────────────────────────────────────────────────────
CLASSIFICATIONS_DIR = Path(__file__).resolve().parent.parent / "ace_classifications"
OUTPUT_DIR = Path(__file__).resolve().parent / "comment_cluster_data"

# ── Canonical subcategory names (fixed order for the vector) ───────────
SUBCATEGORIES = [
    "Visual Observation: Chart Structure & Text",
    "Visual Observation: Data Point Extraction",
    "Visual Observation: Cross-point Pattern Recognition",
    "Prior Knowledge: Background",
    "Prior Knowledge: Personal / Episodic",
    "Evaluative: Prescriptive",
    "Evaluative: Reactive",
    "Inference: Explanatory",
    "Inference: Predictive / Hypothetical",
    "Curiosity",
]

SHORT_LABELS = [
    "VO1", "VO2", "VO3",
    "Background", "Personal",
    "Prescriptive", "Reactive",
    "Explanatory", "Predictive",
    "Curiosity",
]

_LONG_TO_SHORT = dict(zip(SUBCATEGORIES, SHORT_LABELS))

_TAG_CLEANUP = {
    "L3: Trend and pattern analysis": "Visual Observation: Cross-point Pattern Recognition",
    "L1: Elemental and encoded properties": "Visual Observation: Chart Structure & Text",
    "L2: Statistical concepts and relations": "Visual Observation: Data Point Extraction",
    "VO1: Chart Structure, Layout & Text": "Visual Observation: Chart Structure & Text",
    "VO2: Data Point Reading": "Visual Observation: Data Point Extraction",
    "VO3: Comparisons, Trends & Patterns": "Visual Observation: Cross-point Pattern Recognition",
    "Background knowledge": "Prior Knowledge: Background",
    "Personal/episodic retrieval": "Prior Knowledge: Personal / Episodic",
    "Evaluative / affective judgment": "Evaluative: Reactive",
    "Explanatory inference": "Inference: Explanatory",
    "Predictive / counterfactual inference": "Inference: Predictive / Hypothetical",
    "Information need / curiosity": "Curiosity",
    "Meta /Paratext": "Meta / Paratext",
    "Meta / paratext": "Meta / Paratext",
}

EXCLUDED = {"Meta / Paratext", "Uncategorizable"}

MIN_COMMENTS_FOR_UMAP = 10


def _text_preview(texts: list[str], limit: int = 200) -> str:
    joined = " | ".join(dict.fromkeys(texts))
    if len(joined) > limit:
        return joined[:limit] + "…"
    return joined


def process_article(article_id: str, article_df: pd.DataFrame) -> dict | None:
    """Build comment vectors, run UMAP+HDBSCAN, return JSON-ready dict."""
    comment_groups = article_df.groupby("comment_id")

    comment_ids: list[int] = []
    vectors: list[np.ndarray] = []
    n_sentences_list: list[int] = []
    previews: list[str] = []

    for cid, grp in comment_groups:
        counts = grp["comment_tag"].value_counts()
        total = counts.sum()
        if total == 0:
            continue
        vec = np.array([counts.get(cat, 0) / total for cat in SUBCATEGORIES])
        comment_ids.append(int(cid))
        vectors.append(vec)
        n_sentences_list.append(int(total))
        previews.append(_text_preview(grp["original_comment"].tolist()))

    n = len(vectors)
    if n < MIN_COMMENTS_FOR_UMAP:
        return None

    X = np.stack(vectors)

    n_neighbors = min(15, max(2, n // 3))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    embedding = reducer.fit_transform(X)

    min_cluster = max(3, n // 10)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster, metric="euclidean")
    labels = clusterer.fit_predict(embedding)

    records = []
    for i in range(n):
        records.append({
            "comment_id": comment_ids[i],
            "x": round(float(embedding[i, 0]), 4),
            "y": round(float(embedding[i, 1]), 4),
            "cluster": int(labels[i]),
            "vector": {
                short: round(float(X[i, j]), 4)
                for j, short in enumerate(SHORT_LABELS)
            },
            "n_sentences": n_sentences_list[i],
            "text_preview": previews[i],
        })

    n_clusters = len(set(labels) - {-1})
    return {
        "article_id": article_id,
        "subcategories": SHORT_LABELS,
        "n_comments": n,
        "n_clusters": n_clusters,
        "comments": records,
    }


def main() -> None:
    # ── 1. Load & clean ────────────────────────────────────────────────
    rows: list[dict] = []
    for p in sorted(CLASSIFICATIONS_DIR.glob("*.json")):
        with p.open() as f:
            rows.extend(json.load(f))

    df = pd.DataFrame(rows)
    df["article_id"] = df["article_id"].astype(str)
    df["comment_id"] = df["comment_id"].astype(int)
    df["comment_tag"] = df["comment_tag"].replace(_TAG_CLEANUP)
    df = df[~df["comment_tag"].isin(EXCLUDED)].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    article_ids = sorted(df["article_id"].unique(), key=lambda x: int(x))
    index_records = []

    for aid in article_ids:
        article_df = df[df["article_id"] == aid]
        result = process_article(aid, article_df)
        if result is None:
            print(f"  Skipping article {aid} (< {MIN_COMMENTS_FOR_UMAP} comments)")
            continue

        out_path = OUTPUT_DIR / f"{aid}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        index_records.append({
            "article_id": aid,
            "n_comments": result["n_comments"],
            "n_clusters": result["n_clusters"],
        })
        print(f"  Article {aid}: {result['n_comments']} comments, "
              f"{result['n_clusters']} clusters → {out_path.name}")

    # Write a small index so the HTML can list available articles
    index_path = OUTPUT_DIR / "_index.json"
    with open(index_path, "w") as f:
        json.dump(index_records, f, indent=2)

    print(f"\nWrote {len(index_records)} article JSONs + index to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
