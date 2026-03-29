"""
Build per-article subcategory distribution vectors, run UMAP + HDBSCAN clustering,
and emit a JSON file consumed by the Vega-Lite / D3 scatter-plot visualisation.
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
OUTPUT_JSON = Path(__file__).resolve().parent / "cluster_data.json"

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

_TAG_CLEANUP = {
    "L3: Trend and pattern analysis": "Visual Observation: Cross-point Pattern Recognition",
    "L1: Elemental and encoded properties": "Visual Observation: Chart Structure & Text",
    "L2: Statistical concepts and relations": "Visual Observation: Data Point Extraction",
    "VO1: Chart Structure, Layout & Text": "Visual Observation: Chart Structure & Text",
    "VO2: Data Point Reading": "Visual Observation: Data Point Extraction",
    "VO3: Comparisons, Trends & Patterns": "Visual Observation: Cross-point Pattern Recognition",
    "Background knowledge": "Prior Knowledge: Background",
    "Personal/episodic retrieval": "Prior Knowledge: Personal / Episodic",
    "Prior Knowledge: Personal /Episodic": "Prior Knowledge: Personal / Episodic",
    "Evaluative / affective judgment": "Evaluative: Reactive",
    "Explanatory inference": "Inference: Explanatory",
    "Predictive / counterfactual inference": "Inference: Predictive / Hypothetical",
    "Information need / curiosity": "Curiosity",
    "Meta /Paratext": "Meta / Paratext",
    "Meta / paratext": "Meta / Paratext",
}

EXCLUDED = {"Meta / Paratext", "Uncategorizable"}


def main() -> None:
    # ── 1. Load & clean ────────────────────────────────────────────────
    rows: list[dict] = []
    for p in sorted(CLASSIFICATIONS_DIR.glob("*.json")):
        with p.open() as f:
            rows.extend(json.load(f))

    df = pd.DataFrame(rows)
    df["article_id"] = df["article_id"].astype(str)
    df["comment_tag"] = df["comment_tag"].replace(_TAG_CLEANUP)
    df = df[~df["comment_tag"].isin(EXCLUDED)].copy()

    # ── 2. Per-article subcategory proportions ─────────────────────────
    article_ids = sorted(df["article_id"].unique(), key=lambda x: int(x))

    vectors: list[np.ndarray] = []
    valid_ids: list[str] = []

    for aid in article_ids:
        sub = df[df["article_id"] == aid]
        counts = sub["comment_tag"].value_counts()
        total = counts.sum()
        if total == 0:
            continue
        vec = np.array([counts.get(cat, 0) / total for cat in SUBCATEGORIES])
        vectors.append(vec)
        valid_ids.append(aid)

    X = np.stack(vectors)
    print(f"Built {X.shape[0]} article vectors, each of dimension {X.shape[1]}")

    # ── 3. UMAP projection ────────────────────────────────────────────
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    embedding = reducer.fit_transform(X)

    # ── 4. HDBSCAN clustering ─────────────────────────────────────────
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean")
    labels = clusterer.fit_predict(embedding)
    n_clusters = len(set(labels) - {-1})
    print(f"HDBSCAN found {n_clusters} clusters ({(labels == -1).sum()} noise points)")

    # ── 5. Emit JSON ──────────────────────────────────────────────────
    records = []
    for i, aid in enumerate(valid_ids):
        records.append({
            "article_id": aid,
            "x": round(float(embedding[i, 0]), 4),
            "y": round(float(embedding[i, 1]), 4),
            "cluster": int(labels[i]),
            "vector": {short: round(float(X[i, j]), 4) for j, short in enumerate(SHORT_LABELS)},
        })

    payload = {
        "subcategories": SHORT_LABELS,
        "articles": records,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
