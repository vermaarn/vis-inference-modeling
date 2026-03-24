"""
Cluster all ACE statements across articles using SBERT + DBSCAN.

Loads every ace_comments/{article_id}/{comment_index}.json file, extracts
ace_sentences, encodes them with a Sentence-BERT model, and runs DBSCAN
to discover groups of semantically similar statements.

Outputs:
  analysis_output/statement_clusters.json   — per-statement cluster assignments
  analysis_output/cluster_summary.json      — per-cluster summary with members
  analysis_output/embeddings.npy            — raw embedding matrix (for reuse)

Usage:
    python analyze_statement_clusters.py
    python analyze_statement_clusters.py --eps 0.25 --min-samples 3
    python analyze_statement_clusters.py --model all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def collect_statements(data_dir: Path) -> list[dict]:
    """Walk ace_comments/ and pull out every ace_sentence with metadata."""
    comments_root = data_dir / "ace_comments"
    if not comments_root.is_dir():
        raise FileNotFoundError(f"ace_comments directory not found at {comments_root}")

    records: list[dict] = []
    for article_dir in sorted(comments_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
        if not article_dir.is_dir():
            continue
        article_id = article_dir.name
        for json_path in sorted(article_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
            with open(json_path) as f:
                data = json.load(f)
            comment_index = data.get("comment_index", json_path.stem)
            for sentence in data.get("ace_sentences", []):
                records.append({
                    "article_id": article_id,
                    "comment_index": comment_index,
                    "sentence": sentence,
                })
    return records


def encode_statements(
    sentences: list[str],
    model_name: str,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode sentences with SBERT and return the embedding matrix."""
    print(f"Loading SBERT model '{model_name}' ...")
    model = SentenceTransformer(model_name)
    print(f"Encoding {len(sentences):,} statements (batch_size={batch_size}) ...")
    t0 = time.perf_counter()
    embeddings = model.encode(sentences, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    elapsed = time.perf_counter() - t0
    print(f"Encoding finished in {elapsed:.1f}s")
    return np.asarray(embeddings)


def cluster_statements(
    embeddings: np.ndarray,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """Run DBSCAN on cosine distance and return cluster labels."""
    print(f"Computing cosine distance matrix for {embeddings.shape[0]:,} embeddings ...")
    distance_matrix = cosine_distances(embeddings)
    print(f"Running DBSCAN (eps={eps}, min_samples={min_samples}) ...")
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(distance_matrix)
    return labels


def build_cluster_summary(records: list[dict], labels: np.ndarray) -> list[dict]:
    """Group records by cluster label and build a summary."""
    from collections import defaultdict

    clusters: dict[int, list[dict]] = defaultdict(list)
    for rec, label in zip(records, labels):
        clusters[int(label)].append(rec)

    summary = []
    for cluster_id in sorted(clusters.keys()):
        members = clusters[cluster_id]
        article_ids = sorted(set(m["article_id"] for m in members))
        summary.append({
            "cluster_id": cluster_id,
            "size": len(members),
            "num_articles": len(article_ids),
            "article_ids": article_ids,
            "statements": [m["sentence"] for m in members],
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster ACE statements across articles with SBERT + DBSCAN."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"Root directory containing ace_comments/ (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Sentence-BERT model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--eps", type=float, default=0.3,
        help="DBSCAN eps (max cosine distance to be neighbors, default: 0.3)",
    )
    parser.add_argument(
        "--min-samples", type=int, default=3,
        help="DBSCAN min_samples (default: 3)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="SBERT encoding batch size (default: 256)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for output files (default: <data-dir>/analysis_output)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir / "analysis_output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Collect statements ---
    print("Collecting statements from ace_comments/ ...")
    records = collect_statements(data_dir)
    print(f"Found {len(records):,} statements across all articles.\n")

    if not records:
        print("No statements found. Exiting.")
        return

    sentences = [r["sentence"] for r in records]

    # --- 2. Encode with SBERT ---
    embeddings = encode_statements(sentences, args.model, batch_size=args.batch_size)
    np.save(output_dir / "embeddings.npy", embeddings)
    print(f"Saved embeddings to {output_dir / 'embeddings.npy'}\n")

    # --- 3. DBSCAN clustering ---
    labels = cluster_statements(embeddings, eps=args.eps, min_samples=args.min_samples)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"\nClusters found: {n_clusters}")
    print(f"Noise points (unclustered): {n_noise:,} / {len(labels):,} "
          f"({100 * n_noise / len(labels):.1f}%)\n")

    # --- 4. Save per-statement results ---
    statement_results = []
    for rec, label in zip(records, labels):
        statement_results.append({
            "article_id": rec["article_id"],
            "comment_index": rec["comment_index"],
            "sentence": rec["sentence"],
            "cluster_id": int(label),
        })

    stmt_path = output_dir / "statement_clusters.json"
    with open(stmt_path, "w") as f:
        json.dump(statement_results, f, indent=2, ensure_ascii=False)
    print(f"Saved per-statement clusters to {stmt_path}")

    # --- 5. Save cluster summary ---
    summary = build_cluster_summary(records, labels)
    summary_path = output_dir / "cluster_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved cluster summary to {summary_path}")

    # --- 6. Print top clusters ---
    clustered = [s for s in summary if s["cluster_id"] != -1]
    clustered.sort(key=lambda c: c["size"], reverse=True)
    print(f"\n{'='*70}")
    print(f"Top 15 clusters (by size):")
    print(f"{'='*70}")
    for c in clustered[:15]:
        print(f"\n  Cluster {c['cluster_id']}  —  {c['size']} statements, "
              f"{c['num_articles']} articles")
        for s in c["statements"][:5]:
            print(f"    • {s}")
        if c["size"] > 5:
            print(f"    ... and {c['size'] - 5} more")


if __name__ == "__main__":
    main()
