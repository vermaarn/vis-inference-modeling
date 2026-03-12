#!/usr/bin/env python
"""
Step 6b: Combine nodes with the same cluster_id within each comment's dependency graph.

For each comment in the combined JSON, nodes sharing a cluster_id are merged into
a single node that keeps the first node's id and carries the union of dependencies.

Usage:
    python 6_combine_similar_nodes.py --input combined_data/181.json
    python 6_combine_similar_nodes.py --input combined_data/181.json --output reduced_data/181.json
"""

import argparse
import json
import os
from collections import OrderedDict


def merge_nodes_by_cluster(dependency_graph):
    """Merge nodes sharing the same cluster_id inside one comment's graph."""

    # Group by (cluster_id, comment_tag) since cluster_id is scoped per category.
    # Nodes without a cluster_id are kept as standalone entries.
    cluster_groups = OrderedDict()
    solo_counter = 0
    for node in dependency_graph:
        cid = node.get("cluster_id")
        if cid is None:
            key = f"__solo_{solo_counter}"
            solo_counter += 1
        else:
            key = (cid, node.get("comment_tag", ""))
        cluster_groups.setdefault(key, []).append(node)

    # old id → primary id (the first node's id in each cluster group)
    id_remap = {}
    for nodes in cluster_groups.values():
        primary_id = nodes[0]["id"]
        for node in nodes:
            id_remap[node["id"]] = primary_id

    merged_nodes = []
    for cid, nodes in cluster_groups.items():
        primary = nodes[0]
        primary_id = primary["id"]

        combined_deps = set()
        for node in nodes:
            for dep in node.get("depends_on", []):
                remapped = id_remap.get(dep, dep)
                if remapped != primary_id:
                    combined_deps.add(remapped)

        merged = {
            "id": primary_id,
            "sentence": " ".join(n["sentence"] for n in nodes),
            "original_sentences": [n["sentence"] for n in nodes],
            "original_sentence_ids": [n["id"] for n in nodes[1:]],
            "depends_on": sorted(combined_deps),
            "comment_tag": primary["comment_tag"],
        }
        if "cluster_id" in primary:
            merged["cluster_id"] = primary["cluster_id"]
        if "cluster_statement" in primary:
            merged["cluster_statement"] = primary["cluster_statement"]
        merged_nodes.append(merged)

    return merged_nodes


def main():
    parser = argparse.ArgumentParser(
        description="Combine nodes with the same cluster_id in each comment's dependency graph."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the combined JSON file (e.g. combined_data/181.json)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path. Defaults to reduced_data/{filename}.",
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        comments = json.load(f)

    for comment in comments:
        if "dependency_graph" in comment:
            comment["dependency_graph"] = merge_nodes_by_cluster(
                comment["dependency_graph"]
            )

    output_path = args.output
    if output_path is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(args.input)) or ".", "reduced_data")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(args.input))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(comments, f, indent=2, ensure_ascii=False)

    print(f"Wrote merged data to {output_path}")


if __name__ == "__main__":
    main()
