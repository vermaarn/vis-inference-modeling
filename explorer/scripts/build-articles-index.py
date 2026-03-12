#!/usr/bin/env python3
"""
Build explorer/public/articles-index.json from articles.csv.
Each entry has: articleUrl (Comments column), pngPath (/visualizations/Idx.png), commentsPath (/data/comment_data/Idx.json).
Run from repo root: python explorer/scripts/build-articles-index.py
"""
import csv
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTICLES_CSV = os.path.join(REPO_ROOT, "articles.csv")
OUT_JSON = os.path.join(REPO_ROOT, "explorer", "public", "articles-index.json")


def main() -> None:
    index: dict[str, dict[str, str]] = {}
    with open(ARTICLES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row.get("Idx", "").strip()
            comments_url = (row.get("Comments") or row.get("URL") or "").strip()
            if not idx or not comments_url:
                continue
            index[idx] = {
                "articleUrl": comments_url,
                "pngPath": f"/visualizations/{idx}.png",
                "commentsPath": f"/data/comment_data/{idx}.json",
            }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote {len(index)} entries to {OUT_JSON}")


if __name__ == "__main__":
    main()
