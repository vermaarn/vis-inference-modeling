import csv
import json
from pathlib import Path


def build_index(csv_path: Path, output_path: Path) -> None:
    """
    Read the NYT pieces CSV and emit a JSON mapping.

    Output format (one entry per row):

    {
      "<article_id>": {
        "articleId": <int>,
        "title": "<Title>",
        "date": "<Date>",
        "articleUrl": "<Comments column value>",
        "pngPath": "/data/images/<image>",
        "commentsPath": "/data/comment_data/<comment_data>"
      },
      ...
    }
    """
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        records = {}
        for row in reader:
            article_id = str(row["article_id"]).strip()
            if not article_id:
                continue

            # Columns in data.csv:
            # Idx,article_id,Date,Title,comment_data,image,URL,Filler,Comments,...
            title = row.get("Title", "").strip()
            date = row.get("Date", "").strip()
            comment_data = row.get("comment_data", "").strip()
            image = row.get("image", "").strip()
            comments_url = row.get("Comments", "").strip()

            records[article_id] = {
                "articleId": int(article_id) if article_id.isdigit() else article_id,
                "title": title,
                "date": date,
                "articleUrl": comments_url or row.get("URL", "").strip(),
                # Static visualization PNGs are served from /visualizations/... (public dir).
                "pngPath": f"/visualizations/{image}" if image else None,
                "commentsPath": f"/data/comment_data/{comment_data}" if comment_data else None,
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Default locations based on your repo layout
    root = Path(__file__).parent
    csv_path = root / "data.csv"
    output_path = Path(__file__).parent.parent / "explorer/public/articles-index.json"

    build_index(csv_path, output_path)
    print(f"Wrote JSON index to {output_path}")

