# Data scripts

Scripts in this folder process article metadata and NYT comment HTML into JSON for the explorer.

## Prerequisites

- Python 3 with dependencies from the project root (e.g. `uv sync` or `pip install -e .`).
- **html_comments_parser.py** requires `beautifulsoup4` (and is used with HTML that has `data-testid` attributes for comment structure).

---

## 1. `html_comments_parser.py` — HTML comments → JSON

Parses NYT-style comment HTML (with `data-testid` markers) into structured JSON: one file per article.

### What it does

- Reads HTML files from **`data/article_comments_html/`** (e.g. `217.html`, `216.html`, …).
- Extracts for each comment:
  - **name**, **location**, **date posted**, **comment info**
  - **replies** (same fields plus **reply to** = parent author).
- Writes one JSON file per article into **`data/comment_data/`** (e.g. `217.json`, `216.json`, …).

### How to run

From the **project root** (parent of `data/`):

```bash
python data/html_comments_parser.py
```

It will process every `.html` file in `data/article_comments_html/` and overwrite or create the corresponding `.json` in `data/comment_data/`.

### Expected input

- Directory: `data/article_comments_html/`
- Files: `*.html` with structure that includes:
  - `div[data-testid="comment-container"]`
  - Inside each: `div[data-testid="user-header"]`, `span[data-testid="user-header-subtitle"]`, `span[data-testid="todays-date"]`, `p` for the comment body.
  - Optional replies: `div[data-testid="reply-list-threading"]` with `div[data-testid="reply-comment-container"]`.

### Output format (per article JSON)

```json
[
  {
    "name": "Author Name",
    "location": "Location",
    "date posted": "Dec. 11, 2025",
    "comment info": "Comment text...",
    "replies": [
      {
        "name": "Reply Author",
        "location": "...",
        "date posted": "...",
        "comment info": "...",
        "reply to": "Author Name"
      }
    ]
  }
]
```

---

## 2. `update_explorer_data.py` — CSV → explorer articles index

Builds a single JSON index from `data.csv` for the explorer app (article list, URLs, image paths, comment data paths).

### What it does

- Reads **`data/data.csv`** (NYT pieces: article_id, Date, Title, comment_data, image, URL, Comments, etc.).
- Writes **`explorer/public/articles-index.json`** with one entry per row, keyed by `article_id`.

### How to run

From anywhere (paths are resolved from the script’s location):

```bash
python data/update_explorer_data.py
```

Or from the project root:

```bash
python data/update_explorer_data.py
```

### Expected CSV columns

- **article_id** — unique id (used as key in the index).
- **Title**, **Date** — article title and date.
- **comment_data** — filename only (e.g. `217.json`); script turns it into `/data/comment_data/217.json`.
- **image** — filename only (e.g. `217.png`); script turns it into `/visualizations/217.png`.
- **Comments** or **URL** — used for **articleUrl** (Comments preferred if present).

### Output format

`explorer/public/articles-index.json`:

```json
{
  "217": {
    "articleId": 217,
    "title": "Global Nuclear Powers",
    "date": "Nov. 20, 2025",
    "articleUrl": "https://...",
    "pngPath": "/visualizations/217.png",
    "commentsPath": "/data/comment_data/217.json"
  }
}
```

---

## Typical workflow

1. Put new comment HTML files in **`data/article_comments_html/`** (e.g. `218.html`).
2. Run **`python data/html_comments_parser.py`** to generate/update **`data/comment_data/*.json`**.
3. Add or update the corresponding row in **`data/data.csv`** (article_id, Title, Date, comment_data, image, URL/Comments).
4. Run **`python data/update_explorer_data.py`** to refresh **`explorer/public/articles-index.json`**.

The explorer then uses `articles-index.json` to list articles and load comment JSON from `commentsPath`.
