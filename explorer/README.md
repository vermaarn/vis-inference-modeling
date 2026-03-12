# Explorer

React app to browse visualizations, view comments, and explore the UMAP scatter.

## Setup

1. Install dependencies (from repo root or `explorer/`):
   ```bash
   pnpm install
   ```

2. **Static data**: The app expects these under `public/`:
   - `public/visualizations/` — PNG images (symlink or copy from repo root):
     ```bash
     cd explorer && ln -s ../visualizations public/visualizations
     ```
   - `public/articles_data/` — comment JSON files:
     ```bash
     cd explorer && ln -s ../articles_data public/articles_data
     ```
   - `public/articles-index.json` — article links and paths (from `articles.csv`). Generate with:
     ```bash
     python explorer/scripts/build-articles-index.py
     ```
   - `public/analysis/` — UMAP outputs (including slim CSV):
     ```bash
     cd explorer && ln -s ../analysis public/analysis
     ```

3. **Slim UMAP CSV**: Generate a small CSV for the scatter (run from repo root):
   ```bash
   python analysis/export_umap_slim.py
   ```
   This creates `analysis/outputs/umap_2d_slim.csv` from `umap_2d.csv`.

4. **PNG list** (optional): To regenerate `public/visualization-pngs.json`:
   ```bash
   node explorer/scripts/generate-png-list.js
   ```

## Run

From `explorer/`:
```bash
pnpm dev
```

Build:
```bash
pnpm build
```
