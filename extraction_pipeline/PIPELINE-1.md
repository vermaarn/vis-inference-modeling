# Pipeline

High-level flow for turning raw NYT article comments into ACE sentences, semantic tags, dependency graphs, and interactive D3 visualizations.

The pipeline is organized into seven scripts in this folder:

1. `1_extract_ace_comments.py` – extract ACE sentences from raw comments  
2. `2_classify_ace_sentences.py` – classify ACE sentences into semantic categories  
3. `3_cluster_themes.py` – cluster sentences into common propositions (themes)  
4. `4_dependency_classification.py` – build per-comment dependency graphs  
5. `5_combine_dataframe.py` – combine all per-comment data into a single JSON per article  
6. `6_combine_similar_nodes.py` – merge nodes sharing the same cluster into reduced graphs  
7. `10_visualize_graph.py` – render interactive HTML graphs from the combined JSON

All paths below are relative to `extraction_pipeline_v4/`.

---

## Environment & setup

- **Python**: Python ≥ 3.10.
- **Dependencies**: The scripts assume `openai` and `python-dotenv` are installed in your environment.
  - For example:

    ```bash
    pip install openai python-dotenv
    ```

- **API key**: Put your OpenAI API key in a `.env` file at the project root:

  ```bash
  echo 'OPENAI_API_KEY=sk-...' >> .env
  ```

- **Raw NYT comments**:
  - Expected location: `../data/comment_data/{article_id}.json`
  - Each file should contain a list of comment objects with a `comment info` field holding the free-text comment.

You can run the scripts either as `python ...` or with your preferred runner (e.g. `uv run python ...`) from inside `extraction_pipeline_v4/`.

---

## Step 1: Extract ACE comments (`1_extract_ace_comments.py`)

Convert raw NYT article comments into **ACE (Attempto Controlled English)** sentences. Each comment is sent to an LLM with few-shot examples; the model returns one ACE sentence per atomic proposition.

- **Inputs**
  - `../data/comment_data/{article_id}.json`
  - `OPENAI_API_KEY` from `.env` (or `--api-key`)

- **Outputs**
  - Per-comment ACE JSON files in `ace_comments/{article_id}/{comment_index}.json`, each with:
    - `article_id`
    - `comment_index`
    - `raw_comment`
    - `ace_sentences` (list of ACE sentences)

- **Typical commands**
  - Single article:

    ```bash
    python 1_extract_ace_comments.py --article_id 181
    ```

  - All available article JSONs in `../data/comment_data/`:

    ```bash
    python 1_extract_ace_comments.py --all
    ```

You can override the raw data directory with `--articles-data-dir` and the ACE output directory with `--ace-comments-dir`.

---

## Step 2: Classify ACE sentences (`2_classify_ace_sentences.py`)

Assign each ACE sentence to one semantic category. The script reads ACE comment JSONs, loads a classification prompt, and writes one classification JSON per article.

- **Inputs**
  - ACE outputs from step 1: `ace_comments/{article_id}/*.json`
  - Classification prompt: `prompts/classify_ace_sentences.txt`

- **Outputs**
  - Per-article classification JSON:
    - `ace_classifications/ace_sentence_classifications_{article_id}.json`
  - Each entry has:
    - `article_id`
    - `comment_id`
    - `original_comment` (one ACE sentence)
    - `comment_tag` (semantic category)

- **Typical command (per article)**

  ```bash
  python 2_classify_ace_sentences.py --article-id 181
  ```

  With `--article-id` set and the default `--output`, the script automatically saves to `ace_classifications/ace_sentence_classifications_{article_id}.json`.

- **Categories**

# Reading visualization
- **Visual feature detection** — Raw noticing of chart structure or marks.  
  e.g. “The graph has a y-axis.” “The graph has tiered color blocks.”
  - L1

- **Encoding extraction** — Mapping a visual feature to what it means.  
  e.g. “The y-axis is measured in gigawatt hours.” “Yellow-orange blocks represent cleaner fuels.”
  - L2

- **Visual data interpretation** — Reading values, trends, comparisons from the graph.  
  e.g. “Wind and solar increased since 2000.” “China produces more than India.”
  -- L3



# Drawing on prior knowledge
- **Background knowledge** — World knowledge not directly shown in the chart.  
  e.g. “The Lehman Shock happened in 2008.” “Fossil fuels contribute to climate change.”

- **Personal/episodic retrieval** — Self, community, memory, lived association.  
  e.g. “I live in Akita.” “My town has many factories.” “When I was in junior high…”
  - Arvind: what are the sets of chains that encorperate. Peck / NYT article
  - What are the chain the include personal vs not?
  - what are the leafs, what is the root

# engaging with possibilites - combining plot reading with prior knowledge ('insight' -- refine this definition)
- **Explanatory inference** — Causal or abductive inference using either information already in the graph or background knowledge.
<!-- linking  -->
<!-- between observations and background knowledge.   -->
  e.g. “The 2008 dip relates to the financial crisis.” “Population may explain the increase.”
  - insight is not a single node or edge, possible subgraph

- **Predictive / counterfactual inference** — Future or hypothetical reasoning.
  e.g. “If fossil fuels remain steady, warming will worsen.” “Clean energy may dominate later.”

- **Information need / curiosity** — Explicit question or uncertainty.  
  e.g. “I wonder…” / “I ask whether…”
  - episetemic drive
  - modals??
  - related to counterfactual inference

# some stakes  involved / takeaway
- **Evaluative / affective judgment** — Normative stance, concern, alarm, hope.  
  e.g. “This is a problem.” “This increase is alarming.” “This data gives me comfort.”
  - episetemic drive
  - modals?? https://en.wikipedia.org/wiki/Modal_verb

# extra
- **Meta / paratext** — Commentary about the graph or task, not the world.  
  e.g. “The headline is…” “Ember published the graph.” “The graph gives more insight…”

# 

---

## Step 3: Cluster themes (`3_cluster_themes.py`)

Group classified ACE sentences by category, then use an LLM to identify **common propositions (themes)** and assign each sentence to a theme.

- **Inputs**
  - `ace_classifications/ace_sentence_classifications_{article_id}.json`
  - Clustering prompt: `prompts/cluster_ace_themes.txt`

- **Outputs**
  - Per-article theme clusters:
    - `ace_clusters/ace_sentence_theme_clusters_{article_id}.json`
  - For each semantic category, a list of:
    - `common_statements` with `id`, `statement`, and `sentences` (the ACE sentence objects)

- **Typical command (article 181 default)**

  ```bash
  python 3_cluster_themes.py
  ```

  This uses the default input/output for article `181`. For other articles, point the script to the desired input and output:

  ```bash
  python 3_cluster_themes.py \
    --input ace_classifications/ace_sentence_classifications_{ARTICLE_ID}.json \
    --output ace_clusters/ace_sentence_theme_clusters_{ARTICLE_ID}.json
  ```

The script runs several passes per category (`--num-passes`) and uses batching with memory to stabilize themes across passes.

---

## Step 4: Dependency classification (`4_dependency_classification.py`)

Build **dependency graphs (DAGs)** over the ACE sentences in each comment. For each comment, ACE sentences are sent to an LLM with a causal-direction prompt; the model returns nodes (id, sentence, depends_on).

- **Inputs**
  - ACE comment JSONs from step 1: `ace_comments/{article_id}/*.json`
  - Dependency prompt: `prompts/dependency_classification.txt`

- **Outputs**
  - Per-comment dependency graphs:
    - `ace_dependency_graphs/{article_id}/{comment_index}.json`
  - Each file contains:
    - `article_id`
    - `comment_index`
    - `dependency_graph`: list of nodes with `id`, `sentence`, and `depends_on` (list of node ids)

- **Typical command**

  ```bash
  python 4_dependency_classification.py --article-id 181
  ```

You can override source and output locations with `--ace-comments-dir`, `--prompt-file`, and `--output-dir`.

---

## Step 5: Combine per-comment data (`5_combine_dataframe.py`)

Combine all per-comment data (ACE sentences, dependency graphs, sentence-level tags, and theme clusters) into a single JSON list per article.

- **Inputs**
  - ACE comments: `ace_comments/{article_id}/{comment_index}.json`
  - Dependency graphs: `ace_dependency_graphs/{article_id}/{comment_index}.json`
  - Sentence classifications: `ace_classifications/ace_sentence_classifications_{article_id}.json`
  - Theme clusters: `ace_clusters/ace_sentence_theme_clusters_{article_id}.json`

- **Outputs**
  - Combined per-article JSON:
    - `combined_data/{article_id}.json`
  - Each entry (one per comment) includes:
    - `article_id`
    - `comment_index`
    - `raw_comment`
    - `ace_sentences`
    - `dependency_graph` (nodes enriched with `comment_tag`, `cluster_id`, `cluster_statement` when available)

- **Typical command**

  ```bash
  python 5_combine_dataframe.py --article-id 181
  ```

By default, the script assumes the current directory as the data root (containing `ace_comments/`, `ace_dependency_graphs/`, `ace_classifications/`, `ace_clusters/`). You can change this with `--data-dir`.

---

## Step 6: Combine similar nodes (`6_combine_similar_nodes.py`)

Merge dependency-graph nodes that share the same `(cluster_id, comment_tag)` within each comment. This reduces the graph by collapsing semantically equivalent sentences into a single node.

For each merged node:
- `id` – kept from the first node in the group
- `sentence` – all original sentences joined into one string
- `original_sentences` – list of the individual original sentences
- `original_sentence_ids` – ids of the remaining (non-primary) nodes
- `depends_on` – union of all dependencies (internal group references removed, external references remapped to surviving ids)
- `comment_tag`, `cluster_id`, `cluster_statement` – kept from the first node

Nodes without a `cluster_id` are left as-is.

- **Inputs**
  - Combined per-article JSON from step 5: `combined_data/{article_id}.json`

- **Outputs**
  - Reduced per-article JSON:
    - `reduced_data/{article_id}.json`

- **Typical command**

  ```bash
  python 6_combine_similar_nodes.py --input combined_data/181.json
  ```

  Custom output path:

  ```bash
  python 6_combine_similar_nodes.py --input combined_data/181.json --output reduced_data/181.json
  ```

---

## Step 7: Visualize graph (`10_visualize_graph.py`)

Turn the combined dependency-graph data into **interactive D3.js HTML** visualizations.

- **Inputs**
  - Combined or reduced per-article JSON from step 5 or 6: `combined_data/{article_id}.json` or `reduced_data/{article_id}.json`

- **Outputs**
  - HTML files under `graph_visualizations/{article_id}/`, one per comment:
    - `graph_visualizations/{article_id}/comment_{comment_index}.html`

- **Typical commands**
  - All comments for an article:

    ```bash
    python 10_visualize_graph.py --input combined_data/181.json
    ```

  - Single comment:

    ```bash
    python 10_visualize_graph.py --input combined_data/181.json --comment-index 4
    ```

The HTML viewer shows nodes colored by semantic category, supports zooming and panning, and lets you toggle between ACE sentence labels and cluster-level proposition labels.

---

## End-to-end order for one article

For a single article id (e.g. `181`), the typical sequence is:

1. `python 1_extract_ace_comments.py --article_id 181`
2. `python 2_classify_ace_sentences.py --article-id 181`
3. `python 3_cluster_themes.py --input ace_classifications/ace_sentence_classifications_181.json --output ace_clusters/ace_sentence_theme_clusters_181.json`
4. `python 4_dependency_classification.py --article-id 181`
5. `python 5_combine_dataframe.py --article-id 181`
6. `python 6_combine_similar_nodes.py --input combined_data/181.json`
7. `python 10_visualize_graph.py --input reduced_data/181.json`

After step 7, open the generated files in `graph_visualizations/181/` in a browser to explore the comment-level graphs.
