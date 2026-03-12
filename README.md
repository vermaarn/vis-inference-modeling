# NYT Pieces

A data collection and processing repository for New York Times "What's Going On In This Graph?" articles. This project scrapes, parses, and organizes comments from NYT Learning Network articles that feature data visualizations for educational purposes.

## Repository Function

This repository serves as a pipeline for collecting and processing student comments from New York Times "What's Going On In This Graph?" articles. The workflow includes:

1. **Data Collection**: HTML pages containing article comments are stored in the `articles/` folder
2. **Data Parsing**: The `parser.py` script extracts structured comment data (author names, locations, dates, comment text, and replies) from HTML files
3. **Data Aggregation**: The `combine_comments.py` script merges all parsed comment data into a single JSON file (`2025.json`)
4. **Visualization Management**: Graph images from articles are stored and can be combined using `combine_images.py` when articles contain multiple visualizations

## Folder Structure

### `articles/`
Contains HTML files scraped from New York Times article pages. Each file (e.g., `1.html`, `2.html`) represents the full HTML content of an article's page, including the comments section. These files are used as input for the parsing process.

### `articles_data/`
Contains JSON files with parsed comment data extracted from the HTML files. Each JSON file (e.g., `1.json`, `2.json`) contains an array of comment objects, where each object includes:
- `name`: Commenter's name
- `location`: Commenter's location
- `date posted`: Date the comment was posted
- `comment info`: The comment text content
- `replies`: Array of reply objects (if any), each containing similar fields plus a `reply to` field indicating the parent comment author

### `background/`
Contains text files with background information and context for each article. Each file (e.g., `1.txt`, `2.txt`) provides additional details about the graph or visualization featured in the corresponding article, including data sources, methodology, and explanatory text.

### `visualizations/`
Contains image files (PNG and WebP formats) representing the graphs, charts, and visualizations from the NYT articles. Files are named with article indices (e.g., `1.png`, `22.png`). Some articles may have multiple visualizations, indicated by suffixes like `14a.webp` and `14b.webp`, which can be combined using the `combine_images.py` script.

## Articles CSV Columns

The `articles.csv` file contains metadata for all articles in the collection. The columns are:

- **Idx**: A unique index/number identifying each article (1, 2, 3, etc.)
- **Date**: The publication date of the article (e.g., "Nov. 20, 2025", "Jan. 9, 2025")
- **Title**: The title of the article/graph feature (e.g., "Global Nuclear Powers", "Homebodies")
- **URL**: The full URL to the New York Times article page
- **Filler**: A CSS selector or fragment identifier used to locate the comments container on the page (typically `#commentsContainer`)
- **Comments**: The full URL to the article's comments section (includes the fragment identifier for direct navigation to comments)

## Scripts

- **`parser.py`**: Parses HTML files from `articles/` and extracts structured comment data into JSON files in `articles_data/`
- **`combine_comments.py`**: Combines all JSON files from `articles_data/` into a single `2025.json` file containing all comments
- **`combine_images.py`**: Utility script to combine multiple visualization images (e.g., `16a.webp` and `16b.webp`) into a single PNG file
- **`extract_responses.py`**: Uses OpenAI API to extract structured information from comment responses, including:
  - "What do you notice?" observations
  - "What do you wonder?" questions
  - Variables mentioned (both in plot and not in plot)
  - Causal relations between variables in readers' mental models
- **`main.py`**: Placeholder entry point (currently minimal implementation)

## Output

The primary output of this repository is `2025.json`, which contains all parsed comments from all articles in a single consolidated JSON file, making it easy to analyze comment patterns, trends, and student responses across the entire collection of NYT graph articles.

### Extracted Responses

The `extract_responses.py` script generates structured JSON files in the `extracted_responses/` directory. Each file contains:

- **Comments**: Array of comment objects with extracted "notices" and "wonders"
- **Variables**: Categorized list of variables mentioned by readers:
  - `in_plot`: Variables explicitly shown in the visualization
  - `not_in_plot`: Variables mentioned but not shown in the visualization
- **Causal Relations**: Array of proposed causal relationships between variables, including which commenters mentioned them

#### Usage

```bash
# Process a single article
python extract_responses.py 38

# Process all articles
python extract_responses.py --all

# Specify custom directories and model
python extract_responses.py 38 --articles-data-dir articles_data --data-tables-dir data_tables --output-dir extracted_responses --model gpt-4o

# Set API key via environment variable or argument
export OPENAI_API_KEY=your_api_key_here
python extract_responses.py 38
```

### Summaries

The `extract_summaries.py` script processes JSON files from `extracted_responses/` and generates human-readable text summaries in the `summaries/` directory. For each article, three summary files are created:

- **`{article_id}_wonders.txt`**: Lists all unique questions/wonders extracted from comments
- **`{article_id}_notices.txt`**: Lists all unique observations/notices extracted from comments
- **`{article_id}_causal_relations.txt`**: Lists all causal relationships between variables mentioned by readers

#### Example Summary Files

**`summaries/14_wonders.txt`** (excerpt):
```
WONDERS (Questions) - Article 14
================================================================================

1. Why other countries and China are more centralized on production and life satisfaction.
2. What impact these graphs can cause a concern for anyone within the country.
3. Can the wealth of a country change?
4. What countries rank number 1 in each graph?
5. How recent events like the COVID-19 pandemic, shifting political dynamics, and economic instability have contributed to this downward trend.
...

Total: 14 questions
```

**`summaries/14_notices.txt`** (excerpt):
```
NOTICES (Observations) - Article 14
================================================================================

1. The graph emphasizes high incomes from other countries and China.
2. Gross domestic products are classified into production and life satisfaction.
3. The United States has consistently been ranked number 1 for total GDP since 1990.
4. The US's life satisfaction has gone down in recent years.
5. There is a general decline in quality of life across the U.S.
...

Total: 13 observations
```

**`summaries/38_causal_relations.txt`**:
```
CAUSAL RELATIONS - Article 38
================================================================================

1. Birth rate -> Population increase: Higher birth rates are suggested to cause population increases, particularly in Africa.
2. Economic development -> Population growth: Economic development may lead to population growth due to improved living conditions.
3. Cultural factors -> Population growth: Cultural traditions may influence family size and thus population growth.
4. Government policies -> Population decline: Policies like China's one-child policy are suggested to cause population declines.


Total: 4 causal relations
```

#### Usage

```bash
# Process a single article
python extract_summaries.py --article_id 38

# Process all articles (default behavior)
python extract_summaries.py

# Specify custom directories
python extract_summaries.py --article_id 38 --extracted-responses-dir extracted_responses --output-dir summaries
```
