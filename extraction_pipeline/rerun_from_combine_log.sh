#!/usr/bin/env bash
# Rerun pipeline stages based on warnings in a 4_combine_dataframe.py log (e.g. combine.txt).
#
# With only --combine-log PATH (or default combine.txt), scans the log and runs only what is
# needed, in pipeline order:
#   1_extract_ace_comments.py      ← "Skipping comment M: ace_comments file missing" (under "Combining article N")
#   2_classify_ace_sentences.py    ← "classifications file not found ... ace_sentence_classifications_<id>.json"
#   3_dependency_classification.py ← "dependency graph missing for comment M" (under "Combining article N")
#
# Usage:
#   ./rerun_from_combine_log.sh --combine-log PATH
#   ./rerun_from_combine_log.sh [--dry-run] [--deps-strategy STR] [--combine-log PATH]
#
# Optional phase (after options) forces only that step: classify | deps | extract | all
#   extract -- <article_id> [...]   manual article IDs (ignores log for step 1)
#
# Environment:
#   PYTHON         Python executable (default: python3)
#   DEPS_STRATEGY  by-comment (default) | by-article
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
COMBINE_LOG="${COMBINE_LOG:-$SCRIPT_DIR/combine.txt}"
DRY_RUN=0
DEPS_STRATEGY="${DEPS_STRATEGY:-by-comment}"

usage() {
  sed -n '2,25p' "$0" | tail -n +1
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help) usage 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --combine-log)
      COMBINE_LOG="$2"
      shift 2
      ;;
    --deps-strategy)
      DEPS_STRATEGY="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage 1
      ;;
    *)
      break
      ;;
  esac
done

# Optional explicit phase; default auto = infer from log
PHASE="${1:-auto}"
if [[ "$PHASE" != auto ]]; then
  shift
fi

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'DRY-RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

require_log() {
  if [[ ! -f "$COMBINE_LOG" ]]; then
    echo "Error: combine log not found: $COMBINE_LOG" >&2
    exit 1
  fi
}

# Article IDs whose classification JSON is missing.
list_classify_articles() {
  require_log
  sed -n 's/.*ace_sentence_classifications_\([0-9][0-9]*\)\.json.*/\1/p' "$COMBINE_LOG" |
    sort -nu
}

# "article_id comment_index" for missing dependency graphs.
list_dep_pairs_by_comment() {
  require_log
  awk '
    /^Combining article / { article = $3 }
    /dependency graph missing for comment/ { print article, $NF }
  ' "$COMBINE_LOG" | sort -t' ' -k1,1n -k2,2n -u
}

list_dep_articles() {
  list_dep_pairs_by_comment | awk '{print $1}' | sort -nu
}

# "article_id comment_index" for missing ace_comments JSON files.
list_extract_pairs() {
  require_log
  awk '
    /^Combining article / { article = $3 }
    /Skipping comment / && /ace_comments file missing/ {
      line = $0
      sub(/.*Skipping comment /, "", line)
      sub(/:.*/, "", line)
      if (line ~ /^[0-9]+$/) print article, line
    }
  ' "$COMBINE_LOG" | sort -t' ' -k1,1n -k2,2n -u
}

summarize_log() {
  require_log
  local ec ac dc
  ec="$(list_extract_pairs | wc -l | tr -d ' ')"
  ac="$(list_classify_articles | wc -l | tr -d ' ')"
  dc="$(list_dep_pairs_by_comment | wc -l | tr -d ' ')"
  echo "Scanning $COMBINE_LOG"
  echo "  Step 1 extract:     $ec missing ace_comments file(s) (article, comment pairs)"
  echo "  Step 2 classify:    $ac article(s) without classifications JSON"
  echo "  Step 3 dependency:  $dc missing dependency graph(s) (by-comment targets)"
  if [[ "$ec" -eq 0 && "$ac" -eq 0 && "$dc" -eq 0 ]]; then
    echo "Nothing to rerun — no known warnings in this log."
    return 1
  fi
  return 0
}

do_extract_from_log() {
  local pairs
  pairs="$(list_extract_pairs)"
  if [[ -z "$pairs" ]]; then
    echo "=== Step 1: (skip) no ace_comments file missing in log ==="
    return 0
  fi
  echo "=== Step 1: ACE extraction (from log) ==="
  local n=0
  while read -r aid cid; do
    [[ -z "$aid" || -z "$cid" ]] && continue
    n=$((n + 1))
    echo "[$n] 1_extract_ace_comments.py --article-id $aid --comment-index $cid"
    run "$PYTHON" 1_extract_ace_comments.py --article-id "$aid" --comment-index "$cid"
  done <<<"$pairs"
}

do_classify() {
  local ids
  ids="$(list_classify_articles)"
  if [[ -z "$ids" ]]; then
    echo "=== Step 2: (skip) no missing classifications in log ==="
    return 0
  fi
  echo "=== Step 2: classify ACE sentences ==="
  local n=0
  while IFS= read -r aid; do
    [[ -z "$aid" ]] && continue
    n=$((n + 1))
    echo "[$n] 2_classify_ace_sentences.py --article-id $aid"
    run "$PYTHON" 2_classify_ace_sentences.py --article-id "$aid"
  done <<<"$ids"
}

do_deps() {
  echo "=== Step 3: dependency graphs (DEPS_STRATEGY=$DEPS_STRATEGY) ==="
  if [[ "$DEPS_STRATEGY" != "by-comment" && "$DEPS_STRATEGY" != "by-article" ]]; then
    echo "Error: DEPS_STRATEGY must be by-comment or by-article (got: $DEPS_STRATEGY)" >&2
    exit 1
  fi

  if [[ "$DEPS_STRATEGY" == "by-article" ]]; then
    local ids
    ids="$(list_dep_articles)"
    if [[ -z "$ids" ]]; then
      echo "(skip) no missing dependency graphs in log"
      return 0
    fi
    local n=0
    while IFS= read -r aid; do
      [[ -z "$aid" ]] && continue
      n=$((n + 1))
      echo "[$n] 3_dependency_classification.py --article-id $aid (all comments in article)"
      run "$PYTHON" 3_dependency_classification.py --article-id "$aid"
    done <<<"$ids"
    return 0
  fi

  local pairs
  pairs="$(list_dep_pairs_by_comment)"
  if [[ -z "$pairs" ]]; then
    echo "(skip) no missing dependency graphs in log"
    return 0
  fi
  local n=0
  while read -r aid cid; do
    [[ -z "$aid" || -z "$cid" ]] && continue
    n=$((n + 1))
    echo "[$n] 3_dependency_classification.py --article-id $aid --comment-index $cid"
    run "$PYTHON" 3_dependency_classification.py --article-id "$aid" --comment-index "$cid"
  done <<<"$pairs"
}

do_extract_manual() {
  if [[ $# -lt 1 ]]; then
    echo "extract phase needs article IDs, e.g.: $0 extract -- 6 20 33" >&2
    exit 1
  fi
  echo "=== Step 1: ACE extraction (manual article list) ==="
  local aid n=0
  for aid in "$@"; do
    n=$((n + 1))
    echo "[$n] 1_extract_ace_comments.py --article-id $aid"
    run "$PYTHON" 1_extract_ace_comments.py --article-id "$aid"
  done
}

do_auto() {
  summarize_log || return 0
  do_extract_from_log
  do_classify
  do_deps
}

case "$PHASE" in
  auto)
    do_auto
    ;;
  classify)
    require_log
    do_classify
    ;;
  deps)
    require_log
    do_deps
    ;;
  extract)
    do_extract_manual "$@"
    ;;
  all)
    do_auto
    ;;
  *)
    echo "Unknown phase: $PHASE (use auto, classify, deps, extract, or all)" >&2
    exit 1
    ;;
esac

echo "Done."
