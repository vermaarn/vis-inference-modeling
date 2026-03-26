#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

"""
12_label_ace_categories.py

Interactive labeling tool for reviewing ACE sentence classifications.
Launches a local HTTP server with a single-page app that lets you
navigate comments, view their ACE sentences alongside assigned
categories, and mark each classification as correct or incorrect.
When incorrect, you can supply a corrected primary category and
optionally add a secondary category to any sentence.

Usage:
    python 12_label_ace_categories.py
    # Opens browser at http://localhost:8051, select article in the UI

    python 12_label_ace_categories.py --article-id 181
    # Opens with article 181 pre-selected

    python 12_label_ace_categories.py --port 9000 --no-open
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ACE Category Labeling</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { height: 100%; overflow: hidden; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: #f7f7f8;
  color: #1a1a1a;
  display: flex;
  flex-direction: column;
}

/* ---- Header ---- */
#header {
  height: 50px;
  background: #fff;
  border-bottom: 1px solid #e0e0e0;
  display: flex;
  align-items: center;
  padding: 0 16px;
  gap: 10px;
  flex-shrink: 0;
  z-index: 10;
}
#header h1 { font-size: 14px; font-weight: 600; white-space: nowrap; }
.nav-btn {
  padding: 5px 12px;
  border: 1px solid #d0d0d0;
  border-radius: 5px;
  background: #fff;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.12s;
}
.nav-btn:hover:not(:disabled) { background: #f0f0f0; }
.nav-btn:disabled { opacity: 0.35; cursor: default; }
#article-select {
  padding: 5px 8px;
  border: 1px solid #d0d0d0;
  border-radius: 5px;
  font-size: 12px;
  background: #fff;
  max-width: 140px;
  font-weight: 600;
}
.header-sep {
  width: 1px;
  height: 24px;
  background: #e0e0e0;
  flex-shrink: 0;
}
#comment-select {
  padding: 5px 8px;
  border: 1px solid #d0d0d0;
  border-radius: 5px;
  font-size: 12px;
  background: #fff;
  max-width: 200px;
}
#progress {
  font-size: 12px;
  color: #666;
  display: flex;
  align-items: center;
  gap: 10px;
  white-space: nowrap;
}
#progress .stat { font-weight: 600; }
.prog-correct { color: #16a34a; }
.prog-incorrect { color: #dc2626; }
#save-indicator {
  font-size: 11px;
  margin-left: auto;
  padding: 3px 8px;
  border-radius: 4px;
  white-space: nowrap;
}
#save-indicator.saved { color: #16a34a; background: #f0fdf4; }
#save-indicator.saving { color: #666; background: #f5f5f5; }
#save-indicator.error { color: #dc2626; background: #fef2f2; }

/* ---- Main ---- */
#main { flex: 1; display: flex; overflow: hidden; min-height: 0; }

/* ---- Sentence list (left) ---- */
#sentence-list {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 0;
  min-width: 0;
}

#raw-comment-box {
  background: #fffde7;
  border: 1px solid #e0d88a;
  border-radius: 8px;
  padding: 14px 16px;
  font-size: 13px;
  color: #555;
  line-height: 1.55;
  margin-bottom: 14px;
  flex-shrink: 0;
}
#raw-comment-box .rc-label {
  font-size: 11px;
  font-weight: 600;
  color: #999;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 6px;
}
#raw-comment-text .source-hl {
  border-radius: 3px;
  padding: 1px 2px;
  font-weight: 600;
  transition: background 0.15s;
}

.sentence-card {
  background: #fff;
  border: 2px solid #e8e8e8;
  border-radius: 8px;
  padding: 12px 14px;
  margin-bottom: 8px;
  cursor: pointer;
  transition: border-color 0.12s, box-shadow 0.12s;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.sentence-card:hover { border-color: #c0c0c0; box-shadow: 0 2px 6px rgba(0,0,0,0.05); }
.sentence-card.selected { border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59,130,246,0.15); }
.sentence-card.labeled-correct { border-left: 4px solid #22c55e; }
.sentence-card.labeled-incorrect { border-left: 4px solid #ef4444; }

.sc-index {
  font-size: 10px;
  font-weight: 600;
  color: #aaa;
}
.sc-text {
  font-size: 13px;
  line-height: 1.45;
  color: #333;
}
.sc-tag-row {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}
.sc-tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
}
.sc-tag .swatch {
  width: 8px; height: 8px; border-radius: 2px;
  flex-shrink: 0; border: 1px solid rgba(0,0,0,0.1);
}
.sc-badge {
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 3px;
  font-weight: 600;
}
.sc-badge.correct { background: #dcfce7; color: #15803d; }
.sc-badge.incorrect { background: #fef2f2; color: #b91c1c; }
.sc-secondary-badge {
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 3px;
  font-weight: 500;
  background: #eff6ff;
  color: #1d4ed8;
}
.sc-source {
  font-size: 11px;
  color: #777;
  line-height: 1.4;
  padding: 4px 8px;
  background: #fafaf5;
  border-left: 3px solid #e0d88a;
  border-radius: 3px;
}
.sc-source .src-label {
  font-size: 10px;
  font-weight: 600;
  color: #b0a060;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}
.sc-reasoning {
  font-size: 11px;
  color: #999;
  line-height: 1.4;
  font-style: italic;
}

/* ---- Viz context panel ---- */
#viz-context {
  background: #fff;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  margin-bottom: 14px;
  flex-shrink: 0;
  overflow: hidden;
}
#viz-toggle {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 14px;
  cursor: pointer;
  user-select: none;
  background: #fafafa;
  border-bottom: 1px solid #e8e8e8;
  font-size: 12px;
  font-weight: 600;
  color: #555;
}
#viz-toggle:hover { background: #f0f0f0; }
#viz-toggle .chevron {
  font-size: 10px;
  transition: transform 0.2s;
}
#viz-context.collapsed #viz-toggle .chevron { transform: rotate(-90deg); }
#viz-context.collapsed #viz-body { display: none; }
#viz-body {
  display: flex;
  gap: 16px;
  padding: 14px;
}
#viz-image-wrap {
  flex-shrink: 0;
  max-width: 350px;
}
#viz-image-wrap img {
  width: 100%;
  height: auto;
  border-radius: 6px;
  border: 1px solid #e0e0e0;
  cursor: pointer;
}
#viz-description {
  flex: 1;
  font-size: 12px;
  color: #444;
  line-height: 1.55;
  overflow-y: auto;
  white-space: pre-wrap;
}
#viz-description h2, #viz-description h3 {
  font-size: 12px;
  margin: 8px 0 2px 0;
}
#viz-description table {
  font-size: 11px;
  border-collapse: collapse;
  margin: 6px 0;
}
#viz-description th, #viz-description td {
  border: 1px solid #ddd;
  padding: 3px 6px;
  text-align: left;
}
#viz-description th { background: #f5f5f5; font-weight: 600; }

/* Image lightbox */
#lightbox {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.8);
  z-index: 9999;
  align-items: center;
  justify-content: center;
  cursor: zoom-out;
}
#lightbox.active { display: flex; }
#lightbox img {
  max-width: 95vw;
  max-height: 95vh;
  border-radius: 8px;
  box-shadow: 0 4px 30px rgba(0,0,0,0.4);
}

/* ---- Detail panel (right) ---- */
#detail-panel {
  width: 340px;
  background: #fff;
  border-left: 1px solid #e0e0e0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  flex-shrink: 0;
}
#detail-scroll {
  flex: 1;
  overflow-y: auto;
  padding: 18px 16px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.panel-view { display: flex; flex-direction: column; gap: 14px; }
.panel-view[hidden] { display: none !important; }

#detail-panel h3 { font-size: 14px; font-weight: 600; }

.detail-sentence {
  font-size: 12px; color: #444; line-height: 1.45;
  padding: 8px 10px; background: #f8f8f8; border-radius: 5px;
  border-left: 3px solid #ddd;
}
.detail-current { font-size: 12px; color: #777; }
.detail-reasoning {
  font-size: 11px; color: #888; line-height: 1.4;
  padding: 6px 10px; background: #fafafa; border-radius: 5px;
  border-left: 3px solid #e8e8e8;
  font-style: italic;
}

.label-section-title { font-size: 12px; font-weight: 600; color: #555; margin-bottom: 6px; }

.status-buttons { display: flex; gap: 6px; flex-wrap: wrap; }
.cat-btn {
  padding: 6px 13px; border: 2px solid #e0e0e0; border-radius: 6px;
  background: #fff; font-size: 12px; font-weight: 500;
  cursor: pointer; transition: all 0.12s; white-space: nowrap;
}
.cat-btn:hover { border-color: #bbb; }
.cat-btn.active[data-value="correct"] { background: #dcfce7; border-color: #22c55e; color: #15803d; }
.cat-btn.active[data-value="incorrect"] { background: #fef2f2; border-color: #ef4444; color: #b91c1c; }
.clear-btn-small { color: #999 !important; font-size: 11px !important; padding: 4px 10px !important; }

/* Color-swatch dropdown */
.color-select { position: relative; width: 100%; }
.color-select-trigger {
  padding: 6px 8px; border: 1px solid #ddd; border-radius: 5px;
  font-size: 12px; background: #fff; cursor: pointer;
  display: flex; align-items: center; gap: 6px; min-height: 30px;
  user-select: none;
}
.color-select-trigger:hover { border-color: #bbb; }
.color-select-trigger .trigger-swatch {
  width: 11px; height: 11px; border-radius: 2px; flex-shrink: 0;
  border: 1px solid rgba(0,0,0,0.12);
}
.color-select-trigger .trigger-placeholder { color: #999; }
.color-select-dropdown {
  position: absolute; top: calc(100% + 2px); left: 0; right: 0;
  background: #fff; border: 1px solid #ddd; border-radius: 5px;
  max-height: 240px; overflow-y: auto; z-index: 200;
  box-shadow: 0 4px 12px rgba(0,0,0,0.12); display: none;
}
.color-select.open .color-select-dropdown { display: block; }
.color-option {
  padding: 5px 8px; font-size: 12px; cursor: pointer;
  display: flex; align-items: center; gap: 6px;
}
.color-option:hover { background: #f5f5f5; }
.color-option.selected { background: #eff6ff; }
.color-option .option-swatch {
  width: 11px; height: 11px; border-radius: 2px; flex-shrink: 0;
  border: 1px solid rgba(0,0,0,0.12);
}

.panel-textarea {
  width: 100%; padding: 8px 10px; border: 1px solid #ddd; border-radius: 6px;
  font-size: 12px; font-family: inherit; resize: vertical; min-height: 50px; line-height: 1.45;
}
.panel-textarea:focus { outline: none; border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59,130,246,0.1); }
.panel-hint { font-size: 12px; color: #aaa; font-style: italic; }

/* Shortcuts footer */
.shortcuts-help {
  font-size: 11px; color: #aaa; line-height: 1.6;
  border-top: 1px solid #eee; padding: 12px 16px; flex-shrink: 0;
}
.shortcuts-help strong { color: #888; }

/* ---- Summary panel ---- */
#summary-stats { font-size: 12px; color: #666; line-height: 1.6; }
.summary-row { display: flex; justify-content: space-between; align-items: center; padding: 2px 0; }
.summary-row .count { font-weight: 600; }
</style>
</head>
<body>

<div id="header">
  <h1>ACE Category Labeling</h1>
  <select id="article-select"></select>
  <div class="header-sep"></div>
  <button class="nav-btn" id="prev-btn">&larr; Prev</button>
  <select id="comment-select"></select>
  <button class="nav-btn" id="next-btn">Next &rarr;</button>
  <div id="progress"></div>
  <span id="save-indicator" class="saved">Saved</span>
</div>

<div id="lightbox"><img id="lightbox-img" src=""></div>

<div id="main">
  <div id="sentence-list">
    <div id="viz-context">
      <div id="viz-toggle">
        <span>Visualization &amp; Description</span>
        <span class="chevron">&#9660;</span>
      </div>
      <div id="viz-body">
        <div id="viz-image-wrap"><img id="viz-image" src="" alt="Visualization"></div>
        <div id="viz-description"></div>
      </div>
    </div>
    <div id="raw-comment-box">
      <div class="rc-label">Original Comment</div>
      <div id="raw-comment-text"></div>
    </div>
    <div id="cards-container"></div>
  </div>

  <div id="detail-panel">
    <div id="detail-scroll">

      <!-- Summary view (no sentence selected) -->
      <div id="panel-summary" class="panel-view">
        <h3 id="summary-heading">Comment</h3>
        <div id="summary-stats"></div>
        <div class="panel-hint">Click a sentence card on the left to label it.</div>
      </div>

      <!-- Sentence detail view -->
      <div id="panel-sentence" class="panel-view" hidden>
        <h3 id="sent-heading">Sentence 0</h3>
        <div id="sent-text" class="detail-sentence"></div>
        <div id="sent-reasoning" class="detail-reasoning"></div>
        <div id="sent-current-cat" class="detail-current"></div>

        <div>
          <div class="label-section-title">Category correct?</div>
          <div class="status-buttons">
            <button class="cat-btn" data-value="correct">&#10003; Correct</button>
            <button class="cat-btn" data-value="incorrect">&#10007; Incorrect</button>
            <button class="cat-btn clear-btn-small" data-value="">Clear</button>
          </div>
        </div>

        <div id="corrected-cat-section" hidden>
          <div class="label-section-title">Correct category</div>
          <div id="corrected-cat-select" class="color-select">
            <div class="color-select-trigger"><span class="trigger-placeholder">Select category...</span></div>
            <div class="color-select-dropdown"></div>
          </div>
        </div>

        <div>
          <div class="label-section-title">Secondary category (optional)</div>
          <div id="secondary-cat-select" class="color-select">
            <div class="color-select-trigger"><span class="trigger-placeholder">Select secondary category...</span></div>
            <div class="color-select-dropdown"></div>
          </div>
        </div>

        <div>
          <div class="label-section-title">Notes</div>
          <textarea id="sent-notes" class="panel-textarea" placeholder="Notes about this sentence..." rows="2"></textarea>
        </div>
      </div>

    </div>

    <div class="shortcuts-help">
      <div><strong>&larr; &rarr;</strong> Prev/Next comment &nbsp;<strong>&uarr; &darr;</strong> Prev/Next sentence</div>
      <div><strong>1</strong> Correct &nbsp;<strong>2</strong> Incorrect &nbsp;<strong>0</strong> Clear</div>
      <div><strong>Esc</strong> Deselect &nbsp;<strong>N</strong> Focus notes</div>
    </div>
  </div>
</div>

<script>
(function() {

  /* ===== 1. Constants ===== */

  var CATEGORIES = [
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
    "Meta / Paratext",
    "Uncategorizable"
  ];
  var TAB10 = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#aec7e8","#ffbb78"
  ];
  var CATEGORY_COLORS = {};
  CATEGORIES.forEach(function(c, i) { CATEGORY_COLORS[c] = TAB10[i % TAB10.length]; });
  CATEGORY_COLORS["unknown"] = "#b3b3b3";

  function catColor(cat) { return CATEGORY_COLORS[cat] || CATEGORY_COLORS["unknown"]; }
  function escapeHTML(s) { return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }
  function escapeRegex(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

  var rawCommentPlain = "";

  function highlightSourceInComment(aceSentence, color) {
    var el = document.getElementById("raw-comment-text");
    if (!aceSentence) { el.textContent = rawCommentPlain; return; }
    var comment = currentComment();
    var mappings = comment.source_mappings || {};
    var fragments = mappings[aceSentence];
    if (!fragments || fragments.length === 0) { el.textContent = rawCommentPlain; return; }

    var raw = rawCommentPlain;
    var ranges = [];
    fragments.forEach(function(frag) {
      var re = new RegExp(escapeRegex(frag), "gi"), m;
      while ((m = re.exec(raw)) !== null) {
        ranges.push({ start: m.index, end: m.index + m[0].length });
      }
    });
    if (ranges.length === 0) { el.textContent = rawCommentPlain; return; }

    ranges.sort(function(a,b) { return a.start - b.start; });
    var merged = [{ start: ranges[0].start, end: ranges[0].end }];
    for (var i = 1; i < ranges.length; i++) {
      var last = merged[merged.length - 1];
      if (ranges[i].start <= last.end) last.end = Math.max(last.end, ranges[i].end);
      else merged.push({ start: ranges[i].start, end: ranges[i].end });
    }

    var html = "", pos = 0;
    merged.forEach(function(r) {
      html += escapeHTML(raw.substring(pos, r.start));
      html += "<span class='source-hl' style='background:" + color + "30;outline:2px solid " + color + "'>";
      html += escapeHTML(raw.substring(r.start, r.end)) + "</span>";
      pos = r.end;
    });
    html += escapeHTML(raw.substring(pos));
    el.innerHTML = html;
  }

  /* ===== 2. State ===== */

  var allComments = [];        // from /data — grouped by comment
  var classifications = [];    // from /classifications — flat list
  var labels = {};
  var currentIdx = 0;
  var selectedSentIdx = null;
  var saveTimer = null;

  /* ===== 3. Data helpers ===== */

  function currentComment() { return allComments[currentIdx]; }
  function commentKey(c) { return String(c.comment_index); }

  function getSentencesForComment(comment) {
    var cid = comment.comment_index;
    return classifications.filter(function(cl) { return cl.comment_id === cid; });
  }

  function sentenceKey(commentIndex, sentIndex) {
    return commentIndex + ":" + sentIndex;
  }

  /* ===== 4. Label management ===== */

  function ensureCommentEntry(ci) {
    var k = String(ci);
    if (!labels.comments) labels.comments = {};
    if (!labels.comments[k]) labels.comments[k] = {};
    return labels.comments[k];
  }

  function getSentLabel(ci, sentIdx) {
    var cl = labels.comments && labels.comments[String(ci)];
    if (!cl || !cl.sentences) return null;
    return cl.sentences[String(sentIdx)] || null;
  }

  function setSentLabel(ci, sentIdx, data) {
    var entry = ensureCommentEntry(ci);
    if (!entry.sentences) entry.sentences = {};
    if (data) entry.sentences[String(sentIdx)] = data;
    else delete entry.sentences[String(sentIdx)];
    scheduleSave();
  }

  /* ===== 5. Save ===== */

  function scheduleSave() {
    document.getElementById("save-indicator").textContent = "Saving...";
    document.getElementById("save-indicator").className = "saving";
    clearTimeout(saveTimer);
    saveTimer = setTimeout(doSave, 500);
  }
  function doSave() {
    fetch("/labels", {
      method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(labels)
    }).then(function(r) {
      document.getElementById("save-indicator").textContent = r.ok ? "Saved" : "Save error";
      document.getElementById("save-indicator").className = r.ok ? "saved" : "error";
    }).catch(function() {
      document.getElementById("save-indicator").textContent = "Save error";
      document.getElementById("save-indicator").className = "error";
    });
  }

  /* ===== 6. Render sentence cards ===== */

  function renderCards() {
    var container = document.getElementById("cards-container");
    container.innerHTML = "";
    var comment = currentComment();
    var sents = getSentencesForComment(comment);
    var ci = comment.comment_index;

    rawCommentPlain = comment.raw_comment || "";
    document.getElementById("raw-comment-text").textContent = rawCommentPlain;

    sents.forEach(function(cl, idx) {
      var card = document.createElement("div");
      card.className = "sentence-card";
      card.setAttribute("data-idx", String(idx));

      var lbl = getSentLabel(ci, idx);
      if (lbl) {
        if (lbl.category_correct === true) card.classList.add("labeled-correct");
        else if (lbl.category_correct === false) card.classList.add("labeled-incorrect");
      }
      if (idx === selectedSentIdx) card.classList.add("selected");

      var cat = cl.comment_tag || "unknown";
      var color = catColor(cat);

      var srcMappings = comment.source_mappings || {};
      var srcFragments = srcMappings[cl.original_comment] || [];

      var html = "<div class='sc-index'>Sentence " + idx + "</div>";
      html += "<div class='sc-text'>" + escapeHTML(cl.original_comment) + "</div>";
      if (srcFragments.length > 0) {
        html += "<div class='sc-source'><span class='src-label'>Source: </span>" + srcFragments.map(function(f){ return escapeHTML(f); }).join(" &bull; ") + "</div>";
      }
      html += "<div class='sc-tag-row'>";
      html += "<span class='sc-tag' style='background:" + color + "18;color:" + color + "'>";
      html += "<span class='swatch' style='background:" + color + "'></span>" + escapeHTML(cat) + "</span>";

      if (lbl) {
        if (lbl.category_correct === true) {
          html += "<span class='sc-badge correct'>&#10003; Correct</span>";
        } else if (lbl.category_correct === false) {
          html += "<span class='sc-badge incorrect'>&#10007; Incorrect</span>";
          if (lbl.correct_category) {
            var cc = lbl.correct_category;
            var ccColor = catColor(cc);
            html += "<span class='sc-tag' style='background:" + ccColor + "18;color:" + ccColor + "'>";
            html += "<span class='swatch' style='background:" + ccColor + "'></span>" + escapeHTML(cc) + "</span>";
          }
        }
        if (lbl.secondary_category) {
          var sc = lbl.secondary_category;
          html += "<span class='sc-secondary-badge'>2nd: " + escapeHTML(sc) + "</span>";
        }
      }
      html += "</div>";

      if (cl.reasoning) {
        html += "<div class='sc-reasoning'>" + escapeHTML(cl.reasoning) + "</div>";
      }

      card.innerHTML = html;

      (function(sentCl, sentColor) {
        card.addEventListener("mouseenter", function() {
          highlightSourceInComment(sentCl.original_comment, sentColor);
        });
        card.addEventListener("mouseleave", function() {
          if (selectedSentIdx !== null) {
            var selSent = sents[selectedSentIdx];
            var selColor = catColor(selSent ? (selSent.comment_tag || "unknown") : "unknown");
            highlightSourceInComment(selSent ? selSent.original_comment : null, selColor);
          } else {
            highlightSourceInComment(null, null);
          }
        });
        card.addEventListener("click", function() { selectSentence(idx); });
      })(cl, color);

      container.appendChild(card);
    });
  }

  /* ===== 7. Panel management ===== */

  function showSummary() {
    selectedSentIdx = null;
    document.getElementById("panel-summary").hidden = false;
    document.getElementById("panel-sentence").hidden = true;
    updateSummary();
    highlightSelectedCard();
    highlightSourceInComment(null, null);
  }

  function selectSentence(idx) {
    selectedSentIdx = idx;
    document.getElementById("panel-summary").hidden = true;
    document.getElementById("panel-sentence").hidden = false;
    updateSentencePanel();
    highlightSelectedCard();
    var sents = getSentencesForComment(currentComment());
    if (idx !== null && idx < sents.length) {
      var cl = sents[idx];
      highlightSourceInComment(cl.original_comment, catColor(cl.comment_tag || "unknown"));
    }
  }

  function highlightSelectedCard() {
    document.querySelectorAll(".sentence-card").forEach(function(c, i) {
      c.classList.toggle("selected", i === selectedSentIdx);
    });
  }

  /* ===== 8. Summary panel ===== */

  function updateSummary() {
    var comment = currentComment();
    var sents = getSentencesForComment(comment);
    var ci = comment.comment_index;
    document.getElementById("summary-heading").textContent = "Comment " + ci;

    var total = sents.length;
    var correct = 0, incorrect = 0, unlabeled = 0;
    sents.forEach(function(_, idx) {
      var lbl = getSentLabel(ci, idx);
      if (!lbl || lbl.category_correct === undefined) unlabeled++;
      else if (lbl.category_correct === true) correct++;
      else if (lbl.category_correct === false) incorrect++;
    });

    var html = "<div class='summary-row'><span>Total sentences</span><span class='count'>" + total + "</span></div>";
    html += "<div class='summary-row'><span style='color:#16a34a'>&#10003; Correct</span><span class='count'>" + correct + "</span></div>";
    html += "<div class='summary-row'><span style='color:#dc2626'>&#10007; Incorrect</span><span class='count'>" + incorrect + "</span></div>";
    html += "<div class='summary-row'><span style='color:#999'>Unlabeled</span><span class='count'>" + unlabeled + "</span></div>";
    document.getElementById("summary-stats").innerHTML = html;
  }

  /* ===== 9. Sentence detail panel ===== */

  function updateSentencePanel() {
    if (selectedSentIdx === null) return;
    var comment = currentComment();
    var sents = getSentencesForComment(comment);
    if (selectedSentIdx >= sents.length) return;
    var cl = sents[selectedSentIdx];
    var ci = comment.comment_index;

    document.getElementById("sent-heading").textContent = "Sentence " + selectedSentIdx + " of " + sents.length;
    document.getElementById("sent-text").textContent = cl.original_comment;
    document.getElementById("sent-reasoning").textContent = cl.reasoning || "";
    document.getElementById("sent-reasoning").hidden = !cl.reasoning;
    document.getElementById("sent-current-cat").innerHTML = "Assigned: <strong>" + escapeHTML(cl.comment_tag || "unknown") + "</strong>";

    var lbl = getSentLabel(ci, selectedSentIdx);
    var catCorrect = lbl ? lbl.category_correct : undefined;

    document.querySelectorAll(".cat-btn").forEach(function(btn) {
      var val = btn.getAttribute("data-value");
      if (val === "correct") btn.classList.toggle("active", catCorrect === true);
      else if (val === "incorrect") btn.classList.toggle("active", catCorrect === false);
      else btn.classList.remove("active");
    });

    document.getElementById("corrected-cat-section").hidden = (catCorrect !== false);
    if (correctedCatSelect) correctedCatSelect.setValue((lbl && lbl.correct_category) || "");
    if (secondaryCatSelect) secondaryCatSelect.setValue((lbl && lbl.secondary_category) || "");
    document.getElementById("sent-notes").value = (lbl && lbl.notes) || "";
  }

  /* ===== 10. Label change handlers ===== */

  function collectSentLabel() {
    var catCorrect = undefined;
    document.querySelectorAll(".cat-btn.active").forEach(function(btn) {
      var v = btn.getAttribute("data-value");
      if (v === "correct") catCorrect = true;
      else if (v === "incorrect") catCorrect = false;
    });
    var correctCat = (correctedCatSelect ? correctedCatSelect.getValue() : "") || null;
    var secondaryCat = (secondaryCatSelect ? secondaryCatSelect.getValue() : "") || null;
    var notes = document.getElementById("sent-notes").value.trim();

    if (catCorrect === undefined && !secondaryCat && !notes) return null;
    var data = { category_correct: catCorrect, notes: notes };
    if (catCorrect === false && correctCat) data.correct_category = correctCat;
    if (secondaryCat) data.secondary_category = secondaryCat;
    data.timestamp = new Date().toISOString();
    return data;
  }

  function onSentLabelChange() {
    if (selectedSentIdx === null) return;
    var data = collectSentLabel();
    setSentLabel(currentComment().comment_index, selectedSentIdx, data);
    updateSentencePanel();
    renderCards();
    updateProgress();
    updateDropdownStatus();
    updateSummary();
  }

  /* ===== 11. Event listeners ===== */

  document.querySelectorAll(".cat-btn").forEach(function(btn) {
    btn.addEventListener("click", function() {
      var v = btn.getAttribute("data-value");
      document.querySelectorAll(".cat-btn").forEach(function(b) { b.classList.remove("active"); });
      if (v) btn.classList.add("active");
      document.getElementById("corrected-cat-section").hidden = (v !== "incorrect");
      onSentLabelChange();
    });
  });

  var sentNotesTimer = null;
  document.getElementById("sent-notes").addEventListener("input", function() {
    clearTimeout(sentNotesTimer);
    sentNotesTimer = setTimeout(onSentLabelChange, 400);
  });

  /* ===== 12. Navigation ===== */

  function navigate(idx) {
    if (idx < 0 || idx >= allComments.length) return;
    currentIdx = idx;
    selectedSentIdx = null;
    showSummary();
    renderCards();
    document.getElementById("comment-select").value = String(idx);
    document.getElementById("prev-btn").disabled = (idx === 0);
    document.getElementById("next-btn").disabled = (idx === allComments.length - 1);
  }

  document.getElementById("prev-btn").addEventListener("click", function() { navigate(currentIdx - 1); });
  document.getElementById("next-btn").addEventListener("click", function() { navigate(currentIdx + 1); });
  document.getElementById("comment-select").addEventListener("change", function(e) {
    navigate(parseInt(e.target.value, 10));
  });

  function updateProgress() {
    if (!labels.comments) labels.comments = {};
    var totalSents = classifications.length;
    var correct = 0, incorrect = 0;
    Object.keys(labels.comments).forEach(function(ci) {
      var entry = labels.comments[ci];
      if (!entry.sentences) return;
      Object.values(entry.sentences).forEach(function(sl) {
        if (sl.category_correct === true) correct++;
        else if (sl.category_correct === false) incorrect++;
      });
    });
    var labeled = correct + incorrect;
    var html = "<span class='stat'>" + labeled + "</span>/" + totalSents + " labeled";
    if (correct) html += " &nbsp;<span class='prog-correct'>&#10003; " + correct + "</span>";
    if (incorrect) html += " &nbsp;<span class='prog-incorrect'>&#10007; " + incorrect + "</span>";
    document.getElementById("progress").innerHTML = html;
  }

  function updateDropdownStatus() {
    var sel = document.getElementById("comment-select");
    for (var i = 0; i < sel.options.length; i++) {
      var c = allComments[i];
      var sents = getSentencesForComment(c);
      var total = sents.length, done = 0;
      sents.forEach(function(_, idx) {
        var lbl = getSentLabel(c.comment_index, idx);
        if (lbl && lbl.category_correct !== undefined) done++;
      });
      var prefix = done === 0 ? "\u00B7" : done === total ? "\u2713" : "(" + done + "/" + total + ")";
      sel.options[i].textContent = prefix + " Comment " + c.comment_index;
    }
  }

  /* ===== 13. Keyboard shortcuts ===== */

  document.addEventListener("keydown", function(e) {
    if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT" || e.target.tagName === "SELECT") {
      if (e.key === "Escape") { e.target.blur(); e.preventDefault(); }
      return;
    }
    if (e.key === "Escape") { showSummary(); renderCards(); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { navigate(currentIdx - 1); e.preventDefault(); }
    else if (e.key === "ArrowRight") { navigate(currentIdx + 1); e.preventDefault(); }
    else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (selectedSentIdx === null) {
        var sents = getSentencesForComment(currentComment());
        if (sents.length > 0) selectSentence(sents.length - 1);
      } else if (selectedSentIdx > 0) {
        selectSentence(selectedSentIdx - 1);
        scrollCardIntoView(selectedSentIdx);
      }
    }
    else if (e.key === "ArrowDown") {
      e.preventDefault();
      var sents2 = getSentencesForComment(currentComment());
      if (selectedSentIdx === null) {
        if (sents2.length > 0) selectSentence(0);
      } else if (selectedSentIdx < sents2.length - 1) {
        selectSentence(selectedSentIdx + 1);
        scrollCardIntoView(selectedSentIdx);
      }
    }
    else if (e.key === "1" && selectedSentIdx !== null) { clickCatBtn("correct"); e.preventDefault(); }
    else if (e.key === "2" && selectedSentIdx !== null) { clickCatBtn("incorrect"); e.preventDefault(); }
    else if (e.key === "0" && selectedSentIdx !== null) { clickCatBtn(""); e.preventDefault(); }
    else if ((e.key === "n" || e.key === "N") && selectedSentIdx !== null) {
      document.getElementById("sent-notes").focus(); e.preventDefault();
    }
  });

  function clickCatBtn(val) {
    document.querySelectorAll(".cat-btn").forEach(function(btn) {
      if (btn.getAttribute("data-value") === val) btn.click();
    });
  }

  function scrollCardIntoView(idx) {
    var cards = document.querySelectorAll(".sentence-card");
    if (cards[idx]) cards[idx].scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  /* ===== 14. ColorSelect component ===== */

  var correctedCatSelect = null;
  var secondaryCatSelect = null;

  function ColorSelect(containerId, placeholder, options, onChange) {
    var self = this;
    this.el = document.getElementById(containerId);
    this.trigger = this.el.querySelector(".color-select-trigger");
    this.dropdown = this.el.querySelector(".color-select-dropdown");
    this.value = "";
    this.options = options;
    this.onChange = onChange;

    var emptyOpt = document.createElement("div");
    emptyOpt.className = "color-option";
    emptyOpt.setAttribute("data-value", "");
    emptyOpt.innerHTML = "<span style='color:#999'>" + escapeHTML(placeholder) + "</span>";
    emptyOpt.addEventListener("click", function() { self.setValue(""); self.close(); if (self.onChange) self.onChange(""); });
    this.dropdown.appendChild(emptyOpt);

    options.forEach(function(opt) {
      var div = document.createElement("div");
      div.className = "color-option";
      div.setAttribute("data-value", opt.value);
      div.innerHTML = "<span class='option-swatch' style='background:" + opt.color + "'></span><span>" + escapeHTML(opt.label) + "</span>";
      div.addEventListener("click", function() { self.setValue(opt.value); self.close(); if (self.onChange) self.onChange(opt.value); });
      self.dropdown.appendChild(div);
    });

    this.trigger.addEventListener("click", function(e) {
      e.stopPropagation();
      var wasOpen = self.el.classList.contains("open");
      closeAllColorSelects();
      if (!wasOpen) self.el.classList.add("open");
    });
  }
  ColorSelect.prototype.getValue = function() { return this.value; };
  ColorSelect.prototype.setValue = function(val) {
    this.value = val;
    var opt = this.options.find(function(o) { return o.value === val; });
    if (opt) {
      this.trigger.innerHTML = "<span class='trigger-swatch' style='background:" + opt.color + "'></span><span>" + escapeHTML(opt.label) + "</span>";
    } else {
      this.trigger.innerHTML = "<span class='trigger-placeholder'>Select...</span>";
    }
    this.dropdown.querySelectorAll(".color-option").forEach(function(div) {
      div.classList.toggle("selected", div.getAttribute("data-value") === val);
    });
  };
  ColorSelect.prototype.close = function() { this.el.classList.remove("open"); };

  function closeAllColorSelects() {
    if (correctedCatSelect) correctedCatSelect.close();
    if (secondaryCatSelect) secondaryCatSelect.close();
  }
  document.addEventListener("click", closeAllColorSelects);

  /* ===== 15. Viz panel & lightbox ===== */

  document.getElementById("viz-toggle").addEventListener("click", function() {
    document.getElementById("viz-context").classList.toggle("collapsed");
  });

  var lightbox = document.getElementById("lightbox");
  document.getElementById("viz-image").addEventListener("click", function() {
    document.getElementById("lightbox-img").src = this.src;
    lightbox.classList.add("active");
  });
  lightbox.addEventListener("click", function() { lightbox.classList.remove("active"); });
  document.addEventListener("keydown", function(e) {
    if (e.key === "Escape" && lightbox.classList.contains("active")) {
      lightbox.classList.remove("active");
      e.stopImmediatePropagation();
    }
  }, true);

  function simpleMarkdown(md) {
    var lines = md.split("\n");
    var html = "", inTable = false;
    lines.forEach(function(line) {
      var trimmed = line.trim();
      if (trimmed.match(/^\|.*\|$/)) {
        if (trimmed.match(/^\|[\s\-:|]+\|$/)) return;
        var cells = trimmed.replace(/^\|/, "").replace(/\|$/, "").split("|").map(function(c) { return c.trim(); });
        if (!inTable) { html += "<table>"; inTable = true; }
        var tag = html.indexOf("<tr>") === -1 ? "th" : "td";
        html += "<tr>" + cells.map(function(c) { return "<" + tag + ">" + escapeHTML(c) + "</" + tag + ">"; }).join("") + "</tr>";
        return;
      }
      if (inTable) { html += "</table>"; inTable = false; }
      if (trimmed.match(/^## /)) { html += "<h2>" + escapeHTML(trimmed.substring(3)) + "</h2>"; }
      else if (trimmed.match(/^### /)) { html += "<h3>" + escapeHTML(trimmed.substring(4)) + "</h3>"; }
      else if (trimmed === "") { html += "<br>"; }
      else {
        var processed = escapeHTML(trimmed)
          .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
          .replace(/`(.+?)`/g, "<code>$1</code>");
        if (trimmed.match(/^- /)) html += "<div style='padding-left:12px'>&bull; " + processed.substring(2) + "</div>";
        else html += "<div>" + processed + "</div>";
      }
    });
    if (inTable) html += "</table>";
    return html;
  }

  /* ===== 16. Article switching ===== */

  var currentArticleId = null;
  var imageDescription = "";
  var dropdownsPopulated = false;

  function populateDropdowns() {
    if (dropdownsPopulated) return;
    dropdownsPopulated = true;
    var catOptions = CATEGORIES.map(function(c) { return { value: c, label: c, color: catColor(c) }; });
    catOptions.push({ value: "unknown", label: "unknown", color: catColor("unknown") });
    correctedCatSelect = new ColorSelect("corrected-cat-select", "Select category...", catOptions, onSentLabelChange);
    secondaryCatSelect = new ColorSelect("secondary-cat-select", "Select secondary category...", catOptions, onSentLabelChange);
  }

  async function loadArticle(articleId) {
    if (!articleId) return;
    currentArticleId = articleId;
    try {
      var switchResp = await fetch("/switch", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({article_id: articleId})
      });
      if (!switchResp.ok) {
        var errText = await switchResp.text();
        document.getElementById("raw-comment-text").textContent = "Error switching article: " + errText;
        return;
      }
      var dataResp = await fetch("/data");
      allComments = await dataResp.json();
      var classResp = await fetch("/classifications");
      classifications = await classResp.json();
      var labelsResp = await fetch("/labels");
      labels = await labelsResp.json();
      var metaResp = await fetch("/meta");
      var meta = await metaResp.json();
      imageDescription = meta.image_description || "";
    } catch(err) {
      document.getElementById("raw-comment-text").textContent = "Error loading data: " + err.message;
      return;
    }
    if (!labels.comments) labels.comments = {};

    document.getElementById("viz-image").src = "/image?t=" + Date.now();
    if (imageDescription) {
      document.getElementById("viz-description").innerHTML = simpleMarkdown(imageDescription);
    } else {
      document.getElementById("viz-description").textContent = "No visualization description available.";
    }

    populateDropdowns();

    var sel = document.getElementById("comment-select");
    sel.innerHTML = "";
    allComments.forEach(function(c, i) {
      var opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = "\u00B7 Comment " + c.comment_index;
      sel.appendChild(opt);
    });

    currentIdx = 0;
    selectedSentIdx = null;
    updateDropdownStatus();
    updateProgress();
    if (allComments.length > 0) navigate(0);
    else {
      document.getElementById("cards-container").innerHTML = "";
      document.getElementById("raw-comment-text").textContent = "No comments for this article.";
    }
  }

  document.getElementById("article-select").addEventListener("change", function(e) {
    var aid = e.target.value;
    if (aid && aid !== currentArticleId) loadArticle(aid);
  });

  /* ===== 17. Init ===== */

  async function init() {
    var artResp = await fetch("/articles");
    var articles = await artResp.json();
    var artSel = document.getElementById("article-select");
    artSel.innerHTML = "";
    articles.forEach(function(aid) {
      var opt = document.createElement("option");
      opt.value = aid;
      opt.textContent = "Article " + aid;
      artSel.appendChild(opt);
    });

    var initialArticle = articles.length > 0 ? articles[0] : null;
    var metaResp2 = await fetch("/meta");
    var meta2 = await metaResp2.json();
    if (meta2.current_article_id) {
      initialArticle = meta2.current_article_id;
      artSel.value = initialArticle;
    }

    if (initialArticle) {
      artSel.value = initialArticle;
      await loadArticle(initialArticle);
    }
  }

  init();
})();
</script>
</body>
</html>"""


MIME_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".svg": "image/svg+xml",
}


def discover_articles(base: Path) -> list[str]:
    """Return sorted list of article IDs that have both comments and classifications."""
    comments_dir = base / "ace_comments"
    classifications_dir = base / "ace_classifications"
    if not comments_dir.exists() or not classifications_dir.exists():
        return []
    comment_ids = {
        d.name for d in comments_dir.iterdir()
        if d.is_dir() and any(d.glob("*.json"))
    }
    class_ids = set()
    for f in classifications_dir.glob("ace_sentence_classifications_*.json"):
        aid = f.stem.replace("ace_sentence_classifications_", "")
        class_ids.add(aid)
    available = sorted(comment_ids & class_ids, key=lambda x: int(x) if x.isdigit() else x)
    return available


def load_article_data(
    article_id: str, base: Path, labels_dir: Path
) -> dict:
    """Load all data for a given article. Returns a dict of everything needed."""
    comments_dir = base / "ace_comments" / article_id
    classifications_path = (
        base / "ace_classifications" / f"ace_sentence_classifications_{article_id}.json"
    )

    comments_data: list = []
    if comments_dir.exists():
        for p in sorted(comments_dir.glob("*.json"), key=lambda x: int(x.stem)):
            with open(p, "r", encoding="utf-8") as f:
                comments_data.append(json.load(f))
        comments_data.sort(key=lambda c: c["comment_index"])

    classifications_data: list = []
    if classifications_path.exists():
        with open(classifications_path, "r", encoding="utf-8") as f:
            classifications_data = json.load(f)

    images_dir = base.parent / "data" / "images"
    image_path = None
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = images_dir / f"{article_id}{ext}"
        if candidate.exists():
            image_path = candidate
            break

    image_description = ""
    for cl in classifications_data:
        if cl.get("image_description"):
            image_description = cl["image_description"]
            break

    labels_path = (labels_dir / f"{article_id}.json").resolve()
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    else:
        labels = {"article_id": article_id, "comments": {}}

    return {
        "comments_data": comments_data,
        "classifications_data": classifications_data,
        "labels": labels,
        "labels_path": labels_path,
        "image_path": image_path,
        "image_description": image_description,
    }


class LabelHandler(BaseHTTPRequestHandler):
    comments_data: list = []
    classifications_data: list = []
    labels: dict = {}
    labels_path: Path = Path()
    html_content: str = ""
    image_path: Path | None = None
    image_description: str = ""
    current_article_id: str = ""
    available_articles: list[str] = []
    base_dir: Path = Path()
    labels_dir: Path = Path()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._respond(self.html_content, "text/html; charset=utf-8")
        elif path == "/articles":
            self._respond(json.dumps(LabelHandler.available_articles), "application/json")
        elif path == "/data":
            self._respond(json.dumps(LabelHandler.comments_data), "application/json")
        elif path == "/classifications":
            self._respond(json.dumps(LabelHandler.classifications_data), "application/json")
        elif path == "/labels":
            self._respond(json.dumps(LabelHandler.labels), "application/json")
        elif path == "/meta":
            self._respond(json.dumps({
                "image_description": LabelHandler.image_description,
                "current_article_id": LabelHandler.current_article_id,
            }), "application/json")
        elif path == "/image":
            self._serve_image()
        else:
            self.send_error(404)

    def _serve_image(self) -> None:
        img = LabelHandler.image_path
        if not img or not img.exists():
            self.send_error(404, "Image not found")
            return
        ctype = MIME_TYPES.get(img.suffix.lower(), "application/octet-stream")
        data = img.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/labels":
            self._handle_save_labels()
        elif path == "/switch":
            self._handle_switch_article()
        else:
            self.send_error(404)

    def _handle_save_labels(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            new_labels = json.loads(body)
            LabelHandler.labels = new_labels
            LabelHandler.labels_path.parent.mkdir(parents=True, exist_ok=True)
            with open(LabelHandler.labels_path, "w", encoding="utf-8") as f:
                json.dump(new_labels, f, indent=2)
            self._respond('{"ok":true}', "application/json")
        except Exception as exc:
            self.send_error(500, str(exc))

    def _handle_switch_article(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            req = json.loads(body)
            article_id = req.get("article_id", "")
            if article_id not in LabelHandler.available_articles:
                self.send_error(400, f"Unknown article: {article_id}")
                return
            data = load_article_data(
                article_id, LabelHandler.base_dir, LabelHandler.labels_dir
            )
            LabelHandler.current_article_id = article_id
            LabelHandler.comments_data = data["comments_data"]
            LabelHandler.classifications_data = data["classifications_data"]
            LabelHandler.labels = data["labels"]
            LabelHandler.labels_path = data["labels_path"]
            LabelHandler.image_path = data["image_path"]
            LabelHandler.image_description = data["image_description"]
            print(
                f"Switched to article {article_id} "
                f"({len(data['comments_data'])} comments, "
                f"{len(data['classifications_data'])} sentences)",
                flush=True,
            )
            self._respond('{"ok":true}', "application/json")
        except Exception as exc:
            self.send_error(500, str(exc))

    def _respond(self, content: str, ctype: str) -> None:
        data = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args) -> None:  # type: ignore[override]
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive labeling tool for reviewing ACE sentence classifications."
    )
    parser.add_argument(
        "--article-id", type=str, default=None,
        help="Optional initial article ID (e.g. 181). If omitted, select in the UI.",
    )
    parser.add_argument(
        "--labels-dir", type=Path, default=Path("ace_category_labels"),
        help="Directory for label output files (default: ace_category_labels/).",
    )
    parser.add_argument(
        "--port", type=int, default=8051,
        help="Server port (default: 8051).",
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="Don't auto-open the browser.",
    )
    args = parser.parse_args()

    base = Path(__file__).parent
    available = discover_articles(base)
    if not available:
        print("No articles found with both ace_comments and ace_classifications.")
        return

    LabelHandler.base_dir = base
    LabelHandler.labels_dir = args.labels_dir
    LabelHandler.available_articles = available
    LabelHandler.html_content = HTML_TEMPLATE

    initial_id = args.article_id if args.article_id in available else available[0]
    data = load_article_data(initial_id, base, args.labels_dir)
    LabelHandler.current_article_id = initial_id
    LabelHandler.comments_data = data["comments_data"]
    LabelHandler.classifications_data = data["classifications_data"]
    LabelHandler.labels = data["labels"]
    LabelHandler.labels_path = data["labels_path"]
    LabelHandler.image_path = data["image_path"]
    LabelHandler.image_description = data["image_description"]

    server = HTTPServer(("127.0.0.1", args.port), LabelHandler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Labeling server running at {url}", flush=True)
    print(f"Available articles: {len(available)}", flush=True)
    print(f"Initial article: {initial_id} ({len(data['comments_data'])} comments, {len(data['classifications_data'])} sentences)", flush=True)
    print("Press Ctrl+C to stop.\n", flush=True)

    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
