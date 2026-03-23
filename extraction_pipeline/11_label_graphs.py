#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

"""
11_label_graphs.py

Interactive labeling tool for reviewing ACE dependency graphs.
Launches a local HTTP server with a single-page D3 visualization app
that lets you navigate comments, view their dependency graphs, and
label individual nodes (category correctness) and edges (type
correctness, removal).  Labels auto-save to disk.

Usage:
    python 11_label_graphs.py --input combined_data/35.json
    # Opens browser at http://localhost:8050
    # Labels saved to graph_labels/35.json

    python 11_label_graphs.py --input combined_data/35.json --port 9000 --no-open
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
<title>Graph Labeling Tool</title>
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
.prog-review { color: #ca8a04; }
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

/* ---- Graph area ---- */
#graph-area { flex: 1; position: relative; min-width: 0; }
#graph-area svg { display: block; }
#no-graph {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 15px; color: #999;
}

/* Legend */
#legend {
  position: absolute; top: 12px; right: 12px;
  background: rgba(255,255,255,0.94);
  border: 1px solid #ddd; border-radius: 6px;
  padding: 10px 14px; font-size: 11px; line-height: 1.7;
  box-shadow: 0 2px 8px rgba(0,0,0,0.07);
  max-height: calc(100% - 24px); overflow-y: auto; z-index: 50;
}
#legend .leg-title {
  font-weight: 600; margin-bottom: 3px; cursor: pointer;
  display: flex; align-items: center; justify-content: space-between; gap: 8px; user-select: none;
}
#legend .chevron { font-size: 9px; transition: transform 0.2s; }
#legend.collapsed .chevron { transform: rotate(-90deg); }
#legend.collapsed .legend-body { display: none; }
#legend .item { display: flex; align-items: center; gap: 6px; }
#legend .swatch { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; border: 1px solid rgba(0,0,0,0.12); }
#legend .edge-swatch { width: 20px; height: 3px; flex-shrink: 0; border-radius: 1px; }
#legend .section-divider { border: none; border-top: 1px solid #ddd; margin: 5px 0; }
#legend .section-label { font-weight: 600; font-size: 10px; margin-bottom: 1px; }

/* Tooltip */
#tooltip {
  position: absolute; pointer-events: none;
  background: rgba(25,25,25,0.93); color: #fff;
  padding: 8px 12px; border-radius: 6px;
  font-size: 12px; line-height: 1.45; max-width: 340px;
  opacity: 0; transition: opacity 0.12s; z-index: 100;
}
#tooltip .cat { color: #aaa; font-size: 11px; margin-top: 2px; }
#tooltip .src { color: #ccc; font-size: 11px; margin-top: 2px; font-style: italic; }

/* ---- Label panel ---- */
#label-panel {
  width: 320px;
  background: #fff;
  border-left: 1px solid #e0e0e0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  flex-shrink: 0;
}
#label-panel-scroll {
  flex: 1;
  overflow-y: auto;
  padding: 18px 16px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.panel-view { display: flex; flex-direction: column; gap: 14px; }
.panel-view[hidden] { display: none !important; }
#label-panel h3 { font-size: 14px; font-weight: 600; }
#graph-stats { font-size: 12px; color: #777; }
#label-stats { font-size: 11px; color: #999; line-height: 1.5; }
.label-section-title { font-size: 12px; font-weight: 600; color: #555; margin-bottom: 6px; }
.status-buttons { display: flex; gap: 6px; flex-wrap: wrap; }
.status-btn, .cat-btn, .type-btn {
  padding: 6px 13px; border: 2px solid #e0e0e0; border-radius: 6px;
  background: #fff; font-size: 12px; font-weight: 500;
  cursor: pointer; transition: all 0.12s; white-space: nowrap;
}
.status-btn:hover, .cat-btn:hover, .type-btn:hover { border-color: #bbb; }
.status-btn.active[data-status="correct"],
.cat-btn.active[data-value="correct"],
.type-btn.active[data-value="correct"] { background: #dcfce7; border-color: #22c55e; color: #15803d; }
.status-btn.active[data-status="incorrect"],
.cat-btn.active[data-value="incorrect"],
.type-btn.active[data-value="incorrect"] { background: #fef2f2; border-color: #ef4444; color: #b91c1c; }
.status-btn.active[data-status="needs_review"] { background: #fefce8; border-color: #eab308; color: #a16207; }
.clear-btn-small { color: #999 !important; font-size: 11px !important; padding: 4px 10px !important; }

.back-btn {
  background: none; border: none; color: #3b82f6; cursor: pointer;
  font-size: 12px; padding: 0; text-align: left;
}
.back-btn:hover { text-decoration: underline; }
.detail-sentence {
  font-size: 12px; color: #444; line-height: 1.45;
  padding: 8px 10px; background: #f8f8f8; border-radius: 5px;
  border-left: 3px solid #ddd;
}
.detail-current { font-size: 12px; color: #777; margin-top: -4px; }
.alt-select {
  width: 100%; padding: 6px 8px; border: 1px solid #ddd; border-radius: 5px;
  font-size: 12px; background: #fff;
}
/* Custom color-swatch dropdown */
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
.edge-remove-section label {
  display: flex; align-items: center; gap: 8px;
  font-size: 12px; cursor: pointer; padding: 2px 0;
}
.edge-remove-section input { accent-color: #ef4444; }
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

/* ---- Comment bar ---- */
#comment-bar {
  background: #fffde7; border-top: 1px solid #e0d88a;
  padding: 10px 16px; font-size: 12px; color: #555;
  font-style: italic; max-height: 96px; overflow-y: auto;
  line-height: 1.5; flex-shrink: 0;
}
#comment-bar .source-hl {
  border-radius: 3px; padding: 1px 2px; font-style: normal; font-weight: 600;
}
</style>
</head>
<body>

<div id="header">
  <h1>Graph Labeling</h1>
  <button class="nav-btn" id="prev-btn">&larr; Prev</button>
  <select id="comment-select"></select>
  <button class="nav-btn" id="next-btn">Next &rarr;</button>
  <div id="progress"></div>
  <span id="save-indicator" class="saved">Saved</span>
</div>

<div id="main">
  <div id="graph-area">
    <svg id="graph"></svg>
    <div id="no-graph" style="display:none;">No dependency graph for this comment.</div>
    <div id="legend"></div>
    <div id="tooltip"></div>
  </div>

  <div id="label-panel">
    <div id="label-panel-scroll">

      <!-- ===== Comment overview panel ===== -->
      <div id="panel-comment" class="panel-view">
        <div>
          <h3 id="panel-heading">Comment</h3>
          <div id="graph-stats"></div>
          <div id="label-stats"></div>
        </div>
        <div>
          <div class="label-section-title">Overall Status</div>
          <div class="status-buttons">
            <button class="status-btn" data-status="correct">&#10003; Correct</button>
            <button class="status-btn" data-status="incorrect">&#10007; Incorrect</button>
            <button class="status-btn" data-status="needs_review">? Review</button>
            <button class="status-btn clear-btn-small" data-status="">Clear</button>
          </div>
        </div>
        <div>
          <div class="label-section-title">Notes</div>
          <textarea id="comment-notes" class="panel-textarea" placeholder="Overall notes..." rows="3"></textarea>
        </div>
        <div class="panel-hint">Click a node or edge in the graph to label it.</div>
      </div>

      <!-- ===== Node detail panel ===== -->
      <div id="panel-node" class="panel-view" hidden>
        <button class="back-btn" id="node-back-btn">&larr; Back to overview</button>
        <h3 id="node-heading">Node 0</h3>
        <div id="node-sentence" class="detail-sentence"></div>
        <div id="node-current-cat" class="detail-current"></div>
        <div>
          <div class="label-section-title">Category correct?</div>
          <div class="status-buttons">
            <button class="cat-btn" data-value="correct">&#10003; Correct</button>
            <button class="cat-btn" data-value="incorrect">&#10007; Incorrect</button>
            <button class="cat-btn clear-btn-small" data-value="">Clear</button>
          </div>
        </div>
        <div id="node-alt-section" hidden>
          <div class="label-section-title">Correct category</div>
          <div id="node-alt-cat" class="color-select">
            <div class="color-select-trigger"><span class="trigger-placeholder">Select category...</span></div>
            <div class="color-select-dropdown"></div>
          </div>
        </div>
        <div>
          <div class="label-section-title">Notes</div>
          <textarea id="node-notes" class="panel-textarea" placeholder="Notes about this node..." rows="2"></textarea>
        </div>
      </div>

      <!-- ===== Edge detail panel ===== -->
      <div id="panel-edge" class="panel-view" hidden>
        <button class="back-btn" id="edge-back-btn">&larr; Back to overview</button>
        <h3 id="edge-heading">Edge</h3>
        <div id="edge-from" class="detail-sentence"></div>
        <div id="edge-to" class="detail-sentence"></div>
        <div id="edge-current-type" class="detail-current"></div>
        <div>
          <div class="label-section-title">Edge type correct?</div>
          <div class="status-buttons">
            <button class="type-btn" data-value="correct">&#10003; Correct</button>
            <button class="type-btn" data-value="incorrect">&#10007; Incorrect</button>
            <button class="type-btn clear-btn-small" data-value="">Clear</button>
          </div>
        </div>
        <div id="edge-alt-section" hidden>
          <div class="label-section-title">Correct type</div>
          <div id="edge-alt-type" class="color-select">
            <div class="color-select-trigger"><span class="trigger-placeholder">Select edge type...</span></div>
            <div class="color-select-dropdown"></div>
          </div>
        </div>
        <div class="edge-remove-section">
          <label><input type="checkbox" id="edge-remove"> This edge should be removed</label>
        </div>
        <div>
          <div class="label-section-title">Notes</div>
          <textarea id="edge-notes" class="panel-textarea" placeholder="Notes about this edge..." rows="2"></textarea>
        </div>
      </div>

    </div><!-- /label-panel-scroll -->

    <div class="shortcuts-help">
      <div><strong>&larr; &rarr;</strong> Navigate &nbsp;<strong>1 2 3</strong> Status &nbsp;<strong>0</strong> Clear</div>
      <div><strong>Esc</strong> Back to overview &nbsp;<strong>N</strong> Focus notes</div>
    </div>
  </div>
</div>

<div id="comment-bar"></div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"></script>
<script>
(function() {

  /* ===== 1. Constants (synced with 10_visualize_graph.py) ===== */

  var CATEGORIES = [
    "L1: Elemental and encoded properties",
    "L2: Statistical concepts and relations",
    "L3: Trend and pattern analysis",
    "Background knowledge",
    "Personal/episodic retrieval",
    "Explanatory inference",
    "Predictive / counterfactual inference",
    "Evaluative / affective judgment",
    "Information need / curiosity",
    "Meta / paratext",
    "Uncategorizable"
  ];
  var TAB10 = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#aec7e8"
  ];
  var CATEGORY_COLORS = {};
  CATEGORIES.forEach(function(c, i) { CATEGORY_COLORS[c] = TAB10[i % TAB10.length]; });
  CATEGORY_COLORS["unknown"] = "#b3b3b3";

  var EDGE_TYPES = ["Causal","Elaboration","Conditional","Evaluative","Questioning","Contrastive","Narrative/Referential"];
  var EDGE_TYPE_COLORS = {
    "Causal":"#e41a1c","Elaboration":"#377eb8","Conditional":"#ff7f00",
    "Evaluative":"#984ea3","Questioning":"#4daf4a","Contrastive":"#a65628",
    "Narrative/Referential":"#f781bf","unknown":"#999999"
  };

  var NODE_W = 180, PAD_X = 8, PAD_Y = 6, LINE_H = 14, CHAR_W = 6.2;
  var LABEL_BAR_H = 18;
  var CHARS_PER_LINE = Math.floor((NODE_W - PAD_X * 2) / CHAR_W);

  /* ===== 2. State ===== */

  var allComments = [];
  var labels = {};
  var currentIdx = 0;
  var saveTimer = null;

  var selectedNodeId = null;
  var selectedEdgeKey = null;
  var currentPanel = "comment";

  // D3 selections set by renderGraph
  var gNodeGroup = null, gLinkSel = null, gLinkHitSel = null, gEdgeLabelGroup = null;
  var gRectSel = null, gAllFills = null;
  var gCurrentNodes = [], gCurrentLinks = [];

  /* ===== 3. Utilities ===== */

  function wrapText(text) {
    var words = text.split(/\s+/), lines = [], cur = "";
    words.forEach(function(w) {
      var test = cur ? cur + " " + w : w;
      if (test.length > CHARS_PER_LINE && cur) { lines.push(cur); cur = w; }
      else { cur = test; }
    });
    if (cur) lines.push(cur);
    return lines;
  }
  function escapeRegex(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
  function escapeHTML(s) { return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }
  function catColor(cat) { return CATEGORY_COLORS[cat] || CATEGORY_COLORS["unknown"]; }
  function edgeColor(et) { return EDGE_TYPE_COLORS[et] || EDGE_TYPE_COLORS["unknown"]; }
  function commentKey(c) { return String(c.comment_index); }
  function edgeKey(link) {
    var s = (typeof link.source === "object") ? link.source.id : link.source;
    var t = (typeof link.target === "object") ? link.target.id : link.target;
    return s + "->" + t;
  }
  function currentComment() { return allComments[currentIdx]; }

  /* ===== 4. Data processing ===== */

  function buildGraphData(comment) {
    var depGraph = comment.dependency_graph || [];
    var srcMap = comment.source_mappings || {};
    var nodes = [], links = [];
    depGraph.forEach(function(node) {
      var tag = node.comment_tag || "unknown";
      if (Array.isArray(tag)) tag = tag[0] || "unknown";
      if (!tag) tag = "unknown";
      var smList = (typeof srcMap === "object" && !Array.isArray(srcMap)) ? (srcMap[node.sentence] || []) : [];
      var lines = wrapText(node.sentence);
      nodes.push({
        id: node.id, sentence: node.sentence, category: tag, color: catColor(tag),
        source_mappings: smList, lines: lines, nodeW: NODE_W,
        nodeH: Math.max(30 + LABEL_BAR_H, lines.length * LINE_H + PAD_Y * 2 + LABEL_BAR_H)
      });
    });
    depGraph.forEach(function(node) {
      (node.depends_on || []).forEach(function(dep) {
        var depId = typeof dep === "object" ? dep.id : dep;
        var et = typeof dep === "object" ? (dep.edge_type || "unknown") : "unknown";
        links.push({ source: depId, target: node.id, edge_type: et, color: edgeColor(et) });
      });
    });
    return { nodes: nodes, links: links };
  }

  /* ===== 5. Label data management ===== */

  function ensureCommentEntry(ci) {
    var k = String(ci);
    if (!labels.comments) labels.comments = {};
    if (!labels.comments[k]) labels.comments[k] = {};
    return labels.comments[k];
  }

  function getCommentLabel(comment) {
    if (!labels.comments) return null;
    return labels.comments[commentKey(comment)] || null;
  }

  function getNodeLabel(ci, nodeId) {
    var cl = labels.comments && labels.comments[String(ci)];
    if (!cl || !cl.nodes) return null;
    return cl.nodes[String(nodeId)] || null;
  }

  function setNodeLabel(ci, nodeId, data) {
    var entry = ensureCommentEntry(ci);
    if (!entry.nodes) entry.nodes = {};
    if (data) entry.nodes[String(nodeId)] = data;
    else delete entry.nodes[String(nodeId)];
    scheduleSave();
  }

  function getEdgeLabel(ci, ek) {
    var cl = labels.comments && labels.comments[String(ci)];
    if (!cl || !cl.edges) return null;
    return cl.edges[ek] || null;
  }

  function setEdgeLabel(ci, ek, data) {
    var entry = ensureCommentEntry(ci);
    if (!entry.edges) entry.edges = {};
    if (data) entry.edges[ek] = data;
    else delete entry.edges[ek];
    scheduleSave();
  }

  /* ===== 6. Graph rendering ===== */

  function renderGraph(comment) {
    var graphArea = document.getElementById("graph-area");
    var svgEl = document.getElementById("graph");
    var noGraph = document.getElementById("no-graph");

    var svg = d3.select(svgEl);
    svg.selectAll("*").remove();
    svg.on(".zoom", null);
    document.getElementById("legend").innerHTML = "";
    d3.select("#tooltip").style("opacity", 0);

    var gd = buildGraphData(comment);
    gCurrentNodes = gd.nodes; gCurrentLinks = gd.links;

    if (gCurrentNodes.length === 0) {
      noGraph.style.display = "flex"; svgEl.style.display = "none";
      document.getElementById("legend").style.display = "none";
      gNodeGroup = gLinkSel = gLinkHitSel = gEdgeLabelGroup = gRectSel = gAllFills = null;
      return;
    }
    noGraph.style.display = "none"; svgEl.style.display = "block";
    document.getElementById("legend").style.display = "block";

    var width = graphArea.clientWidth, height = graphArea.clientHeight;
    svg.attr("width", width).attr("height", height);

    var g = svg.append("g");
    var zoom = d3.zoom().scaleExtent([0.05, 6]).on("zoom", function(e) { g.attr("transform", e.transform); });
    svg.call(zoom).on("dblclick.zoom", null);

    var defs = svg.append("defs");
    function addMarker(id, color) {
      defs.append("marker").attr("id", id).attr("viewBox","0 -5 10 10").attr("refX",10).attr("refY",0)
        .attr("markerWidth",8).attr("markerHeight",8).attr("orient","auto")
        .append("path").attr("d","M0,-4L10,0L0,4").attr("fill", color);
    }
    addMarker("arrow-dim","#ddd");
    EDGE_TYPES.concat(["unknown"]).forEach(function(et) {
      addMarker("arrow-" + et.replace(/[^a-zA-Z0-9]/g,"_"), edgeColor(et));
    });
    addMarker("arrow-sel","#3b82f6");
    addMarker("arrow-remove","#ef4444");
    function markerId(et) { return "arrow-" + (et || "unknown").replace(/[^a-zA-Z0-9]/g,"_"); }

    // Dagre layout
    var nodeById = new Map(gCurrentNodes.map(function(n) { return [n.id, n]; }));
    var gDag = new dagre.graphlib.Graph();
    gDag.setGraph({ rankdir:"TB", nodesep:25, ranksep:50, marginx:20, marginy:20 });
    gDag.setDefaultEdgeLabel(function() { return {}; });
    gCurrentNodes.forEach(function(n) { gDag.setNode(n.id, { width:n.nodeW, height:n.nodeH }); });
    gCurrentLinks.forEach(function(l) { gDag.setEdge(l.source, l.target); });
    dagre.layout(gDag);
    gCurrentNodes.forEach(function(n) {
      var pos = gDag.node(n.id); n.x = pos.x - n.nodeW/2; n.y = pos.y - n.nodeH/2;
    });
    gCurrentLinks.forEach(function(l) {
      if (typeof l.source !== "object") l.source = nodeById.get(l.source);
      if (typeof l.target !== "object") l.target = nodeById.get(l.target);
    });

    // Visible edges
    gLinkSel = g.append("g").selectAll("line").data(gCurrentLinks).join("line")
      .attr("stroke", function(d) { return d.color; }).attr("stroke-width", 2)
      .attr("stroke-opacity", 0.65)
      .attr("marker-end", function(d) { return "url(#" + markerId(d.edge_type) + ")"; });

    // Edge type labels at midpoints (always show original type)
    gEdgeLabelGroup = g.append("g").selectAll("g").data(gCurrentLinks).join("g")
      .attr("pointer-events", "none");
    gEdgeLabelGroup.each(function(d) {
      var grp = d3.select(this);
      var label = d.edge_type || "unknown";
      var charW = 5.2, padX = 6, h = 15;
      var w = label.length * charW + padX * 2;
      grp.append("rect").attr("class","edge-type-bg")
        .attr("x", -w/2).attr("y", -h/2).attr("width", w).attr("height", h)
        .attr("rx", 3).attr("ry", 3)
        .attr("fill", d.color).attr("fill-opacity", 0.85)
        .attr("stroke", "#fff").attr("stroke-width", 1);
      grp.append("text").attr("class","edge-type-text")
        .attr("text-anchor","middle").attr("dominant-baseline","central")
        .attr("font-size","9px").attr("fill","#fff").attr("font-weight","600")
        .text(label);
    });

    // Invisible edge hit targets
    gLinkHitSel = g.append("g").selectAll("line").data(gCurrentLinks).join("line")
      .attr("stroke","transparent").attr("stroke-width", 14).attr("cursor","pointer");

    var nodeDragged = false;
    // Nodes
    gNodeGroup = g.append("g").selectAll("g").data(gCurrentNodes).join("g").attr("cursor","pointer");

    gCurrentNodes.forEach(function(n) {
      defs.append("clipPath").attr("id","clip-n-"+n.id)
        .append("rect").attr("width",n.nodeW).attr("height",n.nodeH).attr("rx",6).attr("ry",6);
    });

    var clipGroups = gNodeGroup.append("g").attr("clip-path", function(d) { return "url(#clip-n-"+d.id+")"; });
    clipGroups.each(function(d) {
      var grp = d3.select(this);
      grp.append("rect").attr("class","fill-slice")
        .attr("width",d.nodeW).attr("height",d.nodeH).attr("fill",d.color).attr("fill-opacity",0.22);
      // Category label bar at top (always shows original category)
      grp.append("rect").attr("class","cat-bar")
        .attr("width",d.nodeW).attr("height",LABEL_BAR_H)
        .attr("fill",d.color).attr("fill-opacity",0.55);
      var maxChars = Math.floor((d.nodeW - PAD_X * 2) / 5);
      var catText = d.category.length > maxChars ? d.category.substring(0, maxChars - 1) + "\u2026" : d.category;
      grp.append("text").attr("class","cat-bar-text")
        .attr("x", PAD_X).attr("y", LABEL_BAR_H - 5)
        .attr("font-size","9px").attr("fill","#fff").attr("font-weight","600")
        .attr("pointer-events","none").text(catText);
    });
    gAllFills = gNodeGroup.selectAll(".fill-slice");

    gRectSel = gNodeGroup.append("rect").attr("class","node-border")
      .attr("width", function(d) { return d.nodeW; }).attr("height", function(d) { return d.nodeH; })
      .attr("rx",6).attr("ry",6).attr("fill","none")
      .attr("stroke", function(d) { return d.color; }).attr("stroke-width",2).attr("stroke-opacity",0.7);

    gNodeGroup.each(function(d) {
      var t = d3.select(this).append("text").attr("x",d.nodeW/2).attr("text-anchor","middle")
        .attr("font-size","11px").attr("fill","#222").attr("pointer-events","none");
      var startY = LABEL_BAR_H + PAD_Y + LINE_H * 0.8;
      d.lines.forEach(function(line,i) {
        t.append("tspan").attr("x",d.nodeW/2).attr("y",startY+i*LINE_H).text(line);
      });
    });

    // Hover + click interactions
    var adjacency = new Map();
    gCurrentNodes.forEach(function(n) { adjacency.set(n.id, new Set()); });
    gCurrentLinks.forEach(function(l) {
      var s = l.source.id, t = l.target.id;
      adjacency.get(s).add(t); adjacency.get(t).add(s);
    });
    function isConn(a, b) { return a === b || adjacency.get(a).has(b); }

    var commentBar = document.getElementById("comment-bar");
    var origCommentHTML = commentBar.innerHTML;
    var tooltip = d3.select("#tooltip");

    function highlightBar(d) {
      if (!d.source_mappings || d.source_mappings.length === 0) return;
      var raw = commentBar.textContent, color = d.color || "#888", ranges = [];
      d.source_mappings.forEach(function(mapping) {
        var re = new RegExp(escapeRegex(mapping), "gi"), m;
        while ((m = re.exec(raw)) !== null) ranges.push({ start: m.index, end: m.index + m[0].length });
      });
      if (ranges.length === 0) return;
      ranges.sort(function(a,b) { return a.start-b.start; });
      var merged = [{ start:ranges[0].start, end:ranges[0].end }];
      for (var i=1; i<ranges.length; i++) {
        var last = merged[merged.length-1];
        if (ranges[i].start <= last.end) last.end = Math.max(last.end, ranges[i].end);
        else merged.push({ start:ranges[i].start, end:ranges[i].end });
      }
      var html="", pos=0;
      merged.forEach(function(r) {
        html += escapeHTML(raw.substring(pos,r.start));
        html += "<span class='source-hl' style='background:"+color+"33;outline:2px solid "+color+"'>";
        html += escapeHTML(raw.substring(r.start,r.end)) + "</span>";
        pos = r.end;
      });
      html += escapeHTML(raw.substring(pos));
      commentBar.innerHTML = html;
    }

    gNodeGroup
      .on("mouseenter", function(event, d) {
        gAllFills.attr("fill-opacity", function() {
          return isConn(d.id, d3.select(this.parentNode.parentNode).datum().id) ? 0.35 : 0.06;
        });
        gRectSel.attr("stroke-opacity", function(o) { return isConn(d.id,o.id)?1:0.1; });
        gNodeGroup.selectAll("text").attr("fill-opacity", function() {
          return isConn(d.id, d3.select(this.parentNode).datum().id)?1:0.12;
        });
        gLinkSel
          .attr("stroke-opacity", function(l) { return (l.source.id===d.id||l.target.id===d.id)?0.9:0.08; })
          .attr("stroke", function(l) { return (l.source.id===d.id||l.target.id===d.id)?l.color:"#ddd"; })
          .attr("stroke-width", function(l) { return (l.source.id===d.id||l.target.id===d.id)?3:1; })
          .attr("marker-end", function(l) {
            return (l.source.id===d.id||l.target.id===d.id)?"url(#"+markerId(l.edge_type)+")":"url(#arrow-dim)";
          });
        var inEdges = gCurrentLinks.filter(function(l) { return l.target.id === d.id; });
        var edgeInfo = inEdges.length > 0 ? "<div class='cat'>Incoming: " +
          inEdges.map(function(l) { return "<span style='color:"+l.color+";font-weight:600'>"+l.edge_type+"</span>"; }).join(", ") + "</div>" : "";
        var srcInfo = (d.source_mappings && d.source_mappings.length > 0)
          ? "<div class='src'>&#128206; " + d.source_mappings.map(escapeHTML).join(" &bull; ") + "</div>" : "";
        tooltip.style("opacity",1).html("<strong>"+d.id+":</strong> "+escapeHTML(d.sentence)+
          "<div class='cat'>"+escapeHTML(d.category)+"</div>"+edgeInfo+srcInfo);
        highlightBar(d);
      })
      .on("mousemove", function(event) {
        var rect = graphArea.getBoundingClientRect();
        tooltip.style("left",(event.clientX-rect.left+16)+"px").style("top",(event.clientY-rect.top-8)+"px");
      })
      .on("mouseleave", function() {
        gAllFills.attr("fill-opacity",0.22);
        gRectSel.attr("stroke-opacity",0.7);
        gNodeGroup.selectAll("text").attr("fill-opacity",1);
        gLinkSel.attr("stroke",function(d){return d.color;}).attr("stroke-opacity",0.65)
          .attr("stroke-width",2).attr("stroke-dasharray",null)
          .attr("marker-end",function(d){return "url(#"+markerId(d.edge_type)+")";});
        tooltip.style("opacity",0);
        commentBar.innerHTML = origCommentHTML;
        updateVisualState();
      });

    // Node click (distinguish from drag)
    gNodeGroup.call(d3.drag()
      .on("start", function() { nodeDragged = false; })
      .on("drag", function(event, d) { nodeDragged = true; d.x = event.x; d.y = event.y; updatePositions(); })
    );
    gNodeGroup.on("click", function(event, d) {
      if (nodeDragged) return;
      event.stopPropagation();
      selectNode(d.id);
    });

    // Edge click
    gLinkHitSel.on("click", function(event, d) {
      event.stopPropagation();
      selectEdge(edgeKey(d));
    });

    // Position helpers
    function rectEdge(cx,cy,w,h,tx,ty) {
      var dx=tx-cx, dy=ty-cy;
      if(dx===0&&dy===0)return[cx,cy];
      var scale=(Math.abs(dx)/(w/2)>Math.abs(dy)/(h/2))?(w/2)/Math.abs(dx):(h/2)/Math.abs(dy);
      return[cx+dx*scale, cy+dy*scale];
    }

    function updatePositions() {
      gNodeGroup.attr("transform", function(d){return "translate("+d.x+","+d.y+")";});
      function edgeEndpoints(d) {
        var sx=d.source.x+d.source.nodeW/2, sy=d.source.y+d.source.nodeH/2;
        var tx=d.target.x+d.target.nodeW/2, ty=d.target.y+d.target.nodeH/2;
        var p1=rectEdge(sx,sy,d.source.nodeW,d.source.nodeH,tx,ty);
        var p2=rectEdge(tx,ty,d.target.nodeW,d.target.nodeH,sx,sy);
        return {x1:p1[0],y1:p1[1],x2:p2[0],y2:p2[1]};
      }
      function setLine(sel) {
        sel.each(function(d) {
          var ep=edgeEndpoints(d);
          d3.select(this).attr("x1",ep.x1).attr("y1",ep.y1).attr("x2",ep.x2).attr("y2",ep.y2);
        });
      }
      setLine(gLinkSel);
      setLine(gLinkHitSel);
      if (gEdgeLabelGroup) {
        gEdgeLabelGroup.each(function(d) {
          var ep=edgeEndpoints(d);
          d3.select(this).attr("transform","translate("+((ep.x1+ep.x2)/2)+","+((ep.y1+ep.y2)/2)+")");
        });
      }
    }
    updatePositions();

    // Fit to view
    (function() {
      var pad=60, x0=Infinity, y0=Infinity, x1=-Infinity, y1=-Infinity;
      gCurrentNodes.forEach(function(n) {
        if(n.x<x0)x0=n.x; if(n.y<y0)y0=n.y;
        if(n.x+n.nodeW>x1)x1=n.x+n.nodeW; if(n.y+n.nodeH>y1)y1=n.y+n.nodeH;
      });
      var bw=x1-x0+pad*2, bh=y1-y0+pad*2;
      var scale=Math.min(width/bw, height/bh, 1.5);
      var cx=(x0+x1)/2, cy=(y0+y1)/2;
      svg.transition().duration(400).call(zoom.transform,
        d3.zoomIdentity.translate(width/2-scale*cx, height/2-scale*cy).scale(scale));
    })();

    buildLegend(gCurrentNodes, gCurrentLinks);
    updateVisualState();
  }

  /* ===== 7. Legend ===== */

  function buildLegend(nodes, links) {
    var leg = document.getElementById("legend");
    leg.innerHTML = ""; leg.className = "";
    var title = document.createElement("div"); title.className = "leg-title";
    title.innerHTML = "<span>Legend</span><span class='chevron'>&#9660;</span>";
    title.onclick = function(){ leg.classList.toggle("collapsed"); };
    leg.appendChild(title);
    var body = document.createElement("div"); body.className = "legend-body";

    var usedCats = {};
    nodes.forEach(function(n){ usedCats[n.category]=true; });
    var catOrder = CATEGORIES.concat(["unknown"]).filter(function(c){ return usedCats[c]; });
    Object.keys(usedCats).forEach(function(c){ if(catOrder.indexOf(c)===-1) catOrder.push(c); });
    var lbl = document.createElement("div"); lbl.className="section-label"; lbl.textContent="Node categories"; body.appendChild(lbl);
    catOrder.forEach(function(cat) {
      var row=document.createElement("div"); row.className="item";
      row.innerHTML="<div class='swatch' style='background:"+catColor(cat)+"'></div><span>"+escapeHTML(cat)+"</span>";
      body.appendChild(row);
    });

    var usedEt = {};
    links.forEach(function(l){ usedEt[l.edge_type]=true; });
    var etOrder = EDGE_TYPES.concat(["unknown"]).filter(function(e){ return usedEt[e]; });
    if (etOrder.length > 0) {
      var hr=document.createElement("hr"); hr.className="section-divider"; body.appendChild(hr);
      var lbl2=document.createElement("div"); lbl2.className="section-label"; lbl2.textContent="Edge types"; body.appendChild(lbl2);
      etOrder.forEach(function(et) {
        var row=document.createElement("div"); row.className="item";
        row.innerHTML="<div class='edge-swatch' style='background:"+edgeColor(et)+"'></div><span>"+escapeHTML(et)+"</span>";
        body.appendChild(row);
      });
    }
    leg.appendChild(body);
  }

  /* ===== 8. Visual indicators (badges + edge styles + selection) ===== */

  function updateVisualState() {
    if (!gNodeGroup) return;
    var ci = currentComment().comment_index;

    // --- Node indicators ---
    gNodeGroup.selectAll(".label-badge").remove();

    // Update fill and border colors based on labels
    gNodeGroup.each(function(d) {
      var nlbl = getNodeLabel(ci, d.id);
      var grp = d3.select(this);
      var fillColor = d.color;
      var borderColor = d.color;

      if (nlbl && nlbl.category_correct === false && nlbl.correct_category) {
        var corrColor = catColor(nlbl.correct_category);
        fillColor = corrColor;
        borderColor = corrColor;
      }

      grp.select(".fill-slice").attr("fill", fillColor);

      if (nlbl && nlbl.category_correct !== undefined) {
        var badgeColor = nlbl.category_correct ? "#22c55e" : "#ef4444";
        var sym = nlbl.category_correct ? "\u2713" : "\u2717";
        grp.append("circle").attr("class","label-badge")
          .attr("cx",d.nodeW-10).attr("cy",10).attr("r",8)
          .attr("fill",badgeColor).attr("fill-opacity",0.9)
          .attr("stroke","#fff").attr("stroke-width",1.5).attr("pointer-events","none");
        grp.append("text").attr("class","label-badge")
          .attr("x",d.nodeW-10).attr("y",13.5).attr("text-anchor","middle")
          .attr("font-size","10px").attr("fill","#fff").attr("font-weight","bold")
          .attr("pointer-events","none").text(sym);
      }

      // Border: selection takes priority, then label, then default
      if (d.id === selectedNodeId) {
        grp.select(".node-border").attr("stroke","#3b82f6").attr("stroke-width",3);
      } else {
        grp.select(".node-border").attr("stroke", borderColor).attr("stroke-width",2);
      }
    });

    // --- Edge indicators ---
    if (gLinkSel) {
      gLinkSel.each(function(d) {
        var ek = edgeKey(d);
        var elbl = getEdgeLabel(ci, ek);
        var sel = d3.select(this);

        if (selectedEdgeKey === ek) {
          sel.attr("stroke","#3b82f6").attr("stroke-width",4)
            .attr("stroke-dasharray",null).attr("stroke-opacity",1)
            .attr("marker-end","url(#arrow-sel)");
        } else if (elbl && elbl.should_remove) {
          sel.attr("stroke","#ef4444").attr("stroke-dasharray","6,4").attr("stroke-opacity",0.5)
            .attr("stroke-width",2).attr("marker-end","url(#arrow-remove)");
        } else if (elbl && elbl.type_correct === false && elbl.correct_type) {
          var corrEdgeColor = edgeColor(elbl.correct_type);
          sel.attr("stroke", corrEdgeColor).attr("stroke-dasharray","4,3").attr("stroke-opacity",0.8)
            .attr("stroke-width",2.5).attr("marker-end","url(#"+markerId(elbl.correct_type)+")");
        } else if (elbl && elbl.type_correct === true) {
          sel.attr("stroke",d.color).attr("stroke-dasharray",null).attr("stroke-opacity",0.85)
            .attr("stroke-width",2.5).attr("marker-end","url(#"+markerId(d.edge_type)+")");
        } else {
          sel.attr("stroke",d.color).attr("stroke-dasharray",null).attr("stroke-opacity",0.65)
            .attr("stroke-width",2).attr("marker-end","url(#"+markerId(d.edge_type)+")");
        }
      });
    }
  }

  /* ===== 9. Panel management ===== */

  function showPanel(mode) {
    currentPanel = mode;
    document.getElementById("panel-comment").hidden = (mode !== "comment");
    document.getElementById("panel-node").hidden = (mode !== "node");
    document.getElementById("panel-edge").hidden = (mode !== "edge");
  }

  function deselectAll() {
    selectedNodeId = null;
    selectedEdgeKey = null;
    showPanel("comment");
    updateCommentPanel();
    updateVisualState();
  }

  function selectNode(nodeId) {
    selectedNodeId = nodeId;
    selectedEdgeKey = null;
    showPanel("node");
    updateNodePanel();
    updateVisualState();
  }

  function selectEdge(ek) {
    selectedEdgeKey = ek;
    selectedNodeId = null;
    showPanel("edge");
    updateEdgePanel();
    updateVisualState();
  }

  /* ===== 10. Panel content updates ===== */

  function updateCommentPanel() {
    var comment = currentComment();
    var gd = buildGraphData(comment);
    document.getElementById("panel-heading").textContent = "Comment " + comment.comment_index;
    document.getElementById("graph-stats").textContent = gd.nodes.length + " nodes, " + gd.links.length + " edges";

    var ci = comment.comment_index;
    var nlCount = 0, elCount = 0;
    gd.nodes.forEach(function(n) { if (getNodeLabel(ci, n.id)) nlCount++; });
    gd.links.forEach(function(l) { if (getEdgeLabel(ci, edgeKey(l))) elCount++; });
    document.getElementById("label-stats").textContent =
      "Nodes labeled: " + nlCount + "/" + gd.nodes.length + " \u00B7 Edges labeled: " + elCount + "/" + gd.links.length;

    var cl = getCommentLabel(comment);
    var status = cl ? (cl.status || "") : "";
    document.querySelectorAll(".status-btn").forEach(function(btn) {
      btn.classList.toggle("active", btn.getAttribute("data-status") === status);
    });
    document.getElementById("comment-notes").value = cl ? (cl.notes || "") : "";
  }

  function updateNodePanel() {
    var node = gCurrentNodes.find(function(n) { return n.id === selectedNodeId; });
    if (!node) return;
    document.getElementById("node-heading").textContent = "Node " + node.id;
    document.getElementById("node-sentence").textContent = node.sentence;
    document.getElementById("node-current-cat").innerHTML = "Current: <strong>" + escapeHTML(node.category) + "</strong>";

    var nlbl = getNodeLabel(currentComment().comment_index, node.id);
    var catCorrect = nlbl ? nlbl.category_correct : undefined;
    document.querySelectorAll(".cat-btn").forEach(function(btn) {
      var val = btn.getAttribute("data-value");
      if (val === "correct") btn.classList.toggle("active", catCorrect === true);
      else if (val === "incorrect") btn.classList.toggle("active", catCorrect === false);
      else btn.classList.remove("active");
    });
    var showAlt = catCorrect === false;
    document.getElementById("node-alt-section").hidden = !showAlt;
    if (nodeCatSelect) nodeCatSelect.setValue((nlbl && nlbl.correct_category) || "");
    document.getElementById("node-notes").value = (nlbl && nlbl.notes) || "";
  }

  function updateEdgePanel() {
    if (!selectedEdgeKey) return;
    var parts = selectedEdgeKey.split("->");
    var srcId = parseInt(parts[0],10), tgtId = parseInt(parts[1],10);
    var link = gCurrentLinks.find(function(l) { return l.source.id === srcId && l.target.id === tgtId; });
    if (!link) return;
    var srcNode = gCurrentNodes.find(function(n){ return n.id === srcId; });
    var tgtNode = gCurrentNodes.find(function(n){ return n.id === tgtId; });

    document.getElementById("edge-heading").textContent = "Edge: Node " + srcId + " \u2192 Node " + tgtId;
    document.getElementById("edge-from").textContent = "From: " + (srcNode ? srcNode.sentence : "?");
    document.getElementById("edge-to").textContent = "To: " + (tgtNode ? tgtNode.sentence : "?");
    document.getElementById("edge-current-type").innerHTML = "Current type: <strong>" + escapeHTML(link.edge_type) + "</strong>";

    var elbl = getEdgeLabel(currentComment().comment_index, selectedEdgeKey);
    var typeCorrect = elbl ? elbl.type_correct : undefined;
    document.querySelectorAll(".type-btn").forEach(function(btn) {
      var val = btn.getAttribute("data-value");
      if (val === "correct") btn.classList.toggle("active", typeCorrect === true);
      else if (val === "incorrect") btn.classList.toggle("active", typeCorrect === false);
      else btn.classList.remove("active");
    });
    document.getElementById("edge-alt-section").hidden = (typeCorrect !== false);
    if (edgeTypeSelect) edgeTypeSelect.setValue((elbl && elbl.correct_type) || "");
    document.getElementById("edge-remove").checked = !!(elbl && elbl.should_remove);
    document.getElementById("edge-notes").value = (elbl && elbl.notes) || "";
  }

  /* ===== 11. Label change handlers ===== */

  // Comment level
  function onCommentLabelChange() {
    var status = "";
    document.querySelectorAll(".status-btn.active").forEach(function(btn) {
      var s = btn.getAttribute("data-status"); if (s) status = s;
    });
    var notes = document.getElementById("comment-notes").value.trim();
    var comment = currentComment();
    var entry = ensureCommentEntry(comment.comment_index);
    if (!status && !notes && !entry.nodes && !entry.edges) {
      delete labels.comments[commentKey(comment)];
    } else {
      entry.status = status;
      entry.notes = notes;
      entry.timestamp = new Date().toISOString();
    }
    scheduleSave();
    updateProgress();
    updateDropdownStatus();
  }

  // Node level
  function collectNodeLabel() {
    var catCorrect = undefined;
    document.querySelectorAll(".cat-btn.active").forEach(function(btn) {
      var v = btn.getAttribute("data-value");
      if (v === "correct") catCorrect = true;
      else if (v === "incorrect") catCorrect = false;
    });
    var correctCat = (nodeCatSelect ? nodeCatSelect.getValue() : "") || null;
    var notes = document.getElementById("node-notes").value.trim();
    if (catCorrect === undefined && !notes) return null;
    var data = { category_correct: catCorrect, notes: notes };
    if (catCorrect === false && correctCat) data.correct_category = correctCat;
    return data;
  }

  function onNodeLabelChange() {
    var data = collectNodeLabel();
    setNodeLabel(currentComment().comment_index, selectedNodeId, data);
    updateNodePanel();
    updateCommentPanel();
    updateVisualState();
  }

  // Edge level
  function collectEdgeLabel() {
    var typeCorrect = undefined;
    document.querySelectorAll(".type-btn.active").forEach(function(btn) {
      var v = btn.getAttribute("data-value");
      if (v === "correct") typeCorrect = true;
      else if (v === "incorrect") typeCorrect = false;
    });
    var correctType = (edgeTypeSelect ? edgeTypeSelect.getValue() : "") || null;
    var shouldRemove = document.getElementById("edge-remove").checked;
    var notes = document.getElementById("edge-notes").value.trim();
    if (typeCorrect === undefined && !shouldRemove && !notes) return null;
    var data = { type_correct: typeCorrect, should_remove: shouldRemove, notes: notes };
    if (typeCorrect === false && correctType) data.correct_type = correctType;
    return data;
  }

  function onEdgeLabelChange() {
    var data = collectEdgeLabel();
    setEdgeLabel(currentComment().comment_index, selectedEdgeKey, data);
    updateEdgePanel();
    updateCommentPanel();
    updateVisualState();
  }

  /* ===== 12. Event listeners ===== */

  // Comment status buttons
  document.querySelectorAll(".status-btn").forEach(function(btn) {
    btn.addEventListener("click", function() {
      var s = btn.getAttribute("data-status");
      document.querySelectorAll(".status-btn").forEach(function(b) { b.classList.remove("active"); });
      if (s) btn.classList.add("active");
      onCommentLabelChange();
    });
  });
  var commentNotesTimer = null;
  document.getElementById("comment-notes").addEventListener("input", function() {
    clearTimeout(commentNotesTimer);
    commentNotesTimer = setTimeout(onCommentLabelChange, 400);
  });

  // Node cat buttons
  document.querySelectorAll(".cat-btn").forEach(function(btn) {
    btn.addEventListener("click", function() {
      var v = btn.getAttribute("data-value");
      document.querySelectorAll(".cat-btn").forEach(function(b) { b.classList.remove("active"); });
      if (v) btn.classList.add("active");
      document.getElementById("node-alt-section").hidden = (v !== "incorrect");
      onNodeLabelChange();
    });
  });
  var nodeNotesTimer = null;
  document.getElementById("node-notes").addEventListener("input", function() {
    clearTimeout(nodeNotesTimer);
    nodeNotesTimer = setTimeout(onNodeLabelChange, 400);
  });

  // Edge type buttons
  document.querySelectorAll(".type-btn").forEach(function(btn) {
    btn.addEventListener("click", function() {
      var v = btn.getAttribute("data-value");
      document.querySelectorAll(".type-btn").forEach(function(b) { b.classList.remove("active"); });
      if (v) btn.classList.add("active");
      document.getElementById("edge-alt-section").hidden = (v !== "incorrect");
      onEdgeLabelChange();
    });
  });
  document.getElementById("edge-remove").addEventListener("change", onEdgeLabelChange);
  var edgeNotesTimer = null;
  document.getElementById("edge-notes").addEventListener("input", function() {
    clearTimeout(edgeNotesTimer);
    edgeNotesTimer = setTimeout(onEdgeLabelChange, 400);
  });

  // Back buttons
  document.getElementById("node-back-btn").addEventListener("click", deselectAll);
  document.getElementById("edge-back-btn").addEventListener("click", deselectAll);

  /* ===== 13. Save ===== */

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

  /* ===== 14. Navigation ===== */

  function navigate(idx) {
    if (idx < 0 || idx >= allComments.length) return;
    currentIdx = idx;
    selectedNodeId = null;
    selectedEdgeKey = null;
    showPanel("comment");

    var comment = currentComment();
    document.getElementById("comment-bar").textContent = comment.raw_comment || "";
    renderGraph(comment);
    updateCommentPanel();

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
    var total = allComments.length;
    var correct = 0, incorrect = 0, review = 0;
    Object.values(labels.comments).forEach(function(cl) {
      if (cl.status === "correct") correct++;
      else if (cl.status === "incorrect") incorrect++;
      else if (cl.status === "needs_review") review++;
    });
    var labeled = correct + incorrect + review;
    var html = "<span class='stat'>" + labeled + "</span>/" + total + " labeled";
    if (correct) html += " &nbsp;<span class='prog-correct'>\u2713 " + correct + "</span>";
    if (incorrect) html += " &nbsp;<span class='prog-incorrect'>\u2717 " + incorrect + "</span>";
    if (review) html += " &nbsp;<span class='prog-review'>? " + review + "</span>";
    document.getElementById("progress").innerHTML = html;
  }

  function updateDropdownStatus() {
    var sel = document.getElementById("comment-select");
    for (var i = 0; i < sel.options.length; i++) {
      var c = allComments[i], cl = getCommentLabel(c);
      var prefix = "\u00B7";
      if (cl) {
        if (cl.status === "correct") prefix = "\u2713";
        else if (cl.status === "incorrect") prefix = "\u2717";
        else if (cl.status === "needs_review") prefix = "?";
      }
      sel.options[i].textContent = prefix + " Comment " + c.comment_index;
    }
  }

  /* ===== 15. Keyboard shortcuts ===== */

  document.addEventListener("keydown", function(e) {
    if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT" || e.target.tagName === "SELECT") {
      if (e.key === "Escape") { e.target.blur(); e.preventDefault(); }
      return;
    }
    if (e.key === "Escape") { deselectAll(); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { navigate(currentIdx-1); e.preventDefault(); }
    else if (e.key === "ArrowRight") { navigate(currentIdx+1); e.preventDefault(); }
    else if (e.key === "1" && currentPanel === "comment") { clickStatusBtn("correct"); e.preventDefault(); }
    else if (e.key === "2" && currentPanel === "comment") { clickStatusBtn("incorrect"); e.preventDefault(); }
    else if (e.key === "3" && currentPanel === "comment") { clickStatusBtn("needs_review"); e.preventDefault(); }
    else if (e.key === "0" && currentPanel === "comment") { clickStatusBtn(""); e.preventDefault(); }
    else if ((e.key === "n" || e.key === "N")) {
      var notesId = currentPanel === "node" ? "node-notes" : currentPanel === "edge" ? "edge-notes" : "comment-notes";
      document.getElementById(notesId).focus(); e.preventDefault();
    }
  });

  function clickStatusBtn(status) {
    document.querySelectorAll(".status-btn").forEach(function(btn) {
      if (btn.getAttribute("data-status") === status) btn.click();
    });
  }

  /* ===== 16. ColorSelect component ===== */

  var nodeCatSelect = null;
  var edgeTypeSelect = null;

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
    if (nodeCatSelect) nodeCatSelect.close();
    if (edgeTypeSelect) edgeTypeSelect.close();
  }
  document.addEventListener("click", closeAllColorSelects);

  /* ===== 17. Init ===== */

  function populateDropdowns() {
    var catOptions = CATEGORIES.map(function(c) { return { value: c, label: c, color: catColor(c) }; });
    catOptions.push({ value: "unknown", label: "unknown", color: catColor("unknown") });
    nodeCatSelect = new ColorSelect("node-alt-cat", "Select category...", catOptions, onNodeLabelChange);

    var etOptions = EDGE_TYPES.map(function(et) { return { value: et, label: et, color: edgeColor(et) }; });
    etOptions.push({ value: "unknown", label: "unknown", color: edgeColor("unknown") });
    edgeTypeSelect = new ColorSelect("edge-alt-type", "Select edge type...", etOptions, onEdgeLabelChange);
  }

  async function init() {
    try {
      var dataResp = await fetch("/data");
      allComments = await dataResp.json();
      var labelsResp = await fetch("/labels");
      labels = await labelsResp.json();
    } catch(err) {
      document.getElementById("comment-bar").textContent = "Error loading data: " + err.message;
      return;
    }
    if (!labels.comments) labels.comments = {};

    populateDropdowns();

    var sel = document.getElementById("comment-select");
    allComments.forEach(function(c, i) {
      var opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = "\u00B7 Comment " + c.comment_index;
      sel.appendChild(opt);
    });

    updateDropdownStatus();
    updateProgress();
    if (allComments.length > 0) navigate(0);
  }

  init();
})();
</script>
</body>
</html>"""


class LabelHandler(BaseHTTPRequestHandler):
    combined_data: list = []
    labels: dict = {}
    labels_path: Path = Path()
    html_content: str = ""

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._respond(self.html_content, "text/html; charset=utf-8")
        elif path == "/data":
            self._respond(json.dumps(self.combined_data), "application/json")
        elif path == "/labels":
            self._respond(json.dumps(LabelHandler.labels), "application/json")
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/labels":
            self.send_error(404)
            return
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
        description="Interactive labeling tool for reviewing ACE dependency graphs."
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to combined data JSON (e.g. combined_data/35.json).",
    )
    parser.add_argument(
        "--labels-dir", type=Path, default=Path("graph_labels"),
        help="Directory for label output files (default: graph_labels/).",
    )
    parser.add_argument(
        "--port", type=int, default=8050,
        help="Server port (default: 8050).",
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="Don't auto-open the browser.",
    )
    args = parser.parse_args()

    combined_path = args.input.resolve()
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined data not found: {combined_path}")

    with open(combined_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print(f"No comments in {combined_path}")
        return

    article_id = data[0].get("article_id", combined_path.stem)
    labels_path = (args.labels_dir / f"{article_id}.json").resolve()

    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        print(f"Loaded existing labels from {labels_path}", flush=True)
    else:
        labels = {"article_id": article_id, "comments": {}}

    LabelHandler.combined_data = data
    LabelHandler.labels = labels
    LabelHandler.labels_path = labels_path
    LabelHandler.html_content = HTML_TEMPLATE

    server = HTTPServer(("127.0.0.1", args.port), LabelHandler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Labeling server running at {url}", flush=True)
    print(f"Article: {article_id} ({len(data)} comments)", flush=True)
    print(f"Labels: {labels_path}", flush=True)
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
