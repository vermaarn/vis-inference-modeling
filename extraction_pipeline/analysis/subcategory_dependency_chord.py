"""
Build a long-form edge table from combined_data/*.json, save it under dataframes/,
and visualize subcategory → subcategory flows as chord diagrams (matplotlib + D3 HTML).

Category colors match extraction_pipeline/analysis/1_across_plot.ipynb (FINEGRAIN_COLORS).

Visualizations read the saved Parquet so they use the canonical dataframe on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath
import matplotlib.patches as mpatches

# Paths relative to this file
HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
COMBINED_DATA_DIR = PIPELINE / "combined_data"
DATAFRAMES_DIR = PIPELINE / "dataframes"
FIGURES_DIR = HERE / "figures"

# Same cleanup as 1_across_plot.ipynb
_TAG_CLEANUP = {
    "L3: Trend and pattern analysis": "Visual Observation: Cross-point Pattern Recognition",
    "L1: Elemental and encoded properties": "Visual Observation: Chart Structure & Text",
    "L2: Statistical concepts and relations": "Visual Observation: Data Point Extraction",
    "VO1: Chart Structure, Layout & Text": "Visual Observation: Chart Structure & Text",
    "VO2: Data Point Reading": "Visual Observation: Data Point Extraction",
    "VO3: Comparisons, Trends & Patterns": "Visual Observation: Cross-point Pattern Recognition",
    "Background knowledge": "Prior Knowledge: Background",
    "Personal/episodic retrieval": "Prior Knowledge: Personal / Episodic",
    "Evaluative / affective judgment": "Evaluative: Reactive",
    "Explanatory inference": "Inference: Explanatory",
    "Predictive / counterfactual inference": "Inference: Predictive / Hypothetical",
    "Information need / curiosity": "Curiosity",
    "Meta /Paratext": "Meta / Paratext",
    "Meta / paratext": "Meta / Paratext",
}

# 1_across_plot.ipynb — FINEGRAIN_CATS / FINEGRAIN_COLORS (subcategory palette)
FINEGRAIN_CATS = [
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
]

FINEGRAIN_COLORS = [
    "#a9c5f2",
    "#4c78a8",
    "#25476a",
    "#f6b0b0",
    "#b51230",
    "#8fe596",
    "#34803a",
    "#ffe993",
    "#e1b800",
    "#cbb293",
]

EXCLUDED_SUBCATS = {"Meta / Paratext", "Uncategorizable"}

SUBCAT_TO_COLOR = dict(zip(FINEGRAIN_CATS, FINEGRAIN_COLORS, strict=True))

PARQUET_NAME = "subcategory_dependency_edges.parquet"
HTML_NAME = "subcategory_dependency_chord.html"


def _short_label(full: str) -> str:
    return (
        full.replace("Visual Observation: ", "VO: ")
        .replace("Prior Knowledge: ", "PK: ")
        .replace("Inference: ", "Inf: ")
        .replace("Evaluative: ", "Ev: ")
    )


def flow_matrix_from_edges(df: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]] | None:
    """
    Build directed count matrix M[i,j] = edges from parent subcategory i to child j.
    Returns (matrix, labels, colors) for active FINEGRAIN_CATS only, or None if empty.
    """
    labels = list(FINEGRAIN_CATS)
    idx = {c: i for i, c in enumerate(labels)}
    n = len(labels)
    M = np.zeros((n, n), dtype=float)

    known_mask = df["parent_subcategory"].isin(idx) & df["child_subcategory"].isin(idx)
    df_k = df.loc[known_mask]
    for row in df_k.itertuples(index=False):
        M[idx[row.parent_subcategory], idx[row.child_subcategory]] += 1

    active = [i for i in range(n) if M[i, :].sum() + M[:, i].sum() > 0]
    if not active:
        return None

    M_sub = M[np.ix_(active, active)]
    lab_sub = [labels[i] for i in active]
    col_sub = [FINEGRAIN_COLORS[i] for i in active]
    return M_sub, lab_sub, col_sub


def flow_details_matrix_from_edges(df: pd.DataFrame, labels: list[str]) -> list[list[dict | None]]:
    """
    For each (parent_subcategory, child_subcategory) pair in `labels` order, aggregate
    the same fields as the edges dataframe (counts / nunique / edge_category breakdown).
    Shape matches the chord matrix.
    """
    idx = {c: i for i, c in enumerate(labels)}
    n = len(labels)
    grid: list[list[dict | None]] = [[None] * n for _ in range(n)]

    cols = [
        "article_id",
        "comment_index",
        "parent_id",
        "child_id",
        "parent_subcategory",
        "child_subcategory",
        "edge_category",
    ]
    known = df["parent_subcategory"].isin(idx) & df["child_subcategory"].isin(idx)
    sub = df.loc[known, cols].copy()

    for (p, c), grp in sub.groupby(["parent_subcategory", "child_subcategory"], sort=False):
        i, j = idx[p], idx[c]
        ec = grp["edge_category"].fillna("").astype(str)
        edge_counts = ec.value_counts().to_dict()
        grid[i][j] = {
            "parent_subcategory": p,
            "child_subcategory": c,
            "n_edges": int(len(grp)),
            "n_unique_article_id": int(grp["article_id"].nunique()),
            "n_unique_comment_graphs": int(
                grp.groupby(["article_id", "comment_index"], sort=False).ngroups
            ),
            "n_unique_parent_id": int(grp["parent_id"].nunique()),
            "n_unique_child_id": int(grp["child_id"].nunique()),
            "edge_category_counts": {
                str(k): int(v) for k, v in sorted(edge_counts.items(), key=lambda x: -x[1])
            },
        }
    return grid


_D3_CHORD_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Subcategory dependency chord</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <style>
    body {
      margin: 0;
      font-family: system-ui, -apple-system, sans-serif;
      background: #fafafa;
    }
    #wrap {
      max-width: 1100px;
      margin: 0 auto;
      padding: 1rem;
    }
    h1 {
      font-size: 1.1rem;
      font-weight: 600;
      margin: 0 0 0.5rem 0;
    }
    p {
      font-size: 0.85rem;
      color: #444;
      margin: 0 0 1rem 0;
    }
    svg { display: block; margin: 0 auto; }
    .group path { cursor: pointer; }
    .ribbon { mix-blend-mode: multiply; }
    .ribbon:hover { stroke: #333; stroke-width: 0.5px; }
    #tooltip {
      position: fixed;
      display: none;
      z-index: 1000;
      pointer-events: none;
      max-width: 420px;
      padding: 10px 12px;
      font-size: 11px;
      line-height: 1.45;
      background: rgba(255, 255, 255, 0.98);
      border: 1px solid #c8c8c8;
      border-radius: 8px;
      box-shadow: 0 6px 24px rgba(0, 0, 0, 0.12);
    }
    #tooltip .tt-title {
      font-weight: 600;
      margin-bottom: 6px;
      font-size: 12px;
      color: #111;
    }
    #tooltip .tt-k { color: #555; }
    #tooltip .tt-v { font-weight: 500; color: #111; }
    #tooltip .tt-row { margin: 2px 0; }
    #tooltip .tt-sub { margin-top: 8px; font-weight: 600; color: #333; }
    #tooltip .tt-ec { margin: 2px 0 0 8px; font-family: ui-monospace, monospace; font-size: 10px; }
  </style>
</head>
<body>
  <div id="wrap">
    <h1>ACE dependency flows between sentence subcategories</h1>
    <p>Bands are directed parent → child; width is edge count. Ribbon color follows the parent subcategory (same palette as across-plot analysis). Hover a band or rim segment for dataframe-style aggregates.</p>
    <div id="chart"></div>
  </div>
  <div id="tooltip" aria-live="polite"></div>
  <script>
    const DATA = __DATA_JSON__;

    const tooltip = d3.select("#tooltip");

    function moveTooltip(event) {
      const pad = 14;
      let x = event.clientX + pad;
      let y = event.clientY + pad;
      const n = tooltip.node();
      if (n) {
        const w = n.offsetWidth;
        const h = n.offsetHeight;
        if (x + w > window.innerWidth - 8) x = event.clientX - w - pad;
        if (y + h > window.innerHeight - 8) y = event.clientY - h - pad;
      }
      tooltip.style("left", x + "px").style("top", y + "px");
    }

    function formatGroupTooltip(i) {
      const n = DATA.matrix.length;
      let out = 0, inn = 0;
      for (let j = 0; j < n; j++) {
        out += DATA.matrix[i][j];
        inn += DATA.matrix[j][i];
      }
      return `<div class="tt-title">${DATA.labels[i]}</div>
        <div class="tt-row"><span class="tt-k">edges as parent (outgoing)</span> <span class="tt-v">${out}</span></div>
        <div class="tt-row"><span class="tt-k">edges as child (incoming)</span> <span class="tt-v">${inn}</span></div>`;
    }

    function formatFlowTooltip(i, j) {
      const cell = DATA.cellDetails[i][j];
      const count = DATA.matrix[i][j];
      if (!cell) {
        return `<div class="tt-title">${DATA.labels[i]} → ${DATA.labels[j]}</div>
          <div class="tt-row"><span class="tt-k">n_edges</span> <span class="tt-v">${count}</span></div>`;
      }
      const ecLines = Object.entries(cell.edge_category_counts)
        .map(([k, v]) => {
          const label = k === "" ? "(empty)" : k;
          return `<div class="tt-ec">${label}: ${v}</div>`;
        })
        .join("");
      return `<div class="tt-title">${cell.parent_subcategory} → ${cell.child_subcategory}</div>
        <div class="tt-row"><span class="tt-k">n_edges</span> <span class="tt-v">${cell.n_edges}</span></div>
        <div class="tt-row"><span class="tt-k">n_unique_article_id</span> <span class="tt-v">${cell.n_unique_article_id}</span></div>
        <div class="tt-row"><span class="tt-k">n_unique_comment_graphs</span> <span class="tt-v">${cell.n_unique_comment_graphs}</span></div>
        <div class="tt-row"><span class="tt-k">n_unique_parent_id</span> <span class="tt-v">${cell.n_unique_parent_id}</span></div>
        <div class="tt-row"><span class="tt-k">n_unique_child_id</span> <span class="tt-v">${cell.n_unique_child_id}</span></div>
        <div class="tt-sub">edge_category (value counts)</div>
        ${ecLines || '<div class="tt-ec">—</div>'}`;
    }

    const width = 960;
    const height = 960;
    const outerRadius = Math.min(width, height) * 0.5 - 72;
    const innerRadius = outerRadius - 24;

    const chord = d3.chordDirected()
      .padAngle(0.04)
      .sortSubgroups(d3.descending)
      .sortChords(d3.descending);

    const arc = d3.arc()
      .innerRadius(innerRadius)
      .outerRadius(outerRadius - 1);

    const ribbon = d3.ribbon()
      .radius(innerRadius - 0.5)
      .padAngle(1 / innerRadius);

    const chords = chord(DATA.matrix);

    const svg = d3.select("#chart")
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", [-width / 2, -height / 2, width, height]);

    const g = svg.append("g");

    const group = g.append("g")
      .attr("class", "groups")
      .selectAll("g")
      .data(chords.groups)
      .join("g")
      .attr("class", "group");

    group.append("path")
      .attr("fill", d => DATA.colors[d.index])
      .attr("stroke", "#fff")
      .attr("d", arc)
      .on("mouseenter", function(event, d) {
        tooltip.style("display", "block").html(formatGroupTooltip(d.index));
        moveTooltip(event);
      })
      .on("mousemove", function(event) {
        moveTooltip(event);
      })
      .on("mouseleave", function() {
        tooltip.style("display", "none");
      });

    g.append("g")
      .attr("class", "ribbons")
      .attr("fill-opacity", 0.67)
      .selectAll("path")
      .data(chords)
      .join("path")
      .attr("class", "ribbon")
      .attr("fill", d => DATA.colors[d.source.index])
      .attr("d", ribbon)
      .on("mouseenter", function(event, d) {
        const i = d.source.index, j = d.target.index;
        tooltip.style("display", "block").html(formatFlowTooltip(i, j));
        moveTooltip(event);
      })
      .on("mousemove", function(event) {
        moveTooltip(event);
      })
      .on("mouseleave", function() {
        tooltip.style("display", "none");
      });

    const labelRadius = outerRadius + 12;
    group.append("text")
      .each(d => { d.angle = (d.startAngle + d.endAngle) / 2; })
      .attr("dy", "0.35em")
      .attr("pointer-events", "none")
      .attr("transform", d => `
        rotate(${((d.angle * 180) / Math.PI - 90)})
        translate(${labelRadius})${d.angle > Math.PI ? " rotate(180)" : ""}
      `)
      .attr("text-anchor", d => (d.angle > Math.PI ? "end" : "start"))
      .attr("font-size", "9px")
      .text(d => DATA.labelsShort[d.index]);
  </script>
</body>
</html>
"""


def write_d3_chord_html(
    matrix: np.ndarray,
    labels: list[str],
    colors: list[str],
    out_path: Path,
    cell_details: list[list[dict | None]],
) -> None:
    """Self-contained HTML with D3 v7 directed chord (ribbon color = parent / source subcategory)."""
    payload = {
        "labels": labels,
        "labelsShort": [_short_label(l) for l in labels],
        "colors": colors,
        "matrix": matrix.astype(float).tolist(),
        "cellDetails": cell_details,
    }
    data_json = json.dumps(payload, ensure_ascii=False)
    html = _D3_CHORD_HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    out_path.write_text(html, encoding="utf-8")


def _norm_tag(tag: object) -> str | None:
    if tag is None or (isinstance(tag, float) and np.isnan(tag)):
        return None
    s = str(tag).strip()
    return _TAG_CLEANUP.get(s, s)


def edges_from_record(record: dict) -> list[dict]:
    """One row per dependency edge (parent → child) in one comment graph."""
    graph = record.get("dependency_graph") or []
    if not graph:
        return []
    nodes = {n["id"]: n for n in graph if "id" in n}
    article_id = str(record.get("article_id", ""))
    comment_index = record.get("comment_index")

    rows: list[dict] = []
    for node in graph:
        cid = node.get("id")
        if cid is None:
            continue
        child_tag = _norm_tag(node.get("comment_tag"))
        for dep in node.get("depends_on") or []:
            if not isinstance(dep, dict):
                continue
            pid = dep.get("id")
            parent = nodes.get(pid)
            if parent is None:
                continue
            parent_tag = _norm_tag(parent.get("comment_tag"))
            edge_type = dep.get("edge_type")
            rows.append(
                {
                    "article_id": article_id,
                    "comment_index": comment_index,
                    "parent_id": pid,
                    "child_id": cid,
                    "parent_subcategory": parent_tag,
                    "child_subcategory": child_tag,
                    "edge_category": edge_type if edge_type is not None else "",
                }
            )
    return rows


def load_combined_edges(combined_dir: Path) -> pd.DataFrame:
    paths = sorted(combined_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON files under {combined_dir}")

    all_rows: list[dict] = []
    for p in paths:
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for record in data:
            if isinstance(record, dict):
                all_rows.extend(edges_from_record(record))

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    mask = (
        df["parent_subcategory"].notna()
        & df["child_subcategory"].notna()
        & ~df["parent_subcategory"].isin(EXCLUDED_SUBCATS)
        & ~df["child_subcategory"].isin(EXCLUDED_SUBCATS)
    )
    return df.loc[mask].reset_index(drop=True)


def _arc_samples(t0: float, t1: float, radius: float, n: int) -> tuple[list[float], list[float]]:
    ts = np.linspace(t0, t1, max(2, n))
    return (radius * np.cos(ts)).tolist(), (radius * np.sin(ts)).tolist()


def _rim_annulus_sector(
    t0: float, t1: float, r_out: float, r_in: float, n: int = 28
) -> tuple[list[tuple[float, float]], list[int]]:
    """Closed path: outer arc t0→t1, radial in, inner arc t1→t0, radial out."""
    verts: list[tuple[float, float]] = []
    codes: list[int] = []
    ts = np.linspace(t0, t1, max(2, n))
    verts.append((float(r_out * np.cos(ts[0])), float(r_out * np.sin(ts[0]))))
    codes.append(MPath.MOVETO)
    for t in ts[1:]:
        verts.append((float(r_out * np.cos(t)), float(r_out * np.sin(t))))
        codes.append(MPath.LINETO)
    verts.append((float(r_in * np.cos(t1)), float(r_in * np.sin(t1))))
    codes.append(MPath.LINETO)
    ts2 = np.linspace(t1, t0, max(2, n))
    for t in ts2[1:]:
        verts.append((float(r_in * np.cos(t)), float(r_in * np.sin(t))))
        codes.append(MPath.LINETO)
    verts.append((float(r_out * np.cos(t0)), float(r_out * np.sin(t0))))
    codes.append(MPath.LINETO)
    codes.append(MPath.CLOSEPOLY)
    verts.append((0.0, 0.0))
    return verts, codes


def _append_arc(
    verts: list[tuple[float, float]],
    codes: list[int],
    t0: float,
    t1: float,
    radius: float,
    n: int = 16,
) -> None:
    xs, ys = _arc_samples(t0, t1, radius, n)
    first = True
    for x, y in zip(xs, ys, strict=True):
        if first:
            verts.append((float(x), float(y)))
            codes.append(MPath.MOVETO)
            first = False
        else:
            verts.append((float(x), float(y)))
            codes.append(MPath.LINETO)


def _ribbon_patch(
    a0: float,
    a1: float,
    b0: float,
    b1: float,
    radius: float = 1.0,
    shrink: float = 0.62,
) -> mpatches.PathPatch:
    """
    Filled ribbon along the outer circle from arc [a0,a1] on side A to [b0,b1] on side B.
    Boundaries: arc on circle + cubic beziers with control points pulled toward the center.
    """
    p0 = np.array([radius * np.cos(a0), radius * np.sin(a0)], dtype=float)
    p1 = np.array([radius * np.cos(a1), radius * np.sin(a1)], dtype=float)
    q0 = np.array([radius * np.cos(b0), radius * np.sin(b0)], dtype=float)
    q1 = np.array([radius * np.cos(b1), radius * np.sin(b1)], dtype=float)

    verts: list[tuple[float, float]] = []
    codes: list[int] = []

    _append_arc(verts, codes, a0, a1, radius)
    c1 = p1 * shrink
    c2 = q0 * shrink
    verts.extend([(c1[0], c1[1]), (c2[0], c2[1]), (q0[0], q0[1])])
    codes.extend([MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])

    _append_arc(verts, codes, b0, b1, radius)
    d1 = q1 * shrink
    d2 = p0 * shrink
    verts.extend([(d1[0], d1[1]), (d2[0], d2[1]), (p0[0], p0[1])])
    codes.extend([MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])

    codes.append(MPath.CLOSEPOLY)
    verts.append((0.0, 0.0))

    path = MPath(verts, codes)
    return mpatches.PathPatch(path, facecolor="#808080", edgecolor="none", antialiased=True)


def chord_diagram_subcategory_flows(
    M: np.ndarray,
    labels: list[str],
    label_colors: list[str],
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (12.0, 12.0),
    ribbon_alpha: float = 0.55,
) -> plt.Figure:
    """
    Directed chord-style diagram: M[i,j] = count from label i (parent) to label j (child).
    Ribbon fill color = parent node color (label_colors[i]).
    """
    n = M.shape[0]
    if n == 0:
        raise ValueError("Empty matrix")

    involvement = M.sum(axis=0) + M.sum(axis=1)
    if involvement.sum() == 0:
        raise ValueError("All-zero flow matrix")

    pad_total = 0.04 * 2 * np.pi
    usable = 2 * np.pi - pad_total
    gap = pad_total / max(n, 1)

    theta0 = np.zeros(n)
    theta1 = np.zeros(n)
    t = gap / 2
    inv_sum = float(involvement.sum())
    for k in range(n):
        span = usable * (involvement[k] / inv_sum)
        theta0[k] = t
        theta1[k] = t + span
        t = theta1[k] + gap

    out_map: dict[tuple[int, int], tuple[float, float]] = {}
    for i in range(n):
        targets = [j for j in range(n) if M[i, j] > 0]
        if not targets:
            continue
        targets.sort(key=lambda j: float(theta0[j] + theta1[j]) / 2.0)
        total = float(sum(M[i, j] for j in targets))
        w = theta1[i] - theta0[i]
        cur = theta0[i]
        for j in targets:
            frac = M[i, j] / total
            out_map[(i, j)] = (cur, cur + frac * w)
            cur += frac * w

    in_map: dict[tuple[int, int], tuple[float, float]] = {}
    for j in range(n):
        sources = [i for i in range(n) if M[i, j] > 0]
        if not sources:
            continue
        sources.sort(key=lambda i: float(theta0[i] + theta1[i]) / 2.0)
        total = float(sum(M[i, j] for i in sources))
        w = theta1[j] - theta0[j]
        cur = theta0[j]
        for i in sources:
            frac = M[i, j] / total
            in_map[(i, j)] = (cur, cur + frac * w)
            cur += frac * w

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"aspect": "equal"})
    else:
        fig = ax.figure

    ax.set_axis_off()
    lim = 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # Draw ribbons (larger counts on top: sort by value ascending)
    pairs = [(i, j, M[i, j]) for i in range(n) for j in range(n) if M[i, j] > 0 and i != j]
    pairs.sort(key=lambda x: x[2])

    vmax = max(p[2] for p in pairs) if pairs else 1.0
    for i, j, v in pairs:
        a0, a1 = out_map[(i, j)]
        b0, b1 = in_map[(i, j)]
        patch = _ribbon_patch(a0, a1, b0, b1, radius=1.0, shrink=0.62)
        c = label_colors[i]
        patch.set_facecolor(c)
        patch.set_alpha(ribbon_alpha + 0.35 * (v / vmax) * (1 - ribbon_alpha))
        ax.add_patch(patch)

    # Self-loops: small wedge markers (optional thin arc)
    for i in range(n):
        if M[i, i] <= 0:
            continue
        mid = (theta0[i] + theta1[i]) / 2
        ax.scatter(
            [1.02 * np.cos(mid)],
            [1.02 * np.sin(mid)],
            s=20 + 80 * (M[i, i] / max(M.diagonal().max(), 1)),
            c=[label_colors[i]],
            alpha=0.9,
            zorder=5,
        )

    # Rim / group arcs
    rim_w = 0.045
    inner_r = 1.0 - rim_w
    for k in range(n):
        verts_r, codes_r = _rim_annulus_sector(theta0[k], theta1[k], 1.0, inner_r, n=28)
        rim = mpatches.PathPatch(
            MPath(verts_r, codes_r),
            facecolor=label_colors[k],
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        ax.add_patch(rim)

        mid = (theta0[k] + theta1[k]) / 2
        lr = 1.12
        x, y = lr * np.cos(mid), lr * np.sin(mid)
        ha = "left" if -np.pi / 2 < mid <= np.pi / 2 else "right"
        rotation = np.degrees(mid)
        if mid > np.pi / 2 or mid < -np.pi / 2:
            rotation += 180
        short = labels[k].replace("Visual Observation: ", "VO: ").replace("Prior Knowledge: ", "PK: ")
        short = short.replace("Inference: ", "Inf: ").replace("Evaluative: ", "Ev: ")
        ax.text(
            x,
            y,
            short,
            ha=ha,
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=7,
            zorder=6,
        )

    return fig


def main() -> None:
    DATAFRAMES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_combined_edges(COMBINED_DATA_DIR)
    parquet_path = DATAFRAMES_DIR / PARQUET_NAME
    csv_path = DATAFRAMES_DIR / "subcategory_dependency_edges.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df):,} edges → {parquet_path}")
    print(f"Also wrote {csv_path}")

    df_edges = pd.read_parquet(parquet_path)
    fm = flow_matrix_from_edges(df_edges)
    if fm is None:
        print("No edges among FINEGRAIN_CATS; skipping chord outputs.")
        return

    M_sub, lab_sub, col_sub = fm
    cell_details = flow_details_matrix_from_edges(df_edges, lab_sub)

    html_path = FIGURES_DIR / HTML_NAME
    write_d3_chord_html(M_sub, lab_sub, col_sub, html_path, cell_details)
    print(f"Saved D3 chord diagram → {html_path}")

    fig = chord_diagram_subcategory_flows(M_sub, lab_sub, col_sub, figsize=(14, 14))
    fig_path = FIGURES_DIR / "subcategory_dependency_chord.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    pdf_path = FIGURES_DIR / "subcategory_dependency_chord.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved chord diagram → {fig_path}")


if __name__ == "__main__":
    main()
