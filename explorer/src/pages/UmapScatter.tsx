import { useEffect, useRef, useState } from 'react';
import {
  scaleLinear,
  scaleOrdinal,
  schemeCategory10,
  select,
  axisBottom,
  axisLeft,
  extent,
  pointer,
  Delaunay,
} from 'd3';
import type { UmapRow } from '../types';

const WIDTH = 550;
const HEIGHT = 450;
const MARGIN = { top: 20, right: 20, bottom: 40, left: 50 };
const INNER_WIDTH = WIDTH - MARGIN.left - MARGIN.right;
const INNER_HEIGHT = HEIGHT - MARGIN.top - MARGIN.bottom;

function parseCSVLine(line: string): string[] {
  const out: string[] = [];
  let cur = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      inQuotes = !inQuotes;
    } else if ((c === ',' && !inQuotes) || (c === '\n' && !inQuotes)) {
      out.push(cur);
      cur = '';
    } else {
      cur += c;
    }
  }
  out.push(cur);
  return out;
}

export function UmapScatter() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [data, setData] = useState<UmapRow[]>([]);
  const [hoveredPath, setHoveredPath] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/umap_2d_slim.csv')
      .then((r) => {
        if (!r.ok) throw new Error('Failed to load UMAP data');
        return r.text();
      })
      .then((text) => {
        const lines = text.trim().split('\n');
        const header = lines[0].split(',');
        const umap1Idx = header.indexOf('umap_1');
        const umap2Idx = header.indexOf('umap_2');
        const classIdx = header.indexOf('class');
        const pathIdx = header.indexOf('path');
        if ([umap1Idx, umap2Idx, classIdx, pathIdx].some((i) => i === -1)) {
          throw new Error('CSV missing required columns (umap_1, umap_2, class, path)');
        }
        const rows: UmapRow[] = [];
        for (let i = 1; i < lines.length; i++) {
          const parts = parseCSVLine(lines[i]);
          rows.push({
            umap_1: Number(parts[umap1Idx]),
            umap_2: Number(parts[umap2Idx]),
            class: String(parts[classIdx]),
            path: String(parts[pathIdx]),
          });
        }
        setData(rows);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!svgRef.current || data.length === 0) return;

    const xExtent = extent(data, (d: UmapRow) => d.umap_1) as [number, number];
    const yExtent = extent(data, (d: UmapRow) => d.umap_2) as [number, number];
    const xScale = scaleLinear().domain(xExtent).range([0, INNER_WIDTH]);
    const yScale = scaleLinear().domain(yExtent).range([INNER_HEIGHT, 0]);
    const classes = Array.from(new Set(data.map((d: UmapRow) => d.class)));
    const colorScale = scaleOrdinal<string, string>().domain(classes).range(schemeCategory10);

    const svg = select(svgRef.current);
    svg.selectAll('*').remove();

    const g = svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    g.append('g')
      .attr('transform', `translate(0,${INNER_HEIGHT})`)
      .attr('class', 'axis axis-x')
      .call(axisBottom(xScale))
      .selectAll('text')
      .attr('fill', '#374151');

    g.append('g')
      .attr('class', 'axis axis-y')
      .call(axisLeft(yScale))
      .selectAll('text')
      .attr('fill', '#374151');

    const points = data.map((d: UmapRow) => [
      xScale(d.umap_1),
      yScale(d.umap_2),
    ]) as [number, number][];
    const delaunay = Delaunay.from(points);

    g.append('g')
      .attr('class', 'points')
      .selectAll<SVGCircleElement, UmapRow>('circle')
      .data(data)
      .join('circle')
      .attr('cx', (d: UmapRow) => xScale(d.umap_1))
      .attr('cy', (d: UmapRow) => yScale(d.umap_2))
      .attr('r', 5)
      .attr('fill', (d: UmapRow) => colorScale(d.class))
      .attr('stroke', '#fff')
      .attr('stroke-width', 0.5)
      .attr('opacity', 0.85)
      .style('cursor', 'pointer');

    const voronoiOverlay = g
      .append('rect')
      .attr('class', 'voronoi-overlay')
      .attr('width', INNER_WIDTH)
      .attr('height', INNER_HEIGHT)
      .attr('fill', 'none')
      .attr('pointer-events', 'all')
      .style('cursor', 'pointer');

    voronoiOverlay
      .on('mousemove', (event: MouseEvent) => {
        const [x, y] = pointer(event);
        const index = delaunay.find(x, y);
        setHoveredPath(data[index].path);
        g.selectAll<SVGCircleElement, UmapRow>('circle')
          .attr('r', (d: UmapRow) => (d.path === data[index].path ? 7 : 5))
          .attr('opacity', (d: UmapRow) => (d.path === data[index].path ? 1 : 0.85));
      })
      .on('mouseleave', () => {
        setHoveredPath(null);
        g.selectAll<SVGCircleElement, UmapRow>('circle').attr('r', 5).attr('opacity', 0.85);
      });

    return () => {
      svg.selectAll('*').remove();
    };
  }, [data]);

  if (loading) {
    return (
      <div className="p-8 text-center text-gray-500">
        Loading UMAP data…
      </div>
    );
  }
  if (error) {
    return (
      <div className="p-8 text-center text-red-600">
        {error}
      </div>
    );
  }

  return (
    <div className="flex h-screen w-full">
      <div className="w-1/2 h-full min-w-0 flex items-center justify-center p-4">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="w-full h-full block"
          style={{ overflow: 'visible' }}
          preserveAspectRatio="xMidYMid meet"
        >
          <g className="chart" />
        </svg>
      </div>
      <div className="w-1/2 h-full flex items-center justify-center p-4 border-l border-gray-200 bg-gray-50">
        {hoveredPath ? (
          <img
            src={`/visualizations/${hoveredPath}`}
            alt="Hovered"
            className="max-w-full max-h-full object-contain"
          />
        ) : (
          <span className="text-gray-400 text-sm">Hover a point to show image</span>
        )}
      </div>
    </div>
  );
}
