import { useEffect, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import { Link, useParams } from 'react-router-dom';
import type { ArticleEntry, Comment } from '../types';

const TAGS_STORAGE_KEY = 'explorer_tags';

const TAG_HUES = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray'] as const;
type TagHue = (typeof TAG_HUES)[number];

interface TagWithHue {
  text: string;
  hue: TagHue;
  /** When set, removing this tag also removes the linked highlight */
  highlightSpanId?: string;
  /** Stable id so duplicate tag labels work (edit/remove target the right one) */
  id?: string;
}

const HUE_CLASSES: Record<TagHue, string> = {
  red: 'bg-red-100 text-red-800',
  green: 'bg-green-100 text-green-800',
  blue: 'bg-blue-100 text-blue-800',
  orange: 'bg-orange-100 text-orange-800',
  purple: 'bg-purple-100 text-purple-800',
  brown: 'bg-amber-200 text-amber-900',
  pink: 'bg-pink-100 text-pink-800',
  gray: 'bg-gray-100 text-gray-800',
};

const HUE_BUTTON_CLASSES: Record<TagHue, string> = {
  red: 'hover:text-red-600',
  green: 'hover:text-green-600',
  blue: 'hover:text-blue-600',
  orange: 'hover:text-orange-600',
  purple: 'hover:text-purple-600',
  brown: 'hover:text-amber-700',
  pink: 'hover:text-pink-600',
  gray: 'hover:text-gray-600',
};

const HUE_SWATCH_CLASSES: Record<TagHue, string> = {
  red: 'bg-red-400',
  green: 'bg-green-400',
  blue: 'bg-blue-400',
  orange: 'bg-orange-400',
  purple: 'bg-purple-400',
  brown: 'bg-amber-500',
  pink: 'bg-pink-400',
  gray: 'bg-gray-400',
};

/** RGB for each hue so we can layer highlights with opacity when they overlap */
const HUE_RGB: Record<TagHue, string> = {
  red: '254, 226, 226',
  green: '220, 252, 231',
  blue: '219, 234, 254',
  orange: '255, 237, 213',
  purple: '243, 232, 255',
  brown: '253, 230, 138',
  pink: '252, 231, 243',
  gray: '229, 231, 235',
};

const HIGHLIGHTS_STORAGE_KEY = 'explorer_highlights';

interface HighlightSpan {
  start: number;
  end: number;
  hue: TagHue;
  /** Links to a tag; when that tag is removed, this highlight is removed */
  id?: string;
}

// (Previously we had an AnnotatedCommentEntry interface and helpers for pre-tagged
// comment JSON files. Those are no longer used now that we read directly from
// `commentsPath` in `articles-index.json`.)

/** Base URL for comment graphs (extraction_pipeline/extraction_pipeline/{imageId}/comment_graphs/{n}.png). */
const COMMENT_GRAPH_BASE = '/extraction_pipeline/extraction_pipeline';

function getCommentGraphSrc(imageId: string, commentIndex: number): string {
  const oneBased = commentIndex + 1;
  return `${COMMENT_GRAPH_BASE}/${imageId}/comment_graphs/${oneBased}.png`;
}

function getTagsKey(imageId: string, commentIndex: number): string {
  return `${TAGS_STORAGE_KEY}_${imageId}_${commentIndex}`;
}

function getHighlightsKey(imageId: string, commentIndex: number): string {
  return `${HIGHLIGHTS_STORAGE_KEY}_${imageId}_${commentIndex}`;
}

function loadTags(imageId: string, commentIndex: number): TagWithHue[] {
  try {
    const raw = localStorage.getItem(getTagsKey(imageId, commentIndex));
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.map((item: unknown) => {
      if (item && typeof item === 'object' && 'text' in item && 'hue' in item) {
        const obj = item as { text: string; hue: string; highlightSpanId?: string; id?: string };
        const id =
          typeof obj.id === 'string'
            ? obj.id
            : typeof obj.highlightSpanId === 'string'
              ? obj.highlightSpanId
              : `t_${Math.random().toString(36).slice(2, 10)}`;
        return {
          text: String(obj.text),
          hue: TAG_HUES.includes(obj.hue as TagHue) ? (obj.hue as TagHue) : 'gray',
          ...(typeof obj.highlightSpanId === 'string' && obj.highlightSpanId
            ? { highlightSpanId: obj.highlightSpanId }
            : {}),
          id,
        };
      }
      if (typeof item === 'string') {
        return { text: item, hue: 'gray' as TagHue, id: `t_${Math.random().toString(36).slice(2, 10)}` };
      }
      return null;
    }).filter((t): t is NonNullable<typeof t> => t !== null) as TagWithHue[];
  } catch {
    return [];
  }
}

function saveTags(imageId: string, commentIndex: number, tags: TagWithHue[]): void {
  localStorage.setItem(getTagsKey(imageId, commentIndex), JSON.stringify(tags));
}

function loadHighlights(imageId: string, commentIndex: number): HighlightSpan[] {
  try {
    const raw = localStorage.getItem(getHighlightsKey(imageId, commentIndex));
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter(
        (item: unknown): item is HighlightSpan =>
          item != null &&
          typeof item === 'object' &&
          'start' in item &&
          'end' in item &&
          'hue' in item &&
          typeof (item as HighlightSpan).start === 'number' &&
          typeof (item as HighlightSpan).end === 'number' &&
          TAG_HUES.includes((item as HighlightSpan).hue)
      )
      .map((s) => {
        const span = s as HighlightSpan;
        return {
          start: span.start,
          end: span.end,
          hue: span.hue,
          ...(typeof span.id === 'string' ? { id: span.id } : {}),
        };
      }) as HighlightSpan[];
  } catch {
    return [];
  }
}

function saveHighlights(imageId: string, commentIndex: number, spans: HighlightSpan[]): void {
  localStorage.setItem(getHighlightsKey(imageId, commentIndex), JSON.stringify(spans));
}

// (We also used to derive highlight spans automatically from tag text; that logic
// depended on pre-annotated JSON and is no longer referenced.)

// NOTE: We previously hydrated localStorage from pre-annotated JSON files here.
// That flow now loads comments directly from `commentsPath` in `articles-index.json`,
// so the helper has been removed to avoid unused-code warnings.

function getTextOffsets(container: Node, range: Range): { start: number; end: number } | null {
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null);
  let current = 0;
  let startOffset = -1;
  let endOffset = -1;
  let node: Node | null = walker.nextNode();
  while (node) {
    const len = (node.textContent ?? '').length;
    if (node === range.startContainer) {
      startOffset = current + range.startOffset;
    }
    if (node === range.endContainer) {
      endOffset = current + range.endOffset;
    }
    current += len;
    node = walker.nextNode();
  }
  if (startOffset >= 0 && endOffset >= 0) return { start: startOffset, end: endOffset };
  return null;
}

/** Opacity per highlight layer so overlapping regions blend visibly */
const HIGHLIGHT_LAYER_OPACITY = 0.65;

function buildSegments(
  text: string,
  highlights: HighlightSpan[]
): { start: number; end: number; hues: TagHue[] }[] {
  if (highlights.length === 0) return [{ start: 0, end: text.length, hues: [] }];
  const points = new Set<number>();
  points.add(0);
  points.add(text.length);
  for (const h of highlights) {
    if (h.start < h.end) {
      points.add(h.start);
      points.add(h.end);
    }
  }
  const sortedPoints = [...points].sort((a, b) => a - b);
  const segments: { start: number; end: number; hues: TagHue[] }[] = [];
  for (let i = 0; i < sortedPoints.length - 1; i++) {
    const start = sortedPoints[i];
    const end = sortedPoints[i + 1];
    const hues = highlights.filter((h) => h.start < end && h.end > start).map((h) => h.hue);
    segments.push({ start, end, hues });
  }
  return segments;
}

function overlayBackgroundStyle(hues: TagHue[]): CSSProperties {
  if (hues.length === 0) return {};
  const layers = hues.map(
    (h) => `linear-gradient(to right, rgba(${HUE_RGB[h]}, ${HIGHLIGHT_LAYER_OPACITY}), rgba(${HUE_RGB[h]}, ${HIGHLIGHT_LAYER_OPACITY}))`
  );
  return { background: layers.join(', ') };
}

function CommentGraphImage({ imageId, commentIndex }: { imageId: string; commentIndex: number }) {
  const [visible, setVisible] = useState(true);
  const src = getCommentGraphSrc(imageId, commentIndex);
  if (!visible) return null;
  return (
    <div className="mt-3 rounded border border-gray-200 bg-gray-50 overflow-hidden relative">
      <img
        src={src}
        alt={`Comment ${commentIndex + 1} graph`}
        className="max-w-full w-full h-auto block"
        onError={() => setVisible(false)}
      />
      <a
        href={src}
        target="_blank"
        rel="noopener noreferrer"
        className="absolute top-2 right-2 p-1.5 rounded-md bg-white/90 hover:bg-white shadow border border-gray-200 text-gray-600 hover:text-gray-900 transition-colors"
        title="Open image in new tab"
        aria-label="Open image in new tab"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
          <polyline points="15 3 21 3 21 9" />
          <line x1="10" y1="14" x2="21" y2="3" />
        </svg>
      </a>
    </div>
  );
}

function CommentCard({
  comment,
  index,
  imageId,
}: {
  comment: Comment;
  index: number;
  imageId: string;
}) {
  const [tags, setTags] = useState<TagWithHue[]>(() => loadTags(imageId, index));
  const [highlights, setHighlights] = useState<HighlightSpan[]>(() => loadHighlights(imageId, index));
  const [newTag, setNewTag] = useState('');
  const [newTagHue, setNewTagHue] = useState<TagHue>('gray');
  const [editingHueFor, setEditingHueFor] = useState<string | null>(null);
  const [selectionPopup, setSelectionPopup] = useState<{
    text: string;
    tagLabel: string;
    hue: TagHue;
    start: number;
    end: number;
    rect: DOMRect;
  } | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);

  const addTag = () => {
    const t = newTag.trim();
    if (!t || tags.some((x) => x.text === t)) return;
    const tagId = `t_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const next = [...tags, { text: t, hue: newTagHue, id: tagId }];
    setTags(next);
    saveTags(imageId, index, next);
    setNewTag('');
  };

  const getTagId = (tag: TagWithHue, tagIndex: number) => tag.id ?? tag.highlightSpanId ?? `tag-${tagIndex}`;

  const removeTag = (tagId: string) => {
    const tag = tags.find((t, i) => getTagId(t, i) === tagId);
    const nextTags = tags.filter((t, i) => getTagId(t, i) !== tagId);
    setTags(nextTags);
    saveTags(imageId, index, nextTags);
    setEditingHueFor((prev) => (prev === tagId ? null : prev));
    const linkedId = tag?.highlightSpanId;
    const nextHighlights =
      linkedId != null
        ? highlights.filter((span) => span.id !== linkedId)
        : highlights.filter(
            (span) => (comment['comment info'] ?? '').trim().slice(span.start, span.end).trim() !== tag?.text
          );
    if (nextHighlights.length !== highlights.length) {
      setHighlights(nextHighlights);
      saveHighlights(imageId, index, nextHighlights);
    }
  };

  const setTagHue = (tagId: string, hue: TagHue) => {
    const next = tags.map((t, i) => (getTagId(t, i) === tagId ? { ...t, hue } : t));
    setTags(next);
    saveTags(imageId, index, next);
    setEditingHueFor(null);
  };

  const cycleTagHue = (tagId: string) => {
    const tag = tags.find((t, i) => getTagId(t, i) === tagId);
    if (!tag) return;
    const idx = TAG_HUES.indexOf(tag.hue);
    const nextHue = TAG_HUES[(idx + 1) % TAG_HUES.length];
    setTagHue(tagId, nextHue);
  };

  const addTagFromSelection = () => {
    if (!selectionPopup) return;
    const { tagLabel, hue, start, end } = selectionPopup;
    const trimmed = tagLabel.trim();
    if (!trimmed) {
      setSelectionPopup(null);
      return;
    }
    const spanId = `hl_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
    const nextTags = [...tags, { text: trimmed, hue, highlightSpanId: spanId, id: spanId }];
    setTags(nextTags);
    saveTags(imageId, index, nextTags);
    const nextHighlights = [...highlights, { start, end, hue, id: spanId }];
    setHighlights(nextHighlights);
    saveHighlights(imageId, index, nextHighlights);
    setSelectionPopup(null);
    window.getSelection()?.removeAllRanges();
  };

  const handleCommentMouseUp = () => {
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0 || !bodyRef.current) return;
    const range = sel.getRangeAt(0);
    if (range.collapsed) {
      setSelectionPopup(null);
      return;
    }
    if (!bodyRef.current.contains(range.commonAncestorContainer)) {
      setSelectionPopup(null);
      return;
    }
    const offsets = getTextOffsets(bodyRef.current, range);
    if (!offsets) return;
    const text = (comment['comment info'] ?? '').trim();
    const selectedText = text.slice(offsets.start, offsets.end);
    if (!selectedText.trim()) return;
    const rect = range.getBoundingClientRect();
    setSelectionPopup({
      text: selectedText,
      tagLabel: selectedText,
      hue: 'gray',
      start: offsets.start,
      end: offsets.end,
      rect,
    });
  };

  const commentText = (comment['comment info'] ?? '').trim();
  const segments = buildSegments(commentText, highlights);

  return (
    <article className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <div className="flex items-center gap-2 text-sm text-gray-500 mb-2">
        <span className="font-medium text-gray-900">{comment.name}</span>
        <span>{comment.location}</span>
        <span>{comment['date posted']}</span>
      </div>
      <div
        ref={bodyRef}
        className="text-gray-700 whitespace-pre-wrap mb-3 select-text"
        onMouseUp={handleCommentMouseUp}
      >
        {segments.map((seg, i) =>
          seg.hues.length > 0 ? (
            <mark key={i} className="rounded-sm" style={overlayBackgroundStyle(seg.hues)}>
              {commentText.slice(seg.start, seg.end)}
            </mark>
          ) : (
            <span key={i}>{commentText.slice(seg.start, seg.end)}</span>
          )
        )}
      </div>
      {selectionPopup && (
        <div
          className="fixed z-50 rounded-lg border border-gray-200 bg-white p-2 shadow-lg flex flex-col gap-2 min-w-[180px]"
          style={(() => {
            const pad = 8;
            const belowTop = selectionPopup.rect.bottom + pad;
            const aboveBottom = selectionPopup.rect.top - pad;
            const preferBelow = belowTop + 120 <= window.innerHeight;
            return {
              left: Math.max(pad, Math.min(selectionPopup.rect.left, window.innerWidth - 200)),
              top: preferBelow ? belowTop : aboveBottom - 120,
            };
          })()}
        >
          <input
            type="text"
            value={selectionPopup.tagLabel}
            onChange={(e) => setSelectionPopup((p) => (p ? { ...p, tagLabel: e.target.value } : null))}
            onKeyDown={(e) => {
              if (e.key === 'Enter') addTagFromSelection();
              if (e.key === 'Escape') {
                setSelectionPopup(null);
                window.getSelection()?.removeAllRanges();
              }
            }}
            placeholder="Tag label"
            className="w-full border border-gray-300 rounded px-2 py-1.5 text-sm"
            autoFocus
          />
          <div className="flex gap-0.5 items-center flex-wrap">
            {TAG_HUES.map((h) => (
              <button
                key={h}
                type="button"
                onClick={() => setSelectionPopup((p) => (p ? { ...p, hue: h } : null))}
                className={`w-5 h-5 rounded-full border-2 ${selectionPopup.hue === h ? 'border-gray-800 scale-110 ring-1 ring-gray-400' : 'border-gray-300'} ${HUE_SWATCH_CLASSES[h]}`}
                title={h}
                aria-label={`Color ${h}`}
              />
            ))}
          </div>
          <div className="flex gap-1">
            <button
              type="button"
              onClick={addTagFromSelection}
              className={`flex-1 px-2 py-1 text-xs rounded ${HUE_CLASSES[selectionPopup.hue]} font-medium hover:opacity-90`}
            >
              Add tag
            </button>
            <button
              type="button"
              onClick={() => {
                setSelectionPopup(null);
                window.getSelection()?.removeAllRanges();
              }}
              className="px-2 py-1 text-xs text-gray-500 hover:text-gray-700"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
      {comment.replies && comment.replies.length > 0 && (
        <div className="ml-4 pl-3 border-l-2 border-gray-200 space-y-2 mb-3">
          {comment.replies.map((r, i) => (
            <div key={i} className="text-sm">
              <span className="font-medium text-gray-700">{r.name}</span>
              <span className="text-gray-500"> · {r['date posted']}</span>
              <p className="text-gray-600 mt-0.5">{r['comment info']}</p>
            </div>
          ))}
        </div>
      )}
      <div className="flex flex-wrap gap-2 items-center">
        {tags.map((tag, tagIndex) => {
          const tagId = getTagId(tag, tagIndex);
          return (
            <span
              key={tagId}
              className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs ${HUE_CLASSES[tag.hue]}`}
            >
              {editingHueFor === tagId ? (
                <span className="flex flex-wrap gap-0.5">
                  {TAG_HUES.map((h) => (
                    <button
                      key={h}
                      type="button"
                      onClick={() => setTagHue(tagId, h)}
                      className={`w-4 h-4 rounded-full border border-gray-300 ${HUE_SWATCH_CLASSES[h]}`}
                      title={h}
                      aria-label={`Set tag color to ${h}`}
                    />
                  ))}
                </span>
              ) : (
                <>
                  <button
                    type="button"
                    onClick={() => setEditingHueFor(tagId)}
                    className="text-left min-w-0"
                    title="Click to change color"
                  >
                    {tag.text}
                  </button>
                  <button
                    type="button"
                    onClick={() => cycleTagHue(tagId)}
                    className={HUE_BUTTON_CLASSES[tag.hue]}
                    title="Cycle color"
                    aria-label={`Cycle color for ${tag.text}`}
                  >
                    ◐
                  </button>
                </>
              )}
              <button
                type="button"
                onClick={() => removeTag(tagId)}
                className={HUE_BUTTON_CLASSES[tag.hue]}
                aria-label={`Remove tag ${tag.text}`}
              >
                ×
              </button>
            </span>
          );
        })}
        <div className="inline-flex flex-wrap gap-1 items-center">
          <input
            type="text"
            value={newTag}
            onChange={(e) => setNewTag(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && addTag()}
            placeholder="+ Add tag"
            className="border border-gray-300 rounded px-2 py-1 text-sm w-28"
          />
          <span className="flex gap-0.5 items-center" title="Tag color">
            {TAG_HUES.map((h) => (
              <button
                key={h}
                type="button"
                onClick={() => setNewTagHue(h)}
                className={`w-5 h-5 rounded-full border-2 ${newTagHue === h ? 'border-gray-800 scale-110 ring-1 ring-gray-400' : 'border-gray-300'} ${HUE_SWATCH_CLASSES[h]}`}
                title={h}
                aria-label={`Color ${h}`}
              />
            ))}
          </span>
          <button
            type="button"
            onClick={addTag}
            className="px-2 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Add
          </button>
        </div>
      </div>

      {/* Comment graph from extraction_pipeline: extraction_pipeline/{imageId}/comment_graphs/{index+1}.png */}
      <CommentGraphImage imageId={imageId} commentIndex={index} />
    </article>
  );
}

export function ImageDetail() {
  const { imageId } = useParams<{ imageId: string }>();
  const [comments, setComments] = useState<Comment[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [articlesIndex, setArticlesIndex] = useState<Record<string, ArticleEntry> | null>(null);

  useEffect(() => {
    fetch('/articles-index.json')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('No articles-index.json'))))
      .then((index: Record<string, ArticleEntry>) => setArticlesIndex(index))
      .catch(() => setArticlesIndex({}));
  }, []);

  const articleEntry = imageId && articlesIndex ? articlesIndex[imageId] ?? null : null;

  if (imageId && articlesIndex) {
    // #region agent log
    fetch('http://127.0.0.1:7243/ingest/188396c9-e207-4905-baa6-36680b519627', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id: `log_${Date.now()}_image_detail_entry`,
        timestamp: Date.now(),
        runId: 'pre-fix',
        hypothesisId: 'H3',
        location: 'ImageDetail.tsx:articleEntry',
        message: 'Resolved articleEntry for image detail',
        data: {
          imageId,
          hasEntry: !!articleEntry,
          pngPath: articleEntry?.pngPath ?? null,
          commentsPath: articleEntry?.commentsPath ?? null,
        },
      }),
    }).catch(() => {});
    // #endregion
  }

  useEffect(() => {
    if (!imageId) return;
    setLoading(true);
    setError(null);
    let cancelled = false;

    const entry = articleEntry;

    (async () => {
      try {
        // Prefer the path from articles-index.json if available.
        const primaryUrl =
          (entry?.commentsPath && entry.commentsPath.trim()) ||
          `/articles_data/${imageId}.json`;

        // #region agent log
        fetch('http://127.0.0.1:7243/ingest/188396c9-e207-4905-baa6-36680b519627', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            id: `log_${Date.now()}_image_detail_comments_fetch`,
            timestamp: Date.now(),
            runId: 'comments-pre-fix',
            hypothesisId: 'C2',
            location: 'ImageDetail.tsx:comments-fetch',
            message: 'Fetching comments for image detail',
            data: { imageId, primaryUrl },
          }),
        }).catch(() => {});
        // #endregion

        const r = await fetch(primaryUrl);
        if (!r.ok) throw new Error('No comments for this image');
        const text = await r.text();
        let data: Comment[] = [];
        try {
          const parsed = JSON.parse(text) as unknown;
          if (Array.isArray(parsed)) data = parsed as Comment[];
        } catch {
          throw new Error('Invalid response for this image');
        }
        if (!cancelled) setComments(data);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [imageId, articleEntry]);

  const exportCommentsWithTags = () => {
    if (!comments || !imageId) return;
    const data = comments.map((comment, index) => ({
      comment,
      tags: loadTags(imageId, index),
    }));
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `comments_${imageId}_with_tags.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!imageId) {
    return (
      <div className="p-8 text-center text-gray-500">
        Missing image ID
      </div>
    );
  }

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] md:flex-row">
      <div className="flex-1 min-h-0 flex items-center justify-center bg-gray-100 p-4 border-b md:border-b-0 md:border-r border-gray-200 relative">
        <img
          src={articleEntry?.pngPath ?? (imageId ? `/visualizations/${imageId}.png` : '')}
          alt={`Visualization ${imageId}`}
          className="max-w-full max-h-full object-contain"
        />
        {(articleEntry?.pngPath ?? (imageId && `/visualizations/${imageId}.png`)) && (
          <a
            href={articleEntry?.pngPath ?? `/visualizations/${imageId}.png`}
            target="_blank"
            rel="noopener noreferrer"
            className="absolute top-2 right-2 p-1.5 rounded-md bg-white/90 hover:bg-white shadow border border-gray-200 text-gray-600 hover:text-gray-900 transition-colors"
            title="Open image in new tab"
            aria-label="Open image in new tab"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
              <polyline points="15 3 21 3 21 9" />
              <line x1="10" y1="14" x2="21" y2="3" />
            </svg>
          </a>
        )}
      </div>
      <div className="flex-1 min-h-0 overflow-auto p-4">
        <div className="flex items-center gap-2 mb-4">
          <Link
            to="/"
            className="text-sm text-blue-600 hover:underline"
          >
            ← Back to gallery
          </Link>
          {!loading && !error && comments !== null && comments.length > 0 && (
            <button
              type="button"
              onClick={exportCommentsWithTags}
              className="ml-auto text-sm px-3 py-1.5 bg-gray-800 text-white rounded hover:bg-gray-700"
            >
              Export comments (JSON)
            </button>
          )}
        </div>

        <p className="text-xs text-gray-600 mb-4 flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="font-medium text-gray-700 mr-1">Legend:</span>
          <span><span className="inline-block w-2.5 h-2.5 rounded-full bg-red-400 align-middle mr-1" aria-hidden />red = data variables</span>
          <span><span className="inline-block w-2.5 h-2.5 rounded-full bg-green-400 align-middle mr-1" aria-hidden />green = statistical computations</span>
          <span><span className="inline-block w-2.5 h-2.5 rounded-full bg-gray-400 align-middle mr-1" aria-hidden />grey = prior world knowledge</span>
          <span><span className="inline-block w-2.5 h-2.5 rounded-full bg-blue-400 align-middle mr-1" aria-hidden />blue = statistical predictions</span>
          <span><span className="inline-block w-2.5 h-2.5 rounded-full bg-orange-400 align-middle mr-1" aria-hidden />orange = normative inferences</span>
        </p>

        {articleEntry && (
          <p className="mb-2">
            <a
              href={articleEntry.articleUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-blue-600 hover:underline"
            >
              View article on NYTimes.com →
            </a>
          </p>
        )}

        {!loading && !error && comments !== null && comments.length > 0 && (() => {
          const wordCounts = comments.map((c) =>
            (c['comment info'] ?? '').trim().split(/\s+/).filter(Boolean).length
          );
          const avgWords = wordCounts.reduce((a, n) => a + n, 0) / wordCounts.length;
          return (
            <p className="text-sm text-gray-600 mb-3">
              <span className="font-medium text-gray-700">Avg words per comment (excl. replies):</span>{' '}
              {avgWords.toFixed(1)}
              {' · '}
              {comments.length} comment{comments.length !== 1 ? 's' : ''} without replies
              {' · '}
              {comments.length + comments.reduce((s, c) => s + (c.replies?.length ?? 0), 0)} total including replies
            </p>
          );
        })()}

        <h2 className="text-lg font-semibold text-gray-900 mb-3">Comments</h2>
        {loading && (
          <p className="text-gray-500">Loading comments…</p>
        )}
        {error && (
          <p className="text-amber-600">{error}</p>
        )}
        {!loading && !error && comments !== null && (
          comments.length === 0 ? (
            <p className="text-gray-500">No comments.</p>
          ) : (
            <ul className="space-y-4">
              {comments!.map((c, i) => (
                <li key={i}>
                  <CommentCard comment={c} index={i} imageId={imageId} />
                </li>
              ))}
            </ul>
          )
        )}
      </div>
    </div>
  );
}
