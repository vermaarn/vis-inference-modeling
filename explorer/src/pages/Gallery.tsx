import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import type { ArticleEntry, Comment } from '../types';

type CommentCounts = { topLevel: number; total: number; avgWordCount: number } | null;

type SortMode = 'default' | 'mostComments' | 'avgWordsAsc' | 'avgWordsDesc';

export function Gallery() {
  const [pngList, setPngList] = useState<string[]>([]);
  const [articlesIndex, setArticlesIndex] = useState<Record<string, ArticleEntry> | null>(null);
  const [countsByImageId, setCountsByImageId] = useState<Record<string, CommentCounts>>({});
  const [sortMode, setSortMode] = useState<SortMode>('default');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const sortedList =
    sortMode === 'mostComments'
      ? [...pngList].sort((a, b) => {
          const idA = a.replace(/\.png$/i, '');
          const idB = b.replace(/\.png$/i, '');
          const totalA = countsByImageId[idA]?.total ?? -1;
          const totalB = countsByImageId[idB]?.total ?? -1;
          return totalB - totalA;
        })
      : sortMode === 'avgWordsAsc' || sortMode === 'avgWordsDesc'
        ? [...pngList].sort((a, b) => {
            const idA = a.replace(/\.png$/i, '');
            const idB = b.replace(/\.png$/i, '');
            const avgA = countsByImageId[idA]?.avgWordCount ?? -1;
            const avgB = countsByImageId[idB]?.avgWordCount ?? -1;
            return sortMode === 'avgWordsAsc' ? avgA - avgB : avgB - avgA;
          })
        : pngList;

  useEffect(() => {
    Promise.all([
      fetch('/visualization-pngs.json').then((r) => {
        if (!r.ok) throw new Error('Failed to load PNG list');
        return r.json() as Promise<string[]>;
      }),
      fetch('/articles-index.json')
        .then((r) => (r.ok ? r.json() : Promise.resolve({})) as Promise<Record<string, ArticleEntry>>)
        .catch(() => ({})),
    ])
      .then(([list, index]) => {
        setPngList(list);
        setArticlesIndex(index);
        // #region agent log
        fetch('http://127.0.0.1:7243/ingest/188396c9-e207-4905-baa6-36680b519627', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            id: `log_${Date.now()}_gallery_init`,
            timestamp: Date.now(),
            runId: 'pre-fix',
            hypothesisId: 'H1',
            location: 'Gallery.tsx:after-index-load',
            message: 'Loaded pngList and articlesIndex',
            data: { pngCount: list.length, hasIndex: !!index && Object.keys(index).length > 0 },
          }),
        }).catch(() => {});
        // #endregion
      })
      .catch((e: unknown) => {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (pngList.length === 0) return;
    const imageIds = pngList.map((f) => f.replace(/\.png$/i, ''));
    let cancelled = false;

    imageIds.forEach((imageId) => {
      const entry = articlesIndex?.[imageId];
      const commentsUrl =
        (entry?.commentsPath && entry.commentsPath.trim()) ||
        `/articles_data/${imageId}.json`;

      // #region agent log
      fetch('http://127.0.0.1:7243/ingest/188396c9-e207-4905-baa6-36680b519627', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: `log_${Date.now()}_gallery_comments_fetch`,
          timestamp: Date.now(),
          runId: 'comments-pre-fix',
          hypothesisId: 'C1',
          location: 'Gallery.tsx:comments-fetch',
          message: 'Fetching comments for gallery image',
          data: { imageId, commentsUrl },
        }),
      }).catch(() => {});
      // #endregion

      fetch(commentsUrl)
        .then((r) => (r.ok ? r.json() : Promise.resolve(null)))
        .then((data: Comment[] | null) => {
          if (cancelled) return;
          if (!Array.isArray(data)) {
            setCountsByImageId((prev) => ({ ...prev, [imageId]: null }));
            return;
          }
          const topLevel = data.length;
          const total = data.reduce(
            (sum, c) => sum + 1 + (c.replies?.length ?? 0),
            0
          );
          const avgWordCount =
            topLevel > 0
              ? data.reduce((sum, c) => {
                  const text = c['comment info'] ?? '';
                  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
                  return sum + words;
                }, 0) / topLevel
              : 0;
          setCountsByImageId((prev) => ({ ...prev, [imageId]: { topLevel, total, avgWordCount } }));
        })
        .catch(() => {
          if (!cancelled) {
            setCountsByImageId((prev) => ({ ...prev, [imageId]: null }));
          }
        });
    });

    return () => {
      cancelled = true;
    };
  }, [pngList, articlesIndex]);

  if (loading) {
    return (
      <div className="p-8 text-center text-gray-500">
        Loading gallery…
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
    <div className="p-6">
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <h1 className="text-xl font-semibold text-gray-900">Visualizations</h1>
        <button
          type="button"
          onClick={() => setSortMode((m) => (m === 'mostComments' ? 'default' : 'mostComments'))}
          className={`px-3 py-1.5 text-sm rounded-md border transition-colors ${
            sortMode === 'mostComments'
              ? 'bg-blue-600 text-white border-blue-600'
              : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
          }`}
        >
          {sortMode === 'mostComments' ? 'Default order' : 'Sort by most comments'}
        </button>
        <button
          type="button"
          onClick={() => {
            setSortMode((m) => {
              if (m === 'avgWordsAsc') return 'avgWordsDesc';
              if (m === 'avgWordsDesc') return 'default';
              return 'avgWordsAsc';
            });
          }}
          className={`px-3 py-1.5 text-sm rounded-md border transition-colors ${
            sortMode === 'avgWordsAsc' || sortMode === 'avgWordsDesc'
              ? 'bg-blue-600 text-white border-blue-600'
              : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
          }`}
        >
          {sortMode === 'avgWordsAsc'
            ? 'Sort by avg words (desc)'
            : sortMode === 'avgWordsDesc'
              ? 'Default order'
              : 'Sort by avg words (asc)'}
        </button>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
        {sortedList.map((filename) => {
          const imageId = filename.replace(/\.png$/i, '');
          const pngPath = articlesIndex?.[imageId]?.pngPath ?? `/visualizations/${filename}`;
          // Only log for the first image to avoid noise.
          if (imageId === '1') {
            // #region agent log
            fetch('http://127.0.0.1:7243/ingest/188396c9-e207-4905-baa6-36680b519627', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                id: `log_${Date.now()}_gallery_tile`,
                timestamp: Date.now(),
                runId: 'pre-fix',
                hypothesisId: 'H2',
                location: 'Gallery.tsx:tile-render',
                message: 'Rendering gallery tile',
                data: {
                  filename,
                  imageId,
                  pngPathFromIndex: articlesIndex?.[imageId]?.pngPath ?? null,
                  resolvedSrc: pngPath,
                },
              }),
            }).catch(() => {});
            // #endregion
          }
          return (
            <div
              key={filename}
              className="rounded-lg border border-gray-200 bg-white overflow-hidden shadow-sm hover:shadow-md hover:border-blue-300 transition-shadow"
            >
              <Link to={`/gallery/${imageId}`} className="block">
                <div className="aspect-square bg-gray-100 flex items-center justify-center p-1">
                  <img
                    src={pngPath}
                    alt={filename}
                    className="max-w-full max-h-full object-contain"
                    loading="lazy"
                  />
                </div>
                <div className="px-2 py-1.5 text-xs text-gray-500">
                  <span className="block truncate">{filename}</span>
                  {countsByImageId[imageId] != null ? (
                    <>
                      <span className="text-gray-400 block mt-0.5">
                        {countsByImageId[imageId]!.topLevel} comments
                        {countsByImageId[imageId]!.total !== countsByImageId[imageId]!.topLevel && (
                          <> ({countsByImageId[imageId]!.total} with replies)</>
                        )}
                      </span>
                      <span className="text-gray-400 block mt-0.5">
                        Avg: {Math.round(countsByImageId[imageId]!.avgWordCount)} words/comment (excl. replies)
                      </span>
                    </>
                  ) : countsByImageId[imageId] === null ? (
                    <span className="text-gray-400">—</span>
                  ) : (
                    <span className="text-gray-400">…</span>
                  )}
                </div>
              </Link>
              {articlesIndex?.[imageId]?.articleUrl && (
                <div className="px-2 pb-1.5">
                  <a
                    href={articlesIndex[imageId].articleUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-blue-600 hover:underline"
                  >
                    View article →
                  </a>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
