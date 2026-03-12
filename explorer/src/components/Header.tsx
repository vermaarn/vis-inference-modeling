import { Link, useLocation } from 'react-router-dom';

export function Header() {
  const location = useLocation();
  const isGallery = location.pathname === '/' || location.pathname.startsWith('/gallery');
  const isUmap = location.pathname === '/umap';

  return (
    <header className="border-b border-gray-200 bg-white px-4 py-3 shadow-sm">
      <nav className="flex items-center gap-6">
        <Link
          to="/"
          className={`text-sm font-medium ${isGallery ? 'text-blue-600' : 'text-gray-600 hover:text-gray-900'}`}
        >
          Gallery
        </Link>
        <Link
          to="/umap"
          className={`text-sm font-medium ${isUmap ? 'text-blue-600' : 'text-gray-600 hover:text-gray-900'}`}
        >
          UMAP Scatter
        </Link>
      </nav>
    </header>
  );
}
