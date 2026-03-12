import { Routes, Route, Outlet } from 'react-router-dom';
import { Header } from './components/Header';
import { Gallery } from './pages/Gallery';
import { ImageDetail } from './pages/ImageDetail';
import { UmapScatter } from './pages/UmapScatter';

function Layout() {
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Header />
      <main className="flex-1">
        <Outlet />
      </main>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Gallery />} />
        <Route path="gallery" element={<Gallery />} />
        <Route path="gallery/:imageId" element={<ImageDetail />} />
        <Route path="umap" element={<UmapScatter />} />
      </Route>
    </Routes>
  );
}
