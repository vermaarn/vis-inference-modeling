import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({
    base: '/',
    plugins: [react()],
    build: {
        rollupOptions: {
            output: {
                // Inline dynamic imports (e.g. mermaid's flowDiagram) so no separate chunk is
                // requested at runtime; avoids "Failed to fetch dynamically imported module"
                // on GitHub Pages (stale hashes or path resolution).
                inlineDynamicImports: true,
            },
        },
    },
});
