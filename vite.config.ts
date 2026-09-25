import { defineConfig } from 'vite';
import type { Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';

import { READY_EXPLAINERS } from './src/explainers/registry';

// GitHub Pages serves from /<repo>/; set VITE_BASE=/ for a root deployment.
const base = process.env.VITE_BASE ?? '/interactive-ml/';
// The public address, for link previews and the sitemap.
const site = (process.env.VITE_SITE_URL ?? 'https://nbharathik.github.io' + base).replace(/\/?$/, '/');
// Unpublished pages are routed on the dev server, and in a build only with VITE_PREVIEW=1.
const preview = process.env.VITE_PREVIEW === '1';

const escape = (text: string) => text.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/** A catalog summary's first sentence, short enough for a search snippet or a link preview. */
function brief(text: string): string {
  const first = text.split(/(?<=[.!?])\s/)[0];
  return first.length <= 200 ? first : first.slice(0, 197).replace(/\s+\S*$/, '') + '...';
}

/** Fills the site address into index.html's link-preview tags. */
function siteMeta(): Plugin {
  return {
    name: 'mlx-site-meta',
    transformIndexHtml: (html) => html.replace(/__MLX_SITE__/g, site),
  };
}

/** One HTML file per explainer with its own title and preview, a 404 bounce and a sitemap. */
function staticPages(): Plugin {
  let outDir = 'dist';
  return {
    name: 'mlx-static-pages',
    apply: 'build',
    configResolved(config) {
      outDir = path.resolve(config.root, config.build.outDir);
    },
    closeBundle() {
      const index = readFileSync(path.join(outDir, 'index.html'), 'utf8');
      const pages = [
        { route: 'about', title: 'About', description: null },
        ...READY_EXPLAINERS.map((meta) => ({ route: 'explainer/' + meta.slug, title: meta.title, description: brief(meta.summary || meta.tagline) })),
      ];
      mkdirSync(path.join(outDir, 'explainer'), { recursive: true });
      for (const page of pages) {
        const title = escape(page.title + ' · Interactive ML');
        const url = site + page.route;
        let html = index
          .replace(/<title>[^<]*<\/title>/, '<title>' + title + '</title>')
          .replace(/(<meta (?:property="og:title"|name="twitter:title") content=")[^"]*"/g, '$1' + title + '"')
          .replace(/(<meta property="og:url" content=")[^"]*"/, '$1' + url + '"')
          .replace(/(<link rel="canonical" href=")[^"]*"/, '$1' + url + '"');
        if (page.description) {
          const description = escape(page.description);
          html = html.replace(/(<meta (?:name="description"|property="og:description"|name="twitter:description") content=")[^"]*"/g, '$1' + description + '"');
        }
        // GitHub Pages serves explainer/knn.html at explainer/knn, so deep links load with a 200.
        writeFileSync(path.join(outDir, page.route + '.html'), html);
      }
      // Unknown paths come back through index.html, which restores them for the router.
      writeFileSync(
        path.join(outDir, '404.html'),
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8" />\n<title>Interactive ML</title>\n<script>\n' +
          "try { sessionStorage.setItem('mlx-redirect', location.pathname + location.search + location.hash); } catch (e) {}\n" +
          'location.replace(' + JSON.stringify(base) + ');\n</script>\n</head>\n<body></body>\n</html>\n',
      );
      const urls = ['', 'about', ...READY_EXPLAINERS.map((meta) => 'explainer/' + meta.slug)];
      writeFileSync(
        path.join(outDir, 'sitemap.xml'),
        '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' +
          urls.map((u) => '  <url><loc>' + site + u + '</loc></url>\n').join('') +
          '</urlset>\n',
      );
    },
  };
}

/** Tells index.html which chunks each explainer needs, so a deep link can preload them before the app boots. */
function explainerPreloads(): Plugin {
  return {
    name: 'mlx-explainer-preloads',
    transformIndexHtml: {
      order: 'post',
      handler(html, ctx) {
        const bundle = ctx.bundle;
        if (!bundle) return html.replace('__MLX_PRELOAD__', 'null');
        const chunks: Record<string, string[]> = {};
        for (const output of Object.values(bundle)) {
          if (output.type !== 'chunk') continue;
          const match = /explainers[\\/]([^\\/]+)[\\/]Page\.tsx$/.exec(output.facadeModuleId ?? '');
          if (!match) continue;
          const files: string[] = [];
          const visit = (name: string) => {
            if (files.includes(name)) return;
            files.push(name);
            const chunk = bundle[name];
            if (chunk?.type === 'chunk') chunk.imports.forEach(visit);
          };
          visit(output.fileName);
          chunks[match[1]] = files.filter((file) => !html.includes(file));
        }
        const fonts = Object.keys(bundle).filter((file) => /KaTeX_(Main-Regular|Math-Italic)[^/]*\.woff2$/.test(file));
        return html.replace('__MLX_PRELOAD__', JSON.stringify({ base, chunks, fonts }));
      },
    },
  };
}

export default defineConfig({
  base,
  define: { __MLX_PREVIEW__: JSON.stringify(preview) },
  plugins: [react(), siteMeta(), explainerPreloads(), staticPages()],
  server: {
    port: Number(process.env.PORT) || 5175,
    strictPort: false,
    open: false,
    // Test artefacts must not trigger HMR mid-run.
    watch: { ignored: ['**/tests/.shots/**', '**/tests/.build/**', '**/dist/**'] },
  },
  preview: { port: 4175, strictPort: false },
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  build: {
    target: 'es2022',
    sourcemap: false,
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules')) {
            if (id.includes('katex')) return 'katex';
            if (id.includes('react-router')) return 'router';
            if (id.includes('react')) return 'react';
          }
          return undefined;
        },
      },
    },
  },
});
