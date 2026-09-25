/** Bundles the suites with esbuild (Node cannot resolve extensionless imports) and runs `node --test`. */

import { build } from 'esbuild';
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.join(here, '.build');

rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });

const suites = [
  'algorithms.test.ts',
  'lessons.test.ts',
  'lessons-linear.test.ts',
  'lessons-polynomial.test.ts',
  'lessons-network.test.ts',
  'lessons-logistic.test.ts',
  'lessons-kmeans.test.ts',
  'lessons-tree.test.ts',
  'lessons-knn.test.ts',
  'lessons-cnn.test.ts',
  'lessons-regularization.test.ts',
  'lessons-activation.test.ts',
  'training.test.ts',
  'elasticnet.test.ts',
  'polynomial.test.ts',
  'networks.test.ts',
  'cnn.test.ts',
  'transformer.test.ts',
  'lessons-transformer.test.ts',
];

await build({
  entryPoints: suites.map((name) => path.join(here, name)),
  outdir: outDir,
  outExtension: { '.js': '.mjs' },
  bundle: true,
  platform: 'node',
  format: 'esm',
  target: 'node20',
  sourcemap: 'inline',
  external: ['node:*'],
  logLevel: 'warning',
});

const outfiles = suites.map((name) => path.join(outDir, name.replace(/\.ts$/, '.mjs')));
const child = spawn(process.execPath, ['--test', ...outfiles], { stdio: 'inherit' });
child.on('exit', (code) => process.exit(code ?? 1));
