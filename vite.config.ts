import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  optimizeDeps: { exclude: ['occt-wasm'] },
  worker: { format: 'es' },
  build: { target: 'esnext' },
});
