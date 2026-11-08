import { defineConfig } from 'vite';

export default defineConfig({
  optimizeDeps: { exclude: ['occt-wasm'] },
  worker: { format: 'es' },
  build: { target: 'esnext' },
});
