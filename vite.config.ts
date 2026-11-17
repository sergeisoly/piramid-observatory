import { defineConfig } from 'vite';
import { updateGate } from './dev/update-gate.ts';

export default defineConfig({
  // Relative asset paths: the built film runs from any folder of a site, for example
  // https://<user>.github.io/piramid-observatory/, and from the root of a local preview.
  base: './',
  plugins: [updateGate()],
  // three's addons import the core build by the bare name "three"; point it at the WebGPU build
  // so only one copy of the library is loaded.
  resolve: { alias: [{ find: /^three$/, replacement: 'three/webgpu' }] },
  server: { port: 5199, strictPort: true, host: '127.0.0.1' },
  preview: { port: 5199, strictPort: true, host: '127.0.0.1' },
  build: { target: 'esnext', chunkSizeWarningLimit: 4000 },
});
