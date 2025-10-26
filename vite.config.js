import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// explicitly export the config
export default defineConfig({
  plugins: [react()],
  build: {
    target: 'esnext',
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('onnxruntime-web')) {
            return 'onnxruntime-web';
          }
        },
      },
    },
  },
  server: {
    headers: {
      'Cross-Origin-Resource-Policy': 'cross-origin',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },
});
