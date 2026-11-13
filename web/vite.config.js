import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// The browser only ever talks to this one origin; /api/* is forwarded to Express.
// One origin means no CORS setup, and the same relative URLs work in production
// behind a reverse proxy (docker-compose, Phase 6).
const BACKEND = process.env.BACKEND_URL ?? 'http://127.0.0.1:3001';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': { target: BACKEND, changeOrigin: true } },
  },
  test: {
    environment: 'jsdom',
  },
});
