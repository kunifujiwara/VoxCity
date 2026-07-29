import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Backend port can be overridden via VITE_BACKEND_PORT (set by run.py).
const backendPort = process.env.VITE_BACKEND_PORT || '8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,
    allowedHosts: true,
    proxy: {
      '/api': {
        target: `http://localhost:${backendPort}`,
        changeOrigin: true,
        timeout: 600000,
      },
    },
  },
})
