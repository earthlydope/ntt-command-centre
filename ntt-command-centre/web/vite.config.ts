import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API half runs on 127.0.0.1:8808 (CONTRACT §3). Everything the page shows
// is fetched through this proxy, so the browser only ever sees same-origin
// relative URLs: no base-URL constant to get wrong, no CORS to configure, and a
// production reverse proxy reproduces the dev topology exactly.
//
// Port 8808 rather than 8801 only because an unrelated FastAPI process of the
// user's was already holding 8801 on this machine; `run.sh` starts uvicorn on
// the same 8808 and the two numbers must be changed together.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5178,
    strictPort: false,
    proxy: {
      "/api": { target: "http://127.0.0.1:8808", changeOrigin: true },
      "/card": { target: "http://127.0.0.1:8808", changeOrigin: true },
      "/healthz": { target: "http://127.0.0.1:8808", changeOrigin: true },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});
