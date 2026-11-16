import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Las peticiones /api se redirigen al backend FastAPI (uvicorn en :8000).
// Se usa 127.0.0.1 y no "localhost": en Windows, Node resuelve localhost
// primero a IPv6 (::1) y uvicorn escucha solo en IPv4, lo que añade latencia.
const proxy = { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } };

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
  // Pre-empaqueta las dependencias pesadas al arrancar `npm run dev`
  // (evita la recarga completa la primera vez que se abre cada pestaña).
  optimizeDeps: {
    include: ["react", "react-dom/client", "framer-motion", "echarts/core", "echarts/charts", "echarts/components", "echarts/renderers", "echarts-for-react/lib/core", "katex", "lucide-react"],
  },
  build: {
    target: "es2020",
    rollupOptions: {
      output: {
        // Librerías en archivos separados: el navegador las cachea entre visitas.
        manualChunks: {
          react: ["react", "react-dom"],
          motion: ["framer-motion"],
          echarts: ["echarts/core", "echarts/charts", "echarts/components", "echarts/renderers", "echarts-for-react/lib/core"],
          katex: ["katex"],
        },
      },
    },
  },
});
