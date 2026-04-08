// vite.worker.config.ts
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    lib: {
      entry: "src/ai/aiWorker.ts",
      name: "AIWorker", // 👈 ADD THIS
      formats: ["iife"],
      fileName: () => "aiWorker.js",
    },
    outDir: "../app/src/main/assets/workers",
    emptyOutDir: false,
  },
  define: { "import.meta": "{}" },
});
