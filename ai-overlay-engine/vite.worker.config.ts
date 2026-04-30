import { defineConfig } from "vite";

export default defineConfig({
  build: {
    lib: {
      entry: "src/worker/aiWorker.ts",
      name: "AIWorker",
      fileName: "ai-worker",
      formats: ["iife"],
    },
    outDir: "../app/src/main/assets/workers",
    emptyOutDir: true,
  },
  define: { "import.meta": "{}" },
});
