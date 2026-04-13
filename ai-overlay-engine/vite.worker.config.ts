import { defineConfig } from "vite";

export default defineConfig({
  build: {
    lib: {
      entry: "src/worker/aiWorker.ts",
      name: "AIWorker",
      formats: ["iife"],
      fileName: () => "aiWorker.js",
    },
    outDir: "../app/src/main/assets/workers",
    emptyOutDir: false,
  },
  define: { "import.meta": "{}" },
});
