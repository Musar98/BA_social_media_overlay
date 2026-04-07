import { defineConfig } from "vite";

export default defineConfig({
    build: {
        lib: {
            entry: "src/index.ts",
            name: "AiOverlayEngine",
            fileName: "ai-overlay-engine",
            formats: ["iife"],
        },
        outDir: "../app/src/main/assets/ai-overlay-engine",
        emptyOutDir: true,
    },
    define: { "import.meta": "{}" },
});