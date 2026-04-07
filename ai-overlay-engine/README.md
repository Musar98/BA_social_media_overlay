# AI Video Filter Engine (TS Module)

This TypeScript module handles real-time AI parameter prediction and visual transformations for Instagram videos within
an Android WebView.

## 🚀 Overview

The engine acts as a bridge between a raw Instagram video feed and an ONNX-based AI model. It performs the following:

1. **Observation**: Uses a `MutationObserver` to detect `<video>` elements on Instagram.
2. **Prediction**: Captures video frames and runs them through a local ONNX model (`parametricmodel.pt.dyn.onnx`) via *
   *ONNX Runtime Web**.
3. **Transformation**: Generates dynamic parameters (sharpness, exposure, tone curves, etc.) and applies them to a
   canvas overlay using a custom WebGL/Canvas pipeline.

## 🛠 Tech Stack

* **TypeScript**: Source logic and type safety for ONNX tensors.
* **Vite**: Used for bundling into a single, high-performance **IIFE** (Immediately Invoked Function Expression).
* **ONNX Runtime Web**: For client-side inference using WebAssembly (WASM) / or WebGPU when available.

## 📦 Building for Android

The build process is configured to output the bundled JavaScript directly into the Android project's assets folder.
(app/src/main/assets/ai-overlay-engine)

### Prerequisites

Ensure you have [Node.js](https://nodejs.org/) installed.
(used, and therefore recommended: v22.22.1 | lts/jod)

### Build Command

Run the following command inside the `ai-overlay-engine` directory:

```bash
npm run build
```

This will always overwrite all existing files in the `assets/ai-overlay-engine` directory.
If you, for some reason, would like to add files there and preserve them, you would need to change the value of the `emptyOutDir` property in the
`vite.main.config.ts` file to `false` in order to prevent a wipe of the directory `assets/ai-overlay-engine` on each build.

### Testing

This project uses **[Vitest](https://vitest.dev/)** for unit testing.

#### Running tests

```bash
npm run test
```
