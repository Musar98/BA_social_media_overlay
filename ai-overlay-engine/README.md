# AI Overlay Engine (TS Module)

This TypeScript module handles AI parameter prediction and visual transformations for Instagram
videos within an Android WebView.

## Overview

The engine acts as a bridge between a raw Instagram video feed and an ONNX-based AI model. It
performs the following:

1. **Observation**: Uses a `MutationObserver` to detect `<video>` elements on Instagram.
2. **Prediction**: Captures video frames and runs them through a local ONNX model (
   `parametricmodel.pt.dyn.onnx`) via *
   *ONNX Runtime Web**.
3. **Transformation**: Generates dynamic parameters (sharpness, exposure, tone curves, etc.) and
   applies them to a
   canvas overlay using a custom WebGL/Canvas pipeline. The the CLIP shader is found in `src/transformation/ClipToUv.ts` and the fragment shader in `src/transformation/ImageAdjustmentsPipeline.ts`. A more detailed description can be found under `src/transformation/shader.md`

## Tech Stack

* **TypeScript**: Source logic and type safety for ONNX tensors.
* **Vite**: Used for bundling.
* **ONNX Runtime Web**: For client-side inference using WebAssembly (WASM).
* **NPM**: For dependency management and scripts.

For more information see `package.json`and `package-lock.json`.

# Android app integration

The related android application is configured (`app/build.gradle.kts`) to install node, the
npm packages, run the tests and build the `ai-overlay-engine` at build time.

# Development Workflow

The `ai-overlay-engine` can be developed and built in isolation using its npm scripts.  
However, it does not run as a standalone application and is only executed within the Android app.
The following instructions are only relevant for isolated development.

### Prerequisites

Ensure you have [Node.js](https://nodejs.org/) installed.
(used, and therefore recommended: v22.14.0 | part of
lts/jod - [Node 22.14.0](https://github.com/nodejs/nodejs.org/blob/main/apps/site/pages/en/blog/release/v22.14.0.md?utm_source=chatgpt.com)
md)

## Install Packages

This project uses npm as its package manager.
To install the packages, simply run:

```bash
npm i
```

within the `ai-overlay-engine` directory.

## Building and Testing

### Build Command

The build process is configured to output the bundled JavaScript files directly into the Android
project's
assets folder. (app/src/main/assets)
The build produces 2 seperate js files in 2 directories: \
`assets/ai-overlay-engine/ai-overlay-engine.iife.js` \
`assets/workers/ai-worker.iife.js` \
To build the module, run the following command inside the `ai-overlay-engine` directory:

```bash
npm run build
```

This will always overwrite all existing files in the `assets/ai-overlay-engine` directory.
If you would like to add files there and preserve them, you would need to change
the value of the `emptyOutDir` property in the
`vite.main.config.ts` file to `false` in order to prevent a wipe of the directory
`assets/ai-overlay-engine` on each build.

## Testing

This project uses **[Vitest](https://vitest.dev/)** for unit testing.

#### Running tests

```bash
npm run test
```
