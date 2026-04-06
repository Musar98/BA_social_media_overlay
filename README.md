Here’s a clean, professional README you can drop into your repo. I kept it developer-focused and clear about architecture, injection flow, and the separate bundling pipeline.

---

# 📱 Instagram AI Overlay – Android WebView Injection

This project demonstrates how to inject a custom AI-powered overlay engine into Instagram using an Android `WebView`.

It consists of two main parts:

1. **Android host app (Kotlin / Jetpack Compose)**
2. **AI Overlay Engine (TypeScript + Vite, bundled into a single JS file)**

---

## 🧠 Overview

The app loads Instagram inside a `WebView` and injects a custom JavaScript bundle into the page at runtime.

The injected script:

* Observes Instagram video elements
* Runs AI-based predictions (via ONNX/WASM)
* Dynamically applies visual transformations to videos (e.g., filters, tone adjustments, effects)

---

## 🏗 Architecture

```
Android App (Kotlin)
│
├── WebView (Instagram)
│   ├── Loads https://www.instagram.com
│   ├── Intercepts requests for local AI assets
│   └── Injects AI Overlay Engine (JS)
│
├── Asset Loader (/_onnx/)
│   ├── Serves ONNX / WASM / JS files from assets/
│
└── AI Overlay Engine (IIFE bundle)
    ├── Built with Vite (TypeScript)
    ├── Runs inside Instagram DOM
    └── Applies AI-driven video transformations
```

---

## ⚙️ Android Implementation

### Key Features

* Fullscreen `WebView` using Jetpack Compose
* Hardware acceleration enabled
* Custom user agent (to mimic Chrome mobile)
* Cookie + third-party cookie support
* JavaScript + DOM storage enabled

---

### 🔌 Script Injection Flow

The injection happens after each page load:

```kotlin
override fun onPageFinished(view: WebView?, url: String?) {
    if (url != null && url != lastInjectedUrl) {
        lastInjectedUrl = url
        injectAiOverlayEngine(this@apply)
    }
}
```

The engine is loaded from app assets:

```kotlin
val bundleScript = webView.context.assets
    .open("ai-overlay-engine/ai-overlay-engine.iife.js")
    .bufferedReader()
    .use { it.readText() }

webView.evaluateJavascript(bundleScript, null)
```

---

### 📦 Local Asset Serving (ONNX / WASM)

We expose local AI model files under:

```
https://www.instagram.com/_onnx/...
```

Handled via:

```kotlin
shouldInterceptRequest(...)
```

This allows the injected script to fetch models as if they were hosted on Instagram.

Supported types:

* `.js`, `.mjs`
* `.wasm`
* fallback: binary

---

## 🤖 AI Overlay Engine

The overlay engine is a **separate project** built with:

* TypeScript
* Vite

### Responsibilities

* Detect Instagram video elements
* Extract video frames or metadata
* Run AI inference (via ONNX Runtime / WASM)
* Predict visual parameters (e.g., color grading, effects)
* Apply transformations in real-time

---

### 📦 Build Output

The project is bundled into a **single IIFE file**:

```
ai-overlay-engine.iife.js
```

Why IIFE:

* No module loader required
* Runs immediately when injected
* Avoids conflicts with Instagram’s JS environment

---

## 🔄 End-to-End Flow

1. App launches
2. WebView loads Instagram
3. Page finishes loading
4. Kotlin injects `ai-overlay-engine.iife.js`
5. Script initializes inside DOM
6. Script:
    * Finds video elements
    * Loads ONNX/WASM models via `/_onnx/`
    * Runs inference
    * Applies visual transformations

---

## 📁 Project Structure

```
android-app/
├── app/src/main/MainActivity.kt
├── assets/
│   ├── ai-overlay-engine/
│   │   └── ai-overlay-engine.iife.js
│   └── onnx/
│       ├── models/*.onnx (model)
│       └── runtime.wasm (ort.min.js ...)
ai-overlay-engine/ (separate ts module)
├── src/ (package.json , vite.config.ts ...)
├── src/ai
├── src/renderer
├── src/state
├── src/transformations
├── src/ui
├── src/video
```

---

## ⚠️ Notes & Limitations

* Instagram DOM is **not stable** → selectors may break
* WebView behavior may differ from Chrome
* Performance depends on device GPU/CPU
* AI inference in-browser (WASM) can be heavy
* Injection may need re-triggering on navigation

---

## 🚀 Future Improvements

* MutationObserver for dynamic content
* Better video tracking (Reels, Stories)
* WebGL acceleration for effects
* Model optimization (quantization)
* Smarter caching of ONNX assets

---

## 🧩 Key Concepts

* WebView JavaScript injection
* Asset interception via `WebViewAssetLoader`
* IIFE bundling for sandboxed environments
* Client-side AI inference (ONNX + WASM)

---

If you want, I can also:

* Add diagrams (sequence / lifecycle)
* Write the TypeScript side README
* Or document the AI pipeline (model → inference → effect mapping)
