package com.example.ba_social_media_overlay.assets


import android.webkit.WebView

object AiOverlayInjector {

    fun inject(webView: WebView?) {
        if (webView == null) return

        try {
            val script = webView.context.assets
                .open("ai-overlay-engine/ai-overlay-engine.iife.js")
                .bufferedReader()
                .use { it.readText() }

            webView.evaluateJavascript(script, null)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}