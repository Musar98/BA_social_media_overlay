package com.example.ba_social_media_overlay.webview

import android.util.Log
import android.view.ViewGroup
import android.webkit.WebView
import java.util.Collections
import java.util.WeakHashMap

object WebViewRegistry {

    private const val TAG = "InstagramWebView"
    private val webViews = Collections.newSetFromMap(WeakHashMap<WebView, Boolean>())

    @Synchronized
    fun register(webView: WebView) {
        webViews.add(webView)
        Log.i(TAG, "WebView registered. count=${webViews.size}")
    }

    @Synchronized
    fun unregister(webView: WebView) {
        webViews.remove(webView)
        Log.i(TAG, "WebView unregistered. count=${webViews.size}")
    }

    fun disposeAll(reason: String) {
        val snapshot = synchronized(this) {
            webViews.toList().also {
                webViews.clear()
            }
        }

        Log.e(TAG, "Disposing all WebViews. reason=$reason, count=${snapshot.size}")

        snapshot.forEach { webView ->
            disposeTrackedWebView(webView, reason)
        }

        Log.e(TAG, "Disposed all WebViews. reason=$reason")
    }

    fun disposeTrackedWebView(webView: WebView, reason: String) {
        Log.i(TAG, "Disposing WebView. reason=$reason")

        synchronized(this) {
            webViews.remove(webView)
        }

        try {
            (webView.parent as? ViewGroup)?.removeView(webView)
            webView.removeAllViews()
            webView.destroy()
        } catch (err: Throwable) {
            Log.e(TAG, "Failed to dispose WebView. reason=$reason", err)
        }
    }
}
