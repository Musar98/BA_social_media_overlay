package com.example.ba_social_media_overlay.webview


import android.content.Context
import android.util.Log
import android.view.ViewGroup
import android.webkit.CookieManager
import android.webkit.RenderProcessGoneDetail
import android.webkit.WebChromeClient
import android.webkit.WebView
import com.example.ba_social_media_overlay.assets.AssetRouting

object WebViewFactory {

    private const val INSTAGRAM_URL = "https://www.instagram.com"

    fun create(
        context: Context,
        onRenderProcessGone: ((WebView, RenderProcessGoneDetail) -> Unit)? = null
    ): WebView {
        val assetRouting = AssetRouting(context)

        return WebView(context).apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )

            WebViewRegistry.register(this)
            Log.i("InstagramWebView", "Creating WebView for Instagram")

            setupCookies()
            WebViewSettings.apply(this)

            webChromeClient = WebChromeClient()
            webViewClient = WebViewClientFactory.create(assetRouting, onRenderProcessGone)

            loadUrl(INSTAGRAM_URL)
        }
    }

    fun dispose(webView: WebView) {
        WebViewRegistry.unregister(webView)

        try {
            webView.stopLoading()
            webView.loadUrl("about:blank")
            webView.clearHistory()
        } catch (err: Throwable) {
            Log.e("InstagramWebView", "Failed to blank WebView before disposal", err)
        }

        disposeCrashedWebView(webView)
    }

    fun disposeCrashedWebView(webView: WebView) {
        WebViewRegistry.disposeTrackedWebView(webView, "direct-dispose")
    }

    private fun WebView.setupCookies() {
        CookieManager.getInstance().apply {
            setAcceptCookie(true)
            setAcceptThirdPartyCookies(this@setupCookies, true)
        }
    }
}
