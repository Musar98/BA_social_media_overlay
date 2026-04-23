package com.example.ba_social_media_overlay.webview


import android.content.Context
import android.view.ViewGroup
import android.webkit.CookieManager
import android.webkit.WebChromeClient
import android.webkit.WebView
import com.example.ba_social_media_overlay.assets.AssetRouting

object WebViewFactory {

    private const val INSTAGRAM_URL = "https://www.instagram.com"

    fun create(context: Context): WebView {
        val assetRouting = AssetRouting(context)

        return WebView(context).apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )

            setupCookies()
            WebViewSettings.apply(this)

            webChromeClient = WebChromeClient()
            webViewClient = WebViewClientFactory.create(assetRouting)

            loadUrl(INSTAGRAM_URL)
        }
    }

    private fun WebView.setupCookies() {
        CookieManager.getInstance().apply {
            setAcceptCookie(true)
            setAcceptThirdPartyCookies(this@setupCookies, true)
        }
    }
}