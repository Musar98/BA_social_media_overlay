package com.example.ba_social_media_overlay

import android.annotation.SuppressLint
import android.os.Bundle
import android.view.ViewGroup
import android.view.WindowManager
import android.webkit.CookieManager
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import com.example.ba_social_media_overlay.ui.theme.BA_social_media_overlayTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        window.setFlags(
            WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED,
            WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED
        )

        WebView.setWebContentsDebuggingEnabled(true)

        setContent {
            BA_social_media_overlayTheme {
                InstagramWebView()
            }
        }
    }
}

@SuppressLint("SetJavaScriptEnabled")
@Composable
fun InstagramWebView() {
    AndroidView(
        modifier = Modifier.fillMaxSize().systemBarsPadding(),
        factory = { context ->
            WebView(context).apply {
                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )

                CookieManager.getInstance().setAcceptCookie(true)
                CookieManager.getInstance().setAcceptThirdPartyCookies(this, true)

                settings.apply {
                    javaScriptEnabled = true
                    domStorageEnabled = true
                    databaseEnabled = true
                    mediaPlaybackRequiresUserGesture = false
                    mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
                    useWideViewPort = true
                    loadWithOverviewMode = true
                    userAgentString =
                        "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Mobile Safari/537.36"
                }

                webChromeClient = WebChromeClient()
                webViewClient = object : WebViewClient() {
                    override fun onPageFinished(view: WebView?, url: String?) {
                        super.onPageFinished(view, url)
                        injectSobelJavaScript(this@apply)
                    }

                    override fun doUpdateVisitedHistory(view: WebView?, url: String?, isReload: Boolean) {
                        super.doUpdateVisitedHistory(view, url, isReload)
                        injectSobelJavaScript(this@apply)
                    }
                }

                loadUrl("https://www.instagram.com")
            }
        }
    )
}

private fun injectSobelJavaScript(webView: WebView) {
    try {
        val js = webView.context.assets.open("instagramFilter.js")
            .bufferedReader().use { it.readText() }
        webView.evaluateJavascript(js, null)
    } catch (e: Exception) {
        e.printStackTrace()
    }
}

// Optional Kotlin helpers to toggle the filter programmatically
fun WebView.enableSobel() = this.evaluateJavascript("window.enableSobelFilter();", null)
fun WebView.disableSobel() = this.evaluateJavascript("window.disableSobelFilter();", null)