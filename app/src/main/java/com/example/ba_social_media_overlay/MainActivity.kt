package com.example.ba_social_media_overlay

import android.annotation.SuppressLint
import android.os.Bundle
import android.view.ViewGroup
import android.view.WindowManager
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
        // Force hardware acceleration for WebGL performance
        window.setFlags(
            WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED,
            WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED
        )
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

                settings.apply {
                    javaScriptEnabled = true
                    domStorageEnabled = true
                    mediaPlaybackRequiresUserGesture = false
                    mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
                    // Standard mobile optimizations
                    useWideViewPort = true
                    loadWithOverviewMode = true
                }

                webChromeClient = WebChromeClient()
                webViewClient = object : WebViewClient() {
                    override fun onPageFinished(view: WebView?, url: String?) {
                        super.onPageFinished(view, url)
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