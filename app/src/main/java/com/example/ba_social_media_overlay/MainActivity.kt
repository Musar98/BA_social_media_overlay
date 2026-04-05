package com.example.ba_social_media_overlay

import android.annotation.SuppressLint
import android.os.Bundle
import android.view.View.LAYER_TYPE_HARDWARE
import android.view.ViewGroup
import android.view.WindowManager
import android.webkit.CookieManager
import android.webkit.WebChromeClient
import android.webkit.WebResourceResponse
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
import androidx.webkit.WebViewAssetLoader
import com.example.ba_social_media_overlay.ui.theme.BA_social_media_overlayTheme
import java.io.InputStream

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
        modifier = Modifier
            .fillMaxSize()
            .systemBarsPadding(),
        factory = { context ->
            val loader = WebViewAssetLoader.Builder()
                .setDomain("www.instagram.com")
                .addPathHandler("/_onnx/", WebViewAssetLoader.AssetsPathHandler(context))
                .build()

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
                    mediaPlaybackRequiresUserGesture = false
                    setLayerType(LAYER_TYPE_HARDWARE, null)
                    mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
                    useWideViewPort = true
                    loadWithOverviewMode = true
                    userAgentString =
                        "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Mobile Safari/537.36"
                }

                webChromeClient = WebChromeClient()
                webViewClient = object : WebViewClient() {
                    override fun shouldInterceptRequest(
                        view: WebView,
                        request: android.webkit.WebResourceRequest
                    ): WebResourceResponse? {
                        val url = request.url
                        if (url.path?.startsWith("/_onnx/") == true) {
                            val assetPath = url.path!!.removePrefix("/_onnx/")
                            return try {
                                val input: InputStream = context.assets.open("onnx/$assetPath")
                                val mime = when {
                                    assetPath.endsWith(".js") -> "application/javascript"
                                    assetPath.endsWith(".mjs") -> "application/javascript"
                                    assetPath.endsWith(".wasm") -> "application/wasm"
                                    else -> "application/octet-stream"
                                }
                                WebResourceResponse(mime, "UTF-8", input)
                            } catch (_: Exception) {
                                null
                            }
                        }
                        return loader.shouldInterceptRequest(url)
                    }

                    private var lastInjectedUrl: String? = null

                    override fun onPageFinished(view: WebView?, url: String?) {
                        super.onPageFinished(view, url)

                        if (url != null && url != lastInjectedUrl) {
                            lastInjectedUrl = url
                            injectAiOverlayEngine(this@apply)
                        }
                    }
                }

                loadUrl("https://www.instagram.com")
            }
        }
    )
}

private fun injectAiOverlayEngine(webView: WebView) {
    try {
        val bundleScript = webView.context.assets.open("ai-overlay-engine/ai-overlay-engine.iife.js")
            .bufferedReader()
            .use { it.readText() }

        webView.evaluateJavascript(bundleScript, null)
    } catch (e: Exception) {
        e.printStackTrace()
    }
}