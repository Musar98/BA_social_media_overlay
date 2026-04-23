package com.example.ba_social_media_overlay.webview

import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import com.example.ba_social_media_overlay.assets.AiOverlayInjector
import com.example.ba_social_media_overlay.assets.AssetRouting

object WebViewClientFactory {

    private const val ONNX_PREFIX = "/_onnx/"

    fun create(
        assetRouting: AssetRouting
    ): WebViewClient {

        return object : WebViewClient() {

            private var lastInjectedUrl: String? = null

            override fun shouldInterceptRequest(
                view: WebView,
                request: WebResourceRequest
            ): WebResourceResponse? {

                val url = request.url

                url.path?.let { path ->
                    if (path.startsWith(ONNX_PREFIX)) {
                        return assetRouting.handleOnnx(path)
                    }
                }

                return assetRouting.assetLoader.shouldInterceptRequest(url)
            }

            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)

                if (url != null && url != lastInjectedUrl) {
                    lastInjectedUrl = url
                    AiOverlayInjector.inject(view)
                }
            }
        }
    }
}