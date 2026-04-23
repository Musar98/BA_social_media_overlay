package com.example.ba_social_media_overlay.assets


import android.content.Context
import android.webkit.WebResourceResponse
import androidx.webkit.WebViewAssetLoader
import java.io.InputStream

class AssetRouting(private val context: Context) {

    companion object {
        private const val WORKER_PREFIX =
            "/static_resources/webworker_v1/init_script/"
    }

    val assetLoader: WebViewAssetLoader = WebViewAssetLoader.Builder()
        .setDomain("www.instagram.com")
        .addPathHandler("/_onnx/", WebViewAssetLoader.AssetsPathHandler(context))
        .addPathHandler(WORKER_PREFIX) { path ->
            loadWorker(path)
        }
        .build()

    fun handleOnnx(path: String): WebResourceResponse? {
        return try {
            val assetPath = path.removePrefix("/_onnx/")
            val input: InputStream = context.assets.open("onnx/$assetPath")

            val mime = when {
                assetPath.endsWith(".js") || assetPath.endsWith(".mjs") ->
                    "application/javascript"
                assetPath.endsWith(".wasm") ->
                    "application/wasm"
                else -> "application/octet-stream"
            }

            WebResourceResponse(mime, "UTF-8", input)
        } catch (_: Exception) {
            null
        }
    }

    private fun loadWorker(path: String): WebResourceResponse? {
        return try {
            val assetPath = path.removePrefix(WORKER_PREFIX)
            val input = context.assets.open("workers/$assetPath")

            WebResourceResponse(
                "application/javascript",
                "UTF-8",
                input
            )
        } catch (_: Exception) {
            null
        }
    }
}