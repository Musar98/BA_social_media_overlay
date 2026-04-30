package com.example.ba_social_media_overlay.webview

import android.net.Uri
import android.webkit.WebResourceRequest
import android.webkit.WebView
import com.example.ba_social_media_overlay.assets.AssetRouting
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class WebViewClientFactoryTest {

    private lateinit var assetRouting: AssetRouting
    private lateinit var webView: WebView
    private lateinit var request: WebResourceRequest

    @Before
    fun setUp() {
        assetRouting = mockk(relaxed = true)
        webView = mockk(relaxed = true)
        request = mockk()
    }

    @Test
    fun `shouldInterceptRequest intercepts ONNX requests`() {
        val client = WebViewClientFactory.create(assetRouting)
        val uri = mockk<Uri>()
        every { request.url } returns uri
        every { uri.path } returns "/_onnx/test.js"

        client.shouldInterceptRequest(webView, request)

        verify { assetRouting.handleOnnx("/_onnx/test.js") }
    }

    @Test
    fun `shouldInterceptRequest delegates to assetLoader`() {
        val client = WebViewClientFactory.create(assetRouting)
        val uri = mockk<Uri>()
        every { request.url } returns uri
        every { uri.path } returns "/other/path"

        client.shouldInterceptRequest(webView, request)

        verify { assetRouting.assetLoader.shouldInterceptRequest(uri) }
    }
}