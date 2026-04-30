package com.example.ba_social_media_overlay.webview

import android.webkit.WebSettings
import android.webkit.WebView
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class WebViewSettingsTest {

    private lateinit var webView: WebView
    private lateinit var webSettings: WebSettings

    @Before
    fun setUp() {
        webView = mockk(relaxed = true)
        webSettings = mockk(relaxed = true)
        every { webView.settings } returns webSettings
    }

    @Test
    fun `apply sets javascript enabled`() {
        WebViewSettings.apply(webView)

        verify { webSettings.javaScriptEnabled = true }
    }

    @Test
    fun `apply sets user agent`() {
        WebViewSettings.apply(webView)

        verify { webSettings.userAgentString = any() }
    }

    @Test
    fun `apply sets dom storage enabled`() {
        WebViewSettings.apply(webView)

        verify { webSettings.domStorageEnabled = true }
    }
}