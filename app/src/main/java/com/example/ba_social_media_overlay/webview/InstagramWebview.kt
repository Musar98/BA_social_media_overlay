package com.example.ba_social_media_overlay.webview

import android.annotation.SuppressLint
import android.os.Handler
import android.os.Looper
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView

@SuppressLint("SetJavaScriptEnabled")
@Composable
fun InstagramWebView() {
    var webViewGeneration by remember { mutableStateOf(0) }

    key(webViewGeneration) {
        AndroidView(
            modifier = Modifier
                .fillMaxSize()
                .systemBarsPadding(),
            factory = { context ->
                val mainHandler = Handler(Looper.getMainLooper())

                WebViewFactory.create(
                    context = context,
                    onRenderProcessGone = { _, _ ->
                        mainHandler.post {
                            webViewGeneration += 1
                        }
                    }
                )
            }
        )
    }
}
