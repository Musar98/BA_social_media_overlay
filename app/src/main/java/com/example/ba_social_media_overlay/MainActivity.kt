package com.example.ba_social_media_overlay

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.example.ba_social_media_overlay.ui.theme.BA_social_media_overlayTheme
import com.example.ba_social_media_overlay.webview.InstagramWebView

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        WebAppBootstrap.init(this)

        setContent {
            BA_social_media_overlayTheme {
                InstagramWebView()
            }
        }
    }
}