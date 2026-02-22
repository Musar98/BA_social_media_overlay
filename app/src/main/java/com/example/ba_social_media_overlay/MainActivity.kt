package com.example.ba_social_media_overlay

import android.opengl.GLSurfaceView
import android.os.Bundle
import android.view.Surface
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.common.MediaItem
import androidx.media3.exoplayer.ExoPlayer
import com.example.ba_social_media_overlay.ui.theme.BA_social_media_overlayTheme

class MainActivity : ComponentActivity() {

    private lateinit var player: ExoPlayer

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        player = ExoPlayer.Builder(this).build()

        //TODO this loads media, currently free google api media
        val mediaItem = MediaItem.fromUri(
            "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
        )

        player.setMediaItem(mediaItem)
        player.prepare()
        player.play()

        setContent {
            BA_social_media_overlayTheme {
                GPUVideoScreen(player)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        player.release()
    }
}

@Composable
fun GPUVideoScreen(player: ExoPlayer) {

    var filterEnabled by remember { mutableStateOf(true) }
    var isPlaying by remember { mutableStateOf(true) }
    val renderer = remember { SimpleVideoRenderer() }

    Box(modifier = Modifier.fillMaxSize()) {

        AndroidView(
            modifier = Modifier
                .fillMaxSize()
                .clickable {
                    if (isPlaying) player.pause() else player.play()
                    isPlaying = !isPlaying
                },
            factory = { context ->

                val glSurfaceView = GLSurfaceView(context)
                glSurfaceView.setEGLContextClientVersion(2)

                renderer.onSurfaceReady = { surface: Surface ->
                    (context as ComponentActivity).runOnUiThread {
                        player.setVideoSurface(surface)
                    }
                }

                glSurfaceView.setRenderer(renderer)
                glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY

                glSurfaceView
            },
            update = {
                renderer.setFilterEnabled(filterEnabled)
            }
        )

        Button(
            onClick = { filterEnabled = !filterEnabled },
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(24.dp)
        ) {
            Text(if (filterEnabled) "Disable Filter" else "Enable Filter")
        }
    }
}