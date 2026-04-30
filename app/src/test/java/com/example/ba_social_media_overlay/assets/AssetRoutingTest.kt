package com.example.ba_social_media_overlay.assets

import android.content.Context
import android.content.res.AssetManager
import io.mockk.every
import io.mockk.mockk
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.ByteArrayInputStream

@RunWith(RobolectricTestRunner::class)
class AssetRoutingTest {

    private lateinit var context: Context
    private lateinit var assetManager: AssetManager
    private lateinit var assetRouting: AssetRouting

    @Before
    fun setUp() {
        context = mockk()
        assetManager = mockk()
        every { context.assets } returns assetManager
        assetRouting = AssetRouting(context)
    }

    @Test
    fun `handleOnnx maps js path correctly`() {
        val path = "/_onnx/test.js"
        val content = "console.log('test')"
        every { assetManager.open("onnx/test.js") } returns ByteArrayInputStream(content.toByteArray())

        val response = assetRouting.handleOnnx(path)

        assertNotNull(response)
        assertEquals("application/javascript", response?.mimeType)
        assertEquals("UTF-8", response?.encoding)
    }

    @Test
    fun `handleOnnx maps wasm path correctly`() {
        val path = "/_onnx/test.wasm"
        every { assetManager.open("onnx/test.wasm") } returns ByteArrayInputStream(byteArrayOf(0, 1, 2))

        val response = assetRouting.handleOnnx(path)

        assertNotNull(response)
        assertEquals("application/wasm", response?.mimeType)
    }

    @Test
    fun `handleOnnx returns null on missing asset`() {
        val path = "/_onnx/missing.js"
        every { assetManager.open("onnx/missing.js") } throws Exception("File not found")

        val response = assetRouting.handleOnnx(path)

        assertNull(response)
    }
}