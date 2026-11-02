package com.hubertecg

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONArray
import java.io.File
import java.nio.FloatBuffer

/**
 * HuBERT-ECG (SFT Cardio-Learning, small) embarqué en ONNX.
 * Entrée : [1, 6000] = 12 dérivations x 500 échantillons à 100 Hz, aplaties.
 * Sortie : logits [1, 164] → sigmoïde = probabilité multi-label.
 * Prétraitement conforme au papier : passe-bande 0.05-47 Hz, resample 100 Hz,
 * fenêtre centrale 5 s, min-max [-1, 1] par dérivation.
 */
class EcgAnalyzer(private val context: Context) {

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var labels: List<String> = emptyList()

    fun isLoaded() = session != null

    fun load() {
        if (session != null) return
        val assetManager = context.assets
        labels = JSONArray(assetManager.open("models/labels.json").readBytes().toString(Charsets.UTF_8))
            .let { arr -> List(arr.length()) { arr.getString(it) } }

        val modelFile = File(context.cacheDir, "hubert-ecg-small.onnx")
        if (!modelFile.exists()) {
            assetManager.open("models/hubert-ecg-small.onnx").use { input ->
                modelFile.outputStream().use { input.copyTo(it) }
            }
        }
        env = OrtEnvironment.getEnvironment()
        session = env!!.createSession(modelFile.absolutePath, OrtSession.SessionOptions())
    }

    data class Finding(val label: String, val prob: Float)

    fun analyze(leads12: List<FloatArray>, fsIn: Int, topK: Int = 8): List<Finding> {
        val sess = session ?: throw IllegalStateException("Modèle non chargé")
        val input = preprocess(leads12, fsIn)
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 6000))
        val output = sess.run(mapOf("input_values" to tensor))
        val logits = (output[0].value as Array<FloatArray>)[0]
        output.close()
        return logits
            .mapIndexed { i, l -> Finding(labels.getOrElse(i) { "LABEL_$i" }, 1f / (1f + Math.exp(-l.toDouble()).toFloat())) }
            .sortedByDescending { it.prob }
            .take(topK)
    }

    companion object {
        const val TARGET_FS = 100
        const val WINDOW = 5 * TARGET_FS
        const val NUM_LEADS = 12

        fun preprocess(leads12: List<FloatArray>, fsIn: Int): FloatArray {
            val hp = butterHighpass(0.05, fsIn)
            val lp = butterLowpass(47.0, fsIn)
            val out = FloatArray(NUM_LEADS * WINDOW)
            for (l in 0 until NUM_LEADS) {
                val filtered = filtfilt(lp.first, lp.second, filtfilt(hp.first, hp.second, leads12[l]))
                val at100 = resampleLinear(filtered, fsIn, TARGET_FS)
                val win = cropOrPad(at100, WINDOW)
                val scaled = minMaxScale(win)
                System.arraycopy(scaled, 0, out, l * WINDOW, WINDOW)
            }
            return out
        }

        fun parseCsv12(text: String): Pair<List<FloatArray>, Int> {
            val rows = text.lineSequence()
                .map { it.trim() }
                .filter { it.isNotEmpty() && !it.startsWith("#") }
                .map { line -> line.split(Regex("[;,\\t ]+")).map { s -> s.replace(',', '.').toFloatOrNull() ?: 0f } }
                .toList()
            if (rows.isEmpty()) throw IllegalArgumentException("CSV vide")
            val width = rows.maxOf { it.size }
            val leads = (0 until minOf(NUM_LEADS, width)).map { c -> FloatArray(rows.size) { r -> rows[r].getOrElse(c) { 0f } } }
                .toMutableList()
            while (leads.size < NUM_LEADS) leads.add(FloatArray(rows.size))
            return leads to rows.size
        }

        fun estimateHeartRate(signal: FloatArray, fs: Int): Int? {
            if (signal.size < fs * 5) return null
            val mean = signal.average().toFloat()
            val sd = Math.sqrt(signal.map { (it - mean) * (it - mean) }.average()).toFloat()
            val threshold = mean + 0.6f * sd
            val refractory = (fs * 0.3).toInt()
            val peaks = mutableListOf<Int>()
            for (i in 1 until signal.size - 1) {
                if (signal[i] > threshold && signal[i] >= signal[i - 1] && signal[i] >= signal[i + 1]) {
                    if (peaks.isEmpty() || i - peaks.last() > refractory) peaks.add(i)
                }
            }
            if (peaks.size < 3) return null
            val rr = peaks.zipWithNext { a, b -> (b - a).toFloat() / fs * 1000f }
            return (60000f / (rr.average().toFloat())).toInt()
        }

        private fun butterLowpass(fc: Double, fs: Int): Pair<DoubleArray, DoubleArray> {
            val w = 2 * Math.PI * fc / fs
            val cw = Math.cos(w); val sw = Math.sin(w)
            val alpha = sw / Math.sqrt(2.0)
            val a0 = 1 + alpha
            return doubleArrayOf((1 - cw) / 2 / a0, (1 - cw) / a0, (1 - cw) / 2 / a0) to
                doubleArrayOf(1.0, -2 * cw / a0, (1 - alpha) / a0)
        }

        private fun butterHighpass(fc: Double, fs: Int): Pair<DoubleArray, DoubleArray> {
            val w = 2 * Math.PI * fc / fs
            val cw = Math.cos(w); val sw = Math.sin(w)
            val alpha = sw / Math.sqrt(2.0)
            val a0 = 1 + alpha
            return doubleArrayOf((1 + cw) / 2 / a0, -(1 + cw) / a0, (1 + cw) / 2 / a0) to
                doubleArrayOf(1.0, -2 * cw / a0, (1 - alpha) / a0)
        }

        private fun filtfilt(b: DoubleArray, a: DoubleArray, x: FloatArray): FloatArray {
            val fwd = directFilter(b, a, x)
            val rev = directFilter(b, a, fwd.reversedArray())
            return rev.reversedArray()
        }

        private fun directFilter(b: DoubleArray, a: DoubleArray, x: FloatArray): FloatArray {
            val n = x.size
            val y = FloatArray(n)
            val z = DoubleArray(a.size - 1)
            val bb = b.map { it / a[0] }.toDoubleArray()
            val aa = a.map { it / a[0] }.toDoubleArray()
            for (i in 0 until n) {
                val xi = x[i].toDouble()
                val yi = bb[0] * xi + z[0]
                for (k in 0 until z.size - 1) z[k] = bb[k + 1] * xi - aa[k + 1] * yi + z[k + 1]
                z[z.size - 1] = bb[z.size] * xi - aa[z.size] * yi
                y[i] = yi.toFloat()
            }
            return y
        }

        private fun resampleLinear(x: FloatArray, fsIn: Int, fsOut: Int): FloatArray {
            val ratio = fsIn.toDouble() / fsOut
            val nOut = (x.size / ratio).toInt()
            val y = FloatArray(nOut)
            for (i in 0 until nOut) {
                val pos = i * ratio
                val i0 = pos.toInt()
                val i1 = minOf(i0 + 1, x.size - 1)
                val frac = pos - i0
                y[i] = (x[i0] * (1 - frac) + x[i1] * frac).toFloat()
            }
            return y
        }

        private fun cropOrPad(x: FloatArray, len: Int): FloatArray {
            if (x.size >= len) {
                val start = (x.size - len) / 2
                return x.copyOfRange(start, start + len)
            }
            return x + FloatArray(len - x.size)
        }

        private fun minMaxScale(x: FloatArray): FloatArray {
            val mn = x.minOrNull() ?: 0f
            val mx = x.maxOrNull() ?: 0f
            val d = (mx - mn + 1e-8f)
            return x.map { (2f * (it - mn) / d - 1f).toFloat() }.toFloatArray()
        }
    }
}
