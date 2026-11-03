package com.hubertecg

import android.content.Context
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.net.URL
import kotlin.coroutines.resume

/**
 * LLM embarqué via MediaPipe LLM Inference (GPU/NNAPI), 100 % on-device.
 * Le modèle (.task) est mis en cache dans le stockage interne après le premier
 * téléchargement ; ensuite l'app fonctionne hors-ligne.
 */
class LlmClient(private val context: Context) {

    private var llm: LlmInference? = null

    companion object {
        const val DEFAULT_MODEL_URL =
            "https://storage.googleapis.com/mediapipe-assets/gemma3-1b-it-int4.task"
        const val MODEL_FILE = "llm-model.task"
    }

    private fun modelFile(): File = File(context.filesDir, MODEL_FILE)

    suspend fun ensureModel(onProgress: (String) -> Unit = {}) {
        val file = modelFile()
        if (file.exists() && file.length() > 0) return
        onProgress("Téléchargement unique du LLM (ensuite hors-ligne)…")
        withContext(Dispatchers.IO) {
            URL(DEFAULT_MODEL_URL).openStream().use { input ->
                file.outputStream().use { input.copyTo(it) }
            }
        }
    }

    fun load() {
        if (llm != null) return
        val options = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(modelFile().absolutePath)
            .setMaxTokens(1024)
            .build()
        llm = LlmInference.createFromOptions(context, options)
    }

    suspend fun generate(prompt: String): String = suspendCancellableCoroutine { cont ->
        val sessionOptions = LlmInferenceSession.LlmInferenceSessionOptions.builder()
            .setTemperature(0.2f)
            .setTopK(20)
            .setTopP(0.8f)
            .build()
        val session = llm!!.createSession(sessionOptions)
        session.addQueryChunk(prompt)
        session.generateResponseAsync { partial, _ ->
            if (partial != null && cont.isActive) cont.resume(partial)
        }
        cont.invokeOnCancellation { session.close() }
    }.also {
        llm?.close()
        llm = null
    }

    fun buildPrompt(patient: Map<String, String>, ecg: Map<String, String>): String {
        val fmt = { m: Map<String, String> ->
            m.filterValues { it.isNotBlank() }.entries.joinToString("\n") { "- ${it.key}: ${it.value}" }
                .ifBlank { "- non renseigné" }
        }
        return """
            Tu es un médecin généraliste expérimenté en interprétation d'ECG.
            Tu réponds UNIQUEMENT par un JSON compact sur une seule ligne, sans texte avant/après,
            sans markdown, et tu ne répètes jamais un champ.
            Cette analyse est une aide à la décision, jamais un substitut à l'avis d'un cardiologue.

            INFORMATIONS PATIENT :
            ${fmt(patient)}

            DONNÉES / MESURES DE L'ECG :
            ${fmt(ecg)}

            Modèle de réponse EXACT à suivre (une seule ligne, valeurs courtes en français) :
            {"rythme":"...","frequence_cardiaque":"...","diagnostic_rapporte":"...","explication":"...","conduite_a_tenir":"...","niveau_urgence":"...","precautions":"..."}
            Ta réponse commence par { et se termine par }. Aucun autre texte.
        """.trimIndent()
    }

    fun parseReport(text: String): Map<String, Any?> {
        val t = text.replace("```json", "").replace("```", "").trim()
        val start = t.indexOf('{')
        if (start < 0) return mapOf("réponse_brute" to text)
        var depth = 0; var end = -1; var inStr = false; var esc = false
        for (i in start until t.length) {
            val c = t[i]
            if (inStr) {
                if (esc) esc = false
                else if (c == '\\') esc = true
                else if (c == '"') inStr = false
            } else if (c == '"') inStr = true
            else if (c == '{') depth++
            else if (c == '}') { depth--; if (depth == 0) { end = i; break } }
        }
        val candidate = if (end > start) t.substring(start, end + 1) else t.substring(start)
        return try {
            val obj = JSONObject(candidate)
            obj.keys().asSequence().associateWith { k ->
                val v = obj.get(k)
                if (v is org.json.JSONArray) List(v.length()) { i -> v.getString(i) } else v.toString()
            }
        } catch (e: Exception) {
            try {
                var fixed = candidate
                if (end <= start) {
                    if (inStr) fixed += '"'
                    val open = fixed.count { it == '[' } - fixed.count { it == ']' }
                    if (open > 0) fixed += "]".repeat(open)
                    if (depth > 0) fixed += "}".repeat(depth)
                }
                val obj = JSONObject(fixed)
                obj.keys().asSequence().associateWith { k ->
                    val v = obj.get(k)
                    if (v is org.json.JSONArray) List(v.length()) { i -> v.getString(i) } else v.toString()
                }
            } catch (e2: Exception) {
                val kv = linkedMapOf<String, Any?>()
                val regex = Regex("""^[-•*]?\s*([A-Za-zÀ-ÿ_ ]{2,40}?)\s*[:：]\s*(.+)$""")
                for (line in candidate.lineSequence()) {
                    val m = regex.find(line.trim()) ?: continue
                    kv[m.groupValues[1].trim().lowercase().replace(Regex("\\s+"), "_")] = m.groupValues[2].trim()
                }
                if (kv.size >= 2) {
                    kv["note"] = "réponse non-JSON, champs extraits ligne par ligne"
                    return kv
                }
                mapOf("erreur" to "JSON invalide", "réponse_brute" to candidate)
            }
        }
    }
}
