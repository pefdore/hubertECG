package com.hubertecg

import android.content.Context
import com.google.mediapipe.tasks.genai.llminference.LlmInference
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
        val session = llm!!.createSession()
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
            Tu es un médecin généraliste expérimenté spécialisé dans l'interprétation des ECG.
            Rédige un rapport structuré, clair et pédagogique en français. Cette analyse est une
            aide à la décision qui ne remplace pas l'avis d'un cardiologue.

            INFORMATIONS PATIENT :
            ${fmt(patient)}

            DONNÉES / MESURES DE L'ECG :
            ${fmt(ecg)}

            Réponds UNIQUEMENT avec un JSON valide :
            {
              "rythme": "description du rythme",
              "frequence_cardiaque": "FC estimée en bpm",
              "intervalles": "analyse PR, QRS, QT/QTc",
              "morphologie": "ondes P, QRS, T, segment ST",
              "diagnostic_rapporte": "diagnostic le plus probable ou 'ECG normal'",
              "diagnostics_differentiels": ["..."],
              "explication": "explication pédagogique",
              "conduite_a_tenir": "conduite à tenir en médecine générale",
              "niveau_urgence": "urgence vitale / avis spécialisé rapide / surveillance standard",
              "precautions": "limites de l'analyse IA"
            }
        """.trimIndent()
    }

    fun parseReport(text: String): Map<String, Any?> {
        val start = text.indexOf('{')
        val end = text.lastIndexOf('}')
        if (start < 0 || end <= start) return mapOf("réponse_brute" to text)
        return try {
            val obj = JSONObject(text.substring(start, end + 1))
            obj.keys().asSequence().associateWith { k ->
                val v = obj.get(k)
                if (v is org.json.JSONArray) List(v.length()) { i -> v.getString(i) } else v.toString()
            }
        } catch (e: Exception) {
            mapOf("erreur" to "JSON invalide", "réponse_brute" to text.substring(start, end + 1))
        }
    }
}
