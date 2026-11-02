package com.hubertecg

import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.hubertecg.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var ecgAnalyzer: EcgAnalyzer
    private lateinit var llmClient: LlmClient

    private var parsedLeads: List<FloatArray>? = null
    private var parsedSamples = 0
    private var ecgFindings: List<EcgAnalyzer.Finding> = emptyList()
    private var photoUri: Uri? = null

    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        if (ok && photoUri != null) {
            binding.photoPreview.setImageURI(photoUri)
            binding.photoPreview.visibility = android.view.View.VISIBLE
        }
    }

    private val openCsv = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@registerForActivityResult
        lifecycleScope.launch {
            try {
                val text = withContext(Dispatchers.IO) {
                    contentResolver.openInputStream(uri)!!.readBytes().toString(Charsets.UTF_8)
                }
                val (leads, n) = EcgAnalyzer.parseCsv12(text)
                parsedLeads = leads
                parsedSamples = n
                val fs = binding.fsInput.text.toString().toIntOrNull() ?: 500
                val hr = EcgAnalyzer.estimateHeartRate(leads[1].ifEmpty { leads[0] }, fs)
                binding.csvStatus.text = "Signal chargé localement : 12 dérivations × $n échantillons à ${fs} Hz." +
                    (hr?.let { " FC estimée ≈ $it bpm." } ?: "")
            } catch (e: Exception) {
                parsedLeads = null
                binding.csvStatus.text = "Erreur CSV : ${e.message}"
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        ecgAnalyzer = EcgAnalyzer(this)
        llmClient = LlmClient(this)

        binding.photoButton.setOnClickListener {
            val file = File(cacheDir, "ecg_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}.jpg")
            photoUri = androidx.core.content.FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
            takePicture.launch(photoUri)
        }

        binding.csvButton.setOnClickListener {
            openCsv.launch(arrayOf("text/csv", "text/plain", "application/octet-stream"))
        }

        binding.ecgModelButton.setOnClickListener {
            val leads = parsedLeads
            if (leads == null) {
                Toast.makeText(this, "Importez d'abord un CSV 12 dérivations", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            lifecycleScope.launch {
                binding.ecgModelButton.isEnabled = false
                binding.ecgResult.text = "Inférence HuBERT-ECG on-device…"
                try {
                    withContext(Dispatchers.Default) { ecgAnalyzer.load() }
                    val fs = binding.fsInput.text.toString().toIntOrNull() ?: 500
                    ecgFindings = withContext(Dispatchers.Default) { ecgAnalyzer.analyze(leads, fs) }
                    binding.ecgResult.text = "Détections HuBERT-ECG (multi-label) :\n" +
                        ecgFindings.joinToString("\n") { "• ${it.label} — ${(it.prob * 100).toInt()} %" }
                } catch (e: Exception) {
                    binding.ecgResult.text = "Erreur : ${e.message}"
                } finally {
                    binding.ecgModelButton.isEnabled = true
                }
            }
        }

        binding.analyzeButton.setOnClickListener {
            lifecycleScope.launch {
                binding.analyzeButton.isEnabled = false
                binding.statusText.text = "Préparation du LLM local…"
                binding.reportText.text = ""
                try {
                    llmClient.ensureModel { msg -> lifecycleScope.launch { binding.statusText.text = msg } }
                    withContext(Dispatchers.Default) { llmClient.load() }
                    binding.statusText.text = "Génération du rapport par le LLM on-device…"
                    val patient = mapOf(
                        "Âge" to binding.ageInput.text.toString(),
                        "Sexe" to binding.sexInput.text.toString(),
                        "Antécédents" to binding.antecedentsInput.text.toString(),
                        "Symptômes" to binding.symptomesInput.text.toString(),
                        "Traitements" to binding.medicamentsInput.text.toString(),
                    )
                    val ecg = mapOf(
                        "mesures" to binding.mesuresInput.text.toString(),
                        "photo du tracé" to (photoUri?.lastPathSegment ?: "non fournie"),
                        "détections HuBERT-ECG" to
                            (if (ecgFindings.isEmpty()) "non réalisées"
                             else ecgFindings.joinToString(", ") { "${it.label} (${(it.prob * 100).toInt()} %)" }),
                    )
                    val raw = withContext(Dispatchers.Default) {
                        llmClient.generate(llmClient.buildPrompt(patient, ecg))
                    }
                    val report = llmClient.parseReport(raw)
                    binding.statusText.text = "Rapport prêt."
                    binding.reportText.text = buildString {
                        report["niveau_urgence"]?.let { append("🔴 Niveau : $it\n\n") }
                        listOf(
                            "Rythme" to "rythme",
                            "Fréquence cardiaque" to "frequence_cardiaque",
                            "Intervalles" to "intervalles",
                            "Morphologie" to "morphologie",
                            "Diagnostic rapporté" to "diagnostic_rapporte",
                            "Diagnostics différentiels" to "diagnostics_differentiels",
                            "Explication" to "explication",
                            "Conduite à tenir" to "conduite_a_tenir",
                            "Précautions" to "precautions",
                        ).forEach { (label, key) ->
                            report[key]?.let { v ->
                                append("$label :\n")
                                if (v.startsWith("[") || v.contains("\n")) append("$v\n\n") else append("$v\n\n")
                            }
                        }
                        if (report.containsKey("erreur") || report.containsKey("réponse_brute"))
                            append("(réponse brute du modèle)\n${report["réponse_brute"]}")
                    }
                } catch (e: Exception) {
                    binding.statusText.text = "Erreur : ${e.message}"
                } finally {
                    binding.analyzeButton.isEnabled = true
                }
            }
        }
    }
}
