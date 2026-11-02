import * as webllm from "./vendor/web-llm.js";
import { loadEcgModel, isEcgModelLoaded, parseCsv12, runEcgInference } from "./ecg-model.js";

let ecgFindings = [];

const MODELS = [
  { id: "SmolLM2-360M-Instruct-q4f16_1-MLC", label: "SmolLM2 360M (recommandé téléphone, ~250 Mo)" },
  { id: "SmolLM2-135M-Instruct-q4f16_1-MLC", label: "SmolLM2 135M (très léger, ~120 Mo)" },
  { id: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC", label: "Qwen2.5 0.5B (~350 Mo)" },
  { id: "Llama-3.2-1B-Instruct-q4f16_1-MLC", label: "Llama 3.2 1B (~600 Mo, plus lent)" },
];

const SYSTEM_PROMPT = `Tu es un médecin généraliste expérimenté spécialisé dans l'interprétation des
électrocardiogrammes (ECG). Tu analyses les données fournies localement et tu produis un rapport
structuré, clair et pédagogique, en français, avec un ton professionnel mais accessible.
Tu rappelles toujours que ton analyse est une aide à la décision qui ne remplace pas l'avis d'un
cardiologue ni le jugement clinique.`;

let engine = null;

const $ = (id) => document.getElementById(id);
const modelSelect = $("model-select");
const gpuStatus = $("gpu-status");
const loadBtn = $("load-model");
const progressWrap = $("progress-wrap");
const progressBar = $("progress-bar");
const progressText = $("progress-text");
const modelStatus = $("model-status");
const analyzeBtn = $("analyze");
const resultSection = $("result");
const reportEl = $("report");
const fileInput = $("file");
const dropzone = $("dropzone");

MODELS.forEach((m, i) => {
  const opt = document.createElement("option");
  opt.value = m.id;
  opt.textContent = m.label;
  if (i === 0) opt.selected = true;
  modelSelect.appendChild(opt);
});

if (navigator.gpu) {
  const adapter = await navigator.gpu.requestAdapter();
  if (adapter) {
    gpuStatus.textContent = "✅ WebGPU disponible : le modèle tourne localement sur ce téléphone.";
    gpuStatus.classList.add("ok");
  } else {
    gpuStatus.textContent = "❌ WebGPU sans adaptateur : l'analyse IA locale ne pourra pas démarrer sur cet appareil.";
    gpuStatus.classList.add("ko");
  }
} else {
  gpuStatus.textContent = "❌ WebGPU non supporté par ce navigateur : utilisez Chrome/Edge récent sur Android, ou Safari récent sur iPhone.";
  gpuStatus.classList.add("ko");
  loadBtn.disabled = true;
}

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("./sw.js");
}

loadEcgModel((msg) => {
  $("ecg-model-status").textContent = "✅ " + msg;
  $("ecg-model-status").classList.add("ok");
  if (parsedLeads) $("run-ecg-model").disabled = false;
}).catch((err) => {
  $("ecg-model-status").textContent = "❌ HuBERT-ECG indisponible : " + err.message;
});

loadBtn.addEventListener("click", async () => {
  if (engine) return;
  loadBtn.disabled = true;
  progressWrap.classList.remove("hidden");
  modelStatus.classList.remove("ok");
  modelStatus.textContent = "Chargement du modèle…";
  try {
    engine = await webllm.CreateMLCEngine(modelSelect.value, {
      initProgressCallback: (report) => {
        const pct = Math.round((report.progress / report.total) * 100);
        progressBar.style.width = pct + "%";
        progressText.textContent = `${report.text} (${pct} %)`;
      },
    });
    modelStatus.textContent = "✅ Modèle chargé et prêt — fonctionne désormais hors-ligne.";
    modelStatus.classList.add("ok");
    analyzeBtn.disabled = false;
  } catch (err) {
    modelStatus.textContent = "❌ Échec du chargement : " + err.message;
    loadBtn.disabled = false;
  } finally {
    progressWrap.classList.add("hidden");
  }
});

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const f = fileInput.files[0];
  $("filename").textContent = f ? f.name : "";
  const preview = $("preview");
  if (f && f.type.startsWith("image/")) {
    preview.src = URL.createObjectURL(f);
    preview.classList.remove("hidden");
  } else {
    preview.classList.add("hidden");
  }
});

let parsedLeads = null;

$("csvfile").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const text = await f.text();
  try {
    const { leads, nSamples } = parseCsv12(text);
    parsedLeads = leads;
    const fs = parseFloat($("fs").value) || 500;
    const stats = analyzeSignal(leads[1] || leads[0], fs);
    $("csv-result").textContent =
      `Signal chargé localement : 12 dérivations × ${nSamples} échantillons à ${fs} Hz. ` +
      (stats
        ? `Fréquence cardiaque estimée ≈ ${stats.hr} bpm (${stats.rr} intervalles RR, écart-type ${stats.sd} ms).`
        : "Détection de fréquence cardiaque indisponible.");
    if (stats) $("fc").value = stats.hr;
    $("run-ecg-model").disabled = !isEcgModelLoaded();
  } catch (err) {
    parsedLeads = null;
    $("csv-result").textContent = "Erreur : " + err.message;
    $("run-ecg-model").disabled = true;
  }
});

$("run-ecg-model").addEventListener("click", async () => {
  if (!parsedLeads) return;
  const btn = $("run-ecg-model");
  btn.disabled = true;
  btn.textContent = "🧬 Inférence HuBERT-ECG en cours…";
  const box = $("ecg-model-result");
  box.classList.remove("hidden");
  box.innerHTML = "<p>Analyse du signal sur l'appareil…</p>";
  try {
    if (!isEcgModelLoaded()) {
      await loadEcgModel((msg) => ($("ecg-model-status").textContent = msg));
    }
    const fs = parseFloat($("fs").value) || 500;
    ecgFindings = await runEcgInference(parsedLeads, fs, 8);
    box.innerHTML =
      "<h3>Détections HuBERT-ECG (probabilités, multi-label)</h3><ul>" +
      ecgFindings
        .map((f) => `<li><strong>${f.label}</strong> — ${(f.prob * 100).toFixed(1)} %</li>`)
        .join("") +
      "</ul><p class='hint'>Scores bruts du modèle de fondation ECG ; à confronter au contexte clinique.</p>";
  } catch (err) {
    box.innerHTML = `<p>Erreur HuBERT-ECG : ${err.message}</p>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "🧬 Analyser le signal avec HuBERT-ECG (164 pathologies)";
  }
});

function analyzeSignal(values, fs = 250) {
  if (values.length < fs * 5) return null;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const sd = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length);
  const threshold = mean + 0.6 * sd;
  const refractory = Math.floor(fs * 0.3);
  const peaks = [];
  for (let i = 1; i < values.length - 1; i++) {
    if (values[i] > threshold && values[i] >= values[i - 1] && values[i] >= values[i + 1]) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1] > refractory) peaks.push(i);
    }
  }
  if (peaks.length < 3) return null;
  const rr = peaks.slice(1).map((p, i) => ((p - peaks[i]) / fs) * 1000);
  const avgRr = rr.reduce((a, b) => a + b, 0) / rr.length;
  const hr = Math.round(60000 / avgRr);
  const sdRr = Math.sqrt(rr.reduce((a, b) => a + (b - avgRr) ** 2, 0) / rr.length);
  return { hr, rr: rr.length, sd: Math.round(sdRr) };
}

function buildPrompt() {
  const patient = {
    "Âge": $("age").value,
    "Sexe": $("sexe").value,
    "Antécédents": $("antecedents").value,
    "Symptômes actuels": $("symptomes").value,
    "Traitements en cours": $("medicaments").value,
  };
  const ecg = {
    "rythme": $("rythme").value,
    "fréquence cardiaque (bpm)": $("fc").value,
    "intervalle PR (ms)": $("pr").value,
    "durée QRS (ms)": $("qrs").value,
    "QT/QTc (ms)": $("qt").value,
    "segment ST": $("st").value,
    "ondes T": $("ondet").value,
    "axe QRS": $("axe").value,
    "observations libres": $("obs").value,
    "image du tracé": fileInput.files[0] ? `${fileInput.files[0].name} (photo prise localement, non transmise)` : "non fournie",
    "détections du modèle HuBERT-ECG (signal brut, 164 classes)": ecgFindings.length
      ? ecgFindings.map((f) => `${f.label} (${(f.prob * 100).toFixed(0)} %)`).join(", ")
      : "non réalisées (fournir un CSV 12 dérivations)",
  };
  const fmt = (o) => Object.entries(o).filter(([, v]) => v).map(([k, v]) => `- ${k}: ${v}`).join("\n") || "- non renseigné";
  return `INFORMATIONS PATIENT :
${fmt(patient)}

DONNÉES / MESURES DE L'ECG :
${fmt(ecg)}

Produis un rapport structuré en JSON valide avec exactement ces champs :
{
  "rythme": "description du rythme (sinusal ou non, régulier ou non)",
  "frequence_cardiaque": "fréquence cardiaque estimée en bpm",
  "intervalles": "analyse des intervalles PR, QRS, QT/QTc si disponibles",
  "morphologie": "analyse morphologique des ondes P, QRS, T et segment ST",
  "diagnostic_rapporte": "diagnostic rapporté le plus probable, ou 'ECG normal'",
  "diagnostics_differentiels": ["liste des diagnostics différentiels à envisager"],
  "explication": "explication pédagogique en langage clair pour le médecin généraliste",
  "conduite_a_tenir": "conduite à tenir recommandée en médecine générale",
  "niveau_urgence": "urgence vitale / avis spécialisé rapide / surveillance standard",
  "precautions": "limites de l'analyse par IA locale et précautions d'interprétation"
}

Réponds UNIQUEMENT avec le JSON, sans texte avant ou après.`;
}

function parseJson(text) {
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start === -1 || end === -1) return { erreur: "Réponse non parsable", reponse_brute: text };
  try {
    return JSON.parse(text.slice(start, end + 1));
  } catch {
    return { erreur: "JSON invalide", reponse_brute: text.slice(start, end + 1) };
  }
}

function field(label, value) {
  if (value === null || value === undefined || value === "") return "";
  const body = Array.isArray(value)
    ? value.length ? `<ul>${value.map((v) => `<li>${v}</li>`).join("")}</ul>` : ""
    : `<p>${value}</p>`;
  return body ? `<div class="report-field"><h4>${label}</h4>${body}</div>` : "";
}

analyzeBtn.addEventListener("click", async () => {
  if (!engine) return;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "⏳ Analyse locale en cours…";
  resultSection.classList.remove("hidden");
  reportEl.innerHTML = "<p>Le modèle réfléchit sur l'appareil…</p>";

  try {
    const messages = [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: buildPrompt() },
    ];
    const reply = await engine.chat.completions.create({
      messages,
      temperature: 0.3,
      max_tokens: 1024,
    });
    const text = reply.choices[0].message.content;
    const data = parseJson(text);

    const urgence = (data.niveau_urgence || "").toLowerCase();
    let badgeClass = "standard";
    if (urgence.includes("urgence") || urgence.includes("vitale")) badgeClass = "urgence";
    else if (urgence.includes("rapide") || urgence.includes("spécialis")) badgeClass = "rapide";

    reportEl.innerHTML =
      (data.niveau_urgence ? `<span class="badge ${badgeClass}">${data.niveau_urgence}</span>` : "") +
      field("Rythme", data.rythme) +
      field("Fréquence cardiaque", data.frequence_cardiaque) +
      field("Intervalles", data.intervalles) +
      field("Morphologie", data.morphologie) +
      field("Diagnostic rapporté", data.diagnostic_rapporte) +
      field("Diagnostics différentiels", data.diagnostics_differentiels) +
      field("Explication", data.explication) +
      field("Conduite à tenir", data.conduite_a_tenir) +
      field("Précautions", data.precautions) +
      (data.erreur ? field("⚠️ Parsing", data.erreur + " — " + (data.reponse_brute || "")) : "");
  } catch (err) {
    reportEl.innerHTML = `<div class="report-field"><h4>Erreur</h4><p>${err.message}</p></div>`;
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "⚡ Lancer l'analyse locale";
  }
});
