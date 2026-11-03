import * as webllm from "./vendor/web-llm.js";
import { loadEcgModel, isEcgModelLoaded, parseCsv12, runEcgInference } from "./ecg-model.js";

let ecgFindings = [];

const MODELS = [
  { id: "SmolLM2-360M-Instruct-q4f16_1-MLC", label: "SmolLM2 360M (recommandé téléphone, ~250 Mo)" },
  { id: "SmolLM2-135M-Instruct-q4f16_1-MLC", label: "SmolLM2 135M (très léger, ~120 Mo)" },
  { id: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC", label: "Qwen2.5 0.5B (~350 Mo)" },
  { id: "Llama-3.2-1B-Instruct-q4f16_1-MLC", label: "Llama 3.2 1B (~600 Mo, plus lent)" },
];

const SYSTEM_PROMPT = `Tu es un médecin généraliste expérimenté en interprétation d'ECG.
RÈGLE ABSOLUE : ta réponse commence par le caractère { et se termine par }, c'est un JSON
compact, sans aucun texte avant ou après, sans markdown, et tu ne répètes jamais un champ.
Cette analyse est une aide à la décision, jamais un substitut à l'avis d'un cardiologue.`;

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

Modèle de réponse EXACT à suivre (une seule ligne, valeurs courtes en français) :
{"rythme":"...","frequence_cardiaque":"...","diagnostic_rapporte":"...","explication":"...","conduite_a_tenir":"...","niveau_urgence":"...","precautions":"..."}
Ta réponse commence par { et se termine par }. Aucun autre texte.`;
}

function parseJson(text) {
  let t = text.replace(/```json|```/gi, "").trim();
  const start = t.indexOf("{");
  if (start === -1) return { erreur: "Réponse non parsable", reponse_brute: text };
  let depth = 0, end = -1, inStr = false, esc = false;
  for (let i = start; i < t.length; i++) {
    const c = t[i];
    if (inStr) {
      if (esc) esc = false;
      else if (c === "\\") esc = true;
      else if (c === '"') inStr = false;
    } else if (c === '"') inStr = true;
    else if (c === "{") depth++;
    else if (c === "}") { depth--; if (depth === 0) { end = i; break; } }
  }
  let candidate = end !== -1 ? t.slice(start, end + 1) : t.slice(start);
  try {
    return JSON.parse(candidate);
  } catch {
    try {
      let fixed = candidate;
      if (end === -1) {
        if (inStr) fixed += '"';
        fixed += "]".repeat(Math.max(0, (fixed.match(/\[/g) || []).length - (fixed.match(/\]/g) || []).length));
        fixed += "}".repeat(Math.max(0, depth));
      }
      return JSON.parse(fixed);
    } catch {
      const kv = {};
      for (const l of candidate.split(/\r?\n/)) {
        const m = l.trim().match(/^[-•*]?\s*([A-Za-zÀ-ÿ_ ]{2,40}?)\s*[:：]\s*(.+)$/);
        if (m) kv[m[1].trim().toLowerCase().replace(/\s+/g, "_")] = m[2].trim();
      }
      if (Object.keys(kv).length >= 2) return { ...kv, note: "réponse non-JSON, champs extraits ligne par ligne" };
      return { erreur: "JSON invalide", reponse_brute: candidate };
    }
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
      temperature: 0.1,
      top_p: 0.9,
      frequency_penalty: 1.5,
      max_tokens: 768,
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
