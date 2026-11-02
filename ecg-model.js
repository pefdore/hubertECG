import * as ort from "./vendor/ort/ort.min.mjs";

ort.env.wasm.wasmPath = "./vendor/ort/";
ort.env.wasm.numThreads = 1;

const TARGET_FS = 100;
const WINDOW = 5 * TARGET_FS;
const NUM_LEADS = 12;

let session = null;
let labels = null;

export async function loadEcgModel(onProgress) {
  if (session) return session;
  const [sess, lbl] = await Promise.all([
    ort.InferenceSession.create("./models/hubert-ecg-small.onnx", {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    }),
    fetch("./models/labels.json").then((r) => r.json()),
  ]);
  session = sess;
  labels = lbl;
  if (onProgress) onProgress("Modèle HuBERT-ECG prêt (164 pathologies).");
  return session;
}

export function isEcgModelLoaded() {
  return !!session;
}

function butterLowpass(fc, fs) {
  const w = 2 * Math.PI * fc / fs;
  const cw = Math.cos(w), sw = Math.sin(w);
  const alpha = sw / Math.SQRT2;
  const a0 = 1 + alpha;
  return {
    b: [(1 - cw) / 2 / a0, (1 - cw) / a0, (1 - cw) / 2 / a0],
    a: [1, (-2 * cw) / a0, (1 - alpha) / a0],
  };
}

function butterHighpass(fc, fs) {
  const w = 2 * Math.PI * fc / fs;
  const cw = Math.cos(w), sw = Math.sin(w);
  const alpha = sw / Math.SQRT2;
  const a0 = 1 + alpha;
  return {
    b: [(1 + cw) / 2 / a0, -(1 + cw) / a0, (1 + cw) / 2 / a0],
    a: [1, (-2 * cw) / a0, (1 - alpha) / a0],
  };
}

function filtfilt(b, a, x) {
  const n = x.length;
  const zi = new Array(a.length - 1).fill(0);
  const y = new Array(n);
  const scale = a[0];
  const bb = b.map((v) => v / scale);
  const aa = a.map((v) => v / scale);
  const fwd = new Array(n);
  let z = [...zi];
  for (let i = 0; i < n; i++) {
    const xi = x[i];
    const yi = bb[0] * xi + z[0];
    for (let k = 0; k < z.length - 1; k++) z[k] = bb[k + 1] * xi - aa[k + 1] * yi + z[k + 1];
    z[z.length - 1] = bb[z.length] * xi - aa[z.length] * yi;
    fwd[i] = yi;
  }
  const rev = fwd.slice().reverse();
  const out = new Array(n);
  z = [...zi];
  for (let i = 0; i < n; i++) {
    const xi = rev[i];
    const yi = bb[0] * xi + z[0];
    for (let k = 0; k < z.length - 1; k++) z[k] = bb[k + 1] * xi - aa[k + 1] * yi + z[k + 1];
    z[z.length - 1] = bb[z.length] * xi - aa[z.length] * yi;
    out[i] = yi;
  }
  return out.reverse();
}

function resampleLinear(x, fsIn, fsOut) {
  const ratio = fsIn / fsOut;
  const nOut = Math.round(x.length / ratio);
  const y = new Array(nOut);
  for (let i = 0; i < nOut; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, x.length - 1);
    const frac = pos - i0;
    y[i] = x[i0] * (1 - frac) + x[i1] * frac;
  }
  return y;
}

function cropOrPad(x, len) {
  if (x.length >= len) {
    const start = Math.floor((x.length - len) / 2);
    return x.slice(start, start + len);
  }
  const pad = new Array(len - x.length).fill(0);
  return x.concat(pad);
}

function minMaxScale(x) {
  let mn = Infinity, mx = -Infinity;
  for (const v of x) { if (v < mn) mn = v; if (v > mx) mx = v; }
  const d = mx - mn + 1e-8;
  return x.map((v) => (2 * (v - mn)) / d - 1);
}

export function preprocessEcg(leads12, fsIn) {
  const hp = butterHighpass(0.05, fsIn);
  const lp = butterLowpass(47, fsIn);
  const out = new Float32Array(NUM_LEADS * WINDOW);
  for (let l = 0; l < NUM_LEADS; l++) {
    const filtered = filtfilt(lp.b, lp.a, filtfilt(hp.b, hp.a, leads12[l]));
    const at100 = resampleLinear(filtered, fsIn, TARGET_FS);
    const win = cropOrPad(at100, WINDOW);
    const scaled = minMaxScale(win);
    out.set(scaled, l * WINDOW);
  }
  return out;
}

export function parseCsv12(text) {
  const rows = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"));
  const matrix = rows.map((line) =>
    line.split(/[;,\t ]+/).map((s) => parseFloat(s.replace(",", ".")))
  );
  const width = Math.max(...matrix.map((r) => r.length));
  if (width < 2) throw new Error("CSV illisible : moins de 2 colonnes");
  const leads = [];
  for (let c = 0; c < NUM_LEADS && c < width; c++) {
    leads.push(matrix.map((r) => (Number.isNaN(r[c]) ? 0 : r[c])));
  }
  while (leads.length < NUM_LEADS) {
    leads.push(new Array(matrix.length).fill(0));
  }
  return { leads, nSamples: matrix.length };
}

export async function runEcgInference(leads12, fsIn, topK = 8) {
  if (!session) throw new Error("Modèle HuBERT-ECG non chargé");
  const input = preprocessEcg(leads12, fsIn);
  const tensor = new ort.Tensor("float32", input, [1, NUM_LEADS * WINDOW]);
  const results = await session.run({ input_values: tensor });
  const logits = results.logits.data;
  const scored = [];
  for (let i = 0; i < logits.length; i++) {
    const p = 1 / (1 + Math.exp(-logits[i]));
    scored.push({ label: labels[i] || `LABEL_${i}`, prob: p });
  }
  scored.sort((a, b) => b.prob - a.prob);
  return scored.slice(0, topK);
}
