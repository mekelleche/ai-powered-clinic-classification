const el = (id) => document.getElementById(id);

const modelStatus = el("modelStatus");
const modelSelect = el("modelSelect");
const btnReconnect = el("btnReconnect");
const btnShowFeatures = el("btnShowFeatures");
const featuresBox = el("featuresBox");
const featuresList = el("featuresList");
const featuresCount = el("featuresCount");
const heroTitle = el("heroTitle");
const heroDescription = el("heroDescription");

const manualForm = el("manualForm");
const featureSearch = el("featureSearch");
const btnPredictManual = el("btnPredictManual");
const btnAddToBatch = el("btnAddToBatch");
const manualResult = el("manualResult");

const imageFile = el("imageFile");
const ocrStatus = el("ocrStatus");
const ocrPreview = el("ocrPreview");
const ocrPreviewWrap = el("ocrPreviewWrap");
const ocrScanOverlay = el("ocrScanOverlay");
const ocrUploadZone = imageFile.closest(".section-gap");

// ── Button ripple effect ──────────────────────────────────
document.querySelectorAll('.btn').forEach(btn => {
  btn.addEventListener('click', function(e) {
    const ripple = document.createElement('span');
    ripple.className = 'ripple';
    const rect = this.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    ripple.style.cssText = `width:${size}px;height:${size}px;left:${e.clientX - rect.left - size/2}px;top:${e.clientY - rect.top - size/2}px`;
    this.appendChild(ripple);
    setTimeout(() => ripple.remove(), 600);
  });
});

const csvFile = el("csvFile");
const btnPredictCsv = el("btnPredictCsv");
const csvInfo = el("csvInfo");
const csvErrors = el("csvErrors");
const csvResults = el("csvResults");
const csvTbody = el("csvTbody");
const btnDownloadResults = el("btnDownloadResults");

const btnPredictBatch = el("btnPredictBatch");
const btnClearBatch = el("btnClearBatch");
const batchBox = el("batchBox");
const batchTbody = el("batchTbody");

const API_BASE = "/api";

let modelInfo = null; // {features, classes}
let availableModels = [];
let selectedModelKey = "anemia";
let manualInputs = new Map();
let batch = []; // array of { sample, result? }
let lastCsvRows = null; // array of objects
let lastCsvResults = null;

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  let data = null;
  try {
    data = await response.json();
  } catch (_err) {
    data = null;
  }

  if (!response.ok) {
    const message = data && data.error ? data.error : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return data;
}

function setModelStatus(kind, text) {
  modelStatus.classList.remove("pill-warn", "pill-ok", "pill-bad");
  modelStatus.classList.add(kind === "ok" ? "pill-ok" : kind === "bad" ? "pill-bad" : "pill-warn");
  modelStatus.textContent = text;
}

function enableUi() {
  const ok = !!modelInfo;
  modelSelect.disabled = availableModels.length === 0;
  btnShowFeatures.disabled = !ok;
  btnPredictManual.disabled = !ok;
  btnAddToBatch.disabled = !ok;
  btnPredictCsv.disabled = !ok || !lastCsvRows;
  btnPredictBatch.disabled = !ok || batch.length === 0;
  btnClearBatch.disabled = batch.length === 0;
}

function validateInfoModel(m) {
  if (!m || typeof m !== "object") throw new Error("Invalid model info");
  if (!Array.isArray(m.features) || !Array.isArray(m.classes)) throw new Error("Bad model info");
}

function getSelectedModelMeta() {
  return availableModels.find((m) => m.key === selectedModelKey) || null;
}

function resetResults() {
  manualResult.hidden = true;
  manualResult.innerHTML = "";
  csvInfo.textContent = "";
  csvErrors.hidden = true;
  csvResults.hidden = true;
  csvTbody.innerHTML = "";
  btnDownloadResults.disabled = true;
  lastCsvRows = null;
  lastCsvResults = null;
  csvFile.value = "";
  batch = [];
  renderBatchTable();
  ocrStatus.innerHTML = "";
  ocrPreviewWrap.classList.remove("active");
}

function updateModelPresentation() {
  const meta = getSelectedModelMeta();
  if (!meta) return;
  heroTitle.innerHTML = `AI-Powered <span>${meta.title}</span>`;
  heroDescription.textContent = meta.description || "Predict clinical outcomes from structured health data.";
  const supportsOcr = !!modelInfo?.supportsImageOcr;
  ocrUploadZone.hidden = !supportsOcr;
  ocrPreviewWrap.style.display = supportsOcr ? "" : "none";
}

function populateModelSelect(models) {
  modelSelect.innerHTML = "";
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.key;
    option.textContent = model.title;
    modelSelect.appendChild(option);
  }
  modelSelect.value = selectedModelKey;
}

function setFeaturesBoxVisible(visible) {
  featuresBox.hidden = !visible;
  if (!visible) return;
  featuresList.innerHTML = "";
  for (const f of modelInfo.features) {
    const chip = document.createElement("div");
    chip.className = "feature";
    chip.textContent = f;
    featuresList.appendChild(chip);
  }
  featuresCount.textContent = `${modelInfo.features.length} feature`;
}

function buildManualForm() {
  manualForm.innerHTML = "";
  manualInputs = new Map();
  manualForm.className = ""; // Remove wrapper generic layout

  const fnsSet = new Set(["WBC", "NE#", "LY#", "MO#", "EO#", "BA#", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "RDW", "PLT", "MPV", "PCT", "PDW", "SD", "SDTSD", "TSD"]);

  const fnsFieldset = document.createElement("fieldset");
  fnsFieldset.className = "fieldset-group";
  fnsFieldset.innerHTML = "<legend>Core Indicators</legend><div class='form'></div>";
  const fnsForm = fnsFieldset.querySelector(".form");

  const otherFieldset = document.createElement("fieldset");
  otherFieldset.className = "fieldset-group";
  otherFieldset.innerHTML = "<legend>Additional Indicators</legend><div class='form'></div>";
  const otherForm = otherFieldset.querySelector(".form");

  const anemiaLike = modelInfo.features.some((f) => fnsSet.has(f));

  for (const f of modelInfo.features) {
    const field = document.createElement("div");
    field.className = "field";

    const label = document.createElement("label");
    const left = document.createElement("div");
    left.textContent = f;

    const hint = document.createElement("span");
    hint.textContent = f === "GENDER" ? "0/1" : "";

    label.appendChild(left);
    label.appendChild(hint);

    const input = document.createElement("input");
    input.type = "number";
    input.inputMode = "decimal";
    input.step = "any";
    input.placeholder = "Leave empty for auto-imputation";

    field.appendChild(label);
    field.appendChild(input);

    if (anemiaLike && fnsSet.has(f)) {
      fnsForm.appendChild(field);
    } else {
      otherForm.appendChild(field);
    }

    manualInputs.set(f, input);
  }

  if (anemiaLike) {
    manualForm.appendChild(fnsFieldset);
  }
  manualForm.appendChild(otherFieldset);
}

function renderResultCard(container, res) {
  const hasCondition = !!res.hasCondition;
  const ok = hasCondition ? "badge-bad" : "badge-ok";
  const msg = hasCondition
    ? `Prediction: ${res.predictedClass.name}`
    : `Prediction: ${res.predictedClass.name}`;
  const summary = hasCondition ? "Needs attention" : "Clear";

  container.innerHTML = `
    <div class="row" style="justify-content:space-between">
      <div class="big">${msg}</div>
      <div class="badge ${ok}">${summary}</div>
    </div>
    <div class="meta">Predicted Class: <strong>${res.predictedClass.name}</strong> (p=${Number(res.probability).toFixed(3)})</div>
    <div class="meta">Probabilities: ${res.probabilities
      .map((p) => `${p.name}:${Number(p.p).toFixed(3)}`)
      .join(" • ")}</div>
  `;
}

function diagnosisBadge(res) {
  return res.hasCondition
    ? `<span class="badge badge-bad">Needs Attention</span>`
    : `<span class="badge badge-ok">Clear</span>`;
}

function collectManualSample() {
  const sample = {};
  for (const f of modelInfo.features) {
    const input = manualInputs.get(f);
    const raw = input?.value;
    if (raw === "" || raw == null) continue;
    const n = Number(String(raw).replace(",", "."));
    if (Number.isFinite(n)) sample[f] = n;
  }
  return sample;
}

async function connect() {
  try {
    setModelStatus("warn", "Connecting...");
    if (availableModels.length === 0) {
      availableModels = await apiRequest("/models", { method: "GET", headers: {} });
      if (availableModels.length === 0) throw new Error("No models are configured.");
      if (!availableModels.some((m) => m.key === selectedModelKey)) {
        selectedModelKey = availableModels[0].key;
      }
      populateModelSelect(availableModels);
    }
    const info = await apiRequest(`/model-info?model=${encodeURIComponent(selectedModelKey)}`, {
      method: "GET",
      headers: {},
    });
    validateInfoModel(info);
    modelInfo = info;
    setModelStatus("ok", `Connected: ${getSelectedModelMeta()?.title || selectedModelKey}`);
    setFeaturesBoxVisible(false);
    resetResults();
    buildManualForm();
    updateModelPresentation();
    enableUi();
  } catch (e) {
    console.error(e);
    modelInfo = null;
    setModelStatus("bad", "Connection failed");
    enableUi();
    alert(e.message || "Cannot load model. Make sure you trained the model and run python web_app.py");
  }
}

async function predictOne(sample) {
  return await apiRequest("/predict", {
    method: "POST",
    body: JSON.stringify({ model: selectedModelKey, sample }),
  });
}

async function predictBatch(samples) {
  return await apiRequest("/predict-batch", {
    method: "POST",
    body: JSON.stringify({ model: selectedModelKey, samples }),
  });
}

function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length === 0) return { headers: [], rows: [] };

  const first = lines[0];
  const commaCount = (first.match(/,/g) || []).length;
  const semiCount = (first.match(/;/g) || []).length;
  const delim = semiCount > commaCount ? ";" : ",";

  function splitLine(line) {
    const out = [];
    let cur = "";
    let inQ = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (inQ && line[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQ = !inQ;
        }
        continue;
      }
      if (!inQ && ch === delim) {
        out.push(cur);
        cur = "";
        continue;
      }
      cur += ch;
    }
    out.push(cur);
    return out;
  }

  const headers = splitLine(lines[0]).map((h) => h.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = splitLine(lines[i]);
    const obj = {};
    for (let j = 0; j < headers.length; j++) {
      obj[headers[j]] = (cols[j] ?? "").trim();
    }
    rows.push(obj);
  }
  return { headers, rows };
}

function normalizeCsvRow(row) {
  const sample = {};
  for (const f of modelInfo.features) {
    if (!(f in row)) continue;
    const raw = String(row[f] ?? "").trim();
    if (raw === "") continue;
    const n = Number(raw.replace(",", "."));
    if (Number.isFinite(n)) sample[f] = n;
  }
  return sample;
}

function renderCsvResults(results) {
  csvTbody.innerHTML = "";
  for (let i = 0; i < results.length; i++) {
    const r = results[i];
    const tr = document.createElement("tr");

    tr.innerHTML = `
      <td>${i + 1}</td>
      <td>${diagnosisBadge(r)}</td>
      <td><strong>${r.predictedClass.name}</strong></td>
      <td>${Number(r.probability).toFixed(3)}</td>
    `;
    csvTbody.appendChild(tr);
  }
  csvResults.hidden = false;
}

function toCsv(rows) {
  const esc = (s) => {
    const t = String(s ?? "");
    if (/[",\n\r;]/.test(t)) return `"${t.replaceAll('"', '""')}"`;
    return t;
  };

  const headers = ["row", "needs_attention", "class", "probability"];
  const lines = [headers.join(",")];
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    lines.push([
      esc(i + 1),
      esc(r.hasCondition ? 1 : 0),
      esc(r.predictedClass.name),
      esc(r.probability),
    ].join(","));
  }
  return lines.join("\n");
}

function downloadText(filename, text) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function renderBatchTable() {
  batchTbody.innerHTML = "";
  for (let i = 0; i < batch.length; i++) {
    const item = batch[i];
    const tr = document.createElement("tr");
    const res = item.result;
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td>${res ? diagnosisBadge(res) : '-'}</td>
      <td>${res ? `<strong>${res.predictedClass.name}</strong>` : '-'}</td>
      <td>${res ? Number(res.probability).toFixed(3) : '-'}</td>
      <td><button class="smallbtn" data-del="${i}">Delete</button></td>
    `;
    batchTbody.appendChild(tr);
  }
  batchBox.hidden = batch.length === 0;
  btnPredictBatch.disabled = !modelInfo || batch.length === 0;
  btnClearBatch.disabled = batch.length === 0;
}

// Events
btnReconnect.addEventListener("click", connect);
modelSelect.addEventListener("change", async () => {
  selectedModelKey = modelSelect.value;
  await connect();
});

btnShowFeatures.addEventListener("click", () => {
  if (!modelInfo) return;
  setFeaturesBoxVisible(featuresBox.hidden);
});

featureSearch.addEventListener("input", () => {
  const q = featureSearch.value.trim().toLowerCase();
  for (const [name, input] of manualInputs.entries()) {
    const field = input.closest(".field");
    if (!field) continue;
    field.style.display = q === "" || name.toLowerCase().includes(q) ? "" : "none";
  }
});

btnPredictManual.addEventListener("click", async () => {
  try {
    const sample = collectManualSample();
    const res = await predictOne(sample);
    manualResult.hidden = false;
    renderResultCard(manualResult, res);
  } catch (e) {
    console.error(e);
    alert(e.message || "Make sure the model is connected first.");
  }
});

btnAddToBatch.addEventListener("click", () => {
  const sample = collectManualSample();
  batch.push({ sample, result: null });
  renderBatchTable();
  enableUi();
});

btnPredictBatch.addEventListener("click", async () => {
  if (!modelInfo) return;
  const samples = batch.map((b) => b.sample);
  const results = await predictBatch(samples);
  for (let i = 0; i < batch.length; i++) batch[i].result = results[i];
  renderBatchTable();
});

btnClearBatch.addEventListener("click", () => {
  batch = [];
  renderBatchTable();
  enableUi();
});

batchTbody.addEventListener("click", (ev) => {
  const btn = ev.target?.closest("button[data-del]");
  if (!btn) return;
  const idx = Number(btn.getAttribute("data-del"));
  if (!Number.isFinite(idx)) return;
  batch.splice(idx, 1);
  renderBatchTable();
  enableUi();
});

imageFile.addEventListener("change", async () => {
  const f = imageFile.files?.[0];
  if (!f) {
    ocrPreviewWrap.classList.remove("active");
    ocrStatus.innerHTML = "";
    return;
  }
  if (!modelInfo) {
    ocrStatus.innerHTML = `<div class="ocr-status-tag error">⚠ Connect to the model first.</div>`;
    return;
  }

  imageFile.disabled = true;

  try {
    const reader = new FileReader();
    reader.onload = async (e) => {
      // Show image preview immediately
      ocrPreview.src = e.target.result;
      ocrPreviewWrap.classList.add("active");
      // Start scan animation
      ocrScanOverlay.classList.add("running");
      ocrStatus.innerHTML = `<div class="ocr-status-tag scanning">🔬 Reading CBC values with AI OCR...</div>`;

      try {
        const b64 = e.target.result.split(',')[1];
        const res = await apiRequest("/extract-from-image", {
          method: "POST",
          body: JSON.stringify({ model: selectedModelKey, b64 }),
        });

        let count = 0;
        for (const [key, val] of Object.entries(res)) {
          if (manualInputs.has(key) && val !== null && val !== undefined) {
            manualInputs.get(key).value = val;
            count++;
          }
        }

        ocrStatus.innerHTML = count > 0
          ? `<div class="ocr-status-tag success">✅ Extracted ${count} values from report. Please verify before analyzing.</div>`
          : `<div class="ocr-status-tag error">⚠ No standard CBC markers detected. Try a clearer image.</div>`;
      } catch (err) {
        console.error("OCR API Error", err);
        ocrStatus.innerHTML = `<div class="ocr-status-tag error">❌ ${err.message || "Server error during OCR processing."}</div>`;
      } finally {
        imageFile.disabled = false;
        ocrScanOverlay.classList.remove("running");
      }
    };
    reader.onerror = () => {
      ocrStatus.innerHTML = `<div class="ocr-status-tag error">❌ Failed to read image file.</div>`;
      imageFile.disabled = false;
    };
    reader.readAsDataURL(f);
  } catch (err) {
    console.error(err);
    ocrStatus.innerHTML = `<div class="ocr-status-tag error">❌ Error processing image.</div>`;
    imageFile.disabled = false;
  }
});

csvFile.addEventListener("change", async () => {
  const f = csvFile.files?.[0];
  csvInfo.textContent = "";
  csvErrors.hidden = true;
  csvResults.hidden = true;
  lastCsvRows = null;
  lastCsvResults = null;
  btnDownloadResults.disabled = true;

  if (!f) {
    btnPredictCsv.disabled = true;
    return;
  }

  const text = await f.text();
  const { headers, rows } = parseCsv(text);
  lastCsvRows = rows;

  const missing = modelInfo ? modelInfo.features.filter((ff) => !headers.includes(ff)) : [];
  csvInfo.textContent = `Read ${rows.length} cases. columns count: ${headers.length}.` +
    (missing.length ? ` (Missing values auto-imputed: ${missing.slice(0, 6).join(", ")}${missing.length > 6 ? "…" : ""})` : "");

  btnPredictCsv.disabled = !modelInfo || rows.length === 0;
});

btnPredictCsv.addEventListener("click", async () => {
  if (!modelInfo || !lastCsvRows) return;

  const samples = [];
  for (let i = 0; i < lastCsvRows.length; i++) samples.push(normalizeCsvRow(lastCsvRows[i]));

  try {
    const results = await predictBatch(samples);
    lastCsvResults = results;
    csvErrors.hidden = true;
    renderCsvResults(results);
    btnDownloadResults.disabled = results.length === 0;
  } catch (e) {
    console.error(e);
    alert(e.message || "Failed to compute results.");
  }
});

btnDownloadResults.addEventListener("click", () => {
  if (!lastCsvResults) return;
  const csv = toCsv(lastCsvResults);
  downloadText("anemia_results.csv", csv);
});

// Init
connect();
