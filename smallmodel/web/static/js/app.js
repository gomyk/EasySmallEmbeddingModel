// SmallModel Interactive Editor

let teachers = {};
let currentTeacher = null;
let selectedLayers = new Set();
let debounceTimer = null;

// ── Init ────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
    await loadTeachers();
    await loadDatasets();
    await loadEvalTasks();
    // Load current teacher from backend
    const resp = await fetch("/api/teacher");
    const info = await resp.json();
    document.getElementById("teacher-select").value = info.key;
    await switchTeacher(info.key);
});

// ── Teachers ────────────────────────────────────────────────────

async function loadTeachers() {
    const resp = await fetch("/api/teachers");
    teachers = await resp.json();

    const select = document.getElementById("teacher-select");
    select.innerHTML = "";
    for (const [key, t] of Object.entries(teachers)) {
        const opt = document.createElement("option");
        opt.value = key;
        opt.textContent = `${t.short_name} (${t.num_layers}L, ${t.hidden_dim}d, ${formatParams(t.total_params)})`;
        select.appendChild(opt);
    }
    select.addEventListener("change", (e) => switchTeacher(e.target.value));
}

async function switchTeacher(key) {
    currentTeacher = { key, ...teachers[key] };
    selectedLayers = new Set();
    vocabAnalysisData = null;
    document.getElementById("vocab-coverage-result").style.display = "none";

    // Update info
    const t = currentTeacher;
    document.getElementById("teacher-info").innerHTML =
        `<b>${t.model_id}</b><br>` +
        `${t.num_layers} layers | ${t.hidden_dim}d | ${t.intermediate_size} FFN | ` +
        `${t.vocab_size.toLocaleString()} vocab | ${t.fp32_mb}MB | ` +
        `${formatParams(t.total_params)} | ` +
        `${t.is_decoder ? "Decoder" : "Encoder"}${t.has_glu ? " + GLU" : ""}`;

    // Update config inputs
    const origHeads = t.num_attention_heads || (t.hidden_dim / 64);
    document.getElementById("num-heads").value = origHeads;
    document.getElementById("num-heads").max = origHeads;
    document.getElementById("heads-hint").textContent = `original: ${origHeads} (head_dim=${Math.floor(t.hidden_dim / origHeads)})`;

    document.getElementById("hidden-dim").value = t.hidden_dim;
    document.getElementById("hidden-dim").max = t.hidden_dim;
    document.getElementById("hidden-hint").textContent = `original: ${t.hidden_dim}`;

    document.getElementById("intermediate-size").value = t.intermediate_size;
    document.getElementById("intermediate-size").max = t.intermediate_size;
    document.getElementById("inter-hint").textContent = `original: ${t.intermediate_size}`;

    document.getElementById("vocab-size").value = t.vocab_size;
    document.getElementById("vocab-size").max = t.vocab_size;
    document.getElementById("vocab-hint").textContent = `original: ${t.vocab_size.toLocaleString()}`;

    // Update teacher estimate display
    document.getElementById("teacher-params").textContent = formatParams(t.total_params);
    document.getElementById("teacher-mb").textContent = t.fp32_mb;
    document.getElementById("size-bar-max").textContent = `${t.fp32_mb} MB`;

    // Model name
    document.getElementById("model-name").placeholder = `${key}_custom`;

    buildLayerGrid();
    await loadPresets();

    // Default: select all layers
    for (let i = 0; i < t.num_layers; i++) selectedLayers.add(i);
    refreshLayerGrid();
    updateEstimate();
}

// ── Layer Grid ──────────────────────────────────────────────────

function buildLayerGrid() {
    const grid = document.getElementById("layer-grid");
    grid.innerHTML = "";
    const n = currentTeacher.num_layers;

    for (let i = 0; i < n; i++) {
        const cell = document.createElement("div");
        cell.className = "layer-cell inactive";
        cell.dataset.idx = i;
        cell.innerHTML = `<span class="layer-idx">L${i}</span><span>${i}</span>`;
        cell.addEventListener("click", () => toggleLayer(i));
        grid.appendChild(cell);
    }
}

function toggleLayer(idx) {
    if (selectedLayers.has(idx)) {
        selectedLayers.delete(idx);
    } else {
        selectedLayers.add(idx);
    }
    refreshLayerGrid();
    debouncedEstimate();
}

function refreshLayerGrid() {
    const cells = document.querySelectorAll(".layer-cell");
    cells.forEach(cell => {
        const idx = parseInt(cell.dataset.idx);
        cell.className = `layer-cell ${selectedLayers.has(idx) ? "active" : "inactive"}`;
    });

    const sorted = [...selectedLayers].sort((a, b) => a - b);
    const n = currentTeacher.num_layers;
    document.getElementById("layer-summary").textContent =
        `Selected: ${sorted.length} / ${n} layers [${sorted.join(", ")}]`;

    updateArchViz();
}

function selectAll() {
    for (let i = 0; i < currentTeacher.num_layers; i++) selectedLayers.add(i);
    refreshLayerGrid();
    debouncedEstimate();
}

function deselectAll() {
    selectedLayers.clear();
    refreshLayerGrid();
    debouncedEstimate();
}

function toggleEven() {
    selectedLayers.clear();
    for (let i = 0; i < currentTeacher.num_layers; i += 2) selectedLayers.add(i);
    refreshLayerGrid();
    debouncedEstimate();
}

function toggleOdd() {
    selectedLayers.clear();
    for (let i = 1; i < currentTeacher.num_layers; i += 2) selectedLayers.add(i);
    refreshLayerGrid();
    debouncedEstimate();
}

function applyPreset(indices) {
    selectedLayers.clear();
    indices.forEach(i => selectedLayers.add(i));
    refreshLayerGrid();
    debouncedEstimate();
}

// ── Presets ──────────────────────────────────────────────────────

async function loadPresets() {
    const resp = await fetch(`/api/presets?teacher_key=${currentTeacher.key}`);
    const presets = await resp.json();

    const container = document.getElementById("preset-buttons");
    container.innerHTML = "";
    presets.forEach(p => {
        const btn = document.createElement("button");
        btn.className = "btn-preset";
        btn.textContent = p.name;
        btn.title = p.description;
        btn.addEventListener("click", () => applyPreset(p.layer_indices));
        container.appendChild(btn);
    });
}

// ── Size Estimation ─────────────────────────────────────────────

function debouncedEstimate() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(updateEstimate, 150);
}

async function updateEstimate() {
    const indices = [...selectedLayers].sort((a, b) => a - b);
    if (indices.length === 0) {
        document.getElementById("student-params").textContent = "-";
        document.getElementById("student-mb").textContent = "-";
        document.getElementById("compression-ratio").textContent = "-";
        document.getElementById("two-stage-hint").textContent = "";
        document.getElementById("size-bar").style.width = "0%";
        return;
    }

    const numHeads = parseInt(document.getElementById("num-heads").value) || null;
    const hiddenDim = parseInt(document.getElementById("hidden-dim").value) || currentTeacher.hidden_dim;
    const interSize = parseInt(document.getElementById("intermediate-size").value) || currentTeacher.intermediate_size;
    const vocabSize = parseInt(document.getElementById("vocab-size").value) || currentTeacher.vocab_size;

    try {
        const resp = await fetch("/api/estimate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                teacher_key: currentTeacher.key,
                layer_indices: indices,
                num_attention_heads: numHeads,
                hidden_dim: hiddenDim,
                intermediate_size: interSize,
                vocab_size: vocabSize,
            }),
        });
        const est = await resp.json();

        document.getElementById("student-params").textContent = formatParams(est.total_params);
        document.getElementById("student-mb").textContent = est.fp32_mb;
        document.getElementById("compression-ratio").textContent = est.compression_ratio;

        const hint = document.getElementById("two-stage-hint");
        if (est.needs_two_stage) {
            hint.textContent = "2-stage distillation recommended";
            hint.style.color = "var(--warning)";
            hint.style.fontSize = "0.7rem";
        } else {
            hint.textContent = "";
        }

        // Size bar
        const bar = document.getElementById("size-bar");
        const pct = Math.min((est.fp32_mb / currentTeacher.fp32_mb) * 100, 100);
        bar.style.width = `${pct}%`;
        bar.className = est.fp32_mb > 50 ? "size-bar-fill over-limit" : "size-bar-fill";

    } catch (e) {
        console.error("Estimate error:", e);
    }
}

// Config inputs trigger re-estimation + vocab coverage
let vocabDebounce = null;
document.addEventListener("DOMContentLoaded", () => {
    ["num-heads", "hidden-dim", "intermediate-size", "vocab-size"].forEach(id => {
        document.getElementById(id).addEventListener("input", debouncedEstimate);
    });
    document.getElementById("vocab-size").addEventListener("input", () => {
        clearTimeout(vocabDebounce);
        vocabDebounce = setTimeout(onVocabSizeChange, 400);
    });
});

// ── Architecture Visualization ──────────────────────────────────

function updateArchViz() {
    const teacherDiv = document.getElementById("arch-teacher");
    const studentDiv = document.getElementById("arch-student");
    const arrowsDiv = document.getElementById("arch-arrows");

    if (!currentTeacher) return;
    const n = currentTeacher.num_layers;
    const sorted = [...selectedLayers].sort((a, b) => a - b);

    // Teacher side
    let teacherHtml = `<div class="arch-layer embed">Embedding (${currentTeacher.vocab_size.toLocaleString()})</div>`;
    for (let i = 0; i < n; i++) {
        const cls = selectedLayers.has(i) ? "kept" : "pruned";
        teacherHtml += `<div class="arch-layer ${cls}">Layer ${i}</div>`;
    }
    teacherHtml += `<div class="arch-layer pool">Mean Pooling -> ${currentTeacher.hidden_dim}d</div>`;
    teacherDiv.innerHTML = teacherHtml;

    // Student side
    const vocabSize = parseInt(document.getElementById("vocab-size").value) || currentTeacher.vocab_size;
    const hiddenDim = parseInt(document.getElementById("hidden-dim").value) || currentTeacher.hidden_dim;
    let studentHtml = `<div class="arch-layer embed">Embedding (${vocabSize.toLocaleString()})</div>`;
    sorted.forEach((origIdx, newIdx) => {
        studentHtml += `<div class="arch-layer student-layer">Layer ${newIdx} (from L${origIdx})</div>`;
    });
    if (sorted.length === 0) {
        studentHtml += `<div class="arch-layer" style="opacity:0.3">No layers selected</div>`;
    }
    studentHtml += `<div class="arch-layer pool">Mean Pooling -> ${hiddenDim}d</div>`;
    studentDiv.innerHTML = studentHtml;
}

// ── Datasets & Eval Tasks ───────────────────────────────────

let datasets = {};
let evalTasks = {};
let vocabAnalysisData = null;

async function loadDatasets() {
    const resp = await fetch("/api/datasets");
    datasets = await resp.json();
    renderDatasetList();
}

async function loadEvalTasks() {
    const resp = await fetch("/api/eval-tasks");
    evalTasks = await resp.json();
    renderEvalTaskList();
}

function renderDatasetList() {
    const container = document.getElementById("dataset-list");
    container.innerHTML = "";

    // Group by category
    const groups = {};
    for (const [key, ds] of Object.entries(datasets)) {
        if (!groups[ds.group]) groups[ds.group] = [];
        groups[ds.group].push({ key, ...ds });
    }

    for (const [group, items] of Object.entries(groups)) {
        const header = document.createElement("div");
        header.className = "group-header";
        header.textContent = group;
        container.appendChild(header);

        for (const item of items) {
            const label = document.createElement("label");
            label.innerHTML = `<input type="checkbox" class="ds-checkbox" data-key="${item.key}" data-group="${item.group}" checked> ${item.label}`;
            container.appendChild(label);
        }
    }
}

function renderEvalTaskList() {
    const container = document.getElementById("eval-task-list");
    container.innerHTML = "";

    for (const [group, tasks] of Object.entries(evalTasks)) {
        const header = document.createElement("div");
        header.className = "group-header";
        header.textContent = group;
        container.appendChild(header);

        for (const task of tasks) {
            const label = document.createElement("label");
            label.innerHTML = `<input type="checkbox" class="eval-checkbox" data-task="${task}" data-group="${group}" checked> ${task}`;
            container.appendChild(label);
        }
    }
}

function toggleDatasetGroup(group) {
    const boxes = document.querySelectorAll(".ds-checkbox");
    if (group === "all") {
        boxes.forEach(cb => cb.checked = true);
    } else if (group === "none") {
        boxes.forEach(cb => cb.checked = false);
    } else {
        boxes.forEach(cb => {
            if (cb.dataset.group === group) cb.checked = !cb.checked;
        });
    }
}

function toggleEvalGroup(group) {
    const boxes = document.querySelectorAll(".eval-checkbox");
    if (group === "all") {
        boxes.forEach(cb => cb.checked = true);
    } else if (group === "none") {
        boxes.forEach(cb => cb.checked = false);
    } else {
        boxes.forEach(cb => {
            if (cb.dataset.group === group) cb.checked = !cb.checked;
        });
    }
}

function getSelectedDatasets() {
    return [...document.querySelectorAll(".ds-checkbox:checked")].map(cb => cb.dataset.key);
}

function getSelectedEvalTasks() {
    return [...document.querySelectorAll(".eval-checkbox:checked")].map(cb => cb.dataset.task);
}

// ── Vocab Coverage Analysis ─────────────────────────────────

async function analyzeVocab() {
    const selectedDs = getSelectedDatasets();
    if (selectedDs.length === 0) {
        document.getElementById("analyze-status").textContent = "Select at least one dataset.";
        return;
    }

    const vocabSize = parseInt(document.getElementById("vocab-size").value) || currentTeacher.vocab_size;

    document.getElementById("btn-analyze").disabled = true;
    document.getElementById("analyze-status").textContent = `Downloading ${selectedDs.length} datasets from HuggingFace + tokenizing... (first run may take a few minutes)`;

    try {
        const resp = await fetch("/api/vocab-analysis", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                teacher_key: currentTeacher.key,
                datasets: selectedDs,
                target_vocab: vocabSize,
            }),
        });
        vocabAnalysisData = await resp.json();
        renderVocabCoverage(vocabAnalysisData);
        document.getElementById("analyze-status").textContent = "";
    } catch (e) {
        document.getElementById("analyze-status").textContent = `Error: ${e.message}`;
    } finally {
        document.getElementById("btn-analyze").disabled = false;
    }
}

function renderVocabCoverage(data) {
    const resultDiv = document.getElementById("vocab-coverage-result");
    resultDiv.style.display = "block";

    document.getElementById("vc-total-texts").textContent = data.total_texts.toLocaleString();
    document.getElementById("vc-total-tokens").textContent = data.total_tokens.toLocaleString();
    document.getElementById("vc-unique-tokens").textContent = data.unique_tokens.toLocaleString();
    document.getElementById("vc-original-vocab").textContent = data.original_vocab.toLocaleString();

    updateVocabCoverageDisplay(data);
}

function updateVocabCoverageDisplay(data) {
    const vocabSize = parseInt(document.getElementById("vocab-size").value) || currentTeacher.vocab_size;

    document.getElementById("vc-target-vocab").textContent = vocabSize.toLocaleString();

    // Find coverage from curve or use target coverage
    let coverage = data.coverage_at_target;
    if (coverage === null || coverage === undefined) {
        // Interpolate from curve
        const curve = data.coverage_curve;
        for (const point of curve) {
            if (point.vocab >= vocabSize) {
                coverage = point.coverage;
                break;
            }
            coverage = point.coverage;
        }
    }

    const pctEl = document.getElementById("vc-coverage-pct");
    pctEl.textContent = `${coverage}%`;
    pctEl.className = "coverage-pct " + (coverage >= 99 ? "good" : coverage >= 95 ? "warn" : "bad");

    // Bar
    document.getElementById("vc-bar").style.width = `${Math.min(coverage, 100)}%`;
}

// Re-analyze when vocab size changes (if analysis was already done)
async function onVocabSizeChange() {
    if (!vocabAnalysisData) return;

    const vocabSize = parseInt(document.getElementById("vocab-size").value) || currentTeacher.vocab_size;

    // Fetch updated coverage for new target
    try {
        const resp = await fetch("/api/vocab-analysis", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                teacher_key: currentTeacher.key,
                datasets: getSelectedDatasets(),
                target_vocab: vocabSize,
            }),
        });
        const data = await resp.json();
        vocabAnalysisData = data;
        updateVocabCoverageDisplay(data);
    } catch (e) {
        // silent
    }
}

// ── Actions ─────────────────────────────────────────────────────

async function createModel() {
    const indices = [...selectedLayers].sort((a, b) => a - b);
    if (indices.length === 0) {
        showStatus("error", "Please select at least one layer.");
        return;
    }

    const name = document.getElementById("model-name").value ||
                 `${currentTeacher.key}_L${indices.length}_custom`;
    const hiddenDim = parseInt(document.getElementById("hidden-dim").value);
    const vocabSize = parseInt(document.getElementById("vocab-size").value);

    showStatus("loading", "Creating model... This may take a few minutes.");
    document.getElementById("btn-create").disabled = true;

    try {
        const resp = await fetch("/api/create", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                teacher_key: currentTeacher.key,
                layer_indices: indices,
                hidden_dim: hiddenDim,
                vocab_size: vocabSize,
                name: name,
            }),
        });
        const result = await resp.json();

        if (result.status === "ok") {
            showStatus("success", `Model created successfully: ${result.path}`);
        } else {
            showStatus("error", `Error: ${result.message}`);
        }
    } catch (e) {
        showStatus("error", `Request failed: ${e.message}`);
    } finally {
        document.getElementById("btn-create").disabled = false;
    }
}

async function autoCompress() {
    const maxMb = parseFloat(document.getElementById("auto-max-mb").value) || 50;
    const maxParams = parseInt(document.getElementById("auto-max-params").value) || 20000000;
    const minLayers = parseInt(document.getElementById("auto-min-layers").value) || 4;

    showStatus("loading", `Auto-compressing (max ${maxMb}MB, ${formatParams(maxParams)} params, min ${minLayers} layers)... This may take several minutes.`);
    document.getElementById("btn-auto").disabled = true;

    try {
        const resp = await fetch("/api/compress", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                teacher_key: currentTeacher.key,
                max_params: maxParams,
                max_fp32_mb: maxMb,
                min_layers: minLayers,
            }),
        });
        const result = await resp.json();

        if (result.status === "ok") {
            showStatus("success",
                `Auto-compressed: ${result.path}\n` +
                `Layers: [${result.layer_indices.join(", ")}], ` +
                `Hidden: ${result.hidden_dim}, ` +
                `${result.needs_two_stage ? "(2-stage distillation needed)" : ""}`
            );
            // Update UI to reflect
            selectedLayers.clear();
            result.layer_indices.forEach(i => selectedLayers.add(i));
            document.getElementById("hidden-dim").value = result.hidden_dim;
            document.getElementById("intermediate-size").value = result.intermediate_size;
            refreshLayerGrid();
            updateEstimate();
        } else {
            showStatus("error", `Error: ${result.message}`);
        }
    } catch (e) {
        showStatus("error", `Request failed: ${e.message}`);
    } finally {
        document.getElementById("btn-auto").disabled = false;
    }
}

// ── Utilities ───────────────────────────────────────────────────

function formatParams(n) {
    if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
    return n.toString();
}

// Auto-compress params hint
document.addEventListener("DOMContentLoaded", () => {
    const paramsInput = document.getElementById("auto-max-params");
    if (paramsInput) {
        paramsInput.addEventListener("input", () => {
            const hint = document.getElementById("auto-params-hint");
            if (hint) hint.textContent = formatParams(parseInt(paramsInput.value) || 0);
        });
    }
});

function showStatus(type, message) {
    const el = document.getElementById("action-status");
    el.className = `status-box ${type}`;
    el.textContent = message;
}

// ── Distillation ────────────────────────────────────────────────

let distillSessionId = null;
let distillEventSource = null;

async function loadDevices() {
    const resp = await fetch("/api/device-info");
    const data = await resp.json();
    const sel = document.getElementById("distill-device");
    sel.innerHTML = "";
    data.devices.forEach(d => {
        const opt = document.createElement("option");
        opt.value = d.id;
        opt.textContent = d.name;
        sel.appendChild(opt);
    });
    // Default to GPU if available
    if (data.devices.length > 1) sel.value = data.devices[1].id;
}

async function refreshModels() {
    const resp = await fetch("/api/models");
    const data = await resp.json();

    // Teacher select
    const teacherSel = document.getElementById("distill-teacher");
    teacherSel.innerHTML = "";
    data.teachers.forEach(t => {
        const opt = document.createElement("option");
        opt.value = t.key;
        opt.textContent = `${t.short_name} (${t.model_id})`;
        teacherSel.appendChild(opt);
    });

    // Student select
    const studentSel = document.getElementById("distill-student");
    studentSel.innerHTML = "";
    data.local_models.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m.path;
        opt.textContent = `${m.name} (${m.size_mb}MB)`;
        studentSel.appendChild(opt);
    });

    // Upload model select
    const uploadSel = document.getElementById("upload-model");
    uploadSel.innerHTML = "";
    data.local_models.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m.path;
        opt.textContent = `${m.name} (${m.size_mb}MB)`;
        uploadSel.appendChild(opt);
    });
}

async function startDistill() {
    const teacherPath = document.getElementById("distill-teacher").value;
    const studentPath = document.getElementById("distill-student").value;
    const outputPath = document.getElementById("distill-output").value || "";

    if (!teacherPath || !studentPath) {
        alert("Select both teacher and student models.");
        return;
    }

    const payload = {
        teacher_path: teacherPath,
        student_path: studentPath,
        output_path: outputPath,
        datasets: getSelectedDatasets(),
        epochs: parseInt(document.getElementById("distill-epochs").value) || 10,
        batch_size: parseInt(document.getElementById("distill-batch").value) || 32,
        lr: parseFloat(document.getElementById("distill-lr").value) || 2e-5,
        patience: parseInt(document.getElementById("distill-patience").value) || 3,
        device: document.getElementById("distill-device").value || "cpu",
        cos_weight: parseFloat(document.getElementById("distill-cos").value) || 0.5,
        mse_weight: parseFloat(document.getElementById("distill-mse").value) || 1.0,
        max_length: parseInt(document.getElementById("distill-maxlen").value) || 64,
        save_every_epoch: document.getElementById("distill-save-every").checked,
    };

    document.getElementById("btn-distill").disabled = true;
    document.getElementById("btn-distill-stop").disabled = false;
    document.getElementById("distill-progress").style.display = "block";
    document.getElementById("dp-log").innerHTML = "";
    document.getElementById("dp-status").textContent = "Starting...";

    try {
        const resp = await fetch("/api/distill/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const result = await resp.json();

        if (result.status !== "ok") {
            logDistill("error", `Error: ${result.message}`);
            document.getElementById("btn-distill").disabled = false;
            return;
        }

        distillSessionId = result.session_id;
        connectDistillSSE(distillSessionId);
    } catch (e) {
        logDistill("error", `Request failed: ${e.message}`);
        document.getElementById("btn-distill").disabled = false;
    }
}

function connectDistillSSE(sessionId) {
    if (distillEventSource) distillEventSource.close();

    distillEventSource = new EventSource(`/api/distill/stream/${sessionId}`);

    distillEventSource.addEventListener("status", (e) => {
        const d = JSON.parse(e.data);
        document.getElementById("dp-status").textContent = d.message;
        logDistill("info", d.message);
    });

    distillEventSource.addEventListener("config", (e) => {
        const d = JSON.parse(e.data);
        document.getElementById("dp-total-epochs").textContent = d.epochs;
        document.getElementById("dp-total-batches").textContent = d.total_batches;
        logDistill("info", `Config: ${d.total_texts.toLocaleString()} texts, ${d.total_batches} batches/epoch, device=${d.device}`);
    });

    distillEventSource.addEventListener("epoch_start", (e) => {
        const d = JSON.parse(e.data);
        document.getElementById("dp-epoch").textContent = d.epoch;
        document.getElementById("dp-status").textContent = `Epoch ${d.epoch}/${d.total_epochs}`;
        document.getElementById("dp-batch-bar").style.width = "0%";
        logDistill("epoch", `Epoch ${d.epoch}/${d.total_epochs} started`);
    });

    distillEventSource.addEventListener("batch", (e) => {
        const d = JSON.parse(e.data);
        const pct = (d.batch / d.total_batches * 100).toFixed(1);
        document.getElementById("dp-batch").textContent = d.batch;
        document.getElementById("dp-batch-bar").style.width = `${pct}%`;
        document.getElementById("dp-batch-loss").textContent = `loss: ${d.loss}`;
        document.getElementById("dp-current-loss").textContent = d.loss;
        document.getElementById("dp-mse").textContent = d.mse_loss;
        document.getElementById("dp-cosine").textContent = d.cos_loss;
        document.getElementById("dp-lr").textContent = d.lr.toExponential(1);

        // ETA
        if (d.eta_total_sec > 0) {
            document.getElementById("dp-eta").textContent = `ETA: ${formatTime(d.eta_total_sec)}`;
        }

        // Epoch bar
        const epochPct = ((d.epoch - 1) / parseInt(document.getElementById("dp-total-epochs").textContent) * 100 +
                          pct / parseInt(document.getElementById("dp-total-epochs").textContent)).toFixed(1);
        document.getElementById("dp-epoch-bar").style.width = `${Math.min(epochPct, 100)}%`;
    });

    distillEventSource.addEventListener("epoch_end", (e) => {
        const d = JSON.parse(e.data);
        document.getElementById("dp-epoch-loss").textContent = `avg: ${d.avg_loss}`;
        document.getElementById("dp-best-loss").textContent = d.best_loss;
        document.getElementById("dp-patience-count").textContent = `${d.no_improve_count}/${d.patience}`;
        document.getElementById("dp-epoch-bar").style.width = `${d.epoch / d.total_epochs * 100}%`;

        if (d.eta_remaining_sec > 0) {
            document.getElementById("dp-eta").textContent = `ETA: ${formatTime(d.eta_remaining_sec)}`;
        }

        const tag = d.improved ? "save" : "warn";
        logDistill(tag, `Epoch ${d.epoch}: avg_loss=${d.avg_loss}, best=${d.best_loss}${d.improved ? " (improved)" : ` (no improve ${d.no_improve_count}/${d.patience})`} [${formatTime(d.epoch_time_sec)}]`);
    });

    distillEventSource.addEventListener("model_saved", (e) => {
        const d = JSON.parse(e.data);
        logDistill("save", `Model saved: ${d.path} (${d.label}, loss=${d.loss})`);
    });

    distillEventSource.addEventListener("early_stop", (e) => {
        const d = JSON.parse(e.data);
        logDistill("warn", `Early stopping at epoch ${d.epoch} (patience=${d.patience}, best_loss=${d.best_loss})`);
    });

    distillEventSource.addEventListener("complete", (e) => {
        const d = JSON.parse(e.data);
        document.getElementById("dp-status").textContent = "Complete!";
        document.getElementById("dp-eta").textContent = `Total: ${formatTime(d.total_time_sec)}`;
        document.getElementById("dp-epoch-bar").style.width = "100%";
        logDistill("save", `Distillation complete! Best loss: ${d.best_loss}, ${d.total_epochs_run} epochs, ${formatTime(d.total_time_sec)}`);
        logDistill("save", `Output: ${d.output_path}`);
        onDistillDone();
    });

    distillEventSource.addEventListener("error", (e) => {
        try {
            const d = JSON.parse(e.data);
            logDistill("error", `Error: ${d.message}`);
        } catch (_) {}
        document.getElementById("dp-status").textContent = "Error";
        onDistillDone();
    });

    distillEventSource.addEventListener("done", () => {
        onDistillDone();
    });

    distillEventSource.onerror = () => {
        // SSE connection closed
    };
}

function onDistillDone() {
    document.getElementById("btn-distill").disabled = false;
    document.getElementById("btn-distill-stop").disabled = true;
    if (distillEventSource) { distillEventSource.close(); distillEventSource = null; }
    refreshModels(); // Refresh model list to show new distilled models
}

async function stopDistill() {
    if (distillSessionId) {
        await fetch(`/api/distill/stop/${distillSessionId}`, { method: "POST" });
        logDistill("warn", "Stop requested...");
    }
}

function logDistill(type, message) {
    const log = document.getElementById("dp-log");
    const line = document.createElement("div");
    line.className = `log-${type}`;
    const ts = new Date().toLocaleTimeString();
    line.textContent = `[${ts}] ${message}`;
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
}

function formatTime(sec) {
    if (sec < 60) return `${Math.round(sec)}s`;
    if (sec < 3600) return `${Math.floor(sec/60)}m ${Math.round(sec%60)}s`;
    return `${Math.floor(sec/3600)}h ${Math.floor((sec%3600)/60)}m`;
}

// ── HuggingFace Upload ──────────────────────────────────────────

async function uploadModel() {
    const modelPath = document.getElementById("upload-model").value;
    const repoId = document.getElementById("upload-repo").value;
    const token = document.getElementById("upload-token").value;

    if (!modelPath || !repoId || !token) {
        showUploadStatus("error", "All fields are required.");
        return;
    }

    showUploadStatus("loading", "Uploading to HuggingFace...");
    document.getElementById("btn-upload").disabled = true;

    try {
        const resp = await fetch("/api/upload", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_path: modelPath, repo_id: repoId, hf_token: token }),
        });
        const result = await resp.json();

        if (result.status === "ok") {
            showUploadStatus("success", `Uploaded successfully: ${result.url}`);
        } else {
            showUploadStatus("error", `Error: ${result.message}`);
        }
    } catch (e) {
        showUploadStatus("error", `Request failed: ${e.message}`);
    } finally {
        document.getElementById("btn-upload").disabled = false;
    }
}

function showUploadStatus(type, message) {
    const el = document.getElementById("upload-status");
    el.className = `status-box ${type}`;
    el.textContent = message;
}

// ── Init: load devices & models ─────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
    await loadDevices();
    await refreshModels();
});
