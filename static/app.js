/* MEDUSA 2026 web service — single-page UI (vanilla JS + Plotly.js). */
"use strict";

// ------------------------------------------------------------------ //
// state
// ------------------------------------------------------------------ //
const S = {
  session: null,          // session_id
  fileInfo: null,         // {file_name, n_points, mz_range, load_time_s}
  ions: [],               // [{ion_id, mz, charge, n_peaks, mz_min, mz_max}]
  ionProbs: {},           // ion_id -> prob (selected element)
  selectedIon: null,
  element: "Ir",
  threshold: null,        // {threshold, mz_range, element, n_points_above}
  knee: null,
  ranked: [],
  window: null,           // last spectrum window response
  elementSymbols: [],
};

const PALETTE = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
  "#e377c2", "#7f7f7d", "#bcbd22", "#17becf", "#393f7a", "#e6194b",
  "#4363d8", "#9a6324", "#800000", "#469990",
];
const ionColor = (id) => PALETTE[id % PALETTE.length];

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// ------------------------------------------------------------------ //
// helpers
// ------------------------------------------------------------------ //
let toastTimer = null;
function toast(msg, isError = false, ms = 4000) {
  const el = $("#toast");
  el.textContent = msg;
  el.className = isError ? "error" : "";
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, ms);
}

async function api(path, options = {}) {
  const opts = { headers: {}, ...options };
  if (opts.body && typeof opts.body === "object" && !(opts.body instanceof FormData)) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.body);
  }
  const resp = await fetch(path, opts);
  let data = null;
  try { data = await resp.json(); } catch (_) { /* non-JSON */ }
  if (!resp.ok) {
    const detail = data && data.detail ? data.detail : `HTTP ${resp.status}`;
    const err = new Error(detail);
    err.status = resp.status;
    err.data = data;
    throw err;
  }
  return data;
}

function renderLogs(logs) {
  const area = $("#log-area");
  if (!logs || !logs.length) { area.innerHTML = ""; return; }
  const lines = logs.map((l) =>
    `<div class="${l.level}">${new Date(l.ts * 1000).toISOString().slice(11, 19)}  ${escapeHtml(l.message)}</div>`
  ).join("");
  area.innerHTML = lines;
  area.scrollTop = area.scrollHeight;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function setSpin(name, on) {
  const el = document.querySelector(`.spinner[data-spin="${name}"]`);
  if (el) el.hidden = !on;
}
async function withBusy(name, fn) {
  setSpin(name, true);
  try {
    return await fn();
  } finally {
    setSpin(name, false);
  }
}

function fmtMB(b) {
  if (b >= 1e9) return (b / 1e9).toFixed(2) + " GB";
  if (b >= 1e6) return (b / 1e6).toFixed(1) + " MB";
  return (b / 1e3).toFixed(0) + " KB";
}

// ------------------------------------------------------------------ //
// session info header
// ------------------------------------------------------------------ //
function updateHeader() {
  $("#session-id").textContent = S.session ? `session ${S.session}` : "no session";
  $("#session-file").textContent = S.fileInfo ? S.fileInfo.file_name : "";
}

// ------------------------------------------------------------------ //
// step 1: files + upload
// ------------------------------------------------------------------ //
let allFiles = [];
async function refreshFileList() {
  const data = await api("/api/files");
  allFiles = data.files || [];
  renderFileList($("#file-search").value);
}
function renderFileList(query) {
  const list = $("#file-list");
  const q = (query || "").toLowerCase();
  const shown = allFiles.filter((f) => f.path.toLowerCase().includes(q));
  list.innerHTML = shown.length
    ? shown.map((f, i) =>
        `<div class="file-item" data-idx="${allFiles.indexOf(f)}">
           <span>${escapeHtml(f.path)}</span><span class="size">${fmtMB(f.size)}</span>
         </div>`).join("")
    : `<div class="file-item muted">${allFiles.length ? "no match" : "no mzXML files in the spectra folder"}</div>`;
  list.querySelectorAll(".file-item[data-idx]").forEach((el) => {
    el.onclick = () => createSession("folder", { path: allFiles[+el.dataset.idx].path });
  });
}

async function createSession(source, extra) {
  setSpin("load", true);
  try {
    const data = await api("/api/sessions", { method: "POST", body: { source, ...extra } });
    onSessionCreated(data);
  } catch (err) {
    toast(err.message, true);
  } finally {
    setSpin("load", false);
  }
}

function uploadFile(file, onDone) {
  const xhr = new XMLHttpRequest();
  const form = new FormData();
  form.append("file", file);
  const prog = $("#upload-progress"), bar = $("#upload-bar"), label = $("#upload-label");
  prog.hidden = false; bar.style.width = "0%"; label.textContent = "uploading…";
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = Math.round((e.loaded / e.total) * 100);
      bar.style.width = pct + "%";
      label.textContent = `${pct}% — ${fmtMB(e.loaded)} / ${fmtMB(e.total)}`;
    }
  };
  xhr.onload = () => {
    prog.hidden = true;
    if (xhr.status >= 200 && xhr.status < 300) {
      try { onDone(JSON.parse(xhr.responseText)); }
      catch (_) { toast("bad server response", true); }
    } else {
      let detail = `upload failed (HTTP ${xhr.status})`;
      try { detail = JSON.parse(xhr.responseText).detail || detail; } catch (_) {}
      toast(detail, true);
    }
  };
  xhr.onerror = () => { prog.hidden = true; toast("upload failed (network)", true); };
  xhr.open("POST", "/api/upload");
  xhr.send(form);
}

// ------------------------------------------------------------------ //
// session created
// ------------------------------------------------------------------ //
async function onSessionCreated(data) {
  S.session = data.session_id;
  S.fileInfo = {
    file_name: data.file_name, n_points: data.n_points,
    mz_range: data.mz_range, load_time_s: data.load_time_s,
  };
  resetDownstreamState();
  updateHeader();
  renderLogs(data.logs);
  $("#load-summary").hidden = false;
  $("#load-summary").innerHTML =
    `<b>${escapeHtml(data.file_name)}</b><br>` +
    `points: ${data.n_points.toLocaleString()}<br>` +
    `m/z: ${data.mz_range[0].toFixed(2)} – ${data.mz_range[1].toFixed(2)}<br>` +
    `load: ${data.load_time_s}s` +
    (data.n_scans > 1 ? `<br><span class="muted">multi-scan file: using scan 1 of ${data.n_scans}</span>` : "");
  $("#spectrum-status").textContent = "loading overview…";
  await refreshWindow(data.mz_range[0], data.mz_range[1]);
  $("#spectrum-status").textContent = windowStatusText();
}

function resetDownstreamState() {
  S.ions = []; S.ionProbs = {}; S.selectedIon = null; S.threshold = null;
  S.knee = null; S.ranked = null; S.window = null;
  currentRange = null;
  renderLogs([]);
  $("#deiso-summary").hidden = true;
  $("#ion-legend").hidden = true;
  $("#ions-toggle").checked = false;
  $("#layer-ions").checked = false;
  $("#layer-probs").checked = false;
  $("#highlight-summary").hidden = true;
  $("#zoom-highlighted-btn").disabled = true;
  $("#knee-summary").hidden = true;
  $("#formulas-summary").hidden = true;
  $("#ions-table tbody").innerHTML = "";
  $("#formulas-table tbody").innerHTML = "";
  $("#ions-count").textContent = "";
  $("#formulas-count").textContent = "";
  $("#csv-export-btn").disabled = true;
  $("#f-ion").value = 0;
  $("#compare-plot").hidden = true;
  const cmpEl = $("#compare-plot");
  if (cmpEl.data) Plotly.purge(cmpEl);
  const kneeEl = $("#knee-plot");
  if (kneeEl.data) Plotly.purge(kneeEl);
  $("#compare-status").textContent = "step 7 result appears here";
}

// ------------------------------------------------------------------ //
// spectrum viewer
// ------------------------------------------------------------------ //
let zoomTimer = null;
let currentRange = null;

function windowStatusText() {
  const w = S.window;
  if (!w) return "load a spectrum to start";
  const parts = [`${w.n_in_window.toLocaleString()} pts in window`];
  if (w.decimated) parts.push(`decimated ≤ ${w.max_pts}`);
  const [wx0, wx1] = w.window_range || currentRange || [];
  if (wx0 !== undefined && wx1 !== undefined) parts.push(`m/z ${wx0.toFixed(2)}–${wx1.toFixed(2)}`);
  return parts.join(" · ");
}

async function refreshWindow(x0, x1) {
  if (!S.session) return;
  const layers = ["raw"];
  if (S.ions.length) layers.push("ions");
  if (S.threshold) layers.push("probs");
  try {
    const w = await api(
      `/api/sessions/${S.session}/spectrum?x0=${x0}&x1=${x1}&max_pts=2500&layers=${layers.join(",")}`
    );
    S.window = w;
    currentRange = [x0, x1];
    renderSpectrum();
    $("#spectrum-status").textContent = windowStatusText();
  } catch (err) {
    toast(err.message, true);
  }
}

function renderSpectrum() {
  const w = S.window;
  if (!w) return;

  // ion layer requested but the current window has no assignments: refetch
  if (S.ions.length && $("#layer-ions").checked && !w.ion_id && currentRange) {
    refreshWindow(currentRange[0], currentRange[1]);
    return;
  }

  const traces = [];
  const ionsLayerOn = S.ions.length > 0 && $("#layer-ions").checked && w.ion_id;

  // raw spectrum — always shown, deisotoping never changes its rendering;
  // the hover annotation carries the ion index when deisotoping is done
  traces.push({
    x: w.masses, y: w.ints, mode: "lines", name: "raw", showlegend: false,
    line: { color: "#2e6fb7", width: 1 },
    customdata: Array.isArray(w.ion_id)
      ? w.ion_id.map((id) => (id >= 0 ? `ion ${id}` : "")) : undefined,
    hovertemplate: Array.isArray(w.ion_id)
      ? "m/z %{x:.4f}<br>I %{y:.3g}<br>%{customdata}<extra></extra>"
      : "m/z %{x:.4f}<br>I %{y:.3g}<extra></extra>",
  });

  if (ionsLayerOn) {
    // one color per isotopic distribution: a dot at every visible point of
    // the ion; the annotation always names the ion by its base (argmax) peak
    const byIon = new Map();
    for (let i = 0; i < w.masses.length; i++) {
      const id = w.ion_id[i];
      if (id < 0) continue;
      if (!byIon.has(id)) byIon.set(id, []);
      byIon.get(id).push(i);
    }
    byIon.forEach((idxs, id) => {
      const ion = S.ions[id];
      const name = ion ? `ion ${id} · m/z ${ion.mz.toFixed(3)} · z=${ion.charge}` : `ion ${id}`;
      traces.push({
        x: idxs.map((i) => w.masses[i]), y: idxs.map((i) => w.ints[i]),
        mode: "markers", name, showlegend: false,
        marker: { color: ionColor(id), size: 5, opacity: 0.9,
                  line: { color: "#ffffff", width: 0.5 } },
        hovertemplate: `<b>${name}</b><br>m/z %{x:.4f}<br>I %{y:.3g}<br>` +
          `base (argmax) peak m/z ${ion ? ion.mz.toFixed(4) : "?"}<extra></extra>`,
        "medusa-ion": id,
      });
    });
  }

  if (S.threshold && w.prob) {
    const xs = [], ys = [], ps = [];
    for (let i = 0; i < w.masses.length; i++) {
      if (w.prob[i] > S.threshold.threshold) {
        xs.push(w.masses[i]); ys.push(w.ints[i]); ps.push(w.prob[i]);
      }
    }
    if (xs.length) {
      traces.push({
        x: xs, y: ys, mode: "markers",
        name: `P(${S.threshold.element}) > ${S.threshold.threshold}`,
        marker: { color: "#e03131", size: 4, opacity: 0.85 },
        showlegend: true,
        hovertemplate: `m/z %{x:.4f}<br>I %{y:.3g}<br>P %{customdata:.4f}<extra></extra>`,
        customdata: ps,
      });
    }
  }

  const layout = {
    margin: { l: 60, r: 20, t: 30, b: 40 },
    xaxis: { title: "m/z", range: currentRange },
    yaxis: { title: "Intensity" },
    dragmode: "zoom",
    showlegend: true,
    legend: { orientation: "h", y: -0.18, font: { size: 10 } },
    height: $("#spectrum-plot").clientHeight || 420,
  };
  const el = $("#spectrum-plot");
  const bindEvents = () => {
    if (el.__medusaEvents) return;
    el.__medusaEvents = true;
    el.on("plotly_relayout", onRelayout);
    el.on("plotly_click", onPlotClick);
  };
  if (el.data && el.data.length) {
    Plotly.react(el, traces, layout, { responsive: true, scrollZoom: true }).then(bindEvents);
  } else {
    Plotly.newPlot(el, traces, layout, { responsive: true, scrollZoom: true }).then(bindEvents);
  }
}

function onRelayout( eventData ) {
  if (!S.session) return;
  const range = eventData["xaxis.range"] || eventData["xaxis.range[0]"] && [eventData["xaxis.range[0]"], eventData["xaxis.range[1]"]];
  if (!range) return;
  clearTimeout(zoomTimer);
  zoomTimer = setTimeout(() => refreshWindow(range[0], range[1]), 150);
}

function onPlotClick(eventData) {
  if (!eventData || !eventData.length || !S.window) return;
  const pt = eventData[0];
  const ionId = pt.trace["medusa-ion"];
  if (ionId !== undefined && ionId >= 0) { selectIon(ionId); return; }
  // raw trace: use the parallel ion_id array of the current window
  if (S.window.ion_id && pt.pointNumber !== undefined) {
    const id = S.window.ion_id[pt.pointNumber];
    if (id !== undefined && id >= 0) selectIon(id);
  }
}

// ------------------------------------------------------------------ //
// ion selection
// ------------------------------------------------------------------ //
function selectIon(id, scrollTable = true) {
  if (id === null || (S.ions.length && id >= S.ions.length)) return;
  S.selectedIon = id;
  $("#f-ion").value = id;
  $$("#ions-table tbody tr").forEach((tr) =>
    tr.classList.toggle("selected", +tr.dataset.ion === id));
  if (scrollTable) {
    const tr = $(`#ions-table tbody tr[data-ion="${id}"]`);
    if (tr) tr.scrollIntoView({ block: "nearest" });
  }
  if (S.ions[id]) {
    // auto zoom region of interest after selection
    const ion = S.ions[id];
    const span = Math.max(ion.mz_max - ion.mz_min, 2);
    refreshWindow(ion.mz - span, ion.mz + span);
  }
}

// ------------------------------------------------------------------ //
// step 2: deisotoping
// ------------------------------------------------------------------ //
async function runDeisotope() {
  const body = {
    algorithm: $("#d-algorithm").value,
    z_max: +$("#d-zmax").value,
    threshold: +$("#d-threshold").value,
    delta: +$("#d-delta").value,
    min_distance: +$("#d-mindist").value,
    n1: +$("#d-n1").value,
    n2: +$("#d-n2").value,
  };
  await withBusy("deiso", async () => {
    try {
      const r = await api(`/api/sessions/${S.session}/deisotope`, { method: "POST", body });
      S.ions = r.ions;
      renderLogs(r.logs);
      $("#deiso-summary").hidden = false;
      $("#deiso-summary").innerHTML =
        `<b>${r.n_ions}</b> ions in ${r.elapsed_s}s${r.reused ? " (cached)" : ""}<br>` +
        `<span class="muted">peaks: ${S.ions.reduce((a, i) => a + i.n_peaks, 0).toLocaleString()}</span>`;
      renderIonLegend();
      renderIonsTable();
      $("#ions-toggle").disabled = false;
      $("#layer-ions").disabled = false;
      if (r.n_ions) {
        const best = [...S.ions].sort((a, b) => b.n_peaks - a.n_peaks)[0];
        $("#f-ion").value = best.ion_id;
        S.selectedIon = best.ion_id;
      }
      renderSpectrum();
    } catch (err) {
      toast(err.message, true);
    }
  });
}

function renderIonLegend() {
  const ul = $("#ion-legend-list");
  ul.innerHTML = S.ions.map((ion) =>
    `<li data-ion="${ion.ion_id}">
       <span class="swatch" style="background:${ionColor(ion.ion_id)}"></span>
       <span>ion ${ion.ion_id} · m/z ${ion.mz.toFixed(4)} · z=${ion.charge} · ${ion.n_peaks} peaks</span>
     </li>`).join("");
  $$("#ion-legend-list li").forEach((li) =>
    (li.onclick = () => selectIon(+li.dataset.ion)));
  $("#ion-legend").hidden = false;
}

// ------------------------------------------------------------------ //
// ions table (right)
// ------------------------------------------------------------------ //
let ionsSort = { key: "prob", dir: -1 };
function renderIonsTable() {
  const rows = S.ions.map((ion) => ({
    ...ion, prob: S.ionProbs[ion.ion_id] ?? null,
  }));
  const { key, dir } = ionsSort;
  rows.sort((a, b) => {
    const av = a[key] ?? -Infinity, bv = b[key] ?? -Infinity;
    return (av < bv ? -1 : av > bv ? 1 : 0) * dir;
  });
  $("#ions-count").textContent = rows.length ? `(${rows.length})` : "";
  $("#ions-table tbody").innerHTML = rows.map((r) =>
    `<tr data-ion="${r.ion_id}" class="${r.ion_id === S.selectedIon ? "selected" : ""}">
       <td class="num">${r.ion_id}</td>
       <td class="num">${r.mz.toFixed(4)}</td>
       <td class="num">${r.charge}</td>
       <td class="num">${r.n_peaks}</td>
       <td class="num">${r.prob === null ? "—" : r.prob.toExponential(2)}</td>
     </tr>`).join("");
  $$("#ions-table tbody tr").forEach((tr) =>
    (tr.onclick = () => selectIon(+tr.dataset.ion)));
  $$("#ions-table th").forEach((th) => {
    th.textContent = th.dataset.sort === "prob"
      ? `P(${S.element})${ionsSort.key === "prob" ? (ionsSort.dir < 0 ? " ↓" : " ↑") : ""}`
      : th.textContent;
  });
}
$$("#ions-table th").forEach((th) => {
  th.onclick = () => {
    const key = th.dataset.sort;
    ionsSort = { key, dir: ionsSort.key === key ? -ionsSort.dir : -1 };
    renderIonsTable();
  };
});

// ------------------------------------------------------------------ //
// step 3: elements
// ------------------------------------------------------------------ //
async function runElements() {
  const element = ($("#e-element").value || "Ir").trim();
  await withBusy("elements", async () => {
    try {
      const r = await api(`/api/sessions/${S.session}/elements`, { method: "POST", body: { element } });
      S.element = r.element;
      S.ionProbs = {};
      r.rows.forEach((row) => { if (row.prob !== null) S.ionProbs[row.ion_id] = row.prob; });
      renderLogs(r.logs);
      $("#elements-summary").hidden = false;
      $("#elements-summary").className = r.skipped.length ? "summary warn" : "summary";
      $("#elements-summary").innerHTML =
        `P(${escapeHtml(r.element)}) for ${r.rows.length} ions${r.reused ? " (cached)" : ""}` +
        (r.skipped.length ? `<br><span class="warn">${r.skipped.length} ion(s) skipped: [${r.skipped.join(", ")}]</span>` : "");
      renderIonsTable();
      fetchKnee();
    } catch (err) {
      toast(err.message, true);
    }
  });
}

// ------------------------------------------------------------------ //
// step 4: knee + threshold
// ------------------------------------------------------------------ //
async function fetchKnee() {
  if (!S.session || !S.ions.length) return;
  try {
    const r = await api(`/api/sessions/${S.session}/knee?element=${encodeURIComponent(S.element)}`);
    S.knee = r;
    $("#knee-summary").hidden = false;
    $("#knee-summary").innerHTML =
      `knee: P(${escapeHtml(r.element)}) ≥ <b>${r.threshold.toExponential(2)}</b> ` +
      `(idx ${r.knee_idx} of ${r.n_ions})`;
    $("#thr-slider").value = Math.min(1, Math.max(0, r.threshold));
    $("#thr-slider-val").textContent = Number(r.threshold).toFixed(3);
    renderKneePlot();
  } catch (err) {
    toast(err.message, true);
  }
}

function renderKneePlot() {
  const k = S.knee;
  if (!k) return;
  const x = k.probs.map((_, i) => i);
  const traces = [
    { x, y: k.probs, mode: "lines+markers", name: "P (sorted desc)",
      line: { color: "#2e6fb7" }, marker: { size: 4 } },
    { x: [0, k.probs.length - 1], y: [k.threshold, k.threshold], mode: "lines",
      name: `threshold ${k.threshold.toExponential(1)}`,
      line: { color: "#e03131", dash: "dash" }, hoverinfo: "skip" },
    { x: [k.knee_idx], y: [k.probs[k.knee_idx]], mode: "markers",
      name: `knee (${k.knee_idx})`,
      marker: { color: "#f59f00", size: 11, line: { color: "#862e00", width: 2 } } },
  ];
  const layout = {
    margin: { l: 55, r: 15, t: 25, b: 35 },
    yaxis: { type: "log", title: "P (log)" },
    xaxis: { title: "ion rank (sorted by P, descending)" },
    showlegend: false,
    height: 220,
  };
  const el = $("#knee-plot");
  if (el.data && el.data.length) Plotly.react(el, traces, layout);
  else Plotly.newPlot(el, traces, layout);
}

async function applyThreshold() {
  const source = document.querySelector('input[name="thr-source"]:checked').value;
  const body = {
    element: S.element,
    source,
    manual_value: source === "manual" ? +$("#thr-slider").value : null,
  };
  try {
    const r = await api(`/api/sessions/${S.session}/threshold`, { method: "POST", body });
    S.threshold = r;
    renderLogs(r.logs);
    $("#highlight-summary").hidden = false;
    $("#highlight-summary").className = r.n_points_above ? "summary" : "summary warn";
    $("#highlight-summary").innerHTML =
      `P(${escapeHtml(r.element)}) > <b>${r.threshold.toExponential(2)}</b> ` +
      `(${r.source}): <b>${r.n_points_above}</b> points, ${r.n_ions_above} ions` +
      (r.mz_range ? ` · m/z ${r.mz_range[0].toFixed(2)}–${r.mz_range[1].toFixed(2)}` : "");
    $("#zoom-highlighted-btn").disabled = !r.mz_range;
    $("#layer-probs").disabled = false;
    renderSpectrum();
  } catch (err) {
    toast(err.message, true);
  }
}

// ------------------------------------------------------------------ //
// step 6: formulas
// ------------------------------------------------------------------ //
async function loadPresets() {
  const data = await api("/api/formula_presets");
  const sel = $("#f-preset");
  Object.keys(data).forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name; opt.textContent = name;
    sel.appendChild(opt);
  });
  sel.onchange = () => {
    if (sel.value) setLimitsRows(Object.entries(data[sel.value]).map(([el, [lo, hi]]) => [el, lo, hi]));
  };
}
function setLimitsRows(rows) {
  const tbody = $("#f-elements tbody");
  tbody.innerHTML = "";
  rows.forEach((row) => addLimitRow(row[0], row[1], row[2]));
}
function addLimitRow(el = "", lo = 0, hi = 0) {
  const tbody = $("#f-elements tbody");
  const tr = document.createElement("tr");
  tr.innerHTML =
    `<td><input class="sym" value="${escapeHtml(el)}" list="element-list"></td>
     <td><input type="number" class="lo" min="0" value="${lo}"></td>
     <td><input type="number" class="hi" min="0" value="${hi}"></td>
     <td><button class="rm" title="remove row">✕</button></td>`;
  tr.querySelector(".rm").onclick = () => tr.remove();
  tbody.appendChild(tr);
}
function readLimits() {
  const out = {};
  $$("#f-elements tbody tr").forEach((tr) => {
    const el = tr.querySelector(".sym").value.trim();
    const lo = +tr.querySelector(".lo").value;
    const hi = +tr.querySelector(".hi").value;
    if (!el) return;
    if (Number.isNaN(lo) || Number.isNaN(hi)) throw new Error(`invalid limits for ${el}`);
    out[el] = [lo, hi];
  });
  return out;
}

async function runFormulas() {
  const ionId = +$("#f-ion").value;
  const body = {
    ion_id: ionId,
    elements: readLimits(),
    mass_threshold_ppm: +$("#f-ppm").value,
    num_workers: +$("#f-workers").value,
    max_chunk_size: 10000,
  };
  await withBusy("formulas", async () => {
    try {
      const r = await api(`/api/sessions/${S.session}/formulas`, { method: "POST", body });
      S.ranked = r.ranked;
      renderLogs(r.logs);
      $("#formulas-summary").hidden = false;
      $("#formulas-summary").innerHTML =
        `<b>${r.n_valid}</b> valid of ${r.n_candidates} candidates ` +
        `in ${r.elapsed_s}s${r.reused ? " (cached)" : ""}<br>` +
        `<span class="muted">target m/z ${r.target_mz.toFixed(4)}, z=${r.target_charge}, ` +
        `m=${r.target_mass.toFixed(4)}; search space skipped ${r.skipped_pct.toFixed(1)}%` +
        (r.n_failed ? `, ${r.n_failed} check failures` : "") + `</span>`;
      $("#formulas-count").textContent = `(${r.n_valid})`;
      renderFormulasTable();
      if (S.ranked.length) $("#c-formula").value = S.ranked[0].formula;
      $("#csv-export-btn").disabled = false;
    } catch (err) {
      toast(err.message, true);
    }
  });
}

function renderFormulasTable() {
  $("#formulas-table tbody").innerHTML = S.ranked.map((r) =>
    `<tr data-rank="${r.rank}">
       <td class="num">${r.rank}</td>
       <td>${escapeHtml(r.formula)}</td>
       <td class="num">${r.mass.toFixed(4)}</td>
       <td class="num">${r.delta_ppm.toFixed(3)}</td>
       <td class="num">${r.cosine.toFixed(6)}</td>
     </tr>`).join("");
  $$("#formulas-table tbody tr").forEach((tr) =>
    (tr.onclick = () => {
      const row = S.ranked[+tr.dataset.rank - 1];
      if (row) $("#c-formula").value = row.formula;
    }));
}

function exportCSV() {
  if (!S.ranked || !S.ranked.length) return;
  const header = "rank,formula,mass,delta_ppm,cosine";
  const lines = S.ranked.map((r) =>
    `${r.rank},${r.formula},${r.mass.toFixed(6)},${r.delta_ppm.toFixed(4)},${r.cosine.toFixed(8)}`);
  const blob = new Blob([header + "\n" + lines.join("\n") + "\n"], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `formulas_ion${S.selectedIon ?? $("#f-ion").value}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ------------------------------------------------------------------ //
// step 7: compare
// ------------------------------------------------------------------ //
async function runCompare() {
  const formula = $("#c-formula").value.trim();
  const ionId = +$("#f-ion").value;
  await withBusy("compare", async () => {
    try {
      const fig = await api(
        `/api/sessions/${S.session}/compare?ion_id=${ionId}&formula=${encodeURIComponent(formula)}`
      );
      const el = $("#compare-plot");
      el.hidden = false;
      $("#compare-status").textContent =
        `comparing ${formula} with ion ${ionId} (z=${S.ions[ionId] ? S.ions[ionId].charge : "?"})`;
      const layout2 = Object.assign({}, fig.layout,
        { height: Math.max(el.clientHeight || 0, 560) });
      if (el.data && el.data.length) Plotly.react(el, fig.data, layout2, { responsive: true });
      else Plotly.newPlot(el, fig.data, layout2, { responsive: true });
    } catch (err) {
      toast(err.message, true);
    }
  });
}

// ------------------------------------------------------------------ //
// wiring
// ------------------------------------------------------------------ //
async function init() {
  // element symbols for validation
  try {
    const d = await api("/api/elements");
    S.elementSymbols = d.elements;
    $("#element-list").innerHTML =
      d.elements.map((e) => `<option value="${e}">`).join("");
  } catch (_) {}

  await loadPresets();
  setLimitsRows([["C", 0, 90], ["H", 0, 90], ["N", 0, 10], ["O", 0, 10], ["I", 0, 2], ["Ir", 1, 1]]);

  try { await refreshFileList(); }
  catch (err) { toast("could not list server files: " + err.message, true); }

  // tabs
  $$(".tab").forEach((t) => (t.onclick = () => {
    $$(".tab").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    $("#tab-files").hidden = t.dataset.tab !== "files";
    $("#tab-upload").hidden = t.dataset.tab !== "upload";
  }));
  $("#file-search").oninput = () => renderFileList($("#file-search").value);

  // upload
  const dz = $("#dropzone");
  dz.onclick = () => $("#upload-input").click();
  ["dragover", "dragenter"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); }));
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) startUpload(f);
  });
  $("#upload-input").onchange = (e) => {
    if (e.target.files && e.target.files[0]) startUpload(e.target.files[0]);
  };
  function startUpload(file) {
    uploadFile(file, (res) => createSession("upload", { file_id: res.file_id }));
  }

  // runs
  $$("[data-run='deiso']").forEach((b) => (b.onclick = () => S.session && runDeisotope()));
  $$("[data-run='elements']").forEach((b) => (b.onclick = () => S.session && runElements()));
  $$("[data-run='formulas']").forEach((b) => (b.onclick = () => S.session && runFormulas()));
  $$("[data-run='compare']").forEach((b) => (b.onclick = () => S.session && runCompare()));
  $("#apply-threshold-btn").onclick = () => S.session && applyThreshold();
  $("#zoom-highlighted-btn").onclick = () => {
    if (S.threshold && S.threshold.mz_range)
      refreshWindow(S.threshold.mz_range[0] - 0.5, S.threshold.mz_range[1] + 0.5);
  };
  $("#f-add-row").onclick = () => addLimitRow();
  $("#csv-export-btn").onclick = exportCSV;

  // element input validation
  $("#e-element").addEventListener("change", () => {
    const v = $("#e-element").value.trim();
    if (v && !S.elementSymbols.includes(v)) {
      toast(`unknown element ${v}`, true);
      $("#e-element").value = S.element;
    }
  });

  // threshold slider
  const slider = $("#thr-slider");
  slider.oninput = () => { $("#thr-slider-val").textContent = (+slider.value).toFixed(3); };
  $$('input[name="thr-source"]').forEach((r) =>
    r.onchange = () => (slider.disabled = r.value !== "manual" || !r.checked));

  // viewer layer toggles (kept in sync)
  $("#layer-ions").onchange = (e) => { $("#ions-toggle").checked = e.target.checked; renderSpectrum(); };
  $("#ions-toggle").onchange = (e) => { $("#layer-ions").checked = e.target.checked; renderSpectrum(); };
  $("#layer-probs").onchange = () => renderSpectrum();

  // f-ion sync
  $("#f-ion").addEventListener("change", (e) => {
    const v = +e.target.value;
    if (!Number.isNaN(v) && v >= 0) { S.selectedIon = v; renderIonsTable(); }
  });

  // new session
  $("#new-session-btn").onclick = async () => {
    if (S.session && !confirm("Delete the current session and start fresh?")) return;
    if (S.session) await api(`/api/sessions/${S.session}`, { method: "DELETE" }).catch(() => {});
    S.session = null;
    S.fileInfo = null;
    updateHeader();
    resetDownstreamState();
    $("#load-summary").hidden = true;
    $("#spectrum-status").textContent = "load a spectrum to start";
    const el = $("#spectrum-plot");
    if (el.data) Plotly.purge(el);
    const cmp = $("#compare-plot");
    if (cmp.data) Plotly.purge(cmp);
    const kp = $("#knee-plot");
    if (kp.data) Plotly.purge(kp);
  };

  // responsive plot heights
  window.addEventListener("resize", () => {
    ["#spectrum-plot", "#compare-plot"].forEach((sel) => {
      const el = $(sel);
      if (el && el.data) Plotly.Plots.resize(el);
    });
  });

  renderLogs([]);
}

init().catch((err) => toast("init failed: " + err.message, true, 8000));
