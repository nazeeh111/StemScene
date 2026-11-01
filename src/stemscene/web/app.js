"use strict";

const $ = (id) => document.getElementById(id);
const token = document.querySelector('meta[name="stemscene-token"]').content;
const colors = ["#b6cba0", "#b5b3df", "#d7b28b", "#8bc5c4", "#d19eb0", "#c2c88e", "#8eb6dc", "#cba5d7"];
const ui = { project: null, selectedId: null, editId: null, config: { model_ready: false }, pending: false,
  jobId: null, pollTimer: null, span: [0, 0], playbackUrl: null, playbackKey: null, audioRequest: 0 };

function text(id, value) { $(id).textContent = value; }
function showError(error) {
  text("error-message", error instanceof Error ? error.message : String(error));
  $("error-panel").hidden = false;
}
function clearError() { $("error-panel").hidden = true; }
function status(message) { text("status-message", message); }
function busy() { return ui.pending || Boolean(ui.jobId); }
function selectedLayer() { return ui.project?.layers.find((layer) => layer.id === ui.selectedId); }
function time(value, precision = 2) {
  const safe = Math.max(0, Number(value) || 0);
  const minutes = Math.floor(safe / 60);
  return `${minutes}:${(safe % 60).toFixed(precision).padStart(precision ? 3 + precision : 2, "0")}`;
}
function colorFor(id) {
  const index = ui.project?.layers.findIndex((layer) => layer.id === id) ?? 0;
  return colors[Math.max(0, index) % colors.length];
}

async function api(path, options = {}) {
  const headers = { "X-StemScene-Token": token, ...(options.headers || {}) };
  let body = options.body;
  if (body !== undefined && !options.raw) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(body);
  }
  const response = await fetch(path, { method: options.method || "GET", headers, body,
    credentials: "same-origin", cache: "no-store" });
  if (!response.ok) {
    let message = `The local engine returned ${response.status}.`;
    try {
      const problem = await response.json();
      message = typeof problem.error === "string" ? problem.error : problem.message || message;
    } catch (_) { /* A non-JSON error still has a useful status. */ }
    throw new Error(message);
  }
  return options.blob ? response.blob() : response.json();
}

function updateControls() {
  const hasProject = Boolean(ui.project);
  const hasLayer = Boolean(selectedLayer());
  $("request-fields").disabled = busy() || !hasProject || !ui.config.model_ready;
  $("edit-fields").disabled = busy() || !hasLayer;
  $("import-button").disabled = busy();
  $("empty-import-button").disabled = busy();
  $("project-picker").disabled = busy();
  $("undo-button").disabled = busy() || !hasProject || ui.project.revision === 0;
  $("export-button").disabled = busy() || !hasProject;
  $("anchor-present").disabled = busy() || !hasLayer || selectedLayer()?.locked;
  $("anchor-absent").disabled = busy() || !hasLayer || selectedLayer()?.locked;
  $("new-event").disabled = busy() || !hasLayer || !ui.editId;
  document.querySelectorAll(".event-button").forEach((button) => { button.disabled = busy(); });
  $("listen-original").disabled = !hasProject;
  $("listen-mix").disabled = !hasProject;
  document.querySelectorAll(".lock-button").forEach((button) => { button.disabled = busy(); });
  document.querySelectorAll(".anchor-remove").forEach((button) => { button.disabled = busy() || selectedLayer()?.locked; });
  $("job-panel").hidden = !ui.jobId;
}

function renderLayers() {
  const list = $("layer-list");
  list.replaceChildren();
  const layers = ui.project?.layers || [];
  $("layers-empty").hidden = layers.length > 0;
  text("layer-count", layers.length);
  for (const layer of layers) {
    const card = document.createElement("div");
    card.className = `layer-card${layer.id === ui.selectedId ? " selected" : ""}`;
    card.style.setProperty("--layer-color", colorFor(layer.id));
    const select = document.createElement("button");
    select.type = "button"; select.className = "layer-select";
    select.setAttribute("aria-pressed", String(layer.id === ui.selectedId));
    select.setAttribute("aria-label", `Select ${layer.query}`);
    const name = document.createElement("strong"); name.textContent = layer.query;
    const identity = document.createElement("small");
    identity.textContent = `ESTIMATE · ${layer.id.slice(0, 8)}`;
    select.append(name, identity);
    select.addEventListener("click", () => { if(ui.selectedId !== layer.id) ui.editId = null; ui.selectedId = layer.id; renderLayers(); renderInspector(); updateControls(); });
    const actions = document.createElement("div"); actions.className = "layer-card-actions";
    const listen = document.createElement("button");
    listen.type = "button"; listen.className = "button"; listen.textContent = "▶ Listen";
    listen.setAttribute("aria-label", `Listen to estimated ${layer.query}`);
    listen.addEventListener("click", () => listenTo("stem", layer.id, layer.query));
    const lock = document.createElement("button");
    lock.type = "button"; lock.className = "button lock-button";
    lock.textContent = layer.locked ? "Locked" : "Lock estimate";
    lock.setAttribute("aria-pressed", String(Boolean(layer.locked)));
    lock.setAttribute("aria-label", `${layer.locked ? "Unlock" : "Lock"} ${layer.query} estimate`);
    lock.disabled = busy();
    lock.addEventListener("click", () => mutate("/api/lock", { layer_id: layer.id, locked: !layer.locked },
      layer.locked ? "Estimate unlocked." : "Estimate locked. Adding requests will keep its samples."));
    actions.append(listen, lock); card.append(select, actions); list.append(card);
  }
}

function renderInspector() {
  const layer = selectedLayer();
  $("inspector-empty").hidden = Boolean(layer);
  $("inspector-content").hidden = !layer;
  if (!layer) return;
  text("selected-query", layer.query);
  text("selected-id", `LAYER ${layer.id.slice(0, 12)}`);
  $("selected-color").style.background = colorFor(layer.id);
  const saved = ui.project.edits.find((edit) => edit.id === ui.editId && edit.layer_id === layer.id);
  text("event-mode", saved ? "Editing saved event" : "New event");
  text("apply-edit", saved ? "Update event" : "Apply level change");
  if (saved) {
    $("gain-range").max = String(Math.max(1,saved.gain));
    $("gain-range").value = String(saved.gain);
    $("fade-time").value = String(saved.fade_samples/ui.project.sample_rate);
  }
  text("gain-maximum", Number($("gain-range").max)>1 ? `${Math.round(Number($("gain-range").max)*100)}%` : "Original level");
  text("gain-output", `${Math.round(Number($("gain-range").value)*100)}%`);
  const querySet = layer.metadata?.query_set || [];
  const overlaps = layer.metadata?.refinement?.ambiguous_query_pairs || [];
  const related = overlaps.filter(([a,b]) => querySet[a] === layer.query || querySet[b] === layer.query)
    .flatMap(([a,b]) => [querySet[a],querySet[b]]).filter((query) => query && query !== layer.query);
  text("selected-uncertainty", related.length
    ? `This description overlaps ${[...new Set(related)].join(", ")}. These estimates may share a sound. Group them or add a useful anchor.`
    : "Estimated sound class. Listen for other sounds inside this layer before editing.");
  $("start-time").max = String(ui.project.duration);
  $("end-time").max = String(ui.project.duration);
  syncTimeFields();
  const list = $("anchor-list"); list.replaceChildren();
  (layer.anchors || []).forEach((anchor, index) => {
    const row = document.createElement("li");
    const kind = document.createElement("span"); kind.className = "anchor-kind";
    kind.textContent = anchor.present ? "PRESENT" : "ABSENT";
    const span = document.createElement("span"); span.textContent = `${time(anchor.start)}–${time(anchor.end)}`;
    const remove = document.createElement("button"); remove.type = "button"; remove.className = "anchor-remove";
    remove.textContent = "×"; remove.disabled = busy() || layer.locked;
    remove.setAttribute("aria-label", `Remove ${anchor.present ? "present" : "absent"} anchor from ${time(anchor.start)} to ${time(anchor.end)}`);
    remove.addEventListener("click", () => mutate("/api/anchors", {
      layer_id: layer.id, anchors: layer.anchors.filter((_, anchorIndex) => anchorIndex !== index)
    }, "Anchor removed. Estimate again to apply the remaining anchors."));
    row.append(kind, span, remove); list.append(row);
  });
}

function syncTimeFields() {
  const inputTime = (value) => String(Math.min(ui.project.duration, Math.max(0, Math.round(value*1e6)/1e6)));
  $("start-time").value = inputTime(ui.span[0]);
  $("end-time").value = inputTime(ui.span[1]);
  text("selection-summary", `${time(ui.span[0])}–${time(ui.span[1])} selected`);
  drawWaveform();
}

function setSpan(start, end) {
  if (!ui.project) return;
  const duration = ui.project.duration;
  ui.span = [Math.max(0, Math.min(start, duration)), Math.max(0, Math.min(end, duration))].sort((a, b) => a - b);
  syncTimeFields();
}

function renderProject(project) {
  const changedProject = !ui.project || project.id !== ui.project.id;
  const changedRevision = !ui.project || project.revision !== ui.project.revision;
  const oldEditCount = ui.project?.edits.length || 0;
  ui.project = project;
  if (changedProject) {
    const lastEdit = project.edits?.at(-1);
    ui.selectedId = lastEdit?.layer_id || project.layers[0]?.id || null;
    ui.editId = lastEdit?.id || null;
    ui.span = lastEdit ? [lastEdit.start_sample/project.sample_rate, lastEdit.end_sample/project.sample_rate]
      : [0, Math.min(5, project.duration)];
  } else if (!project.layers.some((layer) => layer.id === ui.selectedId)) {
    ui.selectedId = project.layers[0]?.id || null;
  }
  if (!changedProject && project.edits.length>oldEditCount) ui.editId = project.edits.at(-1).id;
  if (ui.editId && !project.edits.some((edit) => edit.id===ui.editId)) ui.editId = null;
  if (changedProject || changedRevision) resetPlayback();
  $("empty-recording").hidden = true;
  $("recording-content").hidden = false;
  text("recording-title", project.title || "Untitled recording");
  text("recording-meta", `${time(project.duration)} · ${Number(project.sample_rate).toLocaleString()} Hz · ${project.channels === 1 ? "Mono" : `${project.channels} channels`} · ${Number(project.frames).toLocaleString()} samples`);
  text("revision-label", `REV ${String(project.revision).padStart(3, "0")}`);
  text("duration-label", time(project.duration));
  $("project-picker").value = project.id;
  $("timeline-ticks").replaceChildren(...[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
    const tick = document.createElement("span"); tick.textContent = time(fraction * project.duration, 1); return tick;
  }));
  renderLayers(); renderInspector(); renderEvents(); updateControls(); syncTimeFields();
}

function newEvent() {
  ui.editId = null;
  $("gain-range").max = "1"; $("gain-range").value = ".25"; $("fade-time").value = ".02";
  renderInspector(); renderEvents(); updateControls();
}

function renderEvents() {
  const list = $("event-list"); list.replaceChildren();
  const events = ui.project?.edits || [];
  text("event-count", String(events.length));
  $("events-empty").hidden = events.length>0;
  $("events-limit").hidden = events.length<=100;
  for (const event of events.slice(-100).reverse()) {
    const layer = ui.project.layers.find((item) => item.id===event.layer_id);
    if (!layer) continue;
    const row = document.createElement("div"); row.className = `event-row${event.id===ui.editId ? " selected" : ""}`;
    const edit = document.createElement("button"); edit.type="button"; edit.className="event-button event-select";
    const name = document.createElement("strong"); name.textContent=layer.query;
    const detail = document.createElement("span"); detail.textContent=`${time(event.start_sample/ui.project.sample_rate)}–${time(event.end_sample/ui.project.sample_rate)} · ${Math.round(event.gain*100)}% gain`;
    edit.append(name,detail); edit.setAttribute("aria-label",`Edit saved ${layer.query} event`);
    edit.addEventListener("click",()=>{ui.selectedId=layer.id;ui.editId=event.id;ui.span=[event.start_sample/ui.project.sample_rate,event.end_sample/ui.project.sample_rate];renderLayers();renderInspector();renderEvents();updateControls();});
    const remove=document.createElement("button");remove.type="button";remove.className="event-button event-remove";remove.textContent="Remove";
    remove.setAttribute("aria-label",`Remove saved ${layer.query} event`);
    remove.addEventListener("click",()=>mutate("/api/remove-edit",{edit_id:event.id},"Event removed. Undo can restore it."));
    row.append(edit,remove);list.append(row);
  }
}

async function refreshProjects(selectedId = ui.project?.id) {
  const result = await api("/api/projects");
  const picker = $("project-picker");
  picker.replaceChildren();
  const empty = document.createElement("option"); empty.value = ""; empty.textContent = "Choose a saved recording"; picker.append(empty);
  for (const project of result.projects) {
    const option = document.createElement("option"); option.value = project.id; option.textContent = project.title || project.id; picker.append(option);
  }
  if (selectedId) picker.value = selectedId;
  return result.projects;
}

async function mutate(path, values, message) {
  if (!ui.project || busy()) return;
  ui.pending = true; updateControls(); clearError();
  try {
    const project = await api(path, { method: "POST", body: { id: ui.project.id, ...values } });
    renderProject(project); status(message);
  } catch (error) { showError(error); }
  finally { ui.pending = false; updateControls(); }
}

async function importFile(file) {
  if (!file || busy()) return;
  if (!/\.(wav|flac)$/i.test(file.name)) { showError("Choose a WAV or FLAC recording."); return; }
  const cap = Number(ui.config.limits?.max_source_bytes);
  if (cap && file.size > cap) { showError("This recording exceeds the local engine's source-file size limit."); return; }
  ui.pending = true; updateControls(); clearError(); status("Preserving and decoding your recording…");
  try {
    const project = await api("/api/import", { method: "POST", raw: true, body: file,
      headers: { "Content-Type": "application/octet-stream", "X-Filename": encodeURIComponent(file.name) } });
    await refreshProjects(project.id); renderProject(project);
    status("Recording imported. Describe the sound layers you want to estimate.");
  } catch (error) { showError(error); status(""); }
  finally { ui.pending = false; $("file-input").value = ""; updateControls(); }
}

async function compute(event) {
  event.preventDefault();
  if (busy() || !ui.project || !ui.config.model_ready) return;
  let queries = $("queries").value.split(/[,\n]/).map((query) => query.trim()).filter(Boolean);
  if (!queries.length) queries = ui.project.layers.map((layer) => layer.query);
  if (!queries.length) { showError("Describe at least one sound to estimate."); $("queries").focus(); return; }
  ui.pending = true; updateControls(); clearError(); status("");
  try {
    const result = await api("/api/extract", { method: "POST", body: { id: ui.project.id, queries } });
    ui.jobId = result.job_id; ui.pending = false;
    text("job-detail", "Computing from the original recording. Existing level edits are preserved.");
    $("job-progress").removeAttribute("value"); $("cancel-button").disabled = false;
    updateControls(); pollJob();
  } catch (error) { showError(error); ui.pending = false; updateControls(); }
}

async function pollJob() {
  if (!ui.jobId) return;
  const jobId = ui.jobId;
  try {
    const job = await api(`/api/job?id=${encodeURIComponent(jobId)}`);
    if (jobId !== ui.jobId) return;
    if (Number(job.total) > 0) {
      $("job-progress").max = job.total; $("job-progress").value = Math.max(0, Number(job.completed) || 0);
      text("job-detail", `${job.completed || 0} of ${job.total} layer steps complete. Local inference can take a while.`);
    }
    if (job.status === "running" || job.status === "pending" || job.status === "cancelling") {
      ui.pollTimer = setTimeout(pollJob, 1500); return;
    }
    ui.jobId = null; updateControls();
    if (job.status === "error") throw new Error(job.error || "Layer estimation did not complete. Your saved session is still available.");
    if (job.status === "cancelled") { status("Estimation cancelled. Saved layer estimates and edits remain available."); return; }
    const projectId = job.project_id || ui.project.id;
    renderProject(await api(`/api/state?id=${encodeURIComponent(projectId)}`));
    $("queries").value = "";
    status("Layer estimates saved. Listen to each layer before applying a level change.");
  } catch (error) {
    showError(error);
    if (ui.jobId) { text("job-detail", "Status is temporarily unavailable. Retrying the local engine…"); ui.pollTimer = setTimeout(pollJob, 4000); }
  }
}

function resetPlayback() {
  const audio = $("audio-player"); audio.pause(); audio.removeAttribute("src"); audio.load();
  audio.hidden = true;
  if (ui.playbackUrl) URL.revokeObjectURL(ui.playbackUrl);
  ui.playbackUrl = null; ui.playbackKey = null; ui.audioRequest += 1;
  text("playback-label", "Choose original, mix, or a layer."); $("stop-button").disabled = true;
  $("listen-original").classList.remove("active"); $("listen-mix").classList.remove("active");
}

async function listenTo(kind, layerId = "", query = "") {
  if (!ui.project) return;
  const projectId = ui.project.id;
  const requestId = ++ui.audioRequest;
  const label = kind === "original" ? "Original recording" : kind === "mix" ? "Edited mix" : `Estimated ${query}`;
  clearError(); text("playback-label", `Loading ${label.toLowerCase()}…`);
  try {
    const blob = await api(`/api/audio?id=${encodeURIComponent(projectId)}&kind=${kind}&layer_id=${encodeURIComponent(layerId)}`, { blob: true });
    if (requestId !== ui.audioRequest || projectId !== ui.project?.id) return;
    const audio = $("audio-player"); audio.pause();
    if (ui.playbackUrl) URL.revokeObjectURL(ui.playbackUrl);
    ui.playbackUrl = URL.createObjectURL(blob); ui.playbackKey = `${kind}:${layerId}`;
    audio.src = ui.playbackUrl; audio.hidden = false; text("playback-label", label); $("stop-button").disabled = false;
    $("listen-original").classList.toggle("active", kind === "original");
    $("listen-mix").classList.toggle("active", kind === "mix");
    try { await audio.play(); }
    catch (_) { status("Audio is ready. Use the playback control to listen."); }
  } catch (error) { if (requestId === ui.audioRequest) { showError(error); text("playback-label", "Audio could not load. Try again."); } }
}

function drawWaveform() {
  const canvas = $("waveform");
  if (!ui.project || !canvas.clientWidth) return;
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d"); context.scale(ratio, ratio); context.clearRect(0, 0, width, height);
  context.strokeStyle = "#343d37"; context.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick++) { const x = tick * width / 4; context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke(); }
  context.beginPath(); context.moveTo(0, height / 2); context.lineTo(width, height / 2); context.stroke();
  const bins = (ui.project.waveform || []).filter((point) => Number.isFinite(Number(point.min)) && Number.isFinite(Number(point.max)));
  $("waveform-empty").hidden = bins.length > 0;
  let peak = 0;
  for (const point of bins) peak = Math.max(peak, Math.abs(point.min), Math.abs(point.max));
  if (bins.length && peak > 0) {
    context.strokeStyle = "#b5c1b3"; context.lineWidth = Math.max(1, width / bins.length);
    const scale = height * 0.37 / peak;
    context.beginPath();
    bins.forEach((point, index) => { const x = (index + 0.5) * width / bins.length;
      context.moveTo(x, height / 2 - point.max * scale); context.lineTo(x, height / 2 - point.min * scale); });
    context.stroke();
  }
  if (ui.project.duration > 0) {
    const start = ui.span[0] / ui.project.duration * width;
    const end = ui.span[1] / ui.project.duration * width;
    context.fillStyle = "rgba(232,202,120,.12)"; context.fillRect(start, 0, end - start, height);
    context.strokeStyle = "#e8ca78"; context.lineWidth = 1;
    for (const x of [start, end]) { context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke(); }
    const audio = $("audio-player");
    if (ui.playbackUrl && Number.isFinite(audio.currentTime)) {
      const cursor = audio.currentTime / ui.project.duration * width;
      context.strokeStyle = "#a2d1be"; context.beginPath(); context.moveTo(cursor, 0); context.lineTo(cursor, height); context.stroke();
    }
  }
}

function validateSpan() {
  const start = Number($("start-time").value), end = Number($("end-time").value);
  if (!ui.project || !Number.isFinite(start) || !Number.isFinite(end) || start < 0 || start >= end || end > ui.project.duration) {
    throw new Error("Choose a start before the end, within this recording.");
  }
  return [start, end];
}

async function addAnchor(present) {
  const layer = selectedLayer(); if (!layer || busy()) return;
  try {
    const [start, end] = validateSpan();
    setSpan(start, end);
    await mutate("/api/anchors", { layer_id: layer.id, anchors: [...(layer.anchors || []), { start, end, present }] },
      "Anchor saved. Leave the request field blank and estimate again to apply anchors to existing layers.");
  } catch (error) { showError(error); }
}

async function exportProject() {
  if (!ui.project || busy()) return;
  ui.pending = true; updateControls(); clearError(); text("export-status", "Writing and checking aligned WAVs…");
  try {
    const result = await api("/api/export", { method: "POST", body: { id: ui.project.id, subtype: "DOUBLE", clipping: $("clipping-policy").value } });
    text("export-status", "Preparing the ZIP download…");
    const blob = await api(`/api/download?id=${encodeURIComponent(result.export_id)}`, { blob: true });
    const url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url; link.download = result.filename || "stemscene-export.zip"; document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
    text("export-status", "Export checked. Your browser is downloading the aligned WAV bundle.");
  } catch (error) { showError(error); text("export-status", "Export did not complete. Your saved session is unchanged."); }
  finally { ui.pending = false; updateControls(); }
}

$("dismiss-error").addEventListener("click", clearError);
$("import-button").addEventListener("click", () => $("file-input").click());
$("empty-import-button").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (event) => importFile(event.target.files[0]));
$("request-form").addEventListener("submit", compute);
$("project-picker").addEventListener("change", async (event) => {
  if (!event.target.value || busy()) return;
  ui.pending = true; updateControls(); clearError();
  try { renderProject(await api(`/api/state?id=${encodeURIComponent(event.target.value)}`)); status(""); }
  catch (error) { showError(error); }
  finally { ui.pending = false; updateControls(); }
});
$("undo-button").addEventListener("click", () => mutate("/api/undo", {}, "Previous session state restored."));
$("listen-original").addEventListener("click", () => listenTo("original"));
$("listen-mix").addEventListener("click", () => listenTo("mix"));
$("stop-button").addEventListener("click", () => { $("audio-player").pause(); $("audio-player").currentTime = 0; drawWaveform(); });
$("audio-player").addEventListener("timeupdate", drawWaveform);
$("gain-range").addEventListener("input", (event) => { text("gain-output", `${Math.round(Number(event.target.value) * 100)}%`); });
for (const id of ["start-time", "end-time"]) $(id).addEventListener("change", () => {
  try { const span = validateSpan(); ui.span = span; syncTimeFields(); } catch (error) { showError(error); }
});
for (const id of ["start-time", "end-time"]) $(id).addEventListener("input", () => {
  try {
    ui.span = validateSpan();
    text("selection-summary", `${time(ui.span[0])}–${time(ui.span[1])} selected`);
    drawWaveform();
  } catch (_) { /* Leave partially typed numbers untouched until submission. */ }
});
$("edit-form").addEventListener("submit", async (event) => {
  event.preventDefault(); const layer = selectedLayer(); if (!layer) return;
  try {
    const [start, end] = validateSpan();
    setSpan(start, end);
    const fade = Number($("fade-time").value), gain = Number($("gain-range").value);
    if (!Number.isFinite(fade) || fade < 0 || !Number.isFinite(gain)) throw new Error("Choose a finite fade and layer level.");
    await mutate("/api/edit", { layer_id: layer.id, start, end, gain, fade_seconds: fade, edit_id: ui.editId },
      `Level change saved for ${time(start)}–${time(end)}. Listen to the edited mix to check it.`);
  } catch (error) { showError(error); }
});
$("new-event").addEventListener("click",newEvent);
$("anchor-present").addEventListener("click", () => addAnchor(true));
$("anchor-absent").addEventListener("click", () => addAnchor(false));
$("export-button").addEventListener("click", exportProject);
$("cancel-button").addEventListener("click", async () => {
  if (!ui.jobId) return; $("cancel-button").disabled = true;
  try { await api("/api/cancel", { method: "POST", body: { job_id: ui.jobId } }); text("job-detail", "Cancellation requested. Finishing the current bounded inference step…"); }
  catch (error) { showError(error); $("cancel-button").disabled = false; }
});

let dragStart = null;
const canvas = $("waveform");
function pointerTime(event) { const box = canvas.getBoundingClientRect(); return Math.min(1, Math.max(0, (event.clientX - box.left) / box.width)) * ui.project.duration; }
canvas.addEventListener("pointerdown", (event) => {
  if (!ui.project || busy()) return;
  newEvent();
  dragStart = pointerTime(event); canvas.setPointerCapture(event.pointerId); setSpan(dragStart, dragStart);
});
canvas.addEventListener("pointermove", (event) => { if (dragStart !== null && ui.project) setSpan(dragStart, pointerTime(event)); });
canvas.addEventListener("pointerup", (event) => {
  if (dragStart === null || !ui.project) return;
  const end = pointerTime(event); setSpan(dragStart, end); dragStart = null;
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
});
canvas.addEventListener("pointercancel", () => { dragStart = null; });
const empty = $("empty-recording");
for (const type of ["dragenter", "dragover"]) empty.addEventListener(type, (event) => { event.preventDefault(); if (!busy()) empty.classList.add("drag-over"); });
for (const type of ["dragleave", "drop"]) empty.addEventListener(type, (event) => { event.preventDefault(); empty.classList.remove("drag-over"); });
empty.addEventListener("drop", (event) => importFile(event.dataTransfer.files[0]));
if (typeof ResizeObserver !== "undefined") new ResizeObserver(drawWaveform).observe($("waveform-wrap"));
else window.addEventListener("resize", drawWaveform);

async function initialize() {
  updateControls();
  try {
    ui.config = await api("/api/config");
    $("model-notice").hidden = Boolean(ui.config.model_ready);
    text("model-message", ui.config.model_message || "The local separation model is not ready. Import and saved-session editing remain available.");
    const projects = await refreshProjects();
    if (projects.length) renderProject(await api(`/api/state?id=${encodeURIComponent(projects[0].id)}`));
    updateControls();
  } catch (error) { showError(error); status("The local engine is unavailable. Keep this page open and reconnect to its local address."); }
}
initialize();
