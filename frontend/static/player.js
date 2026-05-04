"use strict";

const LABELS = [
  "core_content", "intro", "outro", "sponsorship", "self_promotion",
  "recap", "transition", "dead_air", "holding_screen", "filler",
];

const state = {
  videoId: null,
  metadata: null,
  durationSec: null,
  skipMode: false,
  dirty: false,
};

const $ = (id) => document.getElementById(id);

async function loadVideoList() {
  const r = await fetch("/api/videos");
  const list = await r.json();
  const sel = $("video-picker");
  for (const v of list) {
    const opt = document.createElement("option");
    opt.value = v.video_id;
    opt.textContent = `${v.video_id} (${v.duration_sec.toFixed(1)}s)` +
      (v.verified_by_human ? " ✓" : "");
    sel.appendChild(opt);
  }
}

async function loadVideo(videoId) {
  if (!videoId) return;
  state.videoId = videoId;
  state.dirty = false;

  // Load metadata
  const r = await fetch(`/api/metadata/${videoId}`);
  if (!r.ok) {
    setStatus(`metadata load failed: ${r.status}`);
    return;
  }
  state.metadata = await r.json();
  state.durationSec = state.metadata.video_info.duration_sec;

  // Point the <video> at the MP4
  const player = $("player");
  player.src = `/videos/${state.metadata.video_info.filename}`;

  renderTimeline();
  renderTable();
  setStatus(`loaded ${state.metadata.segments.length} segments`);
}

function renderTimeline() {
  const tl = $("timeline");
  tl.innerHTML = "";
  const total = state.durationSec;
  for (const seg of state.metadata.segments) {
    const block = document.createElement("div");
    const widthPct = ((seg.end_sec - seg.start_sec) / total) * 100;
    block.style.width = widthPct + "%";
    block.className = "label-" + seg.label;
    block.title = `[${seg.label}] ${seg.start_sec.toFixed(1)}–${seg.end_sec.toFixed(1)}s`;
    block.addEventListener("click", () => seekTo(seg.start_sec));
    tl.appendChild(block);
  }
}

function renderTable() {
  const tbody = $("segment-table").querySelector("tbody");
  tbody.innerHTML = "";
  for (const seg of state.metadata.segments) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${seg.segment_id}</td>
      <td>${seg.start_sec.toFixed(1)}</td>
      <td>${seg.end_sec.toFixed(1)}</td>
      <td><select data-seg-id="${seg.segment_id}">
        ${LABELS.map(l =>
          `<option value="${l}" ${l === seg.label ? "selected" : ""}>${l}</option>`
        ).join("")}
      </select></td>
      <td>${seg.confidence.toFixed(2)}</td>
      <td>${escapeHtml(seg.summary || "")}</td>
    `;
    tr.classList.add("label-" + seg.label);
    tbody.appendChild(tr);
  }
  for (const sel of tbody.querySelectorAll("select[data-seg-id]")) {
    sel.addEventListener("change", onLabelChange);
  }
}

function seekTo(sec) {
  $("player").currentTime = sec;
  $("player").play().catch(() => {/* user gesture required; ignore */});
}

function setStatus(msg) {
  $("status").textContent = msg;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function nextSegment() {
  if (!state.metadata) return;
  const now = $("player").currentTime;
  const next = state.metadata.segments.find(s => s.start_sec > now + 0.05);
  if (next) seekTo(next.start_sec);
}

function onTimeUpdate() {
  if (!state.skipMode || !state.metadata) return;
  const now = $("player").currentTime;
  const here = state.metadata.segments.find(s => now >= s.start_sec && now < s.end_sec);
  if (here && here.label !== "core_content") {
    $("player").currentTime = here.end_sec + 0.05;
  }
}

function onLabelChange(ev) {
  const sel = ev.target;
  const segId = parseInt(sel.dataset.segId, 10);
  const newLabel = sel.value;
  const seg = state.metadata.segments.find(s => s.segment_id === segId);
  if (!seg) return;
  seg.label = newLabel;
  seg.user_corrected = true;
  state.dirty = true;
  renderTimeline();
  renderTable();
  setStatus("unsaved changes");
}

async function saveEdits() {
  if (!state.metadata) return;
  if (!state.dirty) {
    setStatus("nothing to save");
    return;
  }
  setStatus("saving…");
  const r = await fetch(`/api/metadata/${state.videoId}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(state.metadata),
  });
  if (!r.ok) {
    let detail = `${r.status}`;
    try { detail += " " + JSON.stringify(await r.json()); } catch (_) {}
    setStatus("save failed: " + detail);
    alert("Save failed: " + detail);
    return;
  }
  state.metadata = await r.json();
  state.dirty = false;
  renderTimeline();
  renderTable();
  setStatus("saved ✓ verified_by_human=true");
}

// ── Wire-up ──────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadVideoList();
  $("video-picker").addEventListener("change", (e) => loadVideo(e.target.value));
  $("next-seg").addEventListener("click", nextSegment);
  $("skip-non-content").addEventListener("change", (e) => {
    state.skipMode = e.target.checked;
  });
  $("player").addEventListener("timeupdate", onTimeUpdate);
  $("save-edits").addEventListener("click", saveEdits);
});
