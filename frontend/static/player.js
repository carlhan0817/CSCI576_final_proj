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

async function loadVideoList(selectVideoId) {
  const r = await fetch("/api/videos");
  const list = await r.json();
  const sel = $("video-picker");
  // Reset, keep the placeholder option
  sel.innerHTML = '<option value="">— pick a video —</option>';
  for (const v of list) {
    const opt = document.createElement("option");
    opt.value = v.video_id;
    opt.textContent = `${v.video_id} (${v.duration_sec.toFixed(1)}s)` +
      (v.verified_by_human ? " ✓" : "");
    sel.appendChild(opt);
  }
  if (selectVideoId) {
    sel.value = selectVideoId;
    if (sel.value === selectVideoId) {
      await loadVideo(selectVideoId);
    }
  }
}

// ── Upload + background-job polling ──────────────────────────────────────────
const POLL_INTERVAL_MS = 2000;

function setUploadStatus(msg, kind) {
  const el = $("upload-status");
  el.textContent = msg;
  el.className = kind ? `upload-${kind}` : "";
}

async function uploadVideo() {
  const input = $("upload-file");
  const btn = $("upload-btn");
  const f = input.files && input.files[0];
  if (!f) {
    setUploadStatus("pick an .mp4 file first", "error");
    return;
  }
  if (!f.name.toLowerCase().endsWith(".mp4")) {
    setUploadStatus("only .mp4 accepted", "error");
    return;
  }

  btn.disabled = true;
  input.disabled = true;
  setUploadStatus(`uploading ${f.name} (${(f.size / 1e6).toFixed(1)} MB)…`, "running");

  const fd = new FormData();
  fd.append("file", f, f.name);

  let resp;
  try {
    resp = await fetch("/api/upload", { method: "POST", body: fd });
  } catch (e) {
    setUploadStatus("upload network error: " + e, "error");
    btn.disabled = false;
    input.disabled = false;
    return;
  }
  if (!resp.ok) {
    let detail = String(resp.status);
    try { detail += " " + JSON.stringify(await resp.json()); } catch (_) {}
    setUploadStatus("upload failed: " + detail, "error");
    btn.disabled = false;
    input.disabled = false;
    return;
  }
  const { job_id, video_id, filename } = await resp.json();
  setUploadStatus(`queued (${filename}). analyzing… this can take several minutes.`, "running");
  pollJob(job_id, video_id);
}

function pollJob(jobId, videoId) {
  const btn = $("upload-btn");
  const input = $("upload-file");
  const handle = setInterval(async () => {
    let r;
    try {
      r = await fetch(`/api/jobs/${jobId}`);
    } catch (e) {
      setUploadStatus("poll error: " + e, "error");
      return;
    }
    if (!r.ok) {
      clearInterval(handle);
      setUploadStatus(`job lookup failed: ${r.status}`, "error");
      btn.disabled = false;
      input.disabled = false;
      return;
    }
    const job = await r.json();
    const elapsed = Math.round(job.elapsed_sec || 0);
    if (job.status === "queued") {
      setUploadStatus(`queued… (${elapsed}s)`, "running");
    } else if (job.status === "running") {
      setUploadStatus(`analyzing ${job.filename}… (${elapsed}s elapsed)`, "running");
    } else if (job.status === "done") {
      clearInterval(handle);
      setUploadStatus(`done in ${elapsed}s — ${job.video_id} loaded`, "ok");
      btn.disabled = false;
      input.disabled = false;
      input.value = "";
      await loadVideoList(videoId);
    } else if (job.status === "error") {
      clearInterval(handle);
      setUploadStatus(`failed: ${job.error || "unknown error"}`, "error");
      btn.disabled = false;
      input.disabled = false;
    }
  }, POLL_INTERVAL_MS);
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

function playContentOnly() {
  if (!state.metadata) return;
  const first = state.metadata.segments.find(s => s.label === "core_content");
  if (!first) { setStatus("no core_content segments found"); return; }
  state.skipMode = true;
  $("skip-non-content").checked = true;
  seekTo(first.start_sec);
  setStatus("playing content only — non-content will be skipped");
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
  $("upload-btn").addEventListener("click", uploadVideo);
  $("play-content-only").addEventListener("click", playContentOnly);
});
