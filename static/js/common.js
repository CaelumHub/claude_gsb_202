/* common.js — shared helpers: toasts, formatting, canvas renderers. */

/* ---------------------------------------------------------------- toasts */

function toast(msg, type = "info", ms = 3400) {
  let wrap = document.querySelector(".toast-wrap");
  if (!wrap) {
    wrap = document.createElement("div");
    wrap.className = "toast-wrap";
    document.body.appendChild(wrap);
  }
  const t = document.createElement("div");
  t.className = "toast " + type;
  t.textContent = msg;
  wrap.appendChild(t);
  setTimeout(() => t.remove(), ms);
}

/* -------------------------------------------------------------- formatting */

function fmtTime(s) {
  if (s == null || isNaN(s)) return "--:--";
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1).padStart(4, "0");
  return `${m}:${sec}`;
}

function fmtBytes(n) {
  if (n == null) return "";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return n.toFixed(n >= 10 ? 1 : 2) + " " + units[i];
}

function fmtHz(f) { return f >= 1000 ? (f / 1000).toFixed(2) + " kHz" : f.toFixed(0) + " Hz"; }

function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function $(sel, root) { return (root || document).querySelector(sel); }
function $$(sel, root) { return Array.from((root || document).querySelectorAll(sel)); }

/* ----------------------------------------------------------- canvas utils */

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.floor(rect.width));
  const h = Math.max(1, Math.floor(rect.height));
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

/* Inferno-style colormap for spectrograms / heatmaps. */
function colormap(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [
    [0.00, 11, 12, 43], [0.25, 74, 16, 92], [0.50, 173, 55, 60],
    [0.75, 240, 137, 33], [1.00, 252, 255, 164],
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, r0, g0, b0] = stops[i];
    const [t1, r1, g1, b1] = stops[i + 1];
    if (t <= t1) {
      const k = (t - t0) / (t1 - t0 || 1);
      return [r0 + (r1 - r0) * k, g0 + (g1 - g0) * k, b0 + (b1 - b0) * k];
    }
  }
  return [252, 255, 164];
}

function colorStyle(t) {
  const [r, g, b] = colormap(t);
  return `rgb(${r | 0},${g | 0},${b | 0})`;
}

/* ------------------------------------------------------------- waveform */

function drawWaveform(canvas, env, opts = {}) {
  const { ctx, w, h } = setupCanvas(canvas);
  const mins = env.mins || [], maxs = env.maxs || [];
  ctx.clearRect(0, 0, w, h);
  if (!mins.length) return;

  const color = opts.color || getComputedStyle(document.documentElement)
    .getPropertyValue("--accent").trim() || "#58a6ff";
  const mid = h / 2;
  const step = w / mins.length;

  // selection highlight
  if (opts.selection) {
    const [a, b] = opts.selection;
    ctx.fillStyle = "rgba(88,166,255,0.15)";
    ctx.fillRect(a * w, 0, (b - a) * w, h);
  }
  if (opts.playhead != null) {
    ctx.fillStyle = "rgba(63,185,80,0.7)";
    ctx.fillRect(opts.playhead * w - 1, 0, 2, h);
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < mins.length; i++) {
    const x = i * step;
    const yMin = mid - (maxs[i] || 0) * mid * 0.95;
    const yMax = mid - (mins[i] || 0) * mid * 0.95;
    ctx.moveTo(x, yMin);
    ctx.lineTo(x, yMax);
  }
  ctx.stroke();

  // centre line
  ctx.strokeStyle = "rgba(139,148,158,0.35)";
  ctx.beginPath();
  ctx.moveTo(0, mid); ctx.lineTo(w, mid);
  ctx.stroke();
}

/* ----------------------------------------------------------- spectrogram */

function drawSpectrogram(canvas, spec, opts = {}) {
  const { ctx, w, h } = setupCanvas(canvas);
  const data = spec.data || [];
  const rows = data.length;
  ctx.clearRect(0, 0, w, h);
  const pad = opts.pad || { l: 44, r: 12, t: 10, b: 26 };
  const pw = Math.max(1, Math.floor(w - pad.l - pad.r));
  const ph = Math.max(1, Math.floor(h - pad.t - pad.b));
  const specDuration = Number.isFinite(spec.duration) && spec.duration > 0 ? spec.duration :
    (spec.times && spec.times.length && spec.hop && spec.sr ?
      spec.times[spec.times.length - 1] + spec.hop / spec.sr / 2 : 0);
  const duration = Number.isFinite(opts.duration) && opts.duration > 0 ? opts.duration : specDuration;
  if (!rows || !(duration > 0)) {
    drawTimeAxis(ctx, pad, pw, ph, 0);
    return;
  }

  const cols = data[0].length;
  const edges = spec.time_edges || [];
  const dbMin = opts.dbMin ?? -100, dbMax = opts.dbMax ?? 0;

  const img = ctx.createImageData(pw, ph);
  const timeForColumn = (x) => duration > 0 ? duration * x / pw : 0;
  const timeRow = (t) => {
    if (edges.length === rows + 1) {
      if (t <= edges[0]) return 0;
      if (t >= edges[rows]) return rows - 1;
      let lo = 0, hi = rows - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (t < edges[mid + 1]) hi = mid; else lo = mid + 1;
      }
      return lo;
    }
    return Math.min(rows - 1, Math.floor(t / (duration || 1) * rows));
  };

  const frameRows = Array.from({ length: pw }, (_, x) => timeRow(timeForColumn(x)));

  for (let y = 0; y < ph; y++) {
    for (let x = 0; x < pw; x++) {
      const freqIdx = Math.floor((ph - 1 - y) / ph * cols);
      const v = data[frameRows[x]][freqIdx];
      const t01 = (v - dbMin) / (dbMax - dbMin);
      const [r, g, b] = colormap(t01);
      const idx = (y * pw + x) * 4;
      img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b; img.data[idx + 3] = 255;
    }
  }

  // Put through a temporary canvas so device-pixel-ratio scaling is applied.
  const off = document.createElement("canvas");
  off.width = pw; off.height = ph;
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, pad.l, pad.t, pw, ph);
  drawTimeAxis(ctx, pad, pw, ph, duration || 0);
}

/* -------------------------------------------------------------- line plot */

function timeTickPositions(duration) {
  if (!(duration > 0)) return [0];
  const rawStep = duration / 4;
  const power = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const steps = [1, 2, 2.5, 5, 10].map(v => v * power);
  const step = steps.find(v => v >= rawStep) || steps[steps.length - 1];
  const ticks = [0];
  for (let k = 1; k * step < duration - 1e-9; k++) ticks.push(k * step);
  ticks.push(duration);
  return ticks;
}

function drawTimeAxis(ctx, pad, pw, ph, duration) {
  ctx.save();
  ctx.strokeStyle = "rgba(139,148,158,0.25)";
  ctx.fillStyle = "rgba(139,148,158,0.7)";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "center";
  const y = pad.t + ph;
  ctx.beginPath();
  ctx.moveTo(pad.l, y);
  ctx.lineTo(pad.l + pw, y);
  ctx.stroke();
  for (const t of timeTickPositions(duration)) {
    const x = pad.l + (duration > 0 ? t / duration * pw : 0);
    ctx.fillText(fmtTime(t), Math.max(pad.l + 18, Math.min(pad.l + pw - 18, x)), y + 14);
  }
  ctx.restore();
}

function drawLinePlot(canvas, seriesList, opts = {}) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const pad = opts.pad || { l: 44, r: 12, t: 10, b: 26 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;

  let yMin = Infinity, yMax = -Infinity;
  let indexMax = 0;
  let duration = Number.isFinite(opts.duration) ? opts.duration : null;
  for (const s of seriesList) {
    indexMax = Math.max(indexMax, s.data.length);
    if (s.times && s.times.length) {
      if (!(duration > 0)) duration = s.times[s.times.length - 1];
    }
    for (const v of s.data) {
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    }
  }
  if (!isFinite(yMin)) return;
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  const range = yMax - yMin;
  if (!(duration > 0)) duration = opts.fallbackDuration || 0;

  // grid
  ctx.strokeStyle = "rgba(139,148,158,0.15)";
  ctx.fillStyle = "rgba(139,148,158,0.6)";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "right";
  for (let g = 0; g <= 4; g++) {
    const y = pad.t + ph * (g / 4);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();
    const val = yMax - range * (g / 4);
    ctx.fillText(fmtVal(val), pad.l - 4, y + 3);
  }

  if (duration > 0) {
    for (const t of timeTickPositions(duration)) {
      const x = pad.l + t / duration * pw;
      ctx.beginPath();
      ctx.moveTo(x, pad.t);
      ctx.lineTo(x, pad.t + ph);
      ctx.stroke();
    }
  }

  const xAtIndex = (i) => {
    if (duration > 0) return pad.l;
    return pad.l + (indexMax <= 1 ? 0 : i / (indexMax - 1) * pw);
  };
  const xAtTime = (t) => pad.l + (duration > 0 ? t / duration * pw : 0);
  const yAt = (v) => pad.t + (1 - (v - yMin) / range) * ph;

  for (const s of seriesList) {
    ctx.strokeStyle = s.color || "#58a6ff";
    ctx.lineWidth = s.width || 1.4;
    ctx.beginPath();
    for (let i = 0; i < s.data.length; i++) {
      const t = s.times && s.times[i] != null ? s.times[i] : null;
      const x = t != null && duration > 0 ? xAtTime(t) : xAtIndex(i);
      const y = yAt(s.data[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  if (duration > 0) drawTimeAxis(ctx, pad, pw, ph, duration);
}

function fmtVal(v) {
  if (Math.abs(v) >= 1000) return (v / 1000).toFixed(1) + "k";
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 1) return v.toFixed(1);
  return v.toFixed(3);
}

/* --------------------------------------------------------------- heatmap */

function drawHeatmap(canvas, matrix, opts = {}) {
  const { ctx, w, h } = setupCanvas(canvas);
  const rows = matrix.length;
  ctx.clearRect(0, 0, w, h);
  if (!rows) return;
  const img = ctx.createImageData(w, h);
  const cols = matrix[0].length;
  for (let y = 0; y < h; y++) {
    const r = Math.floor(y / h * rows);
    const row = matrix[r];
    for (let x = 0; x < w; x++) {
      const c = Math.floor(x / w * cols);
      const [cr, cg, cb] = colormap(row[c]);
      const idx = (y * w + x) * 4;
      img.data[idx] = cr; img.data[idx + 1] = cg; img.data[idx + 2] = cb; img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/* ----------------------------------------------------------- play helpers */

function playFile(id, name) {
  const url = API.fileUrl(id);
  const audio = new Audio(url);
  audio.play().catch(() => toast("无法播放该音频", "error"));
  toast("播放 " + name);
  return audio;
}
