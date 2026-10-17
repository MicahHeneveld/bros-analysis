/* SSBU label review app - vanilla JS, no build step.
 *
 * Responsibilities:
 *  - canvas rendering of the image plus its normalized YOLO boxes
 *  - 8-handle move/resize, draw-new, delete
 *  - thumbnail strip with review status dots and filters
 *  - review panel (good / bad + reasons / reviewed empty + note)
 *  - local undo/redo plus server-side journal undo
 *  - debounced autosave to PUT /api/sample/<id>/labels
 */

(() => {
  'use strict';

  // ---------------------------------------------------------------- state
  const state = {
    dataset: null,
    samples: [],
    total: 0,
    offset: 0,
    limit: 200,
    index: 0,
    current: null,          // full sample detail from the server
    image: null,            // HTMLImageElement
    boxes: [],              // working copy: {cls,xc,yc,w,h}
    savedBoxes: [],         // last persisted copy, for dirty detection
    selected: -1,
    drawing: false,
    draft: null,
    dirty: false,
    saving: false,
    saveTimer: null,
    undo: [],
    redo: [],
    view: { zoom: 1, panX: 0, panY: 0, fitted: false },
    drag: null,
    reasons: [],
    reasonCodes: {},
    // Which box set the canvas shows: 'labels' (yours, editable), 'model'
    // (predictions, read-only) or 'both' (overlay, where editing and adopting
    // happen). The counts live on the toggle in the stage footer.
    viewMode: 'both',
    predSummary: null,
  };

  const VIEW_MODES = ['labels', 'model', 'both'];
  const MODE_NAME = { labels: 'Labels', model: 'Model', both: 'Both' };
  // Bump alongside the ?v= on the asset tags in index.html. Worth noticing: the
  // footer prints it, so a stale page is obvious at a glance instead of silent.
  const UI_VERSION = 'v8';

  const $ = (id) => document.getElementById(id);
  const canvas = $('canvas');
  const ctx = canvas.getContext('2d');

  // --------------------------------------------------------- view modes
  // Editing is only meaningful when the labels are on screen, and adopting only
  // when both sets are visible (you need to see what a detection would land
  // next to). Every interaction gate funnels through these three helpers.
  function predsVisible() { return state.viewMode !== 'labels'; }
  function labelsVisible() { return state.viewMode !== 'model'; }
  function editingEnabled() { return state.viewMode !== 'model'; }
  function adoptEnabled() { return state.viewMode === 'both'; }

  // ------------------------------------------------------------- utilities
  function toast(msg, isError = false) {
    const el = $('toast');
    el.textContent = msg;
    el.className = 'toast' + (isError ? ' err' : '');
    el.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { el.hidden = true; }, 2200);
  }

  async function api(path, options = {}) {
    const res = await fetch(path, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch (_) { data = { raw: text }; }
    if (!res.ok) {
      const msg = (data && data.error) || `${res.status} ${res.statusText}`;
      throw new Error(msg);
    }
    return data;
  }

  function cloneBoxes(boxes) {
    return boxes.map((b) => ({ cls: b.cls, xc: b.xc, yc: b.yc, w: b.w, h: b.h }));
  }

  function scoreBand(v) {
    const n = Number(v) || 0;
    if (n >= 0.5) return 'high';
    if (n >= 0.25) return 'mid';
    return 'low';
  }

  function boxesEqual(a, b) {
    if (a.length !== b.length) return false;
    const r = (v) => Math.round(v * 1e6) / 1e6;
    return a.every((x, i) =>
      x.cls === b[i].cls && r(x.xc) === r(b[i].xc) && r(x.yc) === r(b[i].yc) &&
      r(x.w) === r(b[i].w) && r(x.h) === r(b[i].h));
  }

  function clampBox(b) {
    const w = Math.min(Math.max(b.w, 1e-4), 1);
    const h = Math.min(Math.max(b.h, 1e-4), 1);
    const xc = Math.min(Math.max(b.xc, w / 2), 1 - w / 2);
    const yc = Math.min(Math.max(b.yc, h / 2), 1 - h / 2);
    return { cls: b.cls, xc, yc, w, h };
  }

  // --------------------------------------------------------- view mapping
  function imgRect() {
    if (!state.image) return { x: 0, y: 0, w: 0, h: 0 };
    const iw = state.image.naturalWidth;
    const ih = state.image.naturalHeight;
    const vw = canvas.clientWidth;
    const vh = canvas.clientHeight;
    const scale = Math.min(vw / iw, vh / ih) * state.view.zoom;
    const w = iw * scale;
    const h = ih * scale;
    return { x: (vw - w) / 2 + state.view.panX, y: (vh - h) / 2 + state.view.panY, w, h };
  }

  function toScreen(nx, ny) {
    const r = imgRect();
    return { x: r.x + nx * r.w, y: r.y + ny * r.h };
  }

  function boxScreen(b) {
    const c = toScreen(b.xc, b.yc);
    const r = imgRect();
    return { x: c.x - (b.w * r.w) / 2, y: c.y - (b.h * r.h) / 2, w: b.w * r.w, h: b.h * r.h };
  }

  function screenToNorm(x, y) {
    const r = imgRect();
    if (!r.w || !r.h) return { x: 0, y: 0 };
    return { x: (x - r.x) / r.w, y: (y - r.y) / r.h };
  }

  function fitView() {
    state.view.zoom = 1;
    state.view.panX = 0;
    state.view.panY = 0;
    state.view.fitted = true;
    draw();
  }

  // ------------------------------------------------------------ rendering
  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const wrap = $('canvasWrap');
    canvas.width = Math.max(1, Math.floor(wrap.clientWidth * dpr));
    canvas.height = Math.max(1, Math.floor(wrap.clientHeight * dpr));
    canvas.style.width = wrap.clientWidth + 'px';
    canvas.style.height = wrap.clientHeight + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  const HANDLE_ORDER = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];

  function handlePoints(rect) {
    const { x, y, w, h } = rect;
    return {
      nw: { x, y }, n: { x: x + w / 2, y },
      ne: { x: x + w, y }, e: { x: x + w, y: y + h / 2 },
      se: { x: x + w, y: y + h }, s: { x: x + w / 2, y: y + h },
      sw: { x, y: y + h }, w: { x, y: y + h / 2 },
    };
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    ctx.clearRect(0, 0, canvas.width / dpr, canvas.height / dpr);
    ctx.fillStyle = '#0e1013';
    ctx.fillRect(0, 0, canvas.width / dpr, canvas.height / dpr);

    if (!state.image) return;

    const r = imgRect();
    ctx.imageSmoothingEnabled = state.view.zoom < 3;
    ctx.drawImage(state.image, r.x, r.y, r.w, r.h);

    ctx.strokeStyle = 'rgba(255,255,255,.18)';
    ctx.lineWidth = 1;
    ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);

    // Model predictions render underneath the editable labels: dashed, thin,
    // non-interactive, tagged with confidence. Shown alone they are the subject,
    // so they get solid strokes and index numbers instead.
    const preds = (state.current && state.current.predictions) || [];
    if (predsVisible()) {
      const solo = state.viewMode === 'model';
      preds.forEach((p, i) => {
        const s = boxScreen(p);
        ctx.strokeStyle = '#e0a437';
        ctx.setLineDash(solo ? [] : [5, 4]);
        ctx.lineWidth = solo ? 2.5 : 1.5;
        ctx.strokeRect(s.x, s.y, s.w, s.h);
        ctx.setLineDash([]);

        if (solo && preds.length > 1) {
          ctx.fillStyle = '#e0a437';
          ctx.font = '600 11px -apple-system, sans-serif';
          ctx.fillText(String(i + 1), s.x + 3, s.y + 12);
        }
        if (typeof p.conf === 'number') {
          const tag = p.conf.toFixed(2);
          ctx.font = '600 10px -apple-system, sans-serif';
          const tw = ctx.measureText(tag).width + 7;
          ctx.fillStyle = 'rgba(20,22,26,.85)';
          ctx.fillRect(s.x, s.y - 13, tw, 12);
          ctx.fillStyle = '#e0a437';
          ctx.fillText(tag, s.x + 3, s.y - 4);
        }
      });
    }

    if (labelsVisible()) {
      state.boxes.forEach((b, i) => {
        const s = boxScreen(b);
        const isSel = i === state.selected;
        ctx.strokeStyle = isSel ? '#4a9eff' : '#35c07a';
        ctx.lineWidth = isSel ? 2.5 : 2;
        ctx.strokeRect(s.x, s.y, s.w, s.h);

        if (state.boxes.length > 1) {
          ctx.fillStyle = isSel ? '#4a9eff' : '#35c07a';
          ctx.font = '600 11px -apple-system, sans-serif';
          ctx.fillText(String(i + 1), s.x + 3, s.y + 12);
        }

        if (isSel) {
          const pts = handlePoints(s);
          HANDLE_ORDER.forEach((k) => {
            ctx.fillStyle = '#14161a';
            ctx.strokeStyle = '#4a9eff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.rect(pts[k].x - 4, pts[k].y - 4, 8, 8);
            ctx.fill();
            ctx.stroke();
          });
        }
      });
    }

    if (state.draft) {
      const d = state.draft;
      ctx.strokeStyle = '#e0a437';
      ctx.setLineDash([5, 4]);
      ctx.lineWidth = 2;
      ctx.strokeRect(d.x, d.y, d.w, d.h);
      ctx.setLineDash([]);
    }
  }

  // -------------------------------------------------------------- hit test
  function hitTest(mx, my) {
    // handles first (only for the selected box)
    if (state.selected >= 0 && state.boxes[state.selected]) {
      const rect = boxScreen(state.boxes[state.selected]);
      const pts = handlePoints(rect);
      for (const k of HANDLE_ORDER) {
        if (Math.abs(mx - pts[k].x) <= 7 && Math.abs(my - pts[k].y) <= 7) {
          return { type: 'handle', handle: k, index: state.selected };
        }
      }
    }
    // then boxes, topmost first
    for (let i = state.boxes.length - 1; i >= 0; i--) {
      const s = boxScreen(state.boxes[i]);
      if (mx >= s.x - 2 && mx <= s.x + s.w + 2 && my >= s.y - 2 && my <= s.y + s.h + 2) {
        return { type: 'box', index: i };
      }
    }
    return { type: 'empty' };
  }

  function canvasPos(evt) {
    const r = canvas.getBoundingClientRect();
    return { x: evt.clientX - r.left, y: evt.clientY - r.top };
  }

  // -------------------------------------------------- adopting predictions
  // Only a detection that overlaps NO label is adoptable. A duplicate overlaps
  // a character we already labelled, so adopting it would teach the model to
  // double-fire. Indices come from the server, which knows the split between
  // the two cases.
  function unmatchedPredIndices() {
    const m = (state.current && state.current.metrics) || {};
    return new Set(m.missed_player_idx || []);
  }

  function adoptablePreds(minConf = 0) {
    if (!adoptEnabled() || !state.current) return [];
    const un = unmatchedPredIndices();
    return (state.current.predictions || [])
      .map((p, i) => ({ p, i }))
      .filter(({ p, i }) => un.has(i) && (p.conf === undefined || p.conf >= minConf))
      .map(({ i }) => i);
  }

  function predHitTest(mx, my) {
    for (const i of adoptablePreds()) {
      const s = boxScreen(state.current.predictions[i]);
      if (mx >= s.x - 3 && mx <= s.x + s.w + 3 && my >= s.y - 3 && my <= s.y + s.h + 3) {
        return i;
      }
    }
    return -1;
  }

  function adoptPreds(indices) {
    if (!indices.length) return;
    const preds = state.current.predictions || [];
    const existing = new Set(state.boxes.map((b) => `${b.xc.toFixed(4)},${b.yc.toFixed(4)}`));
    pushUndo();
    let added = 0;
    for (const i of indices) {
      const p = preds[i];
      if (!p) continue;
      // Guard against a double-adopt landing the same geometry twice.
      const key = `${Number(p.xc).toFixed(4)},${Number(p.yc).toFixed(4)}`;
      if (existing.has(key)) continue;
      existing.add(key);
      state.boxes.push({
        cls: p.cls | 0,
        xc: Number(p.xc), yc: Number(p.yc), w: Number(p.w), h: Number(p.h),
      });
      added += 1;
    }
    if (!added) return;
    state.selected = state.boxes.length - 1;
    markDirty();
    draw();
    toast(`Adopted ${added} prediction${added === 1 ? '' : 's'} as label${added === 1 ? '' : 's'}`);
  }

  function adoptAllMissing() {
    if (!adoptEnabled()) {
      toast('Adopting works in Both view');
      return;
    }
    const idx = adoptablePreds(0.5);
    if (!idx.length) {
      toast('No confident unlabelled predictions on this image');
      return;
    }
    adoptPreds(idx);
  }

  // -------------------------------------------------------- box mutation
  function pushUndo() {
    state.undo.push(cloneBoxes(state.boxes));
    if (state.undo.length > 60) state.undo.shift();
    state.redo.length = 0;
  }

  function markDirty() {
    state.dirty = !boxesEqual(state.boxes, state.savedBoxes);
    setSaveState(state.dirty ? 'dirty' : 'saved');
    if (state.dirty) scheduleSave();
    renderPanel();
  }

  function setSaveState(kind) {
    const el = $('saveState');
    el.className = 'save-state ' + kind;
    el.textContent = kind === 'dirty' ? 'unsaved…' : kind === 'saving' ? 'saving…'
      : kind === 'error' ? 'save failed' : 'saved';
  }

  function scheduleSave() {
    clearTimeout(state.saveTimer);
    state.saveTimer = setTimeout(() => saveNow(), 900);
  }

  async function saveNow(silent = false) {
    if (!state.current) return;
    // Snapshot everything up front: the selection can change while the write is
    // in flight, and the response must not be applied to a different image.
    const imageId = state.current.image_id;
    const boxes = cloneBoxes(state.boxes);

    if (boxesEqual(boxes, state.savedBoxes)) {
      state.dirty = false;
      setSaveState('saved');
      if (!silent) toast('Nothing to save');
      return;
    }

    clearTimeout(state.saveTimer);
    state.saving = true;
    setSaveState('saving');
    try {
      const resp = await api(`/api/sample/${imageId}/labels`, {
        method: 'PUT',
        body: JSON.stringify({ boxes, reason: 'canvas_edit' }),
      });
      if (state.current && state.current.image_id === imageId) {
        state.savedBoxes = cloneBoxes(boxes);
        // Recompute rather than assume: the user may have kept editing.
        state.dirty = !boxesEqual(state.boxes, state.savedBoxes);
        setSaveState(state.dirty ? 'dirty' : 'saved');
        // The server re-scored against what it wrote. Adopting a prediction
        // changes the disagreement, so pick that up now instead of on reload.
        if (resp && resp.metrics) {
          state.current.metrics = resp.metrics;
          state.current.kind = resp.kind;
          renderPredMetrics();
        }
      }
      refreshRow(imageId, {
        boxes: boxes.length,
        disagreement: (resp && resp.metrics && resp.metrics.disagreement) || 0,
        kind: (resp && resp.kind) || '',
      });
      if (!silent) toast('Saved');
      loadJournal();
    } catch (err) {
      setSaveState('error');
      toast('Save failed: ' + err.message, true);
    } finally {
      state.saving = false;
    }
  }

  // ---------------------------------------------------------- data loading
  /**
   * Populate the split dropdown from what is actually on disk.
   *
   * Hardcoding train/val/test meant an ingested video (which lands in its own
   * split) was invisible in the filter. Counts come from the same /api/dataset
   * payload the rest of the header uses, so the numbers always agree.
   */
  function renderSplitFilter(splits) {
    const select = $('splitFilter');
    const names = Object.keys(splits);
    // Keep a choice the user already made, as long as it still exists.
    const previous = select.value;
    select.innerHTML = '<option value="all">All splits</option>';
    for (const name of names) {
      const n = splits[name].images;
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = `${name} (${n})`;
      select.appendChild(opt);
    }
    const stillThere = names.includes(previous);
    select.value = stillThere ? previous : 'all';
  }

  async function loadDataset() {
    state.dataset = await api('/api/dataset');
    state.reasonCodes = state.dataset.reason_codes;
    $('datasetName').textContent = state.dataset.dataset;
    document.title = `SSBU Label Review — ${state.dataset.dataset}`;
    buildReasonChips();
    renderSplitFilter(state.dataset.splits || {});

    const pred = state.dataset.predictions || {};
    state.predSummary = pred;
    if (pred.available) {
      $('viewToggleWrap').hidden = false;
      $('kindFilter').hidden = false;
      // When predictions exist, the disagreements are the reason to be here.
      $('sortFilter').value = 'unreviewed';
    }
    renderProgress();
  }

  function currentFilters() {
    return {
      split: $('splitFilter').value,
      status: $('statusFilter').value,
      sort: $('sortFilter').value,
      kind: $('kindFilter').value,
      flag: $('flagFilter').value,
      q: $('searchBox').value.trim(),
    };
  }

  async function loadSamples(reset = true) {
    if (reset) {
      state.offset = 0;
      state.samples = [];
      $('stripList').innerHTML = '';
    }
    const f = currentFilters();
    const params = new URLSearchParams({
      split: f.split, status: f.status, q: f.q,
      sort: f.sort, kind: f.kind, flag: f.flag,
      offset: String(state.offset), limit: String(state.limit),
    });
    const data = await api('/api/samples?' + params.toString());
    state.total = data.total;
    state.samples = reset ? data.items : state.samples.concat(data.items);
    state.offset = state.samples.length;
    renderStrip(reset);
    renderProgress();
    $('loadMore').hidden = state.samples.length >= state.total;
    if (reset) {
      if (state.samples.length === 0) {
        state.current = null;
        state.image = null;
        state.boxes = [];
        $('panelTitle').textContent = 'No images match these filters';
        draw();
        return;
      }
      selectIndex(0);
    }
  }

  async function loadSample(imageId) {
    const data = await api('/api/sample/' + imageId);
    state.current = data;
    state.boxes = data.boxes.map((b) => ({ ...b }));
    state.savedBoxes = cloneBoxes(state.boxes);
    state.selected = -1;
    state.dirty = false;
    state.undo = [];
    state.redo = [];
    state.drawing = false;
    $('drawBtn').classList.remove('active');
    canvas.classList.remove('drawing');
    setSaveState('saved');
    state.reasons = (data.review && data.review.reasons) ? [...data.review.reasons] : [];
    $('noteBox').value = (data.review && data.review.note) || '';

    // Load the image, then fit and render.
    await new Promise((resolve) => {
      const img = new Image();
      img.onload = () => { state.image = img; resolve(); };
      img.onerror = () => { state.image = null; resolve(); };
      img.src = '/api/image/' + imageId;
    });

    if (state.view.fitted === false) fitView();
    else { state.view.panX = 0; state.view.panY = 0; state.view.zoom = 1; draw(); }

    renderPanel();
    $('emptyNote').hidden = !(data.box_count === 0);
    if (data.box_count === 0) {
      $('emptyNote').textContent = data.status === 'empty'
        ? 'Reviewed as empty (background image)'
        : 'No boxes on this image — is that correct?';
    }
  }

  function selectIndex(i) {
    if (i < 0 || i >= state.samples.length) return;
    if (state.dirty) saveNow(true);
    state.index = i;
    const row = state.samples[i];
    highlightThumb(row.image_id);
    loadSample(row.image_id).catch((e) => toast('Load failed: ' + e.message, true));
    $('panelTitle').textContent = row.stem;
  }

  function step(delta) { selectIndex(state.index + delta); }

  function nextUnreviewed() {
    for (let i = state.index + 1; i < state.samples.length; i++) {
      if (state.samples[i].status === 'unreviewed') return selectIndex(i);
    }
    for (let i = 0; i < state.index; i++) {
      if (state.samples[i].status === 'unreviewed') return selectIndex(i);
    }
    toast('Everything in this view is reviewed');
  }

  // ------------------------------------------------------------- strip UI
  function renderStrip(reset) {
    const list = $('stripList');
    if (reset) list.innerHTML = '';
    const start = reset ? 0 : list.childElementCount;
    for (let i = start; i < state.samples.length; i++) {
      list.appendChild(thumbEl(state.samples[i], i));
    }
  }

  function rowLabel(row) {
    let s = `${row.stem}: ${row.status}, ${row.boxes} box(es)`;
    if (row.has_prediction) {
      s += `, disagreement ${Number(row.disagreement || 0).toFixed(2)}`;
    }
    return s;
  }

  function thumbEl(row, i) {
    // A real button so the strip is keyboard reachable and shows up in the
    // accessibility tree, rather than a click-only div.
    const el = document.createElement('button');
    el.type = 'button';
    el.className = 'thumb';
    el.dataset.imageId = row.image_id;
    el.dataset.index = String(i);
    el.setAttribute('aria-label', rowLabel(row));

    const img = document.createElement('img');
    img.loading = 'lazy';
    img.src = '/api/thumb/' + row.image_id;
    img.alt = row.stem;
    el.appendChild(img);

    const dot = document.createElement('span');
    dot.className = 'status-dot ' + row.status;
    el.appendChild(dot);

    const badge = document.createElement('span');
    badge.className = 'badge' + (row.boxes === 0 ? ' none' : '');
    badge.textContent = row.boxes === 0 ? 'empty' : String(row.boxes);
    el.appendChild(badge);

    if (row.has_prediction) {
      const sc = document.createElement('span');
      sc.className = 'badge score ' + scoreBand(row.disagreement);
      sc.textContent = Number(row.disagreement || 0).toFixed(2);
      el.appendChild(sc);
    }

    el.addEventListener('click', () => {
      const idx = state.samples.findIndex((s) => s.image_id === row.image_id);
      if (idx >= 0) selectIndex(idx);
    });
    if (i === state.index) el.classList.add('current');
    return el;
  }

  function highlightThumb(imageId) {
    document.querySelectorAll('.thumb.current').forEach((n) => n.classList.remove('current'));
    const el = document.querySelector(`.thumb[data-image-id="${CSS.escape(imageId)}"]`);
    if (el) {
      el.classList.add('current');
      el.scrollIntoView({ block: 'nearest' });
    }
  }

  function refreshRow(imageId, patch) {
    const row = state.samples.find((s) => s.image_id === imageId);
    if (!row) return;
    Object.assign(row, patch);
    const el = document.querySelector(`.thumb[data-image-id="${CSS.escape(imageId)}"]`);
    if (el) {
      const dot = el.querySelector('.status-dot');
      dot.className = 'status-dot ' + row.status;
      const badge = el.querySelector('.badge:not(.score)');
      badge.textContent = row.boxes === 0 ? 'empty' : String(row.boxes);
      badge.className = 'badge' + (row.boxes === 0 ? ' none' : '');
      // Keep the accessible name in sync with the visual state.
      el.setAttribute('aria-label', rowLabel(row));
    }
    renderProgress();
  }

  function renderProgress() {
    if (!state.dataset) return;
    const reviewed = state.samples.filter((s) => s.status !== 'unreviewed').length;
    const total = state.samples.length;
    $('progressFill').style.width = (total ? (reviewed / total) * 100 : 0).toFixed(1) + '%';
    const r = state.dataset.review || {};
    $('progressText').textContent =
      `${reviewed}/${total} reviewed in view · ${r.good || 0} good · ${r.bad || 0} bad · ${r.empty || 0} empty`;
  }

  // ------------------------------------------------------------ panel UI
  function buildReasonChips() {
    const wrap = $('reasonChips');
    wrap.innerHTML = '';
    Object.entries(state.reasonCodes).forEach(([code, desc], i) => {
      const chip = document.createElement('button');
      chip.className = 'chip';
      chip.dataset.code = code;
      chip.title = desc;
      chip.innerHTML = `<kbd>${i + 1}</kbd>${code}`;
      chip.addEventListener('click', () => toggleReason(code));
      wrap.appendChild(chip);
    });
  }

  function toggleReason(code) {
    const i = state.reasons.indexOf(code);
    if (i >= 0) state.reasons.splice(i, 1);
    else state.reasons.push(code);
    syncChips();
    postReview('bad');
  }

  function syncChips() {
    document.querySelectorAll('.chip').forEach((c) => {
      c.classList.toggle('on', state.reasons.includes(c.dataset.code));
    });
  }

  function renderPanel() {
    $('boxCount').textContent = String(state.boxes.length);
    const list = $('boxList');
    list.innerHTML = '';
    state.boxes.forEach((b, i) => {
      const row = document.createElement('div');
      row.className = 'box-row' + (i === state.selected ? ' selected' : '');
      row.innerHTML = `
        <span class="swatch"></span>
        <input data-k="xc" type="number" step="0.001" min="0" max="1" value="${b.xc.toFixed(3)}" />
        <input data-k="yc" type="number" step="0.001" min="0" max="1" value="${b.yc.toFixed(3)}" />
        <input data-k="w" type="number" step="0.001" min="0" max="1" value="${b.w.toFixed(3)}" />
        <input data-k="h" type="number" step="0.001" min="0" max="1" value="${b.h.toFixed(3)}" />
        <button class="del ghost" title="Delete box">×</button>`;
      row.addEventListener('mousedown', () => { state.selected = i; draw(); renderPanel(); });
      row.querySelectorAll('input').forEach((inp) => {
        inp.addEventListener('change', () => {
          pushUndo();
          const v = parseFloat(inp.value);
          if (!Number.isNaN(v)) state.boxes[i][inp.dataset.k] = v;
          state.boxes[i] = clampBox(state.boxes[i]);
          state.selected = i;
          markDirty();
          draw();
        });
      });
      row.querySelector('.del').addEventListener('click', (e) => {
        e.stopPropagation();
        deleteBox(i);
      });
      list.appendChild(row);
    });

    const warns = (state.current && state.current.warnings) || [];
    $('warnBlock').hidden = warns.length === 0;
    $('warnList').innerHTML = warns.map((w) => `<li>${escapeHtml(w)}</li>`).join('');

    // In Model view the labels are off screen, so editing them would be blind.
    const readOnly = !editingEnabled();
    $('addBoxBtn').disabled = readOnly;
    $('drawBtn').disabled = readOnly;
    $('boxList').classList.toggle('readonly', readOnly);

    renderPredMetrics();
    renderViewToggle();
  }

  function hintFor(m) {
    if (!m || m.gt_count === undefined) return '';
    if (m.missed_player_hi) {
      return 'The model is confident about a character that has no label. Unlike "label not '
        + 'found", here a box is probably missing.';
    }
    if (m.missing) {
      return 'A labelled box the model cannot find. Either the label sits on nothing, or the '
        + 'model simply missed the player. Not the same as a missing box.';
    }
    if (m.duplicate_hi) {
      return 'A second box on a character already labelled — model over-detection, not a label problem.';
    }
    if (m.weak) return 'Box matched but loosely fit — worth tightening.';
    if (m.extra) return 'Low-confidence extra detection — usually model noise, not a label error.';
    if (m.gt_count === m.pred_count) return 'Model and labels agree.';
    return '';
  }

  function renderPredMetrics() {
    const cur = state.current;
    const hasPred = !!(cur && cur.predictions_available);
    $('predBlock').hidden = !hasPred;
    if (!hasPred) return;

    const m = cur.metrics || {};
    const score = Number(m.disagreement || 0);
    const pill = $('disagreeScore');
    pill.textContent = score.toFixed(2);
    pill.className = 'score-pill ' + scoreBand(score);

    const parts = [];
    if (m.missed_player_hi) parts.push(`<span class="metric bad">unlabelled ×${m.missed_player_hi}</span>`);
    if (m.missing) parts.push(`<span class="metric bad">not found ×${m.missing}</span>`);
    if (m.duplicate_hi) parts.push(`<span class="metric warn">duplicate ×${m.duplicate_hi}</span>`);
    if (m.weak) parts.push(`<span class="metric warn">loose ×${m.weak}</span>`);
    if (m.matched) parts.push(`<span class="metric ok">matched ×${m.matched}</span>`);
    parts.push(`<span class="metric">labels ${m.gt_count || 0}</span>`);
    parts.push(`<span class="metric">model ${m.pred_count || 0}</span>`);
    $('predMetrics').innerHTML = parts.join('');
    $('predHint').textContent = hintFor(m);

    // Show what the dedupe+cap pass removed, so the correction is visible.
    const merged = cur.predictions_merged || 0;
    const pp = (state.predSummary && state.predSummary.postprocess) || {};
    $('predClean').textContent = merged
      ? `Dedupe merged ${merged} duplicate box${merged === 1 ? '' : 'es'} (IoU ${pp.nms_iou}, cap ${pp.max_boxes}/frame).`
      : '';

    // Offer the adopt action only when there is something to adopt, and only in
    // a view where the labels are visible to adopt into.
    const adoptable = adoptEnabled() ? adoptablePreds(0.5).length : 0;
    const btn = $('adoptBtn');
    btn.hidden = adoptable === 0;
    btn.textContent = `Adopt ${adoptable} missing (a)`;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  function deleteBox(i) {
    pushUndo();
    state.boxes.splice(i, 1);
    state.selected = Math.min(state.selected, state.boxes.length - 1);
    markDirty();
    draw();
  }

  function addBox() {
    if (!editingEnabled()) {
      toast('Switch to Labels or Both to edit boxes');
      return;
    }
    pushUndo();
    // Default a box in the middle at 1/4 size, then select it for dragging.
    state.boxes.push({ cls: 0, xc: 0.5, yc: 0.5, w: 0.25, h: 0.25 });
    state.selected = state.boxes.length - 1;
    markDirty();
    draw();
  }

  async function postReview(status) {
    if (!state.current) return;
    try {
      const data = await api(`/api/sample/${state.current.image_id}/review`, {
        method: 'POST',
        body: JSON.stringify({ status, reasons: state.reasons, note: $('noteBox').value }),
      });
      const imageId = state.current.image_id;
      state.current.status = data.review.status;
      state.current.review = data.review;
      refreshRow(imageId, { status: data.review.status, reasons: data.review.reasons });
      // The dataset-wide tally lives on the server; refresh it before painting
      // progress so the header numbers are not a step behind.
      try {
        state.dataset.review = (await (await fetch('/api/dataset')).json()).review;
      } catch (_) { /* header counts are cosmetic */ }
      renderProgress();
      syncChips();
      return data.review.status;
    } catch (err) {
      toast('Review update failed: ' + err.message, true);
      return null;
    }
  }

  async function markGood() {
    await postReview('good');
    toast('Marked good');
    step(1);
  }

  async function markEmpty() {
    state.reasons = [];
    syncChips();
    await postReview('empty');
    toast('Marked reviewed-empty');
    step(1);
  }

  async function markBad() {
    if (state.reasons.length === 0) {
      toast('Pick a reason first (keys 1-6), then it saves automatically');
      return;
    }
    await postReview('bad');
    toast('Marked bad: ' + state.reasons.join(', '));
  }

  async function loadJournal() {
    try {
      const data = await api('/api/journal');
      const items = data.entries.slice(-10).reverse();
      $('journalList').innerHTML = items.map((e) => {
        const t = new Date((e.ts || 0) * 1000).toLocaleTimeString();
        const n = (e.after || []).length;
        return `<li>${t} · ${escapeHtml(e.image_id.split('/').pop())} · ${escapeHtml(e.action)} → ${n} box(es)</li>`;
      }).join('') || '<li>no edits yet</li>';
    } catch (_) { /* journal is informational */ }
  }

  // -------------------------------------------------------- pointer input
  canvas.addEventListener('pointerdown', (evt) => {
    if (!state.image) return;
    try {
      canvas.setPointerCapture(evt.pointerId);
    } catch (_) {
      // Capture is a nicety; dragging still works without it.
    }
    const p = canvasPos(evt);
    const norm = screenToNorm(p.x, p.y);

    if (state.drawing && editingEnabled()) {
      state.draft = { x: p.x, y: p.y, w: 0, h: 0, nx: norm.x, ny: norm.y };
      state.drag = { type: 'draw' };
      return;
    }

    const hit = editingEnabled() ? hitTest(p.x, p.y) : { type: 'empty' };
    if (hit.type === 'handle') {
      pushUndo();
      state.selected = hit.index;
      state.drag = { type: 'resize', handle: hit.handle, index: hit.index, start: { ...state.boxes[hit.index] } };
      draw();
      renderPanel();
      return;
    }
    if (hit.type === 'box') {
      state.selected = hit.index;
      const s = boxScreen(state.boxes[hit.index]);
      pushUndo();
      state.drag = { type: 'move', index: hit.index, dx: p.x - s.x, dy: p.y - s.y };
      draw();
      renderPanel();
      return;
    }
    // Not a label: a click on an unlabelled prediction adopts it. Only in Both
    // view, where the label set it would join is visible.
    const pi = adoptEnabled() ? predHitTest(p.x, p.y) : -1;
    if (pi >= 0) {
      adoptPreds([pi]);
      return;
    }
    // empty space: pan, and deselect
    if (state.selected >= 0) { state.selected = -1; draw(); renderPanel(); }
    state.drag = { type: 'pan', px: p.x, py: p.y };
  });

  canvas.addEventListener('pointermove', (evt) => {
    if (!state.image) return;
    const p = canvasPos(evt);

    if (!state.drag) {
      const hit = (editingEnabled() && !state.drawing)
        ? hitTest(p.x, p.y) : { type: 'empty' };
      const overPred = hit.type === 'empty' && !state.drawing && adoptEnabled()
        && predHitTest(p.x, p.y) >= 0;
      canvas.style.cursor = state.drawing ? 'cell'
        : hit.type === 'handle' ? handleCursor(hit.handle)
        : hit.type === 'box' ? 'move'
        : overPred ? 'copy' : 'grab';
      return;
    }

    if (state.drag.type === 'pan') {
      state.view.panX += p.x - state.drag.px;
      state.view.panY += p.y - state.drag.py;
      state.drag.px = p.x;
      state.drag.py = p.y;
      draw();
      return;
    }

    if (state.drag.type === 'move') {
      const r = imgRect();
      const s = { x: p.x - state.drag.dx, y: p.y - state.drag.dy };
      const b = state.boxes[state.drag.index];
      b.xc = (s.x + (b.w * r.w) / 2 - r.x) / r.w;
      b.yc = (s.y + (b.h * r.h) / 2 - r.y) / r.h;
      state.boxes[state.drag.index] = clampBox(b);
      draw();
      return;
    }

    if (state.drag.type === 'resize') {
      const r = imgRect();
      const s0 = state.drag.start;
      const c = toScreen(s0.xc, s0.yc);
      const left0 = c.x - (s0.w * r.w) / 2;
      const right0 = c.x + (s0.w * r.w) / 2;
      const top0 = c.y - (s0.h * r.h) / 2;
      const bot0 = c.y + (s0.h * r.h) / 2;
      let left = left0, right = right0, top = top0, bot = bot0;
      const hnd = state.drag.handle;
      if (hnd.includes('w')) left = p.x;
      if (hnd.includes('e')) right = p.x;
      if (hnd.includes('n')) top = p.y;
      if (hnd.includes('s')) bot = p.y;
      const nx = (left - r.x) / r.w;
      const ny = (top - r.y) / r.h;
      const nw = (right - left) / r.w;
      const nh = (bot - top) / r.h;
      state.boxes[state.drag.index] = clampBox({
        cls: s0.cls,
        xc: nx + nw / 2,
        yc: ny + nh / 2,
        w: Math.abs(nw),
        h: Math.abs(nh),
      });
      draw();
      return;
    }

    if (state.drag.type === 'draw' && state.draft) {
      state.draft.w = p.x - state.draft.x;
      state.draft.h = p.y - state.draft.y;
      draw();
    }
  });

  canvas.addEventListener('pointerup', (evt) => {
    if (!state.drag) return;
    const drag = state.drag;
    state.drag = null;

    if (drag.type === 'draw' && state.draft) {
      const d = state.draft;
      state.draft = null;
      const x = Math.min(d.x, d.x + d.w);
      const y = Math.min(d.y, d.y + d.h);
      const w = Math.abs(d.w);
      const h = Math.abs(d.h);
      if (w > 4 && h > 4) {
        const p1 = screenToNorm(x, y);
        const p2 = screenToNorm(x + w, y + h);
        pushUndo();
        state.boxes.push(clampBox({
          cls: 0,
          xc: (p1.x + p2.x) / 2,
          yc: (p1.y + p2.y) / 2,
          w: Math.abs(p2.x - p1.x),
          h: Math.abs(p2.y - p1.y),
        }));
        state.selected = state.boxes.length - 1;
        state.drawing = false;
        $('drawBtn').classList.remove('active');
        canvas.classList.remove('drawing');
        markDirty();
      }
      draw();
      return;
    }

    if (drag.type === 'move' || drag.type === 'resize') {
      markDirty();
    }
  });

  canvas.addEventListener('wheel', (evt) => {
    if (!state.image) return;
    evt.preventDefault();
    const p = canvasPos(evt);
    const before = screenToNorm(p.x, p.y);
    const factor = evt.deltaY < 0 ? 1.12 : 1 / 1.12;
    state.view.zoom = Math.min(Math.max(state.view.zoom * factor, 0.1), 20);
    const after = screenToNorm(p.x, p.y);
    const r = imgRect();
    state.view.panX += (after.x - before.x) * r.w;
    state.view.panY += (after.y - before.y) * r.h;
    draw();
  }, { passive: false });

  function handleCursor(h) {
    if (h === 'nw' || h === 'se') return 'nwse-resize';
    if (h === 'ne' || h === 'sw') return 'nesw-resize';
    if (h === 'n' || h === 's') return 'ns-resize';
    return 'ew-resize';
  }

  // ------------------------------------------------------------- keyboard
  const REASON_KEYS = ['1', '2', '3', '4', '5', '6'];

  window.addEventListener('keydown', (evt) => {
    const tag = (evt.target.tagName || '').toLowerCase();
    const typing = tag === 'input' || tag === 'textarea';

    if (evt.key === 'Escape') {
      if (!$('helpOverlay').hidden) { $('helpOverlay').hidden = true; return; }
      state.selected = -1;
      state.drawing = false;
      state.draft = null;
      $('drawBtn').classList.remove('active');
      canvas.classList.remove('drawing');
      draw();
      renderPanel();
      return;
    }

    if (typing) {
      // Cmd/Ctrl+S should still save while typing in the note field.
      if ((evt.metaKey || evt.ctrlKey) && evt.key.toLowerCase() === 's') {
        evt.preventDefault();
        saveNow();
      }
      return;
    }

    if (evt.metaKey || evt.ctrlKey) {
      const k = evt.key.toLowerCase();
      if (k === 's') { evt.preventDefault(); saveNow(); return; }
      if (k === 'z' && !evt.shiftKey) { evt.preventDefault(); undo(); return; }
      if ((k === 'z' && evt.shiftKey) || k === 'y') { evt.preventDefault(); redo(); return; }
      return;
    }

    switch (evt.key) {
      case 'j': case 'ArrowRight': evt.preventDefault(); step(1); break;
      case 'k': case 'ArrowLeft': evt.preventDefault(); step(-1); break;
      case 'u': nextUnreviewed(); break;
      case 'g': markGood(); break;
      case 'b': markBad(); break;
      case 'n': markEmpty(); break;
      case 'd': toggleDraw(); break;
      case 'p': cycleViewMode(); break;
      case 'a': adoptAllMissing(); break;
      case 'Delete': case 'Backspace':
        if (state.selected >= 0 && editingEnabled()) {
          evt.preventDefault();
          deleteBox(state.selected);
        }
        break;
      case '?': $('helpOverlay').hidden = false; break;
      default:
        if (REASON_KEYS.includes(evt.key)) {
          const codes = Object.keys(state.reasonCodes);
          const code = codes[parseInt(evt.key, 10) - 1];
          if (code) toggleReason(code);
        }
    }
  });

  function toggleDraw() {
    if (!editingEnabled()) {
      toast('Switch to Labels or Both to edit boxes');
      return;
    }
    state.drawing = !state.drawing;
    $('drawBtn').classList.toggle('active', state.drawing);
    canvas.classList.toggle('drawing', state.drawing);
    if (state.drawing) toast('Drag on the image to draw a box');
  }

  // ------------------------------------------------------------ view toggle
  // Each segment carries the number of boxes that mode would draw, so the
  // counts are readable without switching. Live label edits update the Labels
  // and Both counts immediately because they read the working copy.
  function renderViewToggle() {
    const hasPred = !!(state.predSummary && state.predSummary.available);
    $('viewToggleWrap').hidden = !hasPred;
    if (!hasPred) {
      $('legend').hidden = true;
      return;
    }

    const nLabels = state.boxes.length;
    const nModel = state.current ? (state.current.predictions || []).length : 0;
    const counts = { labels: nLabels, model: nModel, both: nLabels + nModel };

    document.querySelectorAll('#viewToggle .seg').forEach((btn) => {
      const mode = btn.dataset.mode;
      const on = mode === state.viewMode;
      btn.classList.toggle('on', on);
      btn.setAttribute('aria-pressed', String(on));
      const badge = btn.querySelector('.seg-count');
      if (badge) {
        badge.textContent = String(counts[mode] || 0);
        badge.classList.toggle('zero', !counts[mode]);
      }
    });

    // The legend must only advertise what is actually painted.
    $('legendGt').hidden = !labelsVisible();
    $('legendPred').hidden = !predsVisible();
    $('legend').hidden = false;
  }

  function setViewMode(mode) {
    if (!VIEW_MODES.includes(mode) || mode === state.viewMode) return;
    state.viewMode = mode;
    // A half-finished drag or draft would be stranded by the mode change.
    state.drag = null;
    state.draft = null;
    if (!editingEnabled()) {
      state.drawing = false;
      $('drawBtn').classList.remove('active');
      canvas.classList.remove('drawing');
    }
    renderViewToggle();
    renderPanel();
    draw();
    announceMode();
  }

  // Say what is now on screen. On a frame where the labels and the predictions
  // sit almost on top of each other the canvas barely changes, so without this
  // the switch reads as a broken button.
  function announceMode() {
    const parts = { labels: null, model: null, both: null };
    const nLabels = labelsVisible() ? state.boxes.length : 0;
    const nModel = predsVisible() && state.current
      ? (state.current.predictions || []).length : 0;
    if (nLabels) parts.labels = `${nLabels} label${nLabels === 1 ? '' : 's'}`;
    if (nModel) parts.model = `${nModel} prediction${nModel === 1 ? '' : 's'}`;
    const what = [parts.labels, parts.model].filter(Boolean).join(' + ') || 'nothing';
    const extra = editingEnabled() ? 'editable' : 'read-only';
    toast(`Showing ${what} (${extra})`);
  }

  function cycleViewMode() {
    const next = VIEW_MODES[(VIEW_MODES.indexOf(state.viewMode) + 1) % VIEW_MODES.length];
    setViewMode(next);
  }

  function undo() {
    if (state.undo.length === 0) { toast('Nothing to undo in this session'); return; }
    state.redo.push(cloneBoxes(state.boxes));
    state.boxes = state.undo.pop();
    state.selected = -1;
    markDirty();
    draw();
    toast('Undo');
  }

  function redo() {
    if (state.redo.length === 0) return;
    state.undo.push(cloneBoxes(state.boxes));
    state.boxes = state.redo.pop();
    markDirty();
    draw();
    toast('Redo');
  }

  // ------------------------------------------------------------ wiring up
  function bind() {
    $('splitFilter').addEventListener('change', () => loadSamples(true));
    $('statusFilter').addEventListener('change', () => loadSamples(true));
    $('sortFilter').addEventListener('change', () => loadSamples(true));
    $('kindFilter').addEventListener('change', () => loadSamples(true));
    $('flagFilter').addEventListener('change', () => loadSamples(true));
    let searchTimer = null;
    $('searchBox').addEventListener('input', () => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => loadSamples(true), 250);
    });
    $('loadMore').addEventListener('click', () => loadSamples(false));

    $('markGood').addEventListener('click', markGood);
    $('markBad').addEventListener('click', markBad);
    $('markEmpty').addEventListener('click', markEmpty);
    $('noteBox').addEventListener('change', () => {
      if (state.current && state.current.status === 'bad') postReview('bad');
    });

    $('saveBtn').addEventListener('click', () => saveNow());
    $('undoBtn').addEventListener('click', async () => {
      if (state.dirty) { undo(); return; }
      if (!state.current) return;
      try {
        const res = await api('/api/undo/' + state.current.image_id, { method: 'POST', body: '{}' });
        if (res.restored) {
          toast('Reverted last saved edit from journal');
          await loadSample(state.current.image_id);
          refreshRow(state.current.image_id, { boxes: state.boxes.length });
        } else {
          toast('No journal edit for this image');
        }
      } catch (e) { toast(e.message, true); }
    });
    $('addBoxBtn').addEventListener('click', addBox);
    $('drawBtn').addEventListener('click', toggleDraw);
    $('viewToggle').addEventListener('click', (e) => {
      const seg = e.target.closest('.seg');
      if (seg) setViewMode(seg.dataset.mode);
    });
    $('adoptBtn').addEventListener('click', adoptAllMissing);

    $('zoomIn').addEventListener('click', () => { state.view.zoom = Math.min(state.view.zoom * 1.25, 20); draw(); });
    $('zoomOut').addEventListener('click', () => { state.view.zoom = Math.max(state.view.zoom / 1.25, 0.1); draw(); });
    $('zoomReset').addEventListener('click', () => { state.view.zoom = 1; draw(); });
    $('fitBtn').addEventListener('click', fitView);

    $('helpBtn').addEventListener('click', () => { $('helpOverlay').hidden = false; });
    $('closeHelp').addEventListener('click', () => { $('helpOverlay').hidden = true; });
    $('helpOverlay').addEventListener('click', (e) => {
      if (e.target === $('helpOverlay')) $('helpOverlay').hidden = true;
    });

    window.addEventListener('resize', resizeCanvas);
    window.addEventListener('beforeunload', (e) => {
      if (state.dirty) { saveNow(true); e.preventDefault(); e.returnValue = ''; }
    });
  }

  /**
   * Apply filters from the URL so a review queue can be linked directly, e.g.
   *   /?flag=conflicting&split=val
   * Only known filter values are accepted, so a bad link cannot put the UI into
   * a state the server will not understand.
   */
  function applyUrlFilters() {
    const params = new URLSearchParams(location.search);
    const targets = {
      split: 'splitFilter',
      status: 'statusFilter',
      sort: 'sortFilter',
      kind: 'kindFilter',
      flag: 'flagFilter',
    };
    const applied = [];
    for (const [key, id] of Object.entries(targets)) {
      const raw = params.get(key);
      if (!raw) continue;
      const el = $(id);
      if (!el || el.hidden) continue;
      const match = [...el.options].find((o) => o.value === raw);
      if (!match) continue;
      el.value = raw;
      applied.push(`${key}=${raw}`);
    }
    const q = params.get('q');
    if (q) {
      $('searchBox').value = q;
      applied.push(`q=${q}`);
    }
    return applied;
  }

  async function boot() {
    bind();
    resizeCanvas();
    const stamp = $('uiVersion');
    if (stamp) stamp.textContent = UI_VERSION;
    window.__uiVersion = UI_VERSION;
    try {
      await loadDataset();
      // After loadDataset, since it may hide filters that do not apply and
      // resets the sort default.
      const applied = applyUrlFilters();
      if (applied.length) toast('Filters from link: ' + applied.join(', '));
      await loadSamples(true);
      await loadJournal();
      setInterval(loadJournal, 15000);
    } catch (err) {
      toast('Startup failed: ' + err.message, true);
    }
  }

  boot();
})();
