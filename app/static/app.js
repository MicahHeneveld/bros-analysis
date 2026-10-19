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

  // ------------------------------------------------------------- location
  // Parsed before `state`, because the initial queue size depends on the mode.
  // The landing screen opens one dataset at a time with ?dataset=<name>. Every
  // API call carries it so several datasets can be open in different tabs
  // without the server guessing which one a request means.
  const PARAMS = new URLSearchParams(location.search);
  const DATASET = PARAMS.get('dataset') || '';
  // A bounty is a frame slice, so the queue is scoped to it server-side.
  const BOUNTY_ID = PARAMS.get('bounty_id') || '';
  // Reopening a finished bounty to double-check it. Same fixed slice, same
  // writes; only the wording changes, plus a note when a PR already exists.
  const EDIT_MODE = PARAMS.get('edit') === '1';
  const PR_EXISTS = PARAMS.get('pr') === '1';
  // `/view` is read-only browsing of a whole dataset (images + source video).
  // It is the only mode that does not require a bounty, because it writes
  // nothing. `/review` is the working mode and always needs one.
  const VIEW_MODE = /^\/view\/?$/.test(location.pathname);
  const REVIEW_MODE = !VIEW_MODE;

  if (REVIEW_MODE && !BOUNTY_ID) {
    // Label mode is bounty-scoped by definition. Without a bounty the strip
    // used to show the whole dataset with a "Load more" button, which is how
    // reviewers ended up labelling frames that no bounty was paying for.
    location.replace('/');
  }

  // ---------------------------------------------------------------- state
  const state = {
    dataset: null,
    samples: [],
    total: 0,
    offset: 0,
    // A bounty slice is fixed, so label mode loads all of it at once: a
    // "Load more" button in a 250-frame bounty is just a way to miss frames.
    limit: REVIEW_MODE ? Math.max(200, Number(PARAMS.get('of')) || 500) : 200,
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
    integrityDismissed: false,
    // Which box set the canvas shows: 'labels' (yours, editable), 'model'
    // (predictions, read-only) or 'both' (overlay, where editing and adopting
    // happen). The counts live on the toggle in the stage footer.
    viewMode: 'both',
    predSummary: null,
    // Top-level mode: 'images' reviews the dataset frame by frame, 'video'
    // plays the source MP4 with the same boxes overlaid.
    appMode: 'images',
    // Interface detail level. 'review' is the contributor view: judge frames,
    // fix boxes, nothing else. 'expert' reveals filters, metrics, journal and
    // backup tooling. Persisted so a power user only sets it once.
    role: 'review',
  };

  const VIEW_MODES = ['labels', 'model', 'both'];
  const MODE_NAME = { labels: 'Labels', model: 'Model', both: 'Both' };
  // Bump alongside the ?v= on the asset tags in index.html. Worth noticing: the
  // footer prints it, so a stale page is obvious at a glance instead of silent.
  const UI_VERSION = 'v24';

  const $ = (id) => document.getElementById(id);
  const canvas = $('canvas');
  const ctx = canvas.getContext('2d');

  /**
   * Encode an image id for use in a URL path.
   *
   * Image ids are built from filenames, which routinely contain '#', spaces,
   * '(' and ')'. An unencoded '#' starts a URL fragment, so the request is cut
   * short at that character and the server answers "unknown image" - which is
   * exactly what happened for every frame of an ingested video. Encode each
   * segment but keep the split separator.
   */
  function encId(id) {
    return String(id).split('/').map(encodeURIComponent).join('/');
  }

  // ------------------------------------------------------------- help text
  // Every explanation that used to sit inline in the UI lives here and is
  // reachable through a "?" button. Keeping it in one table means the copy can
  // be reviewed in one place, and the interface stays free of paragraphs that
  // experienced contributors stop reading after the first day.
  const HELP = {
    decide: {
      title: 'Judging a frame',
      body: [
        '<p><strong>Good</strong> — every player is boxed, tightly, and nothing that is not a player has a box.</p>',
        '<p><strong>Needs fixing</strong> — then pick why. One reason is enough; the reason is what teaches us where the model is weak.</p>',
        '<p><strong>Empty</strong> — a background frame with no players (menus, results, transitions). Use this instead of deleting boxes, so the model is taught that nothing is there.</p>',
      ].join(''),
    },
    adopt: {
      title: 'Adopting a suggestion',
      body: [
        '<p>The model found a player you have not boxed. <em>Adopt</em> adds its box as a label, so you only have to check and nudge it instead of drawing from scratch.</p>',
        '<p>Only confident detections that do not overlap an existing label are offered. Dashed blue boxes on the canvas are the ones being offered — you can also click a single dashed box to adopt just that one.</p>',
      ].join(''),
    },
    boxes: {
      title: 'Editing boxes',
      body: [
        '<p>Drag a box to move it, drag a corner or edge to resize. <kbd>d</kbd> draws a new box, <kbd>Delete</kbd> removes the selected one.</p>',
        '<p>Scroll to zoom, drag empty space to pan, <kbd>Esc</kbd> to deselect. Changes save automatically — the top-right indicator tells you when.</p>',
        '<p>Switch to <strong>Expert</strong> to type exact coordinates.</p>',
      ].join(''),
    },
    viewModes: {
      title: 'Labels / Model / Both',
      body: [
        '<p><strong>Labels</strong> — only what is on disk. This is what the model trains on.</p>',
        '<p><strong>Model</strong> — only the model\u2019s predictions, read-only. Use it to see what the model believes without your labels getting in the way.</p>',
        '<p><strong>Both</strong> — overlay. Needed to adopt suggestions, since you have to see where a prediction would land.</p>',
      ].join(''),
    },
    disagreement: {
      title: 'Disagreement score',
      body: [
        '<p>How far the model and the labels are apart on this frame, from 0 (identical) to 1 (nothing in common). It is recomputed from what is on disk every time you save, so fixing a label immediately moves the number.</p>',
        '<p>Sorting by <em>worst disagreement</em> (Expert) is the fastest way to find frames where either the label or the model is wrong.</p>',
      ].join(''),
    },
    progress: {
      title: 'Review progress',
      body: [
        '<p>A frame counts as reviewed once it has been marked <em>Good</em>, <em>Needs fixing</em>, or <em>Empty</em>. Merely opening it does not count — otherwise progress would claim work that was never judged.</p>',
      ].join(''),
    },
    videoFrames: {
      title: 'Why the boxes disappear',
      body: [
        '<p>Frames are sampled (about 5 per second) and every sampled frame is kept, so the timeline is continuous. Datasets ingested with the old menu filter, or with a low <code>--fps</code>, have holes instead. Boxes are drawn from the nearest kept frame and hidden once you are more than a sampling interval away.</p>',
        '<p>An empty overlay therefore means <em>the dataset has no frame here</em>\u2014not that the model found nothing. Clear <em>Follow sampled frames</em> to hold the nearest box instead.</p>',
      ].join(''),
    },
  };

  let popoverOwner = null;

  function showPopover(btn) {
    const key = btn.dataset.help;
    const entry = HELP[key];
    if (!entry) return;
    // A "?" inside a hidden block has no position; anchoring to it would pin the
    // popover to the top-left corner. Only visible buttons can open one.
    if (!btn.getClientRects().length) return;
    if (popoverOwner === btn) { hidePopover(); return; }
    hidePopover();
    popoverOwner = btn;
    $('popoverTitle').textContent = entry.title;
    $('popoverBody').innerHTML = entry.body;
    const pop = $('popover');
    pop.hidden = false;
    const r = btn.getBoundingClientRect();
    const w = pop.offsetWidth;
    const h = pop.offsetHeight;
    // Prefer below-right; flip up/left rather than run off screen.
    let left = r.left;
    let top = r.bottom + 8;
    if (left + w > window.innerWidth - 10) left = Math.max(10, window.innerWidth - w - 10);
    if (top + h > window.innerHeight - 10) top = Math.max(10, r.top - h - 8);
    pop.style.left = left + 'px';
    pop.style.top = top + 'px';
    btn.setAttribute('aria-expanded', 'true');
  }

  function hidePopover() {
    if (popoverOwner) popoverOwner.setAttribute('aria-expanded', 'false');
    popoverOwner = null;
    const pop = $('popover');
    if (pop) pop.hidden = true;
  }

  // ------------------------------------------------------------------ roles
  // Review and Expert are the same app; only the CSS class on <body> changes.
  // Hiding via CSS (rather than removing nodes) keeps every handler bound and
  // means switching is instant and cannot leave the UI half-initialised.
  function setRole(role, announce = true) {
    if (role !== 'review' && role !== 'expert') role = 'review';
    state.role = role;
    document.body.classList.toggle('role-review', role === 'review');
    document.body.classList.toggle('role-expert', role === 'expert');
    for (const seg of $('roleSwitch').querySelectorAll('.seg')) {
      seg.classList.toggle('on', seg.dataset.role === role);
    }
    try { localStorage.setItem('ssbu.role', role); } catch (e) { /* private mode */ }
    if (announce) {
      toast(role === 'expert'
        ? 'Expert view — filters, metrics and journal are visible'
        : 'Review view — the simple interface');
    }
    // The panel is rebuilt by role: experts see coordinate inputs, reviewers
    // see a compact list they can only delete from.
    renderPanel();
    if (state.image) requestAnimationFrame(resizeCanvas);
  }

  // ------------------------------------------------------------ walkthrough
  // Shown when a contributor picks up a bounty, and replayable from Help. Each
  // step spotlights a real element and, where useful, points at it relative to
  // the current screen rather than describing it abstractly.
  const TOUR_STEPS = [
    {
      target: null,
      title: 'Your bounty',
      body: 'You have a slice of frames to review. Work left to right down the strip: '
        + 'each frame is a decision, and the bar at the top fills as you go. '
        + 'Everything saves automatically — you cannot lose work by closing the tab.',
    },
    {
      target: '#strip',
      title: 'Pick a frame',
      body: 'The strip is your queue. Grey dots are unreviewed, green good, red needs fixing, '
        + 'grey-white not gameplay. A bounty is a few hundred frames, so expect a few sessions.',
    },
    {
      target: '#canvasWrap',
      title: 'Look at the boxes',
      body: 'Solid green boxes are the current labels — what the model will train on. '
        + 'Dashed blue boxes are the model\u2019s own guesses. Your job is to make the green correct.',
    },
    {
      target: '#decideBlock',
      title: 'Judge the frame',
      body: 'Press <kbd>g</kbd> if every player is boxed tightly and nothing else is. '
        + 'Press <kbd>b</kbd> if something is wrong, then hit a number to say why. '
        + 'Press <kbd>n</kbd> for menus and other non-gameplay frames.',
    },
    {
      target: '#fixBlock',
      title: 'Let the model help',
      body: 'When the model spots a player you missed, it offers the box. '
        + '<em>Adopt</em> adds it as a label so you only check it instead of drawing it. '
        + 'Press <kbd>a</kbd> to adopt everything offered.',
    },
    {
      target: '#viewToggleWrap',
      title: 'Switch what you see',
      body: 'Press <kbd>p</kbd> to cycle Labels → Model → Both. Use Labels to check what is on disk, '
        + 'Model to see the model alone, Both to adopt.',
    },
    {
      target: '#progressWrap',
      title: 'Finish and commit',
      body: 'When you are done, your labels are plain .txt files in the repo. '
        + 'Commit and push them the same way you push code — that is what earns the bounty points.',
    },
  ];

  const TOUR = { i: 0, open: false };

  function startTour(force = false) {
    if (!force) {
      try { if (localStorage.getItem('ssbu.tourDone') === '1') return; } catch (e) { /* ignore */ }
    }
    TOUR.i = 0;
    TOUR.open = true;
    $('tourOverlay').hidden = false;
    renderTour();
  }

  function endTour(done) {
    TOUR.open = false;
    $('tourOverlay').hidden = true;
    $('tourSpot').style.cssText = '';
    if (done) {
      try { localStorage.setItem('ssbu.tourDone', '1'); } catch (e) { /* ignore */ }
    }
  }

  function renderTour() {
    const step = TOUR_STEPS[TOUR.i];
    const last = TOUR.i === TOUR_STEPS.length - 1;
    $('tourStepLabel').textContent = `Step ${TOUR.i + 1} of ${TOUR_STEPS.length}`;
    $('tourTitle').textContent = step.title;
    $('tourBody').innerHTML = step.body;
    $('tourBack').disabled = TOUR.i === 0;
    $('tourNext').textContent = last ? 'Got it' : 'Next';
    $('tourSkip').hidden = last;

    const spot = $('tourSpot');
    const target = step.target ? document.querySelector(step.target) : null;
    const r = target ? target.getBoundingClientRect() : null;
    const overlay = $('tourOverlay');
    // A target can exist but be unpainted: no image selected, no suggestions to
    // adopt, or the Video tab is open. Spotlighting a zero-sized box would leave
    // the card pointing at nothing, so those steps get the centred treatment.
    if (!r || !r.width || !r.height) {
      // No element to point at: dim the whole screen instead of relying on the
      // spotlight's shadow, which only exists when there is a spot.
      overlay.classList.add('no-target');
      spot.style.cssText = 'display:none';
      positionTourCard(null, 0);
      return;
    }
    overlay.classList.remove('no-target');
    const pad = 6;
    // Clamp to the viewport: a target flush against an edge would otherwise put
    // a negative offset on the spotlight and leave a sliver un-dimmed.
    const left = Math.max(0, r.left - pad);
    const top = Math.max(0, r.top - pad);
    const width = Math.min(window.innerWidth - left, r.width + pad * 2);
    const height = Math.min(window.innerHeight - top, r.height + pad * 2);
    spot.style.cssText = [
      'display:block',
      `left:${left}px`,
      `top:${top}px`,
      `width:${width}px`,
      `height:${height}px`,
    ].join(';');
    positionTourCard(r, pad);
  }

  function positionTourCard(rect, pad) {
    const card = $('tourCard');
    const w = card.offsetWidth;
    const h = card.offsetHeight;
    let left;
    let top;
    if (!rect) {
      left = (window.innerWidth - w) / 2;
      top = (window.innerHeight - h) / 2;
    } else {
      // Prefer the side with more room so the card never covers the target.
      const roomBelow = window.innerHeight - (rect.bottom + pad + 12);
      const roomAbove = rect.top - pad - 12;
      left = Math.min(Math.max(10, rect.left), window.innerWidth - w - 10);
      top = roomBelow > h || roomBelow >= roomAbove
        ? rect.bottom + pad + 12
        : rect.top - pad - 12 - h;
      top = Math.min(Math.max(10, top), window.innerHeight - h - 10);
    }
    card.style.left = left + 'px';
    card.style.top = top + 'px';
  }


  // --------------------------------------------------------- view modes
  // Editing is only meaningful when the labels are on screen, and adopting only
  // when both sets are visible (you need to see what a detection would land
  // next to). Every interaction gate funnels through these three helpers.
  function predsVisible() { return state.viewMode !== 'labels'; }
  function labelsVisible() { return state.viewMode !== 'model'; }
  // One choke point for "may this session write at all". View mode is read-only,
  // so it fails these checks the same way Model view does, and every caller
  // already renders a sensible explanation. A submitted bounty is read-only for
  // the same reason: its labels are now the contents of a PR. The server refuses
  // these writes too - this only keeps the UI from asking for something it knows
  // will be rejected.
  function editingEnabled() {
    return !VIEW_MODE && !PR_EXISTS && state.viewMode !== 'model';
  }
  function adoptEnabled() {
    return !VIEW_MODE && !PR_EXISTS && state.viewMode === 'both';
  }
  function writeBlockedMessage() {
    if (PR_EXISTS) return 'This bounty was submitted, so its labels are locked';
    return VIEW_MODE ? 'View mode is read-only — open this bounty to label'
      : 'Switch to Labels or Both to edit boxes';
  }

  // ------------------------------------------------------------- utilities
  function toast(msg, isError = false) {
    const el = $('toast');
    el.textContent = msg;
    el.className = 'toast' + (isError ? ' err' : '');
    el.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { el.hidden = true; }, 2200);
  }

  function withDataset(path) {
    if (!DATASET || !path.startsWith('/api/')) return path;
    return path + (path.includes('?') ? '&' : '?') + 'dataset=' + encodeURIComponent(DATASET);
  }

  async function api(path, options = {}) {
    const res = await fetch(withDataset(path), {
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
        ctx.strokeStyle = isSel ? '#ff6b1a' : '#35c07a';
        ctx.lineWidth = isSel ? 2.5 : 2;
        ctx.strokeRect(s.x, s.y, s.w, s.h);

        if (state.boxes.length > 1) {
          ctx.fillStyle = isSel ? '#ff6b1a' : '#35c07a';
          ctx.font = '600 11px -apple-system, sans-serif';
          ctx.fillText(String(i + 1), s.x + 3, s.y + 12);
        }

        if (isSel) {
          const pts = handlePoints(s);
          HANDLE_ORDER.forEach((k) => {
            ctx.fillStyle = '#14161a';
            ctx.strokeStyle = '#ff6b1a';
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
    if (VIEW_MODE || !indices.length) return;
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
    if (VIEW_MODE) {
      toast('View mode is read-only');
      return;
    }
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
    if (VIEW_MODE || PR_EXISTS || !state.current) return;
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
      const resp = await api(`/api/sample/${encId(imageId)}/labels`, {
        method: 'PUT',
        body: JSON.stringify({ boxes, reason: 'canvas_edit', bounty_id: BOUNTY_ID }),
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
    renderIntegrity();
  }

  /**
   * Warn about frames that are "good" with no boxes on disk.
   *
   * This is the visible half of the save-order bug: those frames look finished
   * on the board but contribute nothing to a dataset, so no bounty PR can be
   * paid for them. The repair returns them to the queue instead of inventing
   * labels.
   */
  function renderIntegrity() {
    const bar = $('integrityBar');
    if (!bar) return;
    const n = (state.dataset && state.dataset.integrity
      && state.dataset.integrity.good_without_labels) || 0;
    bar.hidden = n === 0 || state.integrityDismissed;
    if (!n) return;
    $('integrityText').textContent = n === 1
      ? '1 frame is marked good but has no labels'
      : `${n} frames are marked good but have no labels`;
    // Repairing rewrites review state, so it is not offered while browsing.
    $('integrityFix').hidden = VIEW_MODE;
  }

  async function repairIntegrity() {
    const btn = $('integrityFix');
    btn.disabled = true;
    try {
      const res = await api('/api/integrity/repair', { method: 'POST', body: '{}' });
      toast(res.reset === 1
        ? '1 frame returned to the queue'
        : `${res.reset} frames returned to the queue`);
      state.integrityDismissed = true;
      await loadDataset();
      await loadSamples(true);
    } catch (err) {
      toast('Repair failed: ' + err.message, true);
    } finally {
      btn.disabled = false;
    }
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
    if (BOUNTY_ID) params.set('bounty', BOUNTY_ID);
    const data = await api('/api/samples?' + params.toString());
    state.total = data.total;
    state.reviewedTotal = data.reviewed || 0;
    state.samples = reset ? data.items : state.samples.concat(data.items);
    state.offset = state.samples.length;
    renderStrip(reset);
    renderProgress();
    $('loadMore').hidden = REVIEW_MODE || state.samples.length >= state.total;
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
    const data = await api('/api/sample/' + encId(imageId));
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
      img.src = withDataset('/api/image/' + encId(imageId));
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

  async function selectIndex(i) {
    if (i < 0 || i >= state.samples.length) return;
    // Await the write: navigating used to fire-and-forget the PUT, so a quick
    // refresh could abandon the save for the frame you just left.
    if (state.dirty) await saveNow(true);
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
    img.src = withDataset('/api/thumb/' + encId(row.image_id));
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
    // Prefer the server's whole-filter tally; the loaded page is only a window
    // onto a much larger queue in a bounty.
    const total = state.total || state.samples.length;
    const reviewed = state.reviewedTotal != null
      ? state.reviewedTotal
      : state.samples.filter((s) => s.status !== 'unreviewed').length;
    $('progressFill').style.width = (total ? (reviewed / total) * 100 : 0).toFixed(1) + '%';
    const r = state.dataset.review || {};
    // Reviewers need the queue number; the good/bad/empty split is bookkeeping
    // that only means something when auditing, so it is expert-only text.
    $('progressText').textContent = state.role === 'expert'
      ? `${reviewed}/${total} reviewed · ${r.good || 0} good · ${r.bad || 0} bad · ${r.empty || 0} empty`
      : `${reviewed} of ${total} reviewed`;
    updateBountyProgress();
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
    const expert = state.role === 'expert';
    state.boxes.forEach((b, i) => {
      const row = document.createElement('div');
      row.className = 'box-row' + (i === state.selected ? ' selected' : '');
      // Reviewers get a compact row they can select and delete from; the exact
      // coordinates are a power feature (Expert), not something to explain.
      if (expert) {
        row.innerHTML = `
          <span class="swatch"></span>
          <input data-k="xc" type="number" step="0.001" min="0" max="1" value="${b.xc.toFixed(3)}" />
          <input data-k="yc" type="number" step="0.001" min="0" max="1" value="${b.yc.toFixed(3)}" />
          <input data-k="w" type="number" step="0.001" min="0" max="1" value="${b.w.toFixed(3)}" />
          <input data-k="h" type="number" step="0.001" min="0" max="1" value="${b.h.toFixed(3)}" />
          <button class="del ghost" title="Delete box">×</button>`;
      } else {
        const size = `${Math.round(b.w * 100)}×${Math.round(b.h * 100)}%`;
        row.classList.add('compact');
        row.innerHTML = `
          <span class="swatch"></span>
          <span class="box-name">Player ${i + 1}</span>
          <span class="box-meta muted">${size}</span>
          <button class="del ghost" title="Delete box">×</button>`;
      }
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

    renderStatusPill();
    renderPredMetrics();
    renderFixBlock();
    renderViewToggle();
  }

  /**
   * A one-line answer to "have I already judged this?", so a reviewer returning
   * to a half-finished bounty does not have to infer state from dot colours.
   */
  function renderStatusPill() {
    const pill = $('statusPill');
    const st = (state.current && state.current.status) || 'unreviewed';
    const label = {
      good: 'Marked good',
      bad: 'Marked: needs fixing',
      empty: 'Marked: not gameplay',
      unreviewed: 'Not reviewed yet',
    }[st] || st;
    pill.textContent = label;
    pill.className = 'status-pill ' + st;
    pill.hidden = !state.current;

    // Reasons and the note are only actionable while fixing, so they stay out
    // of the way the rest of the time rather than sitting there as dead space.
    $('badDetail').hidden = st !== 'bad';
    // Choosing a reason is the one thing that must not be missed, so prompt it
    // when a frame is bad but no reason has been picked yet.
    $('reasonChips').classList.toggle('needs-reason', st === 'bad' && !state.reasons.length);
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
    pill.title = hintFor(m);

    const parts = [];
    if (m.missed_player_hi) parts.push(`<span class="metric bad">unlabelled ×${m.missed_player_hi}</span>`);
    if (m.missing) parts.push(`<span class="metric bad">not found ×${m.missing}</span>`);
    if (m.duplicate_hi) parts.push(`<span class="metric warn">duplicate ×${m.duplicate_hi}</span>`);
    if (m.weak) parts.push(`<span class="metric warn">loose ×${m.weak}</span>`);
    if (m.matched) parts.push(`<span class="metric ok">matched ×${m.matched}</span>`);
    parts.push(`<span class="metric">labels ${m.gt_count || 0}</span>`);
    parts.push(`<span class="metric">model ${m.pred_count || 0}</span>`);
    $('predMetrics').innerHTML = parts.join('');

    // Show what the dedupe+cap pass removed, so the correction is visible.
    const merged = cur.predictions_merged || 0;
    const pp = (state.predSummary && state.predSummary.postprocess) || {};
    state.cleanNote = merged
      ? `Dedupe merged ${merged} duplicate box${merged === 1 ? '' : 'es'} (IoU ${pp.nms_iou}, cap ${pp.max_boxes}/frame).`
      : '';
  }

  /**
   * The one piece of model output a reviewer needs: "there is a player here you
   * have not boxed, accept it?". Kept separate from the expert metrics block so
   * the suggestion survives when the numbers are hidden.
   */
  function renderFixBlock() {
    const cur = state.current;
    const hasPred = !!(cur && cur.predictions_available);
    // Offer the action only when labels are visible to adopt into.
    const adoptable = hasPred && adoptEnabled() ? adoptablePreds(0.5).length : 0;
    const btn = $('adoptBtn');
    const block = $('fixBlock');

    btn.hidden = adoptable === 0;
    if (adoptable) {
      btn.textContent = adoptable === 1
        ? 'Adopt 1 missing box (a)'
        : `Adopt ${adoptable} missing boxes (a)`;
    }
    const note = $('fixNote');
    if (!hasPred) {
      note.textContent = '';
    } else if (!adoptEnabled()) {
      note.textContent = 'Switch to Both to adopt suggestions.';
    } else if (adoptable) {
      note.textContent = 'Click a dashed box to adopt just that one.';
    } else {
      note.textContent = 'Nothing unlabelled stands out on this frame.';
    }
    $('fixCount').textContent = adoptable ? `(${adoptable})` : '';
    block.hidden = !hasPred;
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
      toast(writeBlockedMessage());
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
      const data = await api(`/api/sample/${encId(state.current.image_id)}/review`, {
        method: 'POST',
        body: JSON.stringify({ status, reasons: state.reasons, note: $('noteBox').value, bounty_id: BOUNTY_ID }),
      });
      const imageId = state.current.image_id;
      const prevStatus = state.current.status;
      state.current.status = data.review.status;
      state.current.review = data.review;
      // Keep the queue tally in step without a round-trip: only the
      // unreviewed -> reviewed boundary changes the count.
      if (state.reviewedTotal != null) {
        const wasReviewed = prevStatus && prevStatus !== 'unreviewed';
        const isReviewed = data.review.status !== 'unreviewed';
        if (!wasReviewed && isReviewed) state.reviewedTotal += 1;
        else if (wasReviewed && !isReviewed) state.reviewedTotal -= 1;
      }
      refreshRow(imageId, { status: data.review.status, reasons: data.review.reasons });
      // The dataset-wide tally lives on the server; refresh it before painting
      // progress so the header numbers are not a step behind.
      try {
        state.dataset.review = (await api('/api/dataset')).review;
      } catch (_) { /* header counts are cosmetic */ }
      renderProgress();
      syncChips();
      // The status pill and the reasons/note panel are driven by the status we
      // just stored, so they have to be repainted here — nothing else redraws
      // the panel on a review change.
      renderStatusPill();
      return data.review.status;
    } catch (err) {
      toast('Review update failed: ' + err.message, true);
      return null;
    }
  }

  async function markGood() {
    if (VIEW_MODE) return toast(writeBlockedMessage(), true);
    // Clearing reasons matters: a stale reason from the previous bad frame
    // would otherwise be recorded against a good one.
    state.reasons = [];
    syncChips();
    if (!state.boxes.length) {
      // "Good" approves the boxes on the frame, so there have to be some. The
      // server enforces this too; saying it here avoids a round-trip and a
      // scary error for what is a normal thing to get wrong.
      toast('No boxes on this frame — draw one, adopt the model’s, or press n for Empty', true);
      return;
    }
    // Persist the labels *before* recording the review. The other order left a
    // window where a refresh could land between the two writes, producing a
    // frame marked good whose label file was still empty.
    if (state.dirty) await saveNow(true);
    const status = await postReview('good');
    if (!status) return;   // the review failed; stay on this frame
    toast('Marked good');
    step(1);
  }

  async function markEmpty() {
    if (VIEW_MODE) return toast(writeBlockedMessage(), true);
    state.reasons = [];
    syncChips();
    if (state.dirty) await saveNow(true);
    const status = await postReview('empty');
    if (!status) return;
    toast('Marked reviewed-empty');
    step(1);
  }

  async function markBad() {
    if (VIEW_MODE) return toast(writeBlockedMessage(), true);
    // A reason is strongly encouraged but not a precondition: gating on it
    // deadlocked the UI, because the chips that set a reason only appear once
    // the frame is already marked bad.
    if (state.dirty) await saveNow(true);
    await postReview('bad');
    toast(state.reasons.length
      ? 'Needs fixing: ' + state.reasons.join(', ')
      : 'Needs fixing — now pick a reason (1–6)');
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

    // The walkthrough is modal: while it is up, keys drive the tour rather than
    // the review queue underneath it.
    if (TOUR.open) {
      if (evt.key === 'Escape') { endTour(false); return; }
      if (evt.key === 'ArrowRight' || evt.key === 'Enter') {
        evt.preventDefault();
        $('tourNext').click();
        return;
      }
      if (evt.key === 'ArrowLeft') { evt.preventDefault(); $('tourBack').click(); return; }
      return;
    }

    // Video mode owns space and the arrow keys; the image bindings below would
    // otherwise pull the review queue out from under the player.
    if (state.appMode === 'video' && !typing) {
      if (evt.key === ' ' || evt.key === 'k') {
        evt.preventDefault();
        toggleVideoPlay();
        return;
      }
      if (evt.key === 'ArrowRight' || evt.key === 'l') {
        evt.preventDefault();
        stepVideoFrame(1);
        return;
      }
      if (evt.key === 'ArrowLeft' || evt.key === 'j') {
        evt.preventDefault();
        stepVideoFrame(-1);
        return;
      }
      if (evt.key === 'p') {
        evt.preventDefault();
        const order = ['labels', 'model', 'both'];
        const next = order[(order.indexOf(VIDEO.viewMode) + 1) % order.length];
        setVideoViewMode(next);
        return;
      }
    }

    if (evt.key === 'Escape') {
      if (TOUR.open) { endTour(false); return; }
      if (!$('popover').hidden) { hidePopover(); return; }
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
      toast(writeBlockedMessage());
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
  // ------------------------------------------------------------ video mode
  //
  // Playback mode replays a source MP4 with the dataset's boxes drawn over it.
  // The ingest samples at a target fps and keeps every sample, but older
  // datasets (or a low fps) leave gaps: boxes are shown for the nearest kept
  // frame and hidden when the playhead is further away than half a sampling
  // interval. That makes the overlay honest - if the overlay is empty, the
  // dataset really has no frame there, instead of quietly showing a stale box
  // over a results screen.

  const VIDEO = {
    loaded: false,
    index: 0,
    videos: [],
    frames: [],
    times: [],
    tolerance: 0.5,
    viewMode: 'both',
    active: null,       // nearest frame for the current playhead
    raf: null,
    seeking: false,
    bounty: null,       // {start, end, count, title} for the open bounty
    playTo: null,       // pause when the playhead reaches this second
  };

  async function loadVideos() {
    try {
      const res = await api('/api/videos');
      VIDEO.videos = res.videos || [];
    } catch (err) {
      VIDEO.videos = [];
    }
    const sel = $('videoSelect');
    sel.innerHTML = '';
    if (!VIDEO.videos.length) {
      $('videoEmpty').textContent =
        'No source video found for this dataset. Ingest a video with ' +
        'scripts/ingest_video.py, or start the server with --video <file>.';
      $('videoEmpty').hidden = false;
      $('videoEl').hidden = true;
      $('videoInfo').textContent = '';
      return;
    }
    $('videoEmpty').hidden = true;
    $('videoEl').hidden = false;
    for (const v of VIDEO.videos) {
      const opt = document.createElement('option');
      opt.value = String(v.index);
      opt.textContent = v.name;
      sel.appendChild(opt);
    }
    sel.value = String(VIDEO.index);
    await openVideo(VIDEO.index);
  }

  async function openVideo(index) {
    if (!VIDEO.videos.length) return;
    VIDEO.index = Math.max(0, Math.min(index, VIDEO.videos.length - 1));
    const el = $('videoEl');
    el.pause();
    el.src = '/api/video/' + VIDEO.index;
    el.load();
    VIDEO.frames = [];
    VIDEO.times = [];
    VIDEO.active = null;
    VIDEO.loaded = false;
    VIDEO.bounty = null;
    VIDEO.playTo = null;
    renderVideoRange();
    $('videoFrameInfo').textContent = 'loading timeline…';
    $('videoEmpty').hidden = true;
    drawVideoOverlay();

    let data;
    try {
      // The bounty span scopes the scrubber and the play button; the timeline
      // itself is unchanged, so a bounty id is optional.
      const query = BOUNTY_ID ? '?bounty=' + encodeURIComponent(BOUNTY_ID) : '';
      data = await api('/api/video/' + VIDEO.index + '/timeline' + query);
    } catch (err) {
      $('videoFrameInfo').textContent = 'timeline failed: ' + err.message;
      return;
    }
    VIDEO.frames = (data.frames || []).filter((f) => typeof f.t === 'number');
    VIDEO.times = VIDEO.frames.map((f) => f.t);
    const span = data.bounty;
    VIDEO.bounty = span && typeof span.start === 'number'
      && typeof span.end === 'number' && span.end >= span.start ? span : null;
    VIDEO.loaded = true;
    // Half the typical gap is the natural cut-off: any further and the box
    // belongs to a meaningfully different moment than what is on screen.
    if (VIDEO.times.length > 1) {
      const gaps = [];
      for (let i = 1; i < VIDEO.times.length; i++) gaps.push(VIDEO.times[i] - VIDEO.times[i - 1]);
      gaps.sort((a, b) => a - b);
      const median = gaps[Math.floor(gaps.length / 2)] || 0.2;
      // One sampling interval each side gives contiguous coverage between
      // kept frames (no flicker) while still hiding the overlay within a
      // fraction of a second of a real gap such as a menu or results screen.
      VIDEO.tolerance = Math.min(0.6, Math.max(0.2, median));
    }
    const meta = VIDEO.videos[VIDEO.index];
    const bits = [meta.name];
    if (meta.duration) bits.push(fmtDuration(meta.duration));
    if (VIDEO.frames.length) bits.push(VIDEO.frames.length + ' sampled frames');
    $('videoInfo').textContent = bits.join(' · ');
    $('videoFrameInfo').textContent = '';
    updateVideoClock();
    renderVideoRange();
    // Open on the bounty's first frame, not the start of the set: the whole
    // match is usually tens of minutes, and 0:00 is never the useful moment.
    if (VIDEO.bounty && (el.currentTime || 0) < 1) {
      try { el.currentTime = VIDEO.bounty.start + 0.001; } catch (_) { /* no metadata yet */ }
      updateVideoClock();
    }
    drawVideoOverlay();
  }

  // ------------------------------------------------- bounty interval
  // The player is opened from a bounty, so its slice of the source video is
  // knowable: the first and last sampled second among its frames. Marking that
  // stretch is the difference between "watch 40 minutes of a tournament set"
  // and "watch the two minutes I am paid for".
  function videoDuration() {
    const el = $('videoEl');
    if (isFinite(el.duration) && el.duration > 0) return el.duration;
    const meta = VIDEO.videos[VIDEO.index] || {};
    return Number(meta.duration) || 0;
  }

  function renderVideoRange() {
    const ribbon = $('videoRibbon');
    const fill = $('videoRibbonFill');
    const badge = $('videoBountyText');
    const btn = $('videoBountyPlay');
    const b = VIDEO.bounty;
    const dur = videoDuration();
    if (!b || !(dur > 0) || !(b.end >= b.start) || !b.count) {
      ribbon.hidden = true;
      badge.hidden = true;
      btn.hidden = true;
      return;
    }
    ribbon.hidden = false;
    badge.hidden = false;
    btn.hidden = false;
    const start = Math.max(0, Math.min(b.start, dur));
    const end = Math.max(start, Math.min(b.end + 0.001, dur));
    fill.style.left = (start / dur) * 100 + '%';
    fill.style.width = Math.max(0.5, ((end - start) / dur) * 100) + '%';
    badge.textContent = `bounty · ${fmtDuration(b.start)}–${fmtDuration(b.end)}`
      + ` · ${Number(b.count).toLocaleString('en-US')} frames`;
    badge.title = b.title || 'The frames this bounty covers';
  }

  /** Play only the bounty's stretch: seek to its first frame, stop after its last. */
  function playBountyRange() {
    const b = VIDEO.bounty;
    if (!b) return;
    // A hair of tail so the final frame is actually on screen when it stops.
    VIDEO.playTo = b.end + 0.15;
    seekVideoTo(b.start + 0.001, true);
  }

  function fmtDuration(sec) {
    if (typeof sec !== 'number' || !isFinite(sec)) return '–';
    const s = Math.max(0, Math.floor(sec));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const r = s % 60;
    const pad = (n) => String(n).padStart(2, '0');
    return h ? `${h}:${pad(m)}:${pad(r)}` : `${m}:${pad(r)}`;
  }

  function nearestFrameIndex(t) {
    const times = VIDEO.times;
    if (!times.length) return -1;
    let lo = 0;
    let hi = times.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (times[mid] < t) lo = mid + 1;
      else hi = mid;
    }
    if (lo > 0 && Math.abs(times[lo - 1] - t) <= Math.abs(times[lo] - t)) return lo - 1;
    return lo;
  }

  function activeVideoFrame() {
    if (!VIDEO.times.length) return null;
    const t = $('videoEl').currentTime || 0;
    const i = nearestFrameIndex(t);
    if (i < 0) return null;
    const f = VIDEO.frames[i];
    const strict = $('videoFollowFrame').checked;
    if (strict && Math.abs(f.t - t) > VIDEO.tolerance) return null;
    return f;
  }

  function videoBoxesVisible(kind) {
    return VIDEO.viewMode === 'both' || VIDEO.viewMode === kind;
  }

  function drawVideoOverlay() {
    const c = $('videoOverlay');
    const v = $('videoEl');
    const rect = c.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const cssW = Math.max(1, Math.round(rect.width));
    const cssH = Math.max(1, Math.round(rect.height));
    if (c.width !== Math.round(cssW * dpr) || c.height !== Math.round(cssH * dpr)) {
      c.width = Math.round(cssW * dpr);
      c.height = Math.round(cssH * dpr);
    }
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, cssW, cssH);
    if (!v.videoWidth || !v.videoHeight) return;

    // object-fit: contain -> the picture is letterboxed inside the element.
    const scale = Math.min(cssW / v.videoWidth, cssH / v.videoHeight);
    const dw = v.videoWidth * scale;
    const dh = v.videoHeight * scale;
    const ox = (cssW - dw) / 2;
    const oy = (cssH - dh) / 2;

    const frame = activeVideoFrame();
    VIDEO.active = frame;
    if (!frame) {
      $('videoFrameInfo').textContent = VIDEO.times.length
        ? 'no sampled frame here'
        : '';
      return;
    }

    const draw = (box, color, dashed, label) => {
      const x = ox + (box.xc - box.w / 2) * dw;
      const y = oy + (box.yc - box.h / 2) * dh;
      const w = box.w * dw;
      const h = box.h * dh;
      g.save();
      g.lineWidth = 2.5;
      g.strokeStyle = color;
      if (dashed) g.setLineDash([7, 5]);
      g.strokeRect(x, y, w, h);
      g.setLineDash([]);
      if (label) {
        g.font = '600 12px ui-sans-serif, system-ui, sans-serif';
        const tw = g.measureText(label).width + 8;
        const ty = y - 6 > 14 ? y - 6 : y + h + 14;
        g.fillStyle = 'rgba(11, 15, 20, 0.82)';
        g.fillRect(x - 1, ty - 12, tw, 16);
        g.fillStyle = color;
        g.fillText(label, x + 3, ty);
      }
      g.restore();
    };

    if (videoBoxesVisible('model')) {
      for (const b of frame.model || []) {
        // Same amber as the static canvas and the legend swatch: one meaning,
        // one colour, regardless of which mode you are looking at.
        draw(b, '#e0a437', true, b.conf ? `${Math.round(b.conf * 100)}%` : 'model');
      }
    }
    if (videoBoxesVisible('labels')) {
      for (const b of frame.labels || []) {
        draw(b, '#35c07a', false, b.cls ? `#${b.cls}` : 'label');
      }
    }

    const i = nearestFrameIndex(frame.t);
    $('videoFrameInfo').textContent =
      `frame ${i + 1}/${VIDEO.frames.length} · t=${frame.t.toFixed(2)}s · ` +
      `${(frame.labels || []).length} label / ${(frame.model || []).length} model`;
  }

  function updateVideoClock() {
    const v = $('videoEl');
    const dur = isFinite(v.duration) ? v.duration : (VIDEO.videos[VIDEO.index] || {}).duration;
    const t = v.currentTime || 0;
    $('videoTime').textContent = `${fmtDuration(t)} / ${fmtDuration(dur)}`;
    if (!VIDEO.seeking && dur > 0) {
      $('videoSeek').value = String(Math.round((t / dur) * 1000));
    }
  }

  function videoTick() {
    updateVideoClock();
    drawVideoOverlay();
    const v = $('videoEl');
    // "Start from bounty" is a bounded play: stop at the slice's last frame
    // instead of running into the next game of the set.
    if (VIDEO.playTo != null && v.currentTime >= VIDEO.playTo) {
      VIDEO.playTo = null;
      v.pause();
      return;
    }
    if (!v.paused && !v.ended) {
      VIDEO.raf = requestAnimationFrame(videoTick);
    } else {
      VIDEO.raf = null;
    }
  }

  function toggleVideoPlay() {
    const v = $('videoEl');
    if (v.paused) v.play().catch((err) => toast('Cannot play: ' + err.message, true));
    else v.pause();
  }

  function seekVideoTo(t, play = false) {
    const v = $('videoEl');
    const dur = isFinite(v.duration) ? v.duration : 0;
    v.currentTime = Math.max(0, dur ? Math.min(t, dur - 0.01) : t);
    drawVideoOverlay();
    updateVideoClock();
    if (play) v.play().catch(() => {});
  }

  function stepVideoFrame(delta) {
    if (!VIDEO.times.length) return;
    VIDEO.playTo = null;
    const v = $('videoEl');
    const i = nearestFrameIndex(v.currentTime || 0);
    const base = delta > 0 && VIDEO.times[i] < (v.currentTime || 0) - 1e-3 ? i + 1 : i;
    const next = Math.max(0, Math.min(VIDEO.frames.length - 1, base + (delta > 0 ? 0 : -1)));
    // Land exactly on the sample so "follow sampled frames" always shows it.
    seekVideoTo(VIDEO.times[next] + 0.001);
  }

  function setVideoViewMode(mode) {
    VIDEO.viewMode = mode;
    for (const seg of $('videoViewToggle').querySelectorAll('.seg')) {
      seg.classList.toggle('on', seg.dataset.vmode === mode);
    }
    drawVideoOverlay();
  }

  function setAppMode(mode) {
    state.appMode = mode;
    for (const seg of $('appMode').querySelectorAll('.seg')) {
      seg.classList.toggle('on', seg.dataset.appmode === mode);
    }
    const isVideo = mode === 'video';
    $('layout').hidden = isVideo;
    $('videoView').hidden = !isVideo;
    if (isVideo) {
      if (!VIDEO.loaded) loadVideos();
      requestAnimationFrame(() => {
        drawVideoOverlay();
        updateVideoClock();
      });
    } else {
      const v = $('videoEl');
      if (v) v.pause();
      resizeCanvas();
      draw();
    }
    try {
      localStorage.setItem('ssbu.appMode', mode);
    } catch (e) { /* private mode */ }
  }

  function bindVideo() {
    $('appMode').addEventListener('click', (e) => {
      const seg = e.target.closest('.seg');
      if (seg) setAppMode(seg.dataset.appmode);
    });
    $('videoViewToggle').addEventListener('click', (e) => {
      const seg = e.target.closest('.seg');
      if (seg) setVideoViewMode(seg.dataset.vmode);
    });
    $('videoSelect').addEventListener('change', (e) => openVideo(Number(e.target.value)));

    const v = $('videoEl');
    v.addEventListener('play', () => {
      $('videoPlay').textContent = 'Pause';
      if (!VIDEO.raf) VIDEO.raf = requestAnimationFrame(videoTick);
    });
    v.addEventListener('pause', () => {
      $('videoPlay').textContent = 'Play';
      if (VIDEO.raf) { cancelAnimationFrame(VIDEO.raf); VIDEO.raf = null; }
      updateVideoClock();
      drawVideoOverlay();
    });
    v.addEventListener('loadedmetadata', () => {
      updateVideoClock();
      renderVideoRange();
      drawVideoOverlay();
    });
    v.addEventListener('seeked', () => {
      VIDEO.seeking = false;
      updateVideoClock();
      drawVideoOverlay();
    });
    // rAF stops when the tab is hidden, so the bound is enforced here too.
    v.addEventListener('timeupdate', () => {
      if (VIDEO.playTo != null && v.currentTime >= VIDEO.playTo) {
        VIDEO.playTo = null;
        v.pause();
      }
      updateVideoClock();
    });
    v.addEventListener('ended', () => {
      VIDEO.playTo = null;
      $('videoPlay').textContent = 'Play';
    });

    $('videoPlay').addEventListener('click', toggleVideoPlay);
    $('videoBountyPlay').addEventListener('click', playBountyRange);
    $('videoPrev').addEventListener('click', () => stepVideoFrame(-1));
    $('videoNext').addEventListener('click', () => stepVideoFrame(1));
    $('videoFollowFrame').addEventListener('change', drawVideoOverlay);
    const seek = $('videoSeek');
    seek.addEventListener('input', () => {
      // Hand-steering the playhead cancels a bounded play: nothing is more
      // annoying than pausing because you scrubbed past the old end.
      VIDEO.playTo = null;
      VIDEO.seeking = true;
      const dur = isFinite(v.duration) ? v.duration : 0;
      if (dur > 0) v.currentTime = (Number(seek.value) / 1000) * dur;
      updateVideoClock();
      drawVideoOverlay();
    });
    seek.addEventListener('change', () => { VIDEO.seeking = false; });
    // The ribbon is a shortcut into the bounty's stretch, not a second slider.
    $('videoRibbon').addEventListener('click', (ev) => {
      const b = VIDEO.bounty;
      const dur = videoDuration();
      if (!b || !(dur > 0)) return;
      const rect = ev.currentTarget.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
      VIDEO.playTo = null;
      seekVideoTo(ratio * dur);
    });
    window.addEventListener('resize', () => {
      if (state.appMode === 'video') drawVideoOverlay();
    });
  }

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
    $('integrityFix').addEventListener('click', repairIntegrity);
    $('integrityHide').addEventListener('click', () => {
      state.integrityDismissed = true;
      renderIntegrity();
    });

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
        const res = await api('/api/undo/' + encId(state.current.image_id), {
          method: 'POST',
          body: JSON.stringify({ bounty_id: BOUNTY_ID }),
        });
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

    // Role switch: the only structural UI control a reviewer is trusted with.
    $('roleSwitch').addEventListener('click', (e) => {
      const seg = e.target.closest('.seg');
      if (seg) setRole(seg.dataset.role);
    });

    // One delegated listener for every "?" button. Delegation means a popover
    // added to any block later needs no extra wiring.
    document.addEventListener('click', (e) => {
      const info = e.target.closest('.info');
      if (info) {
        e.preventDefault();
        showPopover(info);
        return;
      }
      if (!e.target.closest('#popover')) hidePopover();
    });
    $('popoverClose').addEventListener('click', hidePopover);

    // Walkthrough controls.
    $('tourNext').addEventListener('click', () => {
      if (TOUR.i >= TOUR_STEPS.length - 1) { endTour(true); return; }
      TOUR.i += 1;
      renderTour();
    });
    $('tourBack').addEventListener('click', () => {
      if (TOUR.i > 0) { TOUR.i -= 1; renderTour(); }
    });
    $('tourSkip').addEventListener('click', () => endTour(false));
    $('replayTour').addEventListener('click', () => {
      $('helpOverlay').hidden = true;
      startTour(true);
    });
    $('tourBtn').addEventListener('click', () => startTour(true));

    // Leaving a session must not drop the last edit: autosave is a debounce, so
    // a click that lands inside the window would otherwise discard it. If the
    // flush fails we stay put rather than walk away from unsaved work.
    for (const el of document.querySelectorAll('.back-link')) {
      el.addEventListener('click', async (ev) => {
        if (VIEW_MODE || !state.dirty || !state.current) return;
        ev.preventDefault();
        const href = el.getAttribute('href') || '/';
        await saveNow(true);
        if (state.dirty && state.current) {
          toast('Could not save this frame, so you are still here.', true);
          return;
        }
        location.href = href;
      });
    }

    window.addEventListener('resize', resizeCanvas);
    window.addEventListener('resize', () => {
      hidePopover();
      if (TOUR.open) renderTour();
    });
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

  /**
   * A bounty is just a named slice of the queue, plus a reason to work on it.
   * The CLI passes ?bounty=<title>&claim=<n>&of=<total>. Nothing here is
   * authoritative: progress is recomputed from the review store, so a stale
   * link cannot overstate what is done.
   */
  function initBounty() {
    // `bounty` carries the human title; `bounty_id` is always present in label
    // mode. Falling back to the id means a bookmarked or hand-edited URL still
    // shows the bar instead of silently looking like an unscoped session.
    const title = PARAMS.get('bounty') || BOUNTY_ID;
    if (!title) return;
    $('bountyTitle').textContent = title;
    $('bountyBar').hidden = false;
    document.body.classList.add('has-bounty');
    applyEditMode();
    updateBountyProgress();
    // The walkthrough is the onboarding step for a claimed bounty, so it runs
    // once per contributor rather than once per bounty.
    startTour(false);
  }

  /**
   * Editing a completed bounty is the same screen with a different contract:
   * the slice is fixed and already reviewed, so the work is double-checking
   * rather than labelling. Say so, and say what happens on submit - a second
   * submit updates the existing PR instead of opening another one.
   */
  function applyEditMode() {
    if (!EDIT_MODE && !PR_EXISTS) return;
    const pill = $('bountyPill');
    document.body.classList.add('edit-mode');
    const note = $('editNote');
    if (PR_EXISTS) {
      // Submitted: the labels are the content of a PR, so they are frozen. The
      // server refuses writes too; this just stops the screen from inviting
      // them. Reached only from a stale link - the board no longer offers Edit.
      if (pill) pill.textContent = 'Submitted';
      document.body.classList.add('bounty-locked');
      if (note) {
        note.hidden = false;
        note.textContent = 'submitted · labels are locked while the PR is open';
      }
      return;
    }
    if (pill) pill.textContent = 'Editing';
    if (!note) return;
    note.hidden = false;
    note.textContent = 'complete · double-check the labels, then submit from the board';
  }

  /** Read-only chrome for /view: no decisions, no editing, no autosave. */
  function applyViewMode() {
    document.body.classList.add('view-mode');
    $('viewBar').hidden = false;
    $('viewDataset').textContent = DATASET || 'dataset';
    const back = $('backToBoard');
    back.textContent = '← All datasets';
    back.title = 'Back to datasets and bounties';
    // Anything that would write is removed rather than disabled: a greyed-out
    // "Good" button invites a click and then an explanation.
    for (const id of ['decideBlock', 'fixBlock']) {
      const el = $(id);
      if (el) el.hidden = true;
    }
    for (const id of ['drawBtn', 'addBoxBtn', 'saveBtn', 'undoBtn']) {
      const el = $(id);
      if (el) el.hidden = true;
    }
    $('statusPill').hidden = true;
    document.title = `${DATASET || 'Dataset'} — view`;
  }

  function updateBountyProgress() {
    if ($('bountyBar').hidden) return;
    const total = state.total || 0;
    const done = state.reviewedTotal != null
      ? state.reviewedTotal
      : state.samples.filter((s) => s.status && s.status !== 'unreviewed').length;
    $('bountyProgress').textContent = total
      ? `${done} of ${total} reviewed`
      : '';
  }

  async function boot() {
    bind();
    bindVideo();
    resizeCanvas();
    const stamp = $('uiVersion');
    if (stamp) stamp.textContent = UI_VERSION;
    window.__uiVersion = UI_VERSION;

    // Restore the detail level before anything renders, so a returning expert
    // does not see the panel flip from simple to full.
    let role = 'review';
    try {
      const savedRole = localStorage.getItem('ssbu.role');
      if (savedRole === 'expert') role = 'expert';
    } catch (e) { /* private mode */ }
    if (new URLSearchParams(location.search).get('role') === 'expert') role = 'expert';
    setRole(role, false);

    try {
      await loadDataset();
      if (VIEW_MODE) applyViewMode();
      // After loadDataset, since it may hide filters that do not apply and
      // resets the sort default.
      const applied = applyUrlFilters();
      if (applied.length) toast('Filters from link: ' + applied.join(', '));
      await loadSamples(true);
      await loadJournal();
      setInterval(loadJournal, 15000);
      updateBountyProgress();
      initBounty();
      // An explicit ?mode= wins in both directions. The remembered mode is only
      // used while browsing: landing in Video because the last mp4 was watched
      // would hide the strip exactly when someone opened a bounty to label it.
      let mode = 'images';
      const modeParam = PARAMS.get('mode');
      if (modeParam === 'video' || modeParam === 'images') {
        mode = modeParam;
      } else if (VIEW_MODE) {
        try {
          const saved = localStorage.getItem('ssbu.appMode');
          if (saved === 'video') mode = 'video';
        } catch (e) { /* private mode */ }
      }
      setAppMode(mode);
    } catch (err) {
      toast('Startup failed: ' + err.message, true);
    }
  }

  boot();
})();
