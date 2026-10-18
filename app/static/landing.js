/* Landing screen: dataset catalogue, bounty board, progress rings, add-dataset.
 *
 * Vanilla JS to match the review app - no build step, no framework. The visual
 * pieces (concentric progress rings, animated bars) follow the pattern used by
 * React dashboard components: stroke-dashoffset transitions for arcs and a
 * width transition for bars, both driven from a single render pass.
 */

(() => {
  'use strict';

  const REPO = 'https://github.com/MicahHeneveld/bros-analysis';
  const state = {
    board: null,
    selectedDataset: null,
    selectedBounty: null,   // bounty object, not just id
    bountyPinned: false,    // true once the user picks a bounty explicitly
    job: null,
    jobTimer: null,
    uploadXhr: null,
    file: null,
    ghUser: '',
    ghError: '',
    ghKind: '',
    ghHint: '',
    ghRetries: 0,
    ghRetryTimer: null,
  };

  const $ = (id) => document.getElementById(id);
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ------------------------------------------------------------- helpers
  function esc(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, (c) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
  }

  function fmt(n) {
    const v = Number(n) || 0;
    return v.toLocaleString('en-US');
  }

  function pct(value) {
    return Math.max(0, Math.min(100, Number(value) || 0));
  }

  function toast(message, isError = false) {
    const el = $('toast');
    el.textContent = message;
    el.className = 'toast' + (isError ? ' err' : '');
    el.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { el.hidden = true; }, 4200);
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
      throw new Error((data && data.error) || `${res.status} ${res.statusText}`);
    }
    return data;
  }

  /** Count a number up to its target, retargeting from wherever it is now. */
  function countTo(el, target, suffix = '') {
    const to = Number(target) || 0;
    if (reduceMotion) { el.textContent = fmt(Math.round(to)) + suffix; return; }
    const from = Number(el.dataset.value || 0);
    if (from === to) { el.textContent = fmt(Math.round(to)) + suffix; return; }
    el.dataset.value = to;
    const start = performance.now();
    const duration = 650;
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = fmt(Math.round(from + (to - from) * eased)) + suffix;
      if (t < 1 && Number(el.dataset.value) === to) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  /** Animate every bar from zero to its data-target on the next frame. */
  function animateBars() {
    const bars = document.querySelectorAll('.bar > div[data-target]');
    if (reduceMotion) {
      bars.forEach((el) => { el.style.width = el.dataset.target + '%'; });
      return;
    }
    bars.forEach((el) => { el.style.width = '0%'; });
    requestAnimationFrame(() => requestAnimationFrame(() => {
      bars.forEach((el) => { el.style.width = el.dataset.target + '%'; });
    }));
  }

  function setRing(el, percent) {
    const r = Number(el.getAttribute('r')) || 100;
    const circumference = 2 * Math.PI * r;
    const value = pct(percent);
    el.style.strokeDasharray = `${circumference}`;
    // Start empty, then let the CSS transition fill the arc.
    el.style.strokeDashoffset = `${circumference}`;
    if (reduceMotion) {
      el.style.strokeDashoffset = `${circumference * (1 - value / 100)}`;
      return;
    }
    requestAnimationFrame(() => requestAnimationFrame(() => {
      el.style.strokeDashoffset = `${circumference * (1 - value / 100)}`;
    }));
  }

  // --------------------------------------------------------------- load
  async function loadBoard({ quiet = false } = {}) {
    try {
      const data = await api('/api/board');
      state.board = data;
      state.ghUser = (data.github && data.github.user) || '';
      state.ghError = (data.github && data.github.error) || '';
      state.ghKind = (data.github && data.github.kind) || '';
      state.ghHint = (data.github && data.github.hint) || '';

      const datasets = data.datasets || [];
      const names = datasets.map((d) => d.name);
      if (!state.selectedDataset || !names.includes(state.selectedDataset)) {
        const first = datasets.find((d) => d.images > 0) || datasets[0];
        state.selectedDataset = first ? first.name : null;
        state.selectedBounty = null;
      }
      if (state.selectedBounty) {
        const found = (data.bounties || []).find((b) => b.id === state.selectedBounty.id);
        state.selectedBounty = found || null;
      }
      // With a dataset selected but no bounty, track the first one still open so
      // the inner ring shows something useful on first load.
      if (!state.selectedBounty && state.selectedDataset) {
        const mine = (data.bounties || []).filter((b) => b.dataset === state.selectedDataset);
        // Prefer unclaimed work: a claimed bounty is already someone's.
        state.selectedBounty = mine.find((b) => b.status === 'open')
          || mine.find((b) => b.status !== 'done') || mine[0] || null;
      }
      render();
      if (!quiet && state.job) showJob(state.job);
      // Opening the page while GitHub is failing should start the self-heal
      // clock, not wait for the user to notice the chip.
      scheduleGhRetry();
    } catch (err) {
      toast('Could not load the board: ' + err.message, true);
    }
  }

  // ------------------------------------------------------------- render
  function datasetBounties(name) {
    return ((state.board && state.board.bounties) || []).filter((b) => b.dataset === name);
  }

  function selectedDatasetInfo() {
    return ((state.board && state.board.datasets) || [])
      .find((d) => d.name === state.selectedDataset) || null;
  }

  function render() {
    if (!state.board) return;
    renderDatasets();
    renderOverview();
    renderLeaderboard();
    renderSubmit();
    renderGh();
  }

  // --------------------------------------------------------------- submit
  // Only a *finished, claimed* bounty can be submitted. A partial bounty keeps
  // its labels local: a PR is paid for the whole bounty, so shipping half a
  // slice would either underpay the reviewer or land unfinished work. Unclaimed
  // bounties are excluded too - their frames can hit 100% because other people
  // reviewed them, and submitting that would open a PR nobody earned.
  function submitCandidates() {
    const me = String(state.ghUser || '').toLowerCase();
    const ghReady = !!state.ghUser && !state.ghError;
    // Submitted bounties are dropped rather than listed with a reason: their
    // labels are locked, so there is no action to offer and nothing to explain.
    const open = ((state.board && state.board.bounties) || [])
      .filter((b) => !b.pr_url && !b.submitted_at);
    return open.map((b) => {
      const owner = String(b.claimed_by || '').trim();
      const mine = owner && (owner === 'you' || !me || owner.toLowerCase() === me);
      const p = b.progress || { done: 0, total: 0, percent: 0 };
      const row = { bounty: b, ready: false, reason: '', owner, mine, p };
      if (b.status !== 'done') {
        row.reason = b.status === 'claimed'
          ? `${fmt(p.done)}/${fmt(p.total)} frames — still in progress`
          : 'not started';
        return row;
      }
      if (!owner) { row.reason = 'unclaimed — nobody to credit'; return row; }
      if (!mine) { row.reason = `claimed by ${owner}`; return row; }
      if (!b.github_issue) { row.reason = 'no GitHub issue yet — post it first'; return row; }
      if (!ghReady) {
        row.reason = state.ghError ? 'GitHub unavailable' : 'GitHub not connected';
        return row;
      }
      row.ready = true;
      return row;
    });
  }

  function renderSubmit() {
    const card = $('submitCard');
    if (!card) return;
    const rows = submitCandidates();
    card.hidden = rows.length === 0;
    if (card.hidden) return;

    const ready = rows.filter((r) => r.ready);
    const active = rows.filter((r) => r.bounty.status === 'claimed' && !r.bounty.pr_url);
    const points = ready.reduce((sum, r) => sum + (Number(r.bounty.points) || 0), 0);
    const btn = $('submitOpen');

    $('submitScope').textContent = ready.length || active.length ? 'all datasets' : '';
    // Kept clickable even with nothing ready: the dialog is where "what about
    // my unfinished bounties?" gets answered, so it must be reachable.
    btn.disabled = false;
    btn.textContent = ready.length
      ? (ready.length === 1 ? 'Submit completed bounty' : `Submit ${ready.length} completed`)
      : 'Nothing to submit';

    if (ready.length) {
      $('submitHint').textContent = `${ready.length} complete · ${fmt(points)} pts · one PR each`;
    } else if (active.length) {
      $('submitHint').textContent =
        `${active.length} in progress — finish one to submit.`;
    } else {
      $('submitHint').textContent = 'Finish a bounty to open its PR.';
    }
  }

  function submitRow(row) {
    const b = row.bounty;
    const p = row.p;
    const when = b.pr_url
      ? `<a class="quiet-link" href="${esc(b.pr_url)}" target="_blank" rel="noopener">PR ↗</a>`
      : `<span class="submit-reason mini">${esc(row.reason)}</span>`;
    const control = row.ready
      ? `<input type="checkbox" class="submit-pick" value="${esc(b.id)}" checked
                aria-label="Submit ${esc(b.title)}" />`
      : `<span class="status-dot ${b.status === 'done' ? 'done' : b.status === 'claimed' ? 'claimed' : 'open'}"
               aria-hidden="true"></span>`;
    return `<li class="submit-item${row.ready ? ' ready' : ''}" data-submit-row="${esc(b.id)}">
      ${control}
      <div class="submit-item-main">
        <span class="submit-item-title">${esc(b.title)}</span>
        <span class="mini muted">${esc(b.dataset)} · ${fmt(p.done)}/${fmt(p.total)} frames</span>
      </div>
      <span class="points-chip">${fmt(b.points)} pts</span>
      ${when}
    </li>`;
  }

  function renderSubmitList() {
    const rows = submitCandidates();
    // Ready work first: the dialog should open on what you can actually do.
    const ordered = rows.slice().sort((a, b) => (b.ready - a.ready)
      || (Number(b.bounty.points) || 0) - (Number(a.bounty.points) || 0));
    $('submitList').innerHTML = ordered.map(submitRow).join('');
    const picks = $('submitList').querySelectorAll('.submit-pick');
    $('submitConfirm').disabled = picks.length === 0;
    // "Open 0 PRs" reads like a bug; with nothing selected the count is
    // meaningless, so fall back to the plain label.
    $('submitConfirm').textContent = picks.length > 1
      ? `Open ${picks.length} PRs` : 'Open PR';
    $('submitError').textContent = picks.length
      ? '' : 'Nothing is ready to submit yet.';
  }

  function openSubmit() {
    renderSubmitList();
    $('submitOverlay').hidden = false;
    const first = $('submitList').querySelector('.submit-pick');
    if (first) first.focus();
  }

  function closeSubmit() {
    $('submitOverlay').hidden = true;
  }

  async function doSubmit(ids) {
    const btn = $('submitConfirm');
    const label = btn.textContent;
    btn.disabled = true;
    btn.textContent = ids.length === 1 ? 'Opening PR…' : 'Opening PRs…';
    $('submitError').textContent = '';
    try {
      const res = await api('/api/board/submit', {
        method: 'POST',
        body: JSON.stringify({ bounty_ids: ids }),
      });
      const results = (res && res.results) || [];
      const failed = results.filter((r) => !r.ok);
      const opened = results.filter((r) => r.ok && r.pr_url);

      if (failed.length) {
        // Keep the dialog open so the reason stays attached to its bounty.
        failed.forEach((r) => {
          const row = $('submitList').querySelector(`[data-submit-row="${r.id}"]`);
          if (row) {
            const slot = row.querySelector('.submit-reason');
            const pick = row.querySelector('.submit-pick');
            row.classList.add('failed');
            row.classList.remove('ready');
            if (pick) { pick.checked = false; pick.disabled = true; }
            if (slot) slot.textContent = r.reason || 'submit failed';
          }
        });
        $('submitError').textContent = failed.length === 1
          ? (failed[0].reason || 'submit failed')
          : `${failed.length} bounties could not be submitted.`;
      }
      if (opened.length) {
        toast(opened.length === 1
          ? `PR opened: ${opened[0].pr_url}`
          : `${opened.length} PRs opened.`);
      }
      await loadBoard({ quiet: true });
      if (!failed.length) closeSubmit();
    } catch (err) {
      $('submitError').textContent = 'Could not submit: ' + err.message;
    } finally {
      btn.textContent = label;
      const left = $('submitList').querySelectorAll('.submit-pick:checked').length;
      btn.disabled = left === 0 || $('submitOverlay').hidden;
      if (left && !$('submitOverlay').hidden) {
        btn.textContent = left === 1 ? 'Open PR' : `Open ${left} PRs`;
      }
    }
  }

  // ---------------------------------------------------------- leaderboard
  // A bounty is worth its frame count, so points are already the unit of work.
  // Earned = completed bounties; in progress = claimed but not finished. People
  // rank by total secured points, tie-broken by work already completed. The
  // 'you' placeholder only exists for claims made before a GitHub identity is
  // known, so it resolves to the signed-in login when possible.
  function renderLeaderboard() {
    const card = $('leaderboard');
    const list = $('leaderList');
    const empty = $('leaderEmpty');
    if (!card || !list) return;

    const me = state.ghUser;
    const rows = new Map();
    (state.board.bounties || []).forEach((b) => {
      const raw = String(b.claimed_by || '').trim();
      if (!raw) return;
      const who = (raw === 'you' && me) ? me : raw;
      const entry = rows.get(who) || {
        who, earned: 0, progress: 0, done: 0, active: 0, isMe: false,
      };
      const points = Number(b.points) || 0;
      if (b.status === 'done') {
        entry.earned += points;
        entry.done += 1;
      } else {
        entry.progress += points;
        entry.active += 1;
      }
      entry.isMe = entry.isMe || raw === 'you' || (me && who === me);
      rows.set(who, entry);
    });

    const people = [...rows.values()]
      .map((p) => ({ ...p, total: p.earned + p.progress }))
      .sort((a, b) => (
        b.total - a.total
        || b.earned - a.earned
        || a.who.localeCompare(b.who)
      ));

    card.hidden = false;
    empty.hidden = people.length > 0;
    if (!people.length) { list.innerHTML = ''; return; }

    list.innerHTML = people.slice(0, 6).map((p, i) => {
      const initials = p.who.slice(0, 2).toUpperCase();
      const title = p.who === 'you' ? 'You (local claim)' : p.who;
      const sub = p.done && p.active
        ? `${p.done} done · ${p.active} in review`
        : p.done
          ? `${p.done} done`
          : `${p.active} claimed`;
      return `
        <li class="leader-row${p.isMe ? ' me' : ''}" title="${esc(title)}">
          <span class="leader-rank">${i + 1}</span>
          <span class="leader-avatar" aria-hidden="true">${esc(initials)}</span>
          <span class="leader-who">
            <span class="leader-name">${esc(p.who === 'you' ? 'You' : p.who)}</span>
            <span class="leader-sub">${sub}</span>
          </span>
          <span class="leader-points">${fmt(p.total)} pts</span>
        </li>`;
    }).join('');
  }

  // `gh` reports failures as `Post "https://api.github.com/graphql": Forbidden`.
  // The URL is noise in a 200px chip; the last clause is the part that tells you
  // what actually went wrong.
  function ghReason(raw) {
    const text = String(raw || '').trim();
    const tail = text.match(/:\s*([^:]+)$/);
    return (tail ? tail[1] : text).trim() || 'unknown error';
  }

  // A 403 and a bad token need different fixes, so the chip names the kind
  // rather than repeating one generic "sync failed" for every failure.
  const GH_LABELS = {
    blocked: 'GitHub blocked',
    auth: 'GitHub auth needed',
    network: 'GitHub unreachable',
    unknown: 'GitHub sync failed',
  };
  const GH_LABELS_HINT = {
    blocked: 'Retry; if it persists, restart the server and check `gh auth status`.',
    auth: 'Run `gh auth login`, then retry.',
    network: 'Check the connection, then retry.',
    unknown: 'Retry, or run `gh issue list --label bounty` for the raw error.',
  };

  function renderGh() {
    const chip = $('ghChip');
    if (state.ghError) {
      chip.className = 'gh-chip warn retryable';
      const label = GH_LABELS[state.ghKind] || 'GitHub sync failed';
      chip.textContent = label + ' — ' + ghReason(state.ghError);
      chip.title = [
        state.ghError,
        state.ghHint || GH_LABELS_HINT[state.ghKind] || '',
        'Local work is unaffected. Click to retry.',
      ].filter(Boolean).join('\n\n');
      chip.setAttribute('role', 'button');
      chip.tabIndex = 0;
    } else if (state.ghUser) {
      chip.className = 'gh-chip ok';
      chip.textContent = 'GitHub @' + state.ghUser;
      chip.title = 'Claims can be posted to GitHub issue comments';
      chip.removeAttribute('role');
      chip.removeAttribute('tabindex');
    } else {
      chip.className = 'gh-chip';
      chip.textContent = 'GitHub not synced';
      chip.title = 'Run gh auth login, then Sync';
      chip.removeAttribute('role');
      chip.removeAttribute('tabindex');
    }
    const publishNote = $('publishNote');
    if (publishNote) {
      publishNote.textContent = (state.ghUser && !state.ghError)
        ? 'Issues will be opened for each bounty and the frame bundle published as a release.'
        : 'GitHub is not connected, so bounties will be created locally only. '
          + 'You can publish later with scripts/bounty.py.';
    }
  }

  function renderDatasets() {
    const list = $('datasetList');
    const datasets = state.board.datasets || [];
    const totalFrames = datasets.reduce((n, d) => n + d.images, 0);
    const totalReviewed = datasets.reduce((n, d) => n + d.reviewed, 0);
    $('datasetsSummary').textContent = datasets.length
      ? `${datasets.length} dataset${datasets.length === 1 ? '' : 's'} · `
        + `${fmt(totalReviewed)} of ${fmt(totalFrames)} frames reviewed`
      : 'No datasets yet';

    $('emptyState').hidden = datasets.length > 0;
    list.hidden = datasets.length === 0;
    if (!datasets.length) { animateBars(); return; }

    list.innerHTML = datasets.map((ds, i) => {
      const bounties = datasetBounties(ds.name);
      const open = bounties.filter((b) => b.status === 'open').length;
      const claimed = bounties.filter((b) => b.status === 'claimed').length;
      const selected = ds.name === state.selectedDataset;
      const noImages = ds.images === 0;
      const expand = selected || bounties.length <= 3;

      const rows = bounties.map((b, bi) => bountyRow(b, bi, expand)).join('');
      const body = bounties.length
        ? `<details class="bounty-menu" ${expand ? 'open' : ''} data-menu="${esc(ds.name)}">
             <summary>
               <svg class="caret" viewBox="0 0 24 24" aria-hidden="true"><path d="M9 5l7 7-7 7"
                 fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
                 stroke-linejoin="round"/></svg>
               <span>Bounties</span>
               <span class="count-pill">${bounties.length}</span>
               ${open ? `<span class="mini muted">${open} open</span>` : ''}
               ${claimed ? `<span class="mini muted">${claimed} claimed</span>` : ''}
             </summary>
             <div class="bounty-list">${rows}</div>
           </details>`
        : noImages
          ? `<div class="empty-note-card">No frames on disk yet. Fetch the published bundle:
               <code>python scripts/bounty.py fetch --dataset data/datasets/${esc(ds.name)}</code></div>`
          : `<div class="empty-note-card">No bounties yet.
               <button class="ghost btn-sm" data-seed="${esc(ds.name)}">Create bounties</button></div>`;

      return `<article class="dataset-card${selected ? ' selected' : ''}" data-dataset="${esc(ds.name)}"
                       style="--delay:${Math.min(i * 55, 330)}ms">
        <div class="ds-head">
          <div class="ds-title">
            <span class="ds-name" title="${esc(ds.name)}">${esc(ds.name)}</span>
            <span class="ds-meta">${fmt(ds.images)} frames
              · ${fmt(ds.reviewed)} reviewed
              ${ds.has_predictions ? ' · model predictions' : ''}</span>
          </div>
          <span class="kind-chip ${esc(ds.kind)}">${esc(ds.kind)}</span>
          <div class="ds-actions">
            <span class="ds-pct">${ds.percent.toFixed(1)}%</span>
            <!-- Top of the card, always available, never writes. -->
            <button class="view-btn" data-open="${esc(ds.name)}"
                    title="Browse every frame and the source video"
                    ${noImages ? 'disabled title="No frames on disk"' : ''}>
              <svg viewBox="0 0 16 16" width="13" height="13" fill="none"
                   stroke="currentColor" stroke-width="1.6" stroke-linecap="round"
                   stroke-linejoin="round" aria-hidden="true">
                <path d="M1.5 8S3.9 3.8 8 3.8 14.5 8 14.5 8 12.1 12.2 8 12.2 1.5 8 1.5 8Z"/>
                <circle cx="8" cy="8" r="2.1"/>
              </svg>
              View
            </button>
          </div>
        </div>
        <div class="ds-progress">
          <div class="bar"><div data-target="${pct(ds.percent)}"
               style="width:${reduceMotion ? pct(ds.percent) : 0}%"></div></div>
          <div class="ds-progress-row">
            <span>${fmt(ds.reviewed)} of ${fmt(ds.images)} reviewed</span>
            <span>${fmt(ds.unreviewed)} left</span>
          </div>
        </div>
        ${body}
        <div class="ds-foot">
          <span class="ds-hint">${noImages ? 'Download frames before reviewing'
            : ds.unreviewed ? `${fmt(ds.unreviewed)} frames waiting` : 'Fully reviewed'}</span>
          <span class="spacer"></span>
          <button class="ghost btn-sm" data-select="${esc(ds.name)}">${selected ? 'Tracking' : 'Track'}</button>
        </div>
      </article>`;
    }).join('');

    animateBars();
    // A one-off scan-in for the rings whenever the catalogue changes.
    requestAnimationFrame(() => {
      const ds = selectedDatasetInfo();
      if (ds) setRing($('ringDataset'), ds.percent);
    });
  }

  function bountyRow(b, index, expanded) {
    const p = b.progress || { done: 0, total: 0, percent: 0 };
    const isSelected = state.selectedBounty && state.selectedBounty.id === b.id;
    const statusClass = b.status === 'done' ? 'done' : b.status === 'claimed' ? 'claimed' : 'open';
    // A complete bounty keeps its entry point so a labeler can double-check the
    // work - until it is submitted. After that the labels are the content of a
    // PR, so they are frozen and there is nothing left to edit.
    const done = b.status === 'done';
    const submitted = !!(b.pr_url || b.submitted_at);
    const barClass = b.status === 'done' ? 'green' : b.status === 'claimed' ? 'gold' : '';
    const claimLabel = b.status === 'done' ? 'Complete'
      : b.status === 'claimed' ? (b.claimed_by ? `Claimed · ${b.claimed_by}` : 'Claimed')
        : 'Claim';
    const pending = b.claim_pending_github
      ? '<span class="bounty-pending" title="Claimed here, not yet posted to GitHub">unposted</span>'
      : '';
    // A submitted bounty has already shipped its labels, so the PR (not the
    // issue) is the thing worth opening from its row.
    const github = (submitted && b.pr_url)
      ? `<a class="quiet-link btn-sm" href="${esc(b.pr_url)}" target="_blank"
            rel="noopener" title="Open the pull request for these labels">PR ↗</a>`
      : b.github_url
        ? `<a class="quiet-link btn-sm" href="${esc(b.github_url)}" target="_blank"
              rel="noopener" title="Open the GitHub issue">Issue ↗</a>`
        : `<button class="ghost btn-sm" data-github="${esc(b.id)}" title="Create a GitHub issue">Post ↗</button>`;

    // One state chip and one entry point, so the row always reads the same way:
    // where the work stands, then the only thing you can do about it.
    const statusChip = submitted
      ? `<span class="done-chip submitted" title="Submitted - its labels are locked until the PR is merged">Submitted</span>`
      : done
        ? `<span class="done-chip" title="Every frame in this bounty is reviewed">Complete</span>`
        : `<button class="btn-sm claim-btn ${b.status === 'open' ? 'primary' : statusClass}" data-claim="${esc(b.id)}">${esc(claimLabel)}</button>`;
    const entryAction = submitted
      ? ''
      : done
        ? `<button class="ghost btn-sm edit-btn" data-label="${esc(b.id)}" data-edit="1"
               title="Reopen this bounty to double-check and edit its labels">Edit</button>`
        : `<button class="ghost btn-sm" data-label="${esc(b.id)}" data-edit="0"
               title="Label this bounty's frames">Label</button>`;

    return `<div class="bounty${isSelected ? ' selected' : ''}" data-bounty="${esc(b.id)}"
                 data-dataset="${esc(b.dataset)}"
                 style="--delay:${expanded ? Math.min(index * 32, 220) : 0}ms"
                 tabindex="0" role="button"
                 aria-label="Track bounty ${esc(b.title)}">
      <span class="status-dot ${statusClass}" title="${esc(b.status)}" aria-hidden="true"></span>
      <div class="bounty-main">
        <span class="bounty-title" title="${esc(b.title)}">${esc(b.title)}</span>
        <span class="bounty-sub">
          ${b.points ? `<span class="points-chip">${fmt(b.points)} pts</span>` : ''}
          <span>${p.estimated ? '≈' : ''}${fmt(p.done)}/${fmt(p.total)} frames</span>
          ${pending}
        </span>
      </div>
      <div class="bounty-progress">
        <div class="bar ${barClass}"><div data-target="${pct(p.percent)}"
             style="width:${reduceMotion ? pct(p.percent) : 0}%"></div></div>
        <span class="mini muted">${p.percent.toFixed(0)}%${p.estimated ? ' est.' : ''}</span>
      </div>
      <div class="bounty-actions">
        ${statusChip}
        ${entryAction}
        ${github}
      </div>
    </div>`;
  }

  function renderOverview() {
    const ds = selectedDatasetInfo();
    const bounty = state.selectedBounty;
    const datasets = (state.board && state.board.datasets) || [];

    if (!ds) {
      $('overviewTitle').textContent = 'Overview';
      $('overviewSub').textContent = 'Pick a dataset to see its progress';
      $('overviewBadge').hidden = true;
      $('ringPercent').textContent = '0%';
      setRing($('ringDataset'), 0);
      setRing($('ringBounty'), 0);
      ['statFrames', 'statReviewed', 'statLeft', 'statBounties'].forEach((id) => {
        $(id).textContent = '—';
        $(id).dataset.value = 0;
      });
      return;
    }

    $('overviewTitle').textContent = ds.name;
    $('overviewSub').textContent = bounty
      ? bounty.title
      : `${fmt(ds.images)} frames · ${fmt(ds.reviewed)} reviewed`;
    const badge = $('overviewBadge');
    badge.hidden = false;
    badge.className = 'kind-chip ' + ds.kind;
    badge.textContent = ds.kind;

    const bounties = datasetBounties(ds.name);
    countTo($('statFrames'), ds.images);
    countTo($('statReviewed'), ds.reviewed);
    countTo($('statLeft'), ds.unreviewed);
    countTo($('statBounties'), bounties.length);

    const datasetPct = ds.percent;
    const bountyPct = bounty ? (bounty.progress.percent || 0) : 0;
    setRing($('ringDataset'), datasetPct);
    setRing($('ringBounty'), bountyPct);

    // The big number follows an explicitly picked bounty; on first load it
    // stays on the dataset total so the screen does not greet you with 0%.
    const pinned = bounty && state.bountyPinned;
    const center = pinned ? bountyPct : datasetPct;
    $('ringPercent').textContent = `${center.toFixed(0)}%`;
    $('ringCaption').textContent = pinned ? 'bounty' : 'dataset';

    $('legendDataset').textContent = `${datasetPct.toFixed(1)}%`;
    // The row is a third of the sidebar wide, so lead with what differs between
    // bounties (part number, size) instead of repeating "Bounty: <dataset> — ",
    // which is already in the pane title. Full title stays on hover.
    const bountyLabel = bounty
      ? bounty.title.replace(/^Bounty:\s*/, '').replace(`${ds.name} — `, '')
      : 'Bounty (none selected)';
    $('legendBountyLabel').textContent = bountyLabel;
    $('legendBountyLabel').title = bounty ? bounty.title : 'No bounty selected';
    $('legendBounty').textContent = bounty ? `${bountyPct.toFixed(1)}%` : '—';

    const foot = $('ringsFoot');
    if (bounty) {
      const p = bounty.progress;
      foot.textContent = `${fmt(p.remaining)} frame${p.remaining === 1 ? '' : 's'} left in this bounty`
        + (p.estimated ? ' · estimate from reviewed frames' : '')
        + (bounty.claimed_by ? ` · claimed by ${bounty.claimed_by}` : '');
    } else if (ds.unreviewed) {
      foot.textContent = `${fmt(ds.unreviewed)} frames still unreviewed across ${bounties.length || 0} bounties.`;
    } else {
      foot.textContent = 'Every frame in this dataset has been reviewed.';
    }
  }

  // -------------------------------------------------------------- select
  function selectDataset(name) {
    state.selectedDataset = name;
    const bounties = datasetBounties(name);
    state.selectedBounty = bounties.find((b) => b.status === 'open')
      || bounties.find((b) => b.status !== 'done') || bounties[0] || null;
    state.bountyPinned = false;
    render();
  }

  function selectBounty(bountyId) {
    const bounty = (state.board.bounties || []).find((b) => b.id === bountyId);
    if (!bounty) return;
    state.selectedBounty = bounty;
    state.selectedDataset = bounty.dataset;
    state.bountyPinned = true;
    render();
  }

  // --------------------------------------------------------------- claim
  async function claimBounty(bountyId) {
    const bounty = (state.board.bounties || []).find((b) => b.id === bountyId);
    if (!bounty) return;
    // Publish whenever GitHub is connected. Requiring an existing issue number
    // meant a freshly seeded bounty - which has none - could only ever be
    // claimed locally, so the claim never appeared on GitHub.
    const post = !!state.ghUser && !state.ghError;

    if (bounty.status === 'done') {
      toast('That bounty is already complete.');
      return;
    }
    try {
      if (bounty.status === 'claimed' && bounty.claimed_by === state.ghUser) {
        await api(`/api/board/bounty/${encodeURIComponent(bountyId)}/unclaim`, { method: 'POST' });
        toast('Released ' + bounty.title);
      } else {
        const res = await api(`/api/board/bounty/${encodeURIComponent(bountyId)}/claim`, {
          method: 'POST',
          body: JSON.stringify({ by: state.ghUser || '', post_to_github: post }),
        });
        const gh = res.github || {};
        if (gh.posted) {
          toast(gh.linked ? `Claimed and posted to GitHub issue #${gh.issue}.` : 'Claimed and posted to GitHub.');
        } else if (gh.linked) {
          // The issue exists, but commenting failed - the claim is queued and
          // the next successful sync retries it.
          toast('Claimed locally. GitHub post failed (' + (gh.reason || 'unknown')
            + '); it will retry on the next sync.', true);
        } else if (!post) {
          toast('Claimed locally — GitHub is not connected, so it stays local.', true);
        } else {
          toast('Claimed locally. GitHub: ' + (gh.reason || 'could not post'), true);
        }
      }
      await loadBoard({ quiet: true });
      selectBounty(bountyId);
    } catch (err) {
      toast('Could not claim: ' + err.message, true);
    }
  }

  async function postIssue(bountyId) {
    toast('Creating the GitHub issue…');
    try {
      const res = await api('/api/board/sync', { method: 'POST' });
      await loadBoard({ quiet: true });
      toast(res.ok ? 'Synced with GitHub.' : 'GitHub: ' + res.error, !res.ok);
    } catch (err) {
      toast('Sync failed: ' + err.message, true);
    }
  }

  // One sync path for the toolbar button and the (failed) connection chip, so
  // retrying from either place behaves identically.
  async function syncNow(btn, { quiet = false } = {}) {
    if (btn) {
      btn.disabled = true;
      btn.textContent = 'Syncing…';
    }
    try {
      const res = await api('/api/board/sync', { method: 'POST' });
      await loadBoard({ quiet: true });
      if (res.ok) state.ghRetries = 0;
      if (!quiet) {
        toast(res.ok
          ? `Synced with GitHub (${res.imported} new, ${res.updated} updated).`
          : 'GitHub unavailable: ' + res.error, !res.ok);
      }
    } catch (err) {
      if (!quiet) toast('Sync failed: ' + err.message, true);
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = 'Sync';
      }
    }
    scheduleGhRetry();
  }

  // A 403 from a proxy or abuse filter is usually temporary, and asking the
  // user to hit retry on a timer is a worse product than retrying quietly. Cap
  // it so a genuinely broken setup does not poll GitHub forever.
  const GH_RETRY_LIMIT = 3;
  const GH_RETRY_DELAY = 45000;

  function scheduleGhRetry() {
    clearTimeout(state.ghRetryTimer);
    if (!state.ghError || state.ghRetries >= GH_RETRY_LIMIT) return;
    state.ghRetryTimer = setTimeout(() => {
      // A background tab retrying is wasted traffic; check again on focus.
      if (document.hidden) return scheduleGhRetry();
      state.ghRetries += 1;
      syncNow($('syncBtn'), { quiet: true });
    }, GH_RETRY_DELAY);
  }

  async function seedBounties(name) {
    try {
      const res = await api('/api/board/seed', {
        method: 'POST',
        body: JSON.stringify({ dataset: name }),
      });
      toast(res.created ? `Created ${res.created} bount${res.created === 1 ? 'y' : 'ies'}.`
        : 'Nothing left to turn into bounties.');
      await loadBoard({ quiet: true });
    } catch (err) {
      toast('Could not create bounties: ' + err.message, true);
    }
  }

  // ------------------------------------------------------------- routing
  /** Read-only browse: every frame, the labels, and the source video. */
  function openView(name) {
    location.href = '/view?dataset=' + encodeURIComponent(name);
  }

  /**
   * Label mode is always tied to one bounty. Opening it from a bounty row
   * rather than a dataset card is what keeps the queue to that bounty's fixed
   * slice, so a reviewer can never wander into frames nobody is paying for.
   */
  function openBountyLabels(bountyId, edit) {
    const bounty = (state.board.bounties || []).find((b) => b.id === bountyId);
    if (!bounty) return;
    const params = new URLSearchParams({
      dataset: bounty.dataset,
      bounty: bounty.title,
      bounty_id: bounty.id,
      of: String((bounty.progress && bounty.progress.total) || bounty.target || 0),
    });
    // Editing a finished bounty reopens the same fixed slice. The flag lets the
    // review screen say "editing" rather than "labelling" and warn when a PR
    // already exists, because a second submit updates that PR rather than
    // opening a new one.
    if (edit) {
      params.set('edit', '1');
      if (bounty.pr_url) params.set('pr', '1');
    }
    location.href = '/review?' + params.toString();
  }

  // --------------------------------------------------------- add dataset
  function openAdd() {
    $('addOverlay').hidden = false;
    $('addForm').hidden = false;
    $('addProgress').hidden = true;
    $('addStart').hidden = false;
    $('addStart').disabled = !state.file;
    $('addCancel').textContent = 'Close';
    $('addError').textContent = '';
    setTimeout(() => $('videoFile').focus(), 30);
  }

  function closeAdd() {
    $('addOverlay').hidden = true;
    if (state.uploadXhr) { try { state.uploadXhr.abort(); } catch (_) { /* ignore */ } }
    stopPolling();
  }

  function slugify(text) {
    return String(text).replace(/[^A-Za-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 64);
  }

  function pickFile(file) {
    if (!file) return;
    state.file = file;
    $('dropText').textContent = file.name;
    $('dropMeta').textContent = `${(file.size / (1024 * 1024)).toFixed(1)} MB`;
    const base = slugify(file.name.replace(/\.[^.]+$/, ''));
    if (!$('datasetName').value.trim() || $('datasetName').dataset.auto !== '0') {
      $('datasetName').value = base;
      $('datasetName').dataset.auto = '1';
    }
    $('addStart').disabled = false;
  }

  function startIngest() {
    if (!state.file) return;
    const name = slugify($('datasetName').value || state.file.name.replace(/\.[^.]+$/, ''));
    if (!name) { $('addError').textContent = 'Give the dataset a name.'; return; }
    const fps = $('fpsSelect').value;
    const publish = $('publishCheck').checked ? '1' : '0';

    $('addError').textContent = '';
    $('addForm').hidden = true;
    $('addProgress').hidden = false;
    $('addStart').hidden = true;
    $('addCancel').textContent = 'Close';
    $('uploadLabel').textContent = 'Uploading';
    $('uploadPercent').textContent = '0%';
    $('uploadFill').style.width = '0%';
    $('jobSteps').innerHTML = '';

    const xhr = new XMLHttpRequest();
    state.uploadXhr = xhr;
    xhr.open('POST', `/api/board/upload?name=${encodeURIComponent(name)}&fps=${encodeURIComponent(fps)}&publish=${publish}`);
    xhr.setRequestHeader('Content-Type', 'video/mp4');
    xhr.setRequestHeader('X-Filename', state.file.name);
    xhr.upload.onprogress = (ev) => {
      if (!ev.lengthComputable) return;
      const p = (ev.loaded / ev.total) * 100;
      $('uploadFill').style.width = p.toFixed(1) + '%';
      $('uploadPercent').textContent = Math.round(p) + '%';
    };
    xhr.onload = () => {
      state.uploadXhr = null;
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch (_) { data = null; }
      if (xhr.status >= 400 || !data || !data.job) {
        $('addError').textContent = (data && data.error) || `Upload failed (${xhr.status})`;
        $('addForm').hidden = false;
        $('addStart').hidden = false;
        $('uploadLabel').textContent = 'Upload failed';
        return;
      }
      $('uploadFill').style.width = '100%';
      $('uploadPercent').textContent = '100%';
      $('uploadLabel').textContent = 'Uploaded — ingesting';
      state.job = data.job;
      showJob(data.job);
      pollJob(data.job.id);
    };
    xhr.onerror = () => {
      state.uploadXhr = null;
      $('addError').textContent = 'Upload failed — is the server still running?';
      $('addForm').hidden = false;
      $('addStart').hidden = false;
    };
    xhr.send(state.file);
  }

  function renderSteps(job) {
    const wanted = [
      { key: 'upload', label: 'Upload video' },
      { key: 'ingest', label: 'Sample frames and run the model' },
      { key: 'index', label: 'Index labels' },
      { key: 'bounties', label: 'Create bounties' },
      { key: 'publish', label: 'Publish to GitHub' },
    ];
    const msg = (job.message || '').toLowerCase();
    let stage = 'ingest';
    if (job.status === 'done') stage = 'publish';
    else if (msg.includes('bount')) stage = 'bounties';
    else if (msg.includes('indexing')) stage = 'index';
    else if (msg.includes('publishing') || msg.includes('github')) stage = 'publish';
    const order = wanted.map((w) => w.key);
    const activeIdx = order.indexOf(stage);
    const failed = job.status === 'error';

    return wanted.map((step, i) => {
      let cls = '';
      if (job.status === 'done') cls = 'done';
      else if (i < activeIdx) cls = 'done';
      else if (i === activeIdx) cls = failed ? 'failed' : 'active';
      if (failed && i > activeIdx) cls = '';
      const show = !(step.key === 'publish' && !job.result.release && i > activeIdx && !job.warnings.length);
      return show ? `<div class="job-step ${cls}">${esc(step.label)}</div>` : '';
    }).join('');
  }

  function showJob(job) {
    state.job = job;
    const card = $('jobCard');
    card.hidden = false;
    $('jobTitle').textContent = job.status === 'error' ? 'Ingest failed'
      : job.status === 'done' ? 'Dataset ready' : 'Ingesting ' + (job.label || '');
    $('jobPercent').textContent = Math.round(job.percent || 0) + '%';
    $('jobFill').style.width = `${pct(job.percent)}%`;
    $('jobMessage').textContent = job.error || job.message || '';
    $('jobLog').textContent = (job.log || []).join('\n');

    const steps = $('jobSteps');
    if (steps) steps.innerHTML = renderSteps(job);

    const upload = $('addProgress');
    if (upload && !upload.hidden) {
      $('uploadLabel').textContent = job.status === 'error' ? 'Failed'
        : job.status === 'done' ? 'Done' : 'Ingesting';
      $('uploadPercent').textContent = Math.round(job.percent || 0) + '%';
    }
  }

  function stopPolling() {
    if (state.jobTimer) { clearTimeout(state.jobTimer); state.jobTimer = null; }
  }

  function pollJob(jobId) {
    stopPolling();
    const tick = async () => {
      try {
        const job = await api('/api/jobs/' + encodeURIComponent(jobId));
        showJob(job);
        if (job.status === 'done') {
          stopPolling();
          await loadBoard({ quiet: true });
          const name = job.result.dataset || job.label;
          // Only jump to the new dataset if it is really there: a job replayed
          // from a previous session may refer to a folder since deleted.
          if (name && (state.board.datasets || []).some((d) => d.name === name)) {
            state.selectedDataset = name;
            state.selectedBounty = datasetBounties(name).find((b) => b.status !== 'done') || null;
          }
          render();
          const warn = job.warnings && job.warnings.length;
          toast(warn ? `Dataset ready, with ${job.warnings.length} warning(s).`
            : 'Dataset ready — bounties created.', !!warn);
          setTimeout(closeAdd, 1600);
          return;
        }
        if (job.status === 'error' || job.status === 'cancelled') {
          stopPolling();
          $('addError').textContent = job.error || 'Ingest stopped.';
          $('addCancel').textContent = 'Close';
          return;
        }
      } catch (err) {
        stopPolling();
        $('addError').textContent = 'Lost track of the job: ' + err.message;
        return;
      }
      state.jobTimer = setTimeout(tick, 1200);
    };
    state.jobTimer = setTimeout(tick, 600);
  }

  // ---------------------------------------------------------------- bind
  function bind() {
    $('refreshBtn').addEventListener('click', () => { loadBoard({ quiet: true }); toast('Rescanned datasets.'); });
    $('syncBtn').addEventListener('click', () => syncNow($('syncBtn')));
    // The chip turns into a retry target only while a sync is failing, which is
    // exactly when you want to poke it again after fixing your network.
    const retryChip = () => {
      if (state.ghError) syncNow($('syncBtn'));
    };
    $('ghChip').addEventListener('click', retryChip);
    $('ghChip').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        retryChip();
      }
    });

    // Warning banner: dismiss once, remembered locally.
    let dismissed = false;
    try { dismissed = localStorage.getItem('ssbu.claimWarnDismissed') === '1'; } catch (_) { /* private */ }
    if (dismissed) $('claimWarn').hidden = true;
    $('warnDismiss').addEventListener('click', () => {
      $('claimWarn').hidden = true;
      try { localStorage.setItem('ssbu.claimWarnDismissed', '1'); } catch (_) { /* private */ }
    });
    $('warnBoardLink').href = REPO + '/issues?q=label%3Abounty';
    $('repoLink').href = REPO;

    // Dataset + bounty interactions (event delegation: rows are re-rendered).
    $('datasetList').addEventListener('click', (ev) => {
      const open = ev.target.closest('[data-open]');
      if (open) { openView(open.dataset.open); return; }

      const label = ev.target.closest('[data-label]');
      if (label) { openBountyLabels(label.dataset.label, label.dataset.edit === '1'); return; }

      const select = ev.target.closest('[data-select]');
      if (select) { selectDataset(select.dataset.select); return; }

      const seed = ev.target.closest('[data-seed]');
      if (seed) { seedBounties(seed.dataset.seed); return; }

      const claim = ev.target.closest('[data-claim]');
      if (claim) { claimBounty(claim.dataset.claim); return; }

      const gh = ev.target.closest('[data-github]');
      if (gh) { postIssue(gh.dataset.github); return; }

      const row = ev.target.closest('.bounty');
      if (row) { selectBounty(row.dataset.bounty); return; }

      const card = ev.target.closest('.dataset-card');
      if (card && !ev.target.closest('a, button, summary')) {
        selectDataset(card.dataset.dataset);
      }
    });

    $('datasetList').addEventListener('keydown', (ev) => {
      if (ev.key !== 'Enter' && ev.key !== ' ') return;
      const row = ev.target.closest('.bounty');
      if (row) { ev.preventDefault(); selectBounty(row.dataset.bounty); }
    });

    // Add dataset dialog.
    $('addDatasetBtn').addEventListener('click', openAdd);
    $('emptyAddBtn').addEventListener('click', openAdd);
    $('addClose').addEventListener('click', closeAdd);
    $('addCancel').addEventListener('click', closeAdd);
    $('addOverlay').addEventListener('click', (ev) => {
      if (ev.target === $('addOverlay')) closeAdd();
    });
    document.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape' && !$('addOverlay').hidden) closeAdd();
    });

    // Submit dialog.
    $('submitOpen').addEventListener('click', openSubmit);
    $('submitClose').addEventListener('click', closeSubmit);
    $('submitCancel').addEventListener('click', closeSubmit);
    $('submitOverlay').addEventListener('click', (ev) => {
      if (ev.target === $('submitOverlay')) closeSubmit();
    });
    $('submitList').addEventListener('change', () => {
      const picks = $('submitList').querySelectorAll('.submit-pick:checked');
      const btn = $('submitConfirm');
      btn.disabled = picks.length === 0;
      btn.textContent = picks.length > 1 ? `Open ${picks.length} PRs` : 'Open PR';
    });
    $('submitConfirm').addEventListener('click', () => {
      const ids = [...$('submitList').querySelectorAll('.submit-pick:checked')]
        .map((el) => el.value);
      if (ids.length) doSubmit(ids);
    });
    document.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape' && !$('submitOverlay').hidden) closeSubmit();
    });

    $('dropzone').addEventListener('click', () => $('videoFile').click());
    $('videoFile').addEventListener('change', (ev) => pickFile(ev.target.files[0]));
    ['dragenter', 'dragover'].forEach((name) => {
      $('dropzone').addEventListener(name, (ev) => {
        ev.preventDefault();
        $('dropzone').classList.add('drag');
      });
    });
    ['dragleave', 'drop'].forEach((name) => {
      $('dropzone').addEventListener(name, (ev) => {
        ev.preventDefault();
        $('dropzone').classList.remove('drag');
      });
    });
    $('dropzone').addEventListener('drop', (ev) => {
      const file = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0];
      if (file) pickFile(file);
    });
    $('datasetName').addEventListener('input', (ev) => {
      ev.target.dataset.auto = '0';
    });
    $('addStart').addEventListener('click', startIngest);
  }

  // ---------------------------------------------------------------- boot
  async function boot() {
    bind();
    await loadBoard();
    // A job left running by a previous page load is still tracked server-side.
    const active = state.board && state.board.active_job;
    if (active && (active.status === 'queued' || active.status === 'running')) {
      showJob(active);
      pollJob(active.id);
    }
  }

  boot();
})();
