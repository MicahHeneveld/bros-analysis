/* Theme bootstrap.
   Loaded synchronously in <head> so the resolved palette lands on <html>
   before the first paint — otherwise a light-mode user gets a black flash.
   A stored choice wins; until the user makes one we follow the OS. */
(function () {
  var KEY = 'ssbu-theme';
  var root = document.documentElement;

  function stored() {
    try {
      var v = localStorage.getItem(KEY);
      return v === 'light' || v === 'dark' ? v : null;
    } catch (e) {
      return null;
    }
  }

  function system() {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches
      ? 'light'
      : 'dark';
  }

  function resolve() {
    return stored() || system();
  }

  function apply(theme) {
    root.dataset.theme = theme;
    var btn = document.getElementById('themeToggle');
    if (!btn) return;
    var dark = theme === 'dark';
    // The accessible name stays "Dark theme" (from .sr-only); aria-pressed
    // carries the state, so screen readers never hear a contradictory label.
    btn.setAttribute('aria-pressed', String(dark));
    btn.title = dark ? 'Switch to light theme' : 'Switch to dark theme';
  }

  apply(resolve());

  function wire() {
    apply(resolve());
    var btn = document.getElementById('themeToggle');
    if (!btn) return;
    btn.addEventListener('click', function () {
      var next = root.dataset.theme === 'dark' ? 'light' : 'dark';
      try { localStorage.setItem(KEY, next); } catch (e) {}
      apply(next);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wire);
  } else {
    wire();
  }

  // Follow the OS until an explicit choice is stored.
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: light)');
    var onChange = function () { if (!stored()) apply(system()); };
    if (mq.addEventListener) mq.addEventListener('change', onChange);
    else if (mq.addListener) mq.addListener(onChange);
  }
})();
