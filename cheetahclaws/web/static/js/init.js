/* Bootstrap: instantiate ChatApp once all mixin modules have loaded,
 * then wire keyboard + sidebar click-outside handlers. */

const app = new ChatApp();
app.initTheme();
app.bootstrap();
app._showWelcome();

const promptInput = document.getElementById('prompt-input');
promptInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    app.send();
  }
});
promptInput.addEventListener('input', () => {
  promptInput.style.height = 'auto';
  promptInput.style.height = Math.min(promptInput.scrollHeight, 200) + 'px';
});

document.getElementById('main').addEventListener('click', () => {
  document.getElementById('sidebar').classList.remove('open');
});

/* ── Sidebar resizer ───────────────────────────────────────────── */
(function initSidebarResizer() {
  const sidebar = document.getElementById('sidebar');
  const resizer = document.getElementById('sidebar-resizer');
  if (!sidebar || !resizer) return;
  const MIN = 200, MAX = 600;
  // Restore saved width
  const saved = parseInt(localStorage.getItem('cc-sidebar-w') || '0', 10);
  if (saved >= MIN && saved <= MAX) {
    sidebar.style.width = saved + 'px';
    sidebar.style.minWidth = saved + 'px';
  }
  let startX = 0, startW = 0, dragging = false;
  const onMove = (e) => {
    if (!dragging) return;
    const x = e.touches ? e.touches[0].clientX : e.clientX;
    const w = Math.min(MAX, Math.max(MIN, startW + (x - startX)));
    sidebar.style.width = w + 'px';
    sidebar.style.minWidth = w + 'px';
  };
  const onUp = () => {
    if (!dragging) return;
    dragging = false;
    resizer.classList.remove('dragging');
    document.body.classList.remove('resizing');
    const w = Math.round(sidebar.getBoundingClientRect().width);
    localStorage.setItem('cc-sidebar-w', String(w));
  };
  const onDown = (e) => {
    dragging = true;
    startX = e.touches ? e.touches[0].clientX : e.clientX;
    startW = sidebar.getBoundingClientRect().width;
    resizer.classList.add('dragging');
    document.body.classList.add('resizing');
    e.preventDefault();
  };
  resizer.addEventListener('mousedown', onDown);
  resizer.addEventListener('touchstart', onDown, {passive: false});
  document.addEventListener('mousemove', onMove);
  document.addEventListener('touchmove', onMove, {passive: false});
  document.addEventListener('mouseup', onUp);
  document.addEventListener('touchend', onUp);
  // Double-click resets to default
  resizer.addEventListener('dblclick', () => {
    sidebar.style.width = '';
    sidebar.style.minWidth = '';
    localStorage.removeItem('cc-sidebar-w');
  });
})();
