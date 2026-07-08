/* CheetahClaws PWA service worker.
 *
 * Goals:
 *   - Make the web UI installable + launchable offline (the app *shell*).
 *   - NEVER touch dynamic/auth traffic: /api/*, SSE, and WebSockets always
 *     go straight to the network so streaming and login keep working.
 *
 * Strategy:
 *   - navigation (loading the page): network-first, fall back to the cached
 *     /chat shell when offline.
 *   - static assets (js/css/img/fonts/manifest): stale-while-revalidate —
 *     serve from cache instantly, refresh in the background.
 *
 * Bump CACHE_VERSION on any shell change to roll clients onto fresh assets.
 */
const CACHE_VERSION = "cheetahclaws-shell-v1";

// Core app shell — best-effort precache (a single 404 must not fail install).
const SHELL_ASSETS = [
  "/chat",
  "/manifest.webmanifest",
  "/marked.min.js",
  "/static/js/csrf.js",
  "/static/js/chat.js",
  "/static/js/util.js",
  "/static/js/auth.js",
  "/static/js/sidebar.js",
  "/static/js/tools.js",
  "/static/js/approval.js",
  "/static/js/settings.js",
  "/static/js/welcome.js",
  "/static/js/init.js",
  "/static/favicon.png",
  "/static/icon-192.png",
  "/static/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION).then((cache) =>
      // Add each asset independently so one missing file doesn't abort install.
      Promise.allSettled(SHELL_ASSETS.map((url) => cache.add(url)))
    ).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k))
      ))
      .then(() => self.clients.claim())
  );
});

function isBypassed(url) {
  // Dynamic/auth/streaming endpoints must never be served from cache.
  return url.pathname.startsWith("/api/") ||
         url.pathname === "/api" ||
         url.pathname.startsWith("/health") ||
         url.pathname.startsWith("/metrics");
}

self.addEventListener("fetch", (event) => {
  const req = event.request;

  // Only GET is cacheable; everything else (POST/PUT/…) and cross-origin
  // requests pass through untouched. WebSocket upgrades never hit fetch.
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (isBypassed(url)) return;

  // Navigations: network-first, offline fallback to the cached shell.
  if (req.mode === "navigate") {
    event.respondWith(
      fetch(req)
        .then((res) => {
          const copy = res.clone();
          caches.open(CACHE_VERSION).then((c) => c.put("/chat", copy)).catch(() => {});
          return res;
        })
        .catch(() => caches.match(req).then((hit) => hit || caches.match("/chat")))
    );
    return;
  }

  // Static assets: stale-while-revalidate.
  event.respondWith(
    caches.match(req).then((cached) => {
      const network = fetch(req)
        .then((res) => {
          if (res && res.status === 200 && res.type === "basic") {
            const copy = res.clone();
            caches.open(CACHE_VERSION).then((c) => c.put(req, copy)).catch(() => {});
          }
          return res;
        })
        .catch(() => cached);
      return cached || network;
    })
  );
});
