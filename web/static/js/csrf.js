/* Double-submit cookie CSRF helper.
 *
 * The server mints a non-HttpOnly `ccsrf` cookie on first GET. State-changing
 * requests (POST/PUT/PATCH/DELETE) must echo the cookie value in an
 * X-CSRF-Token header. This file patches window.fetch so existing call sites
 * pick it up automatically — no per-call edits required.
 */
(function patchFetchForCsrf() {
  if (window.__cc_csrf_patched) return;
  window.__cc_csrf_patched = true;

  const origFetch = window.fetch.bind(window);

  function tokenFromCookie() {
    const m = document.cookie.match(/(?:^|;\s*)ccsrf=([^;]+)/);
    return m ? decodeURIComponent(m[1]) : "";
  }

  window.fetch = function(input, init) {
    init = init || {};
    const method = ((init.method || (typeof input === "object" && input.method) || "GET") + "").toUpperCase();
    if (["POST", "PUT", "PATCH", "DELETE"].indexOf(method) !== -1) {
      const tok = tokenFromCookie();
      if (tok) {
        const hdrs = init.headers || {};
        // Headers may be a plain object or a Headers instance — handle both.
        if (typeof Headers !== "undefined" && hdrs instanceof Headers) {
          hdrs.set("X-CSRF-Token", tok);
          init.headers = hdrs;
        } else if (Array.isArray(hdrs)) {
          init.headers = hdrs.concat([["X-CSRF-Token", tok]]);
        } else {
          init.headers = Object.assign({}, hdrs, {"X-CSRF-Token": tok});
        }
      }
    }
    return origFetch(input, init);
  };
})();
