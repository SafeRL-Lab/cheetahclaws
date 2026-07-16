'use strict';

// Smoke test for the sidecar manager — no Electron, no GUI required.
// Launches the real `cheetahclaws --web --no-auth`, waits for readiness,
// hits the served endpoints, then shuts the server down.
//
//   node test/sidecar.smoke.js          (DEBUG=1 to echo server logs)
//
// Requires the `cheetahclaws` CLI on PATH with the web extra installed:
//   pip install 'cheetahclaws[web]'

const http = require('http');
const { startSidecar, stopChild } = require('../src/sidecar');

function get(url) {
  return new Promise((resolve, reject) => {
    const req = http.get(url, (res) => { res.resume(); resolve(res.statusCode); });
    req.on('error', reject);
    req.setTimeout(5000, () => req.destroy(new Error('request timeout')));
  });
}

(async () => {
  console.log('→ starting sidecar: cheetahclaws --web --no-auth --host 127.0.0.1');
  const sc = await startSidecar({
    readyTimeoutMs: 30000,
    onLog: (l) => process.env.DEBUG && console.log('   |', l),
  });
  console.log(`→ ready at ${sc.url} (port ${sc.port})`);

  try {
    for (const ep of ['/chat', '/', '/health']) {
      const code = await get(`http://127.0.0.1:${sc.port}${ep}`);
      console.log(`   GET ${ep.padEnd(8)} -> ${code}`);
      if (code !== 200) throw new Error(`expected 200 for ${ep}, got ${code}`);
    }
    console.log('✅ SMOKE PASS — shell can launch the server and reach the chat UI');
  } finally {
    await stopChild(sc.child);
    console.log('→ sidecar stopped');
  }
})().catch((err) => {
  console.error('❌ SMOKE FAIL —', err.message);
  process.exit(1);
});
