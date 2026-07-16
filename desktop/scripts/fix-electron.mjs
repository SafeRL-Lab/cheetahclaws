#!/usr/bin/env node
// Repair a half-installed Electron — writes node_modules/electron/path.txt
// when the binary was extracted but the pointer file is missing/wrong.
//
// Why this exists: on very new Node versions Electron's postinstall has been
// observed to extract dist/ but never write path.txt, so `electron .` throws
// "Electron failed to install correctly" even though the binary is right
// there. Hardened-npm setups (allow-scripts) and CN networks make the normal
// postinstall flaky too. This script is the deterministic last-mile fix.
//
//   node scripts/fix-electron.mjs        (run via: npm run fix-electron)

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const elDir = path.resolve(here, '..', 'node_modules', 'electron');
const distDir = path.join(elDir, 'dist');

// index.js resolves the binary as path.join(__dirname, 'dist', <path.txt>),
// so path.txt must be RELATIVE TO dist/ — no leading "dist/".
const REL = {
  darwin: 'Electron.app/Contents/MacOS/Electron',
  win32: 'electron.exe',
  linux: 'electron',
}[process.platform] || 'electron';

if (!fs.existsSync(distDir)) {
  console.error(`✗ ${distDir} is missing — Electron's binary was never extracted.`);
  console.error('  Reinstall it (CN mirror recommended):');
  console.error('    rm -rf node_modules/electron ~/Library/Caches/electron');
  console.error('    ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/ \\');
  console.error('      npm install electron --foreground-scripts');
  console.error('  then re-run: npm run fix-electron');
  process.exit(1);
}

const binary = path.join(distDir, REL);
if (!fs.existsSync(binary)) {
  console.error(`✗ extracted dist/ exists but the binary is missing at dist/${REL}`);
  console.error('  The extraction looks incomplete — re-download the full zip:');
  console.error('    https://npmmirror.com/mirrors/electron/  (pick your platform/arch)');
  console.error('    unzip it into node_modules/electron/dist/, then re-run this.');
  process.exit(1);
}

const pathFile = path.join(elDir, 'path.txt');
const current = fs.existsSync(pathFile) ? fs.readFileSync(pathFile, 'utf8') : '<missing>';
fs.writeFileSync(pathFile, REL);
console.log(`✓ wrote node_modules/electron/path.txt = "${REL}"`);
if (current !== REL) console.log(`  (was: "${current}")`);
console.log('  Electron should now launch: npm start');
