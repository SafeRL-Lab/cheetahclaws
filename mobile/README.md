# CheetahClaws Mobile (iOS / Android)

A native mobile shell built with [Capacitor](https://capacitorjs.com/). It wraps
the **existing** web Chat UI in a real App-Store/Play-Store-installable app — no
UI is reimplemented. This is the same idea as the [`desktop/`](../desktop) Electron
shell, with one architectural difference:

| | Desktop (Electron) | Mobile (Capacitor) |
|---|---|---|
| Where the agent runs | local `cheetahclaws --web` **sidecar** on the same machine | a **remote** `cheetahclaws --web` server you point the app at |
| How the UI loads | webview → `http://127.0.0.1:<port>/chat` | webview → `https://<your-server>/chat` |
| New code | window + sidecar management | a small "connect to your server" launcher (`www/index.html`) |

A phone can't run the Python server locally, so the app is a **thin client**: it
connects to a CheetahClaws server running on your machine or a VM (the
Cursor/OpenClaw model). The launcher stores the server URL and navigates the
webview into that server's `/chat`, reusing the entire web UI (streaming
WebSocket/SSE, JWT auth, tool cards, sessions — all of it).

## What's here

```
mobile/
├── capacitor.config.json   app id, webDir, allowed navigation, splash/status bar
├── package.json            Capacitor deps + helper scripts
├── www/
│   ├── index.html          the launcher shell (connect / test / remember server)
│   └── icon.png            logo shown on the launcher
└── assets/
    ├── icon.png            1024² source for app icons
    └── splash.png          2732² source for splash screens
```

The native `ios/` and `android/` projects are **generated** (not committed) — see setup.

## Prerequisites

- **Node.js 18+** and npm.
- **iOS:** macOS + Xcode + CocoaPods (`sudo gem install cocoapods`), an Apple
  Developer account to run on a physical device / submit to the App Store.
- **Android:** Android Studio + JDK 17.

## One-time setup

```bash
cd mobile
npm install

# Generate the native projects (creates ios/ and android/)
npx cap add ios
npx cap add android

# Generate app icons + splash screens from assets/icon.png + assets/splash.png
npx @capacitor/assets generate --iconBackgroundColor '#0b0b0e' --splashBackgroundColor '#0b0b0e'

# Copy web assets + config into the native projects
npx cap sync
```

## Run

```bash
# iOS (opens Xcode → pick a simulator or your device → Run)
npx cap open ios

# Android (opens Android Studio → Run)
npx cap open android
```

After editing anything in `www/` or `capacitor.config.json`, run `npx cap sync`
again (or `npx cap copy` for web-only changes).

## Using the app

1. On your machine: `cheetahclaws --web` (keep auth **on**).
2. Expose it over **HTTPS** — required, because the mobile webview loads a
   secure origin and the server needs to be reachable from the phone:
   ```bash
   cloudflared tunnel --url http://localhost:8080
   ```
   (or ngrok / `tailscale serve` / a Caddy+certs reverse proxy).
3. Launch the app, paste the `https://…` URL, tap **Test** then **Connect**.
4. First connection prompts you to register an admin account (handled by the web
   UI). The URL is remembered; next launch auto-connects.

**Change server / re-open the launcher:** Android — hardware back button. iOS —
enable the edge-swipe-back gesture in Xcode once
(`ios/App/App/…`: set `webView.allowsBackForwardNavigationGestures = true`), or
launch the shell with `?setup=1`. Clearing app data also resets it.

## Publishing

- **iOS:** in Xcode set your Team + a unique bundle id (default `ai.saferl.cheetahclaws`),
  Archive → distribute to TestFlight / App Store.
- **Android:** in Android Studio, Build → Generate Signed Bundle (`.aab`) → upload
  to Play Console.
- Both stores generally require the app to *do something* beyond wrapping a
  website — the launcher + native shell is the minimum; see the roadmap below for
  features that strengthen a submission.

## Roadmap (the parts that make it feel native, like Cursor/OpenClaw)

These are deliberately **not** implemented yet — the scaffold above is a working
thin client; these are the next increments:

- **Push notifications** ("your agent finished" / "approval needed"). Needs
  `@capacitor/push-notifications` + APNs (iOS) / FCM (Android) **and** a
  server-side hook in `cheetahclaws/web` to POST a push on `turn_done` /
  permission-request events. This is the highest-value native feature.
- **Biometric lock** (`@capacitor/…` FaceID/fingerprint before showing chat).
- **Native share target** (share a file/URL into a new prompt).
- **Saved-servers list** instead of a single remembered URL.
- **Deep links** (`cheetahclaws://` to open a specific session).

## Security notes

- `capacitor.config.json` sets `server.allowNavigation: ["*"]` because the server
  URL is user-configured at runtime (unknown at build time). If you ship to a
  fixed backend, tighten this to just your host(s).
- Always run the exposed server **with authentication on** (never `--no-auth` on a
  public tunnel). The web UI's JWT login + CSRF apply unchanged over the tunnel.
