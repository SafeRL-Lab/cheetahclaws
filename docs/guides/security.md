# Security & Environment Variables

This page is the single reference for cheetahclaws' security model and every
environment variable that gates a sensitive subsystem. Read this before you
expose cheetahclaws to a remote bridge, a multi-user web deployment, or an
untrusted plugin.

## Threat model — at a glance

cheetahclaws is built around a single trusted operator running on a single
host. Inside that boundary the defaults favour usability; once you cross it
(remote bridge, web UI on a LAN, shared daemon, third-party plugins) the
hardening below kicks in.

| Surface | Default posture | Tighten with |
|---|---|---|
| Local REPL `!command` | Allowed (you typed it) | NUL / length / control-char filter |
| Bash tool (`Bash`) | LLM-driven, owner-confirmed | Hard denylist (`rm -rf /`, fork bomb, `dd of=/dev/sd…`, `mkfs`, `chmod -R / 777` etc.) |
| Remote bridge `!cmd` (Telegram / Slack / WeChat) | **Enabled by default**, owner-only (chat_id whitelist) + hard denylist | `CHEETAHCLAWS_BRIDGE_TERMINAL=0` to disable entirely |
| File I/O (`Read` / `Write` / `Edit`) | CWD-relative, credential paths denied | `allowed_root` config / `CHEETAHCLAWS_FS_NO_SANDBOX=1` |
| Plugins (`plugin/loader.py`) | Manifest-driven, runs arbitrary Python | `CHEETAHCLAWS_DISABLE_PLUGINS=1` / `CHEETAHCLAWS_PLUGIN_ALLOWLIST=…` |
| MCP server env | Dangerous keys (`LD_PRELOAD`, `PYTHONPATH`, …) stripped | `CHEETAHCLAWS_MCP_TRUST_ENV=1` to allow |
| Web UI (chat) | JWT cookie + CSRF double-submit + owner-bound terminal sessions | `--no-auth` (testing only) |

## Environment variables

| Variable | Default | Behaviour |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | (unset) | Recommended way to provide the Telegram bot token. Token from env never lands in `readline` history or `~/.cheetahclaws/config.json`. Beats the `/telegram <token> <chat_id>` syntax. |
| `SLACK_BOT_TOKEN` | (unset) | Same idea for Slack (`xoxb-…`). |
| `CHEETAHCLAWS_BRIDGE_TERMINAL` | `1` | Set to `0` to hard-disable remote `!cmd` shell from any bridge (Telegram / Slack / WeChat). Useful when a bridge owner is *not* the same person as the host operator. |
| `CHEETAHCLAWS_FS_NO_SANDBOX` | `0` | Bypass the credential-path denylist (SSH private keys, `~/.aws`, `~/.gnupg`, `/etc/shadow`, etc.). Only set if you're deliberately auditing your own secrets. |
| `CHEETAHCLAWS_DISABLE_PLUGINS` | `0` | Hard-disable plugin loading regardless of what's installed in `~/.cheetahclaws/plugins/`. |
| `CHEETAHCLAWS_PLUGIN_ALLOWLIST` | (unset) | Comma-separated plugin names. When set, only these plugins are loaded — everything else is silently skipped, even if enabled in the registry. |
| `CHEETAHCLAWS_MCP_TRUST_ENV` | `0` | Set to `1` to allow MCP server configs to inject `LD_PRELOAD`, `LD_LIBRARY_PATH`, `DYLD_*`, `PYTHONPATH`, `PYTHONSTARTUP`, `PYTHONHOME`, `NODE_OPTIONS`, `NODE_PATH`, `BASH_ENV`, or `ENV`. Default is to silently strip these from `mcp.json`-supplied env. |
| `CHEETAHCLAWS_WEB_SECRET` | (unset) | Per-deployment JWT signing secret for the web UI. Strongly recommended for production — overrides the auto-generated file at `~/.cheetahclaws/web_secret`. |

## Bot tokens — recommended setup

Tokens passed as `/telegram <TOKEN> <chat_id>` in the REPL land in your
`readline` history file (`~/.cheetahclaws/input_history.txt`). Subsequent
processes that read this file (including a curious shell user with read
access) can extract the token.

Recommended:

```bash
export TELEGRAM_BOT_TOKEN=7812345678:AAFxyz...
cheetahclaws
[myproject] ❯ /telegram 987654321        # only chat_id on the command line
```

If you do use the old two-arg form, cheetahclaws will:

- print a deprecation warning,
- run `readline.remove_history_item` on every line in history that contains
  the token (so it disappears the moment the command runs).

WeChat does not have a token-in-argv path — its session token is obtained
via QR-code scan and stored in `~/.cheetahclaws/config.json` (0600).

## Bash tool — hard denylist

A small set of command patterns is rejected by `tools/shell.py` even when
`permission_mode=accept-all` is active. The denylist is intentionally
narrow to avoid false positives:

- `rm -rf /` (and variations like `rm --recursive --force /`)
- `mkfs.*`
- `dd of=/dev/sd*` / `dd of=/dev/nvme*` / `dd of=/dev/hd*` / `dd of=/dev/vd*` / `dd of=/dev/mmcblk*` / `dd of=/dev/xvd*`
- `> /dev/sd*` / `> /dev/nvme*` / `> /dev/hd*` (and similar)
- `chmod -R 777 /`
- `chown -R <user> /`
- `:(){:|:&};:` fork bomb

These also apply to bridge-originated `!cmd` invocations and to tmux pane
commands. Set `permission_mode` correctly if you need a one-off override;
the denylist has no opt-out (by design).

## File-system sandbox

When `config["allowed_root"]` is set, `Read` / `Write` / `Edit` / `Glob`
will refuse paths outside that root. This is the strict mode for
multi-user deployments and is what `cheetahclaws daemon` uses.

Even without `allowed_root`, a credential denylist is enforced:

- `~/.ssh/id_*` (private keys)
- `~/.aws/*`, `~/.gnupg/*`, `~/.kube/*`, `~/.docker/*`
- `~/.netrc`, `~/.pgpass`
- `/etc/shadow`, `/etc/gshadow`, `/etc/sudoers`, `/etc/sudoers.d/*`
- `/root/*`

Public-by-convention SSH files (`~/.ssh/config`, `known_hosts`,
`authorized_keys`) are still readable.

Set `CHEETAHCLAWS_FS_NO_SANDBOX=1` to disable the denylist (e.g. when
intentionally auditing your own keys).

## Web UI

See [`docs/guides/web-ui.md`](web-ui.md) for the full chat-UI auth flow.
Quick summary of the security pieces:

| Defence | Mechanism |
|---|---|
| Login | `username + password` → bcrypt verify → HS256 JWT |
| JWT cookie | `ccjwt=…; Path=/; HttpOnly; SameSite=Strict; Max-Age=604800` |
| CSRF | Double-submit cookie: `ccsrf=…; SameSite=Strict` non-HttpOnly cookie minted on first GET, every POST/PUT/PATCH/DELETE must echo it in the `X-CSRF-Token` header. Exempted: `/api/auth/bootstrap`, `/api/auth/register`, `/api/auth/login`, `/api/auth/logout`, `/api/auth` (legacy terminal). Frontend `web/static/js/csrf.js` patches `window.fetch` so this is automatic for the bundled UI. |
| Terminal session ownership | `/api/session` creator's JWT uid is tagged onto the `_PtySession`. `/api/stream`, `/api/input`, `/api/resize` reject any other authenticated caller with `403`. Password-only mode (`cctoken` auth, no JWT) skips this check since all callers share the same secret. |
| Terminal password | 32-char `secrets.token_urlsafe(32)` (~190 bits of entropy) generated at startup, displayed once. |
| JWT signing secret | `secrets.token_urlsafe(32)` saved to `~/.cheetahclaws/web_secret` with `O_CREAT \| O_EXCL` + 0o600. If the post-write `stat` shows world-readable mode the file is removed and the secret falls back to in-memory only; override with `CHEETAHCLAWS_WEB_SECRET`. |

## Plugins

Plugins are loaded by `importlib` from `~/.cheetahclaws/plugins/<name>/`
or any directory on `$CHEETAHCLAWS_PLUGIN_PATH`. Each module runs with
the same privileges as the cheetahclaws process — there is no sandbox.

Mitigations on by default:

- Module paths are confined to the plugin's `install_dir` (no
  `../../etc/passwd` traversal).
- EXTERNAL-scope plugins (loaded via `$CHEETAHCLAWS_PLUGIN_PATH`) print a
  one-time stderr warning on first load so a stolen env var doesn't
  load code silently.

Use `CHEETAHCLAWS_DISABLE_PLUGINS=1` for an emergency kill switch or
`CHEETAHCLAWS_PLUGIN_ALLOWLIST=a,b,c` to whitelist exactly the plugins
you trust.

## MCP

MCP server configs (`.mcp.json`, plugin manifests) supply a `command`,
`args`, and an `env` map. cheetahclaws merges `env` over `os.environ`
before spawning, but first strips a hard-coded set of process-hijack
keys: `LD_PRELOAD`, `LD_LIBRARY_PATH`, `LD_AUDIT`, `DYLD_INSERT_LIBRARIES`,
`DYLD_LIBRARY_PATH`, `PYTHONPATH`, `PYTHONSTARTUP`, `PYTHONHOME`,
`PYTHONEXECUTABLE`, `NODE_OPTIONS`, `NODE_PATH`, `BASH_ENV`, `ENV`.

A dropped key prints a one-line `[mcp] Dropped potentially-dangerous env
keys …` notice to stderr. Set `CHEETAHCLAWS_MCP_TRUST_ENV=1` if a
legitimate MCP server actually needs one of these.

## Permission mode `accept-all`

`/permissions accept-all` (or clicking "Accept all" at the permission
prompt) is now **session-scoped**. The value is *not* persisted to
`~/.cheetahclaws/config.json`, so launching cheetahclaws again starts
back in `auto` (the safe default).
