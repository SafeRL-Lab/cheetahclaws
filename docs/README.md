# CheetahClaws Documentation Index

This directory holds all project documentation. For the project overview,
start with the [top-level README](../README.md).

## Layout

| Path | What |
|------|------|
| [`news.md`](news.md) | Full release notes (the top-level README has one-line summaries linking here) |
| [`agent-os.md`](agent-os.md) | Agent-OS layer (`cc_kernel/`) reference |
| [`architecture.md`](architecture.md) | System architecture deep dive |
| [`contributor_guide.md`](contributor_guide.md) | How to set up dev environment and submit PRs |
| [`guides/`](guides/) | User-facing how-to guides (Web UI, bridges, trading, research lab, plugin authoring, voice/video, recipes, reference, Docker, advanced) |
| [`RFC/`](RFC/) | Design proposals (numbered, ~30 documents covering daemon, kernel, agent OS) |
| [`roadmap/`](roadmap/) | Roadmap snapshots |
| [`superpowers/`](superpowers/) | Plans and specs for in-flight initiatives |
| [`PR/`](PR/) | Per-PR design notes |
| [`media/`](media/) | Images, logos, demo GIFs (`demos/`, `logos/`, `screenshots/`) |
| [`i18n/`](i18n/) | README translations (CN · DE · ES · FR · JP · KO · PT) |
| [`archive/`](archive/) | Superseded docs (old release notes, old comparison tables) |

## Where to start

- **Just want to use it?** Start with the [top-level README](../README.md), then [`guides/reference.md`](guides/reference.md) for the full slash-command and config reference.
- **Setting up the Web UI?** [`guides/web-ui.md`](guides/web-ui.md).
- **Deploying it past your laptop?** [`guides/security.md`](guides/security.md) — every `CHEETAHCLAWS_*` env var, bot-token handling, Bash denylist, plugin / MCP / file-system hardening, CSRF, terminal session owner-binding.
- **Building a plugin / tool?** [`guides/plugin-authoring.md`](guides/plugin-authoring.md).
- **Hacking on the core?** [`architecture.md`](architecture.md) + [`contributor_guide.md`](contributor_guide.md).
- **Curious about the agent-OS direction?** [`agent-os.md`](agent-os.md) + the `RFC/` series (start with `0002-daemon-foundation-roadmap.md`).
