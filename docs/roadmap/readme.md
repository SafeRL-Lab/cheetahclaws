
# CheetahClaws Roadmap (v3, 2026-04-26)

> **CheetahClaws is becoming a secure, multi-model agent runtime that engineering and research teams trust to actually run on their CI, repos, and experiments.**



---

## TL;DR

| Horizon | Window | What we will have |
|---|---|---|
| **Now**   | 2026-Q2 (next ~9 weeks) | Two flagship workflows running on at least one real repo end-to-end, with a structured trace anyone can read. |
| **Next**  | 2026-Q3 (Jul–Sep)       | Permission/approval system, audit log, and replay-from-checkpoint. The runtime is safe enough for a small team to point at their own CI. |
| **Later** | 2026-Q4 → 2027-Q1       | Local-deploy package (Docker), model tier guarantees, and the first paid / sponsored team using CheetahClaws on real engineering work. |
| **North Star** | 2027-H2+         | An industrial product: secure agent runtime for engineering and research automation, deployable in customer infrastructure, with audit, approval, and ROI metrics. |

The path to industrial product **is not** "build an enterprise feature checklist now." It is "ship two workflows that solve real pain, instrument them so customers can audit what happened, then customers will tell us which enterprise features they actually need."

---

## 1. The User Problem We Are Solving

We solve **specific** problems. Not "improving developer productivity in general."

| User | Pain today | What CheetahClaws delivers | How we measure it |
|---|---|---|---|
| Backend developer | A flaky CI run blocks a PR; takes 30–90 min to triage logs, reproduce locally, and patch. | `cheetah investigate-ci <run-url>` returns: failure summary, suspected files, proposed patch, local reproduction command, test command. | Median time from CI failure → PR review request. Target: < 10 min on the supported repo set. |
| Maintainer of a small library | Inbox of "easy" GitHub issues never gets cleared because each one needs context loading. | `cheetah solve-issue <url>` produces a draft PR with diff + tests + summary, awaiting human approval. | Issues triaged-to-PR per week. Target: 5/week per maintainer with < 30% rejection. |
| Research engineer (lab) | Reproducing experiments across models is glue-code-heavy and traces vanish. | One workflow YAML runs across 3+ providers, emits a comparable trace + result table. | Number of model-pairs benchmarked from one config in one command. Target: 5+. |

These three are the **only** product wedges we commit to in 2026. Everything else (skills, plugins, dashboards, RBAC) is supporting infrastructure or future work.

---

## 2. Strategy: Earn Trust First, Productize Second

The mistake to avoid: treating "industrial product" as a feature list (SSO, RBAC, Helm, dashboards) and building those first. Enterprise buyers do not buy roadmaps; they buy **a tool their engineers already secretly use** plus the wrapper that makes it auditable.

So the sequence is:

1. **OSS wedge** (now). Two flagship workflows that solve real pain. Free.
2. **Trust layer** (Q3). Trace, audit, approval, replay. Still free.
3. **Self-host bundle** (Q4). Docker, config templates, local model gateway. Free.
4. **Industrial wrapper** (2027+). Only when customers have already adopted (1)–(3) do we add SSO/RBAC/admin console — paid.

Building (4) before (1) is how OSS projects die looking like enterprise sales decks with no users.

---

## 3. Where We Are Today (2026-04-26)

Honest state of the repo.

**Working and stable**
- REPL with prompt_toolkit input + slash command completion (`ui/input.py`)
- Web terminal bridge over PTY + SSE/WebSocket (`web/server.py`)
- Multiple model provider adapters (`providers.py`) — Anthropic, OpenAI-compat, DeepSeek, etc.
- Skills loader (`skill/loader.py`) supporting flat + nested layout
- MCP integration (`mcp_client/`)
- Checkpoint, compaction, session_store modules
- ~570 tests passing on Linux; 565 + 2 skipped on Windows after PR #66
- Plugin system (external plugin loader + tests)

**Half-built or unstable**
- Multi-agent module (`multi_agent/`) — exists, unclear how stable
- Memory module — present but not consolidated (memory.py + memory/)
- Web UI — terminal works; chat UI partial (`web/chat.html`, `web/api.py`)

**Missing for the wedges in §1**
- No structured trace event schema — logs are line-based, hard to replay
- No tool-permission policy — no read-only / approve-edits / sandboxed modes
- No durable task graph — nothing to checkpoint mid-workflow
- No CI log parser, no GitHub issue ingester
- No benchmark harness that can run the same config across providers

**Out of scope but currently in repo**
- `demos/`, `video/`, `voice/`, trading examples, telegram bridge
  → these should move to `examples/` or a separate repo. They distract from the wedges.

---

## 4. The Wedge: Two Flagship Workflows

We ship **two** workflows in 2026, not five. Each must be runnable end-to-end with a single command, against a public repo we don't own, producing artifacts a human can review in under five minutes.

### 4.1 CI Failure Investigator (primary wedge)

```bash
cheetah investigate-ci https://github.com/<owner>/<repo>/actions/runs/<id>
```

**Inputs**: a CI run URL or local log file.
**Outputs (artifacts/<run-id>/)**:
- `summary.md` — what failed, in plain English
- `suspected_files.json` — ranked list with line ranges
- `patch.diff` — proposed minimal fix
- `repro.sh` — local reproduction command
- `trace.json` — structured trace of every model + tool call

**Definition of done (Q2 2026)**: works end-to-end on 5 hand-picked OSS repos with > 70% useful-patch rate on a curated failure set.

### 4.2 GitHub Issue Resolver (secondary wedge)

```bash
cheetah solve-issue https://github.com/<owner>/<repo>/issues/<n>
```

**Inputs**: issue URL.
**Outputs**: branch + draft PR + tests + summary, plus the same `trace.json`.

**Definition of done (Q3 2026)**: produces a mergeable PR on 50% of issues from a labeled "good-first-issue" set across 3 OSS repos.

These two workflows share infrastructure: model runtime, tool runtime, trace, repo-aware context. **Every infra task we list below must serve at least one of these two workflows**. If a feature can't be tied to (4.1) or (4.2), it's deferred.

---

## 5. Now — Q2 2026 (target: end of June)

The next 9 weeks. Five concrete deliverables.

| # | Deliverable | Why | Done when |
|---|---|---|---|
| N1 | **Trace event schema v1** + JSON writer wired into model + tool calls | Without traces, neither wedge produces auditable output. | A run of `cheetah investigate-ci` writes `trace.json` containing every model call, tool call, file diff, and error, and `cheetah trace show` prints a readable timeline. |
| N2 | **CI log parser tool** (read-only, no execution) | Wedge 4.1 needs this. | Tool can ingest GitHub Actions raw log + return structured `{stage, command, exit_code, error_blocks}`. Tested on 10 real failure logs. |
| N3 | **`cheetah investigate-ci` MVP** | Wedge 4.1 first cut. | Runs end-to-end on the 5 curated repos. May still produce bad patches; that's OK. Trace must be complete. |
| N4 | **Repo cleanup**: move trading/voice/telegram demos to `examples/`; update README to drop "personal assistant" framing in favor of "agent runtime for engineering automation" | Current README contradicts the direction. | README + repo structure match this roadmap. Tests still green. |
| N5 | **Provider tier doc** (see §9) committed; tier-1 providers gated by a smoke test in CI | Without this, every provider regresses silently. | `tests/test_provider_smoke.py` runs in CI for all T1 providers and fails the build on regression. |

Anti-deliverables for Q2 (we will say **no** to these):
- RBAC, SSO, admin console, ROI dashboard
- Helm chart, Kubernetes deployment
- New web UI features beyond fixing the existing terminal
- Adding new model providers beyond the T1 set
- Skill registry / plugin marketplace

---

## 6. Next — Q3 2026 (Jul–Sep)

| # | Deliverable | Done when |
|---|---|---|
| Q3-1 | **Permission modes**: `read-only`, `approve-edits`, `approve-bash`, `workspace-write` | Toggling mode demonstrably changes what `investigate-ci` does (e.g., read-only mode never writes a patch file, only prints diff). |
| Q3-2 | **Approval queue** in the REPL + web UI | `approve-edits` mode pauses on each file write and shows the diff; user types `y/n/edit`. |
| Q3-3 | **`cheetah solve-issue` MVP** (wedge 4.2) | End-to-end on 3 repos. |
| Q3-4 | **Replay from checkpoint** | A workflow that crashes mid-way can be resumed via `cheetah resume <run-id>`. |
| Q3-5 | **HTML trace report** | `cheetah trace export <run-id> --format html` produces a self-contained file with model calls, file diffs, tool timeline. |
| Q3-6 | **Secret redaction in trace + logs** | Run with `OPENAI_API_KEY` set; redacted in all written artifacts. |

Phase exit criterion: a small team (3–10 engineers) can point CheetahClaws at their CI, run it in `approve-bash` mode, and feel safe leaving it on overnight.

---

## 7. Later — Q4 2026 → 2027-Q1

The "pre-product" phase. Goal: one or two design partner teams using us on real engineering work.

| # | Deliverable | Done when |
|---|---|---|
| L1 | **Docker self-host bundle** | `docker compose up` brings up the runtime + web UI + a local model gateway. Documented in `docs/deployment/docker.md`. |
| L2 | **Local model gateway** for vLLM/Ollama/LM Studio behind a single OpenAI-compatible URL | Workflow YAML routes "local-qwen" through the gateway with no code changes. |
| L3 | **Multi-model benchmark harness** (wedge 4.3 graduates here) | `cheetah benchmark --suite ci-failures --models claude-sonnet,gpt,qwen-vllm` produces a comparable report. |
| L4 | **Workflow YAML format v1** | `investigate-ci` and `solve-issue` are both expressible as a YAML task graph (not only Python). |
| L5 | **Design-partner program** | At least 1 team is running CheetahClaws weekly on their own repo and giving us trace data + bug reports. |

Phase exit criterion: a design partner says "we'd pay for this if it had X, Y, Z" — and X, Y, Z are concrete enterprise features (RBAC, SSO, on-prem support), not core capability gaps. That's the signal we've earned the right to build §8.

---

## 8. North Star — 2027-H2 and beyond (Industrial)

Only enter this phase after §7 has produced a paying or contracted design partner. Otherwise we're shipping into a void.

The shape of the industrial product:

- **Secure deployment**: Docker / Helm / on-prem / VPC / air-gapped. Local model gateway is the default deploy.
- **Identity and access**: SSO (SAML/OIDC), RBAC, workspace + project + tool + model permissions, two-person approval for high-risk actions.
- **Audit and compliance**: signed traces, approval records, retention policies, exportable incident reviews.
- **Admin console**: model config, integration config, policy templates, audit dashboard, ROI dashboard.
- **Integrations**: GitHub/GitLab Enterprise, Slack approvals, Jira/Linear, internal MCP servers.
- **SLA**: workflow-level success metrics, regression test suite, support bundle.

What we will **not** commit to ahead of customer ask: the specific dashboard panels, the exact RBAC matrix, the integration list past the first three. Customers will tell us.

---

## 9. Provider Support Tiers

We support 11 providers across both v1 and v2 roadmaps. That's a lie — we cannot keep 11 providers regression-tested with current capacity. Instead:

| Tier | Providers | Commitment |
|---|---|---|
| **T1 — Supported** | Anthropic Claude, OpenAI GPT, DeepSeek | Smoke test on every PR. Tool-call regressions block the build. Workflow YAMLs use these as default. |
| **T2 — Experimental** | Google Gemini, Qwen (DashScope), Ollama, vLLM, LM Studio | Best-effort. Tested on release candidates only. Breakages opened as issues, not blockers. |
| **T3 — Community** | Kimi, Zhipu, MiniMax, any OpenAI-compatible endpoint | Provided as configuration templates. Supported by community PRs. We don't run CI for these. |

A provider can be promoted T3 → T2 → T1 if a contributor commits to maintaining the test fixture.

---

## 10. Non-Goals (Things We Will Not Do This Year)

Listing these explicitly so PRs and issues that propose them get a clear "not now":

- General-purpose chatbot / personal assistant features
- Trading bot, voice assistant, Telegram bridge as core offerings (move to `examples/`)
- A proprietary skill marketplace
- A replacement for VS Code / Cursor IDE integration (we run inside terminals)
- Browser automation as a core tool (out of scope for the two wedges)
- Mobile app
- Custom UI framework — the web terminal stays minimal
- Speculative work on agent-to-agent protocols / swarms before single-agent reliability is solved

---

## 11. Success Metrics (demo-able only)

We commit to metrics that someone can run a command and read a number. No vanity.

**Q2 2026 (Now)**
- `investigate-ci` end-to-end success rate ≥ 70% on the 5 curated repos (curated set published in `benchmarks/ci-failures-2026q2/`)
- Trace coverage: 100% of model + tool calls in a run appear in `trace.json`
- Test pass rate: 100% on Linux T1 providers; ≥ 95% on macOS

**Q3 2026 (Next)**
- `solve-issue` produces a mergeable PR on ≥ 50% of curated good-first-issues
- Median time-to-fix for `investigate-ci` ≤ 10 min on T1 models
- Zero unredacted secrets in traces (verified by redaction unit tests + a CI grep)

**Q4 2026 + 2027-Q1 (Later)**
- ≥ 1 design partner running CheetahClaws weekly
- Reproducible benchmark report from a single YAML across ≥ 3 providers
- Runtime starts cleanly via `docker compose up` on a fresh Ubuntu 22.04 VM

Metrics we explicitly are **not** tracking yet (will track post-§7): RBAC adoption, SSO MAU, admin actions per day, ROI dashboard usage.

---

## 12. How We Decide What's Next

The two questions every issue and PR must answer:

1. **Which wedge does this serve?** If neither (4.1) nor (4.2), it's deferred to 2027.
2. **Can we test it end-to-end?** If the only test is a unit test of an internal abstraction, we're building plumbing without water.

Roadmap exceptions are allowed — but they must be opened as a PR to this file with the reasoning.

---

## 13. Contributing

The contribution priority order for 2026 is:

1. Anything that makes `investigate-ci` more reliable on the curated repo set
2. Anything that improves `trace.json` completeness or readability
3. Permission/approval mechanics
4. `solve-issue` workflow components
5. T1 provider stability
6. Documentation, examples, and curated repo additions

If you're new, pick a `good-first-issue` labelled with `q2-2026` — those are pre-vetted to fit one of the deliverables in §5.

We do not currently accept PRs for: T3 provider stability, RBAC, dashboards, or new core demos. Open a discussion first.

---

## 14. Roadmap Maintenance

This roadmap is reviewed and updated **at the end of each quarter**:

- 2026-06-30 — Q2 review, set Q3 deliverables
- 2026-09-30 — Q3 review, set Q4 deliverables
- 2026-12-31 — annual review, set 2027 direction

If a deliverable slips two consecutive quarter reviews, it gets cut or rescoped — not silently moved forward. Roadmaps that never delete anything stop being roadmaps.

---

## North Star (one paragraph)

> CheetahClaws will be the open-source agent runtime that engineering and research teams trust to run on their CI, repos, and experiments — because every action it takes is gated by explicit permission, recorded in an auditable trace, and reproducible from a checkpoint. The industrial product is the secure, governed, deployable wrapper that customers buy after their engineers have already adopted the OSS runtime.
