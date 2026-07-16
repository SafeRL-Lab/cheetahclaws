# CheetahClaws Roadmap

> **CheetahClaws is evolving into a Python-native, local-first Agent OS for autonomous engineering, research workflows, and reproducible multi-model agent infrastructure.**

This roadmap defines the community direction, technical architecture, industrial product path, and contribution priorities for CheetahClaws.

The goal is not only to build an impressive agent demo.  
The goal is to build infrastructure that solves real user problems.

---

## Table of Contents

1. [Vision](#1-vision)
2. [Core Positioning](#2-core-positioning)
3. [Who We Serve](#3-who-we-serve)
4. [User Problems We Aim to Solve](#4-user-problems-we-aim-to-solve)
5. [Product Principles](#5-product-principles)
6. [Agent OS Primitives](#6-agent-os-primitives)
7. [Technical Architecture Roadmap](#7-technical-architecture-roadmap)
8. [Industrial Product Roadmap](#8-industrial-product-roadmap)
9. [Flagship Workflows](#9-flagship-workflows)
10. [Development Phases](#10-development-phases)
11. [Success Metrics](#11-success-metrics)
12. [Contribution Priorities](#12-contribution-priorities)
13. [Suggested GitHub Issues](#13-suggested-github-issues)
14. [Recommended Repository Structure](#14-recommended-repository-structure)
15. [Long-Term Direction](#15-long-term-direction)
16. [Call for Contributors](#16-call-for-contributors)

---

## 1. Vision

CheetahClaws aims to become a **Python-native Agent OS / Agent Runtime** for researchers, developers, engineering teams, and advanced users who want to build, run, inspect, control, and reproduce autonomous agents across different models and environments.

In product language, CheetahClaws is an **Agent OS**.

In systems language, CheetahClaws is a **durable agent runtime infrastructure layer** that manages:

- models
- tools
- memory
- context
- workflows
- permissions
- traces
- checkpoints
- artifacts
- human approvals
- evaluations
- integrations

CheetahClaws should not only answer prompts.  
It should help users complete real engineering and research tasks safely, reproducibly, and transparently.

---

## 2. Core Positioning

### One-line positioning

> **CheetahClaws is a Python-native, local-first Agent OS for secure, reproducible, multi-model autonomous workflows.**

### Open-source positioning

> **A hackable agent runtime for researchers and advanced developers to build, study, benchmark, and customize autonomous agents.**

### Industrial product positioning

> **A secure agent runtime for engineering and research automation, with enterprise deployment, workflow templates, human approval, audit logs, and multi-model execution.**

### What CheetahClaws is

CheetahClaws is:

- **Python-native**: easy to read, modify, extend, debug, and research.
- **Local-first**: supports local files, local tools, local models, and private deployments.
- **Multi-model**: supports proprietary and open models through unified interfaces.
- **Hackable**: users can modify the agent loop, tools, memory, skills, and workflow logic.
- **Workflow-oriented**: focuses on completing real tasks, not only chatting.
- **Reproducible**: agent runs should be traceable, comparable, replayable, and debuggable.
- **Safety-aware**: tool execution, file editing, shell commands, and network access should be governed by explicit permissions.
- **Industrializable**: should support deployment, integrations, audit, governance, and reliability metrics.

### What CheetahClaws is not

CheetahClaws should not primarily be:

- only a Claude Code clone
- only a chatbot
- only a trading bot
- only a personal assistant
- only a collection of unrelated demos
- a closed, model-specific agent product
- a generic agent framework with no opinionated workflows
- a toy system that cannot be audited, recovered, or deployed

---

## 3. Who We Serve

CheetahClaws should be useful for both open-source users and future industrial customers.

### Primary open-source users

| User | Need |
|---|---|
| AI researchers | Study agent behavior, memory, tools, safety, multi-model performance, and long-horizon workflows. |
| ML engineers | Run local/proprietary models on coding, debugging, evaluation, and experiment workflows. |
| Advanced developers | Automate repo tasks, CI failures, code changes, documentation, and PR preparation. |
| Open-source model community | Test whether local models can perform Claude-Code-style autonomous workflows. |
| AI safety researchers | Inspect, constrain, evaluate, and red-team autonomous agent behavior. |

### Primary industrial users

| User | Need |
|---|---|
| Engineering teams | Reduce repetitive debugging, CI investigation, code migration, documentation, and PR work. |
| Research teams | Automate paper reading, experiment setup, log analysis, result tables, and reports. |
| AI infrastructure teams | Compare models, deploy local agent runtimes, and evaluate agent reliability. |
| Platform teams | Provide governed agent infrastructure to internal users. |
| Security and compliance teams | Control tool permissions, audit actions, enforce policies, and prevent unsafe execution. |
| Team leads and managers | Understand agent ROI, reliability, failure modes, and human intervention rate. |

---

## 4. User Problems We Aim to Solve

CheetahClaws should be guided by real user problems, not only infrastructure ambitions.

| User | Pain Point | CheetahClaws Solution | Success Metric |
|---|---|---|---|
| Developer | CI failures take too long to investigate. | Agent inspects logs, identifies likely cause, edits code, runs tests, and proposes a fix. | Reduced time-to-fix; test pass rate. |
| Developer | Small repo issues consume repetitive engineering time. | Agent resolves GitHub/GitLab issues with traceable plans, diffs, tests, and PR summaries. | Issues resolved per week; human approval rate. |
| Engineering team | Code migrations are tedious and error-prone. | Agent performs scoped refactors, updates call sites, runs tests, and summarizes risk. | Migration completion rate; rollback rate. |
| Researcher | Experiments require repetitive setup and result formatting. | Agent creates configs, runs scripts, parses logs, generates tables, and writes summaries. | Faster experiment iteration; reproducible reports. |
| AI researcher | Hard to compare agentic ability across models. | Same workflow runs across Claude, GPT, Gemini, Qwen, DeepSeek, Ollama, vLLM, and other endpoints. | Model comparison report; benchmark success rate. |
| Platform team | Agents are hard to govern safely. | Permission policies, sandboxing, approval gates, audit logs, and rollback. | Unsafe action rate; policy violation rate. |
| Security team | Agent actions are hard to explain after failure. | Full trace, tool-call timeline, file diffs, approval records, and replay. | Incident review time; audit completeness. |
| Team lead | Unclear whether agents create business value. | ROI dashboard with time saved, cost per task, success rate, failure rate, and human intervention rate. | Cost saved per task; automation ROI. |

---

## 5. Product Principles

CheetahClaws should follow these principles.

### 5.1 Runtime before app

Build stable primitives for running agents before adding many demos.

The project should avoid becoming a collection of disconnected assistants.  
Every major feature should strengthen the runtime.

### 5.2 Workflows before prompts

Useful agents need durable workflows, task graphs, checkpoints, tools, tests, recovery mechanisms, and human approval gates.

The main unit of value should be a completed task, not a long chat.

### 5.3 Local-first by default

Users should be able to run CheetahClaws with local models, local files, and local tools whenever possible.

This matters for:

- privacy
- cost control
- enterprise deployment
- research reproducibility
- offline or air-gapped environments
- open-source model evaluation

### 5.4 Multi-model by design

The same workflow should be runnable with:

- Claude
- GPT
- Gemini
- Qwen
- DeepSeek
- Kimi
- Zhipu
- MiniMax
- Ollama
- vLLM
- LM Studio
- any OpenAI-compatible endpoint

CheetahClaws should make it easy to compare models under the same workflow.

### 5.5 Safety is infrastructure

Permission control, sandboxing, audit logs, and secret redaction should be core runtime features.

Safety should not be an optional enterprise add-on.

### 5.6 Everything should be inspectable

Agent traces, tool calls, file diffs, command logs, cost, latency, context snapshots, and memory changes should be visible.

Users should be able to answer:

- What did the agent see?
- What did the agent decide?
- What tool did the agent call?
- What file did the agent change?
- What command did the agent run?
- Who approved it?
- Why did it fail?
- Can we replay or roll back?

### 5.7 Product value before feature count

Industrial users care about solved problems, reliability, security, and return on investment.

CheetahClaws should prioritize a few high-value workflows over many shallow demos.

### 5.8 Human-in-the-loop first, full autonomy later

For real users, especially in industrial settings, the first useful product is often:

```text
agent drafts -> human approves -> agent executes -> human reviews -> agent learns
```

Full autonomy should be introduced only when permissions, rollback, audit, and reliability are mature.

---

## 6. Agent OS Primitives

CheetahClaws should expose a set of OS-like primitives for agentic systems.

| Traditional OS Concept | CheetahClaws Agent OS Equivalent |
|---|---|
| Process | Agent instance |
| Thread | Sub-agent / worker |
| File system | Workspace and artifact store |
| Memory | Short-term context, long-term memory, task state |
| Scheduler | Task planner and workflow executor |
| System call | Tool call / MCP call |
| Permission | Tool permission, sandbox policy, approval gate |
| IPC | Agent-to-agent communication |
| Shell | CLI, REPL, web UI, remote bridge |
| Logs | Trace, audit log, replay record |
| Package manager | Skill/plugin registry |
| Service manager | Long-running task manager |
| User account | Workspace identity / team identity |
| Access control | RBAC, policy, approval workflow |

These primitives should guide the architecture and implementation.

---

## 7. Technical Architecture Roadmap

CheetahClaws should evolve around six infrastructure layers.

---

### Layer 1: Unified Model Runtime

Goal: provide a consistent runtime interface for different LLM providers and local models.

Supported targets should include:

- Anthropic Claude
- OpenAI GPT models
- Google Gemini
- Qwen
- DeepSeek
- Kimi
- Zhipu
- MiniMax
- Ollama
- vLLM
- LM Studio
- any OpenAI-compatible endpoint

Key features:

- streaming response normalization
- tool-call normalization
- provider-specific error handling
- retry and fallback
- model routing
- cost tracking
- latency tracking
- context length tracking
- model capability registry
- local model configuration templates
- OpenAI-compatible endpoint support
- per-model safety and tool-use compatibility notes

Example commands:

```bash
cheetah run task.yaml --model claude
cheetah run task.yaml --model qwen-vllm
cheetah run task.yaml --model deepseek
cheetah compare runs/ --models claude,gpt,qwen-vllm
```

Priority:

- [ ] Define a stable `ModelRuntime` interface.
- [ ] Add provider capability metadata.
- [ ] Add model routing and fallback.
- [ ] Add token, cost, latency, and context usage logging.
- [ ] Add reproducible model configuration files.
- [ ] Add local model templates for Ollama, vLLM, and LM Studio.
- [ ] Add compatibility tests for tool calling across providers.

---

### Layer 2: Safe Tool Runtime

Goal: make tool execution powerful, controllable, observable, and safe.

Core tools:

- file read/write/edit
- bash/shell
- git
- Python execution
- notebook execution
- web/search/browser tools
- LaTeX tools
- repo inspection
- diagnostics
- MCP tools
- custom Python tools
- CI log parser
- issue tracker tools
- experiment log parser

Runtime requirements:

- typed tool schema
- tool permission levels
- command allowlist/blocklist
- approval-before-write
- approval-before-bash
- read-only mode
- sandbox mode
- network-disabled mode
- timeout and resource limits
- secret redaction
- tool failure recovery
- audit logs
- rollback for file edits
- dry-run mode
- policy-based approval
- per-workspace policy config

Suggested permission modes:

| Mode | Description |
|---|---|
| `read-only` | Agent can inspect files but cannot modify or execute risky commands. |
| `approve-edits` | Agent must ask before modifying files. |
| `approve-bash` | Agent must ask before running shell commands. |
| `workspace-write` | Agent can edit files inside the workspace. |
| `sandboxed` | Agent runs commands only inside a controlled sandbox. |
| `network-off` | Agent cannot access external network resources. |
| `dry-run` | Agent proposes actions but does not execute them. |
| `full-auto` | Agent can act autonomously under configured policy. |

Priority:

- [ ] Add `SECURITY.md`.
- [ ] Add tool permission policy config.
- [ ] Add dangerous command detection.
- [ ] Add secret redaction.
- [ ] Add file edit rollback.
- [ ] Add per-tool audit logs.
- [ ] Add sandbox execution mode.
- [ ] Add dry-run execution mode.
- [ ] Add approval queue.
- [ ] Add test suite for safety policies.

---

### Layer 3: Memory and Context OS

Goal: provide explicit, inspectable memory and context management.

Memory should not only mean chat history. CheetahClaws should support multiple memory types:

| Memory Type | Description |
|---|---|
| Working memory | Current task state, active goals, open files, recent decisions. |
| Episodic memory | Past runs, traces, failures, successes, and user interactions. |
| Semantic memory | Project knowledge, repo structure, user preferences, documentation. |
| Procedural memory | Skills, debugging recipes, workflows, tool-use patterns. |
| Artifact memory | Generated files, patches, experiment outputs, logs, reports. |
| Team memory | Shared project-level knowledge for labs or engineering teams. |

Context construction should answer:

- Which files are relevant?
- Which memories are relevant?
- Which past traces are useful?
- What should be compressed?
- What should be excluded for privacy or safety?
- What should be pinned into context?
- What should be retrieved only when needed?
- Which context sources are allowed under current policy?

Priority:

- [ ] Define memory interfaces.
- [ ] Add task-local working memory.
- [ ] Add persistent project memory.
- [ ] Add run trace memory.
- [ ] Add context snapshot export.
- [ ] Add memory diff viewer.
- [ ] Add privacy-aware context filtering.
- [ ] Add repo-aware context retrieval.
- [ ] Add artifact memory for generated outputs.
- [ ] Add memory reset and export controls.

---

### Layer 4: Durable Workflow Scheduler

Goal: support long-running, recoverable, multi-step agent workflows.

The runtime should support:

- task graph
- dependency graph
- plan/execute loop
- parallel workers
- checkpoints
- resume
- pause
- cancel
- retry
- fork
- rollback
- human approval gates
- scheduled or remote-triggered execution
- run status tracking
- workflow templates
- failure recovery policy

Example workflow:

```text
plan -> inspect repo -> edit code -> run tests -> diagnose failures -> fix -> summarize diff -> create PR
```

Example commands:

```bash
cheetah task create "fix failing pytest"
cheetah task status
cheetah task pause
cheetah task resume
cheetah task fork
cheetah task replay
```

Priority:

- [ ] Define task graph format.
- [ ] Add checkpoint and resume for task runs.
- [ ] Add retry policy.
- [ ] Add task fork/replay.
- [ ] Add parallel worker execution.
- [ ] Add human approval nodes.
- [ ] Add long-running autonomous mode with safety constraints.
- [ ] Add workflow template format.
- [ ] Add task status dashboard.
- [ ] Add cancellation and rollback.

---

### Layer 5: Observability, Audit, Trace, and Replay

Goal: make every agent run inspectable, debuggable, auditable, and reproducible.

CheetahClaws should provide a trace system that records:

- model used
- prompts and responses
- tool calls
- tool results
- shell commands
- file diffs
- errors and retries
- cost
- latency
- token usage
- context snapshots
- memory changes
- checkpoints
- human approvals
- policy decisions
- final artifacts
- run outcome

Core features:

- terminal trace summary
- HTML trace report
- web trace viewer
- run comparison
- replay from checkpoint
- failure diagnosis
- export to JSON/Markdown
- benchmark report generation
- compliance/audit export
- tool-call timeline
- file-diff timeline
- approval timeline

Example commands:

```bash
cheetah trace show runs/2026-04-25-001
cheetah trace export runs/2026-04-25-001 --format html
cheetah trace compare run_a run_b
cheetah replay run_a --from checkpoint_3
```

Priority:

- [ ] Define trace event schema.
- [ ] Log all model/tool/file/shell events.
- [ ] Add run summary.
- [ ] Add HTML trace export.
- [ ] Add run comparison.
- [ ] Add replay from checkpoint.
- [ ] Add failure taxonomy.
- [ ] Add approval and policy logs.
- [ ] Add incident review export.
- [ ] Add dashboard-ready metrics.

---

### Layer 6: Skill and Plugin Ecosystem

Goal: let users package reusable workflows, tools, prompts, policies, and evaluations.

A CheetahClaws skill should include:

- instructions
- tool requirements
- workflow steps
- examples
- tests
- permission requirements
- model requirements
- expected artifacts
- safety policy
- evaluation criteria

Example skill packs:

- `coding`
- `repo-debugging`
- `github-issue-resolver`
- `ci-failure-investigator`
- `research`
- `paper-reading`
- `latex-writing`
- `benchmarking`
- `vllm`
- `notebook`
- `safety-redteam`
- `agent-evaluation`
- `release-notes`
- `documentation-update`

Example commands:

```bash
cheetah skill list
cheetah skill install github:SafeRL-Lab/cc-research-skills
cheetah skill run paper-reading paper.pdf
cheetah skill test github-issue-resolver
```

Priority:

- [ ] Define skill package format.
- [ ] Add skill install/list/run commands.
- [ ] Add skill permission metadata.
- [ ] Add official skill packs.
- [ ] Add skill tests.
- [ ] Add security scan for skills.
- [ ] Add community contribution template.
- [ ] Add versioning for skills.
- [ ] Add workflow-template integration.
- [ ] Add enterprise-approved skill registry.

---

## 8. Industrial Product Roadmap

The open-source roadmap focuses on agent infrastructure.  
The industrial product roadmap focuses on deployment, reliability, governance, integrations, and ROI.

The industrial product should not start as a generic personal assistant.  
It should start as a secure agent runtime for engineering and research automation.

---

### 8.1 Industrial Product Positioning

Recommended positioning:

> **CheetahClaws Enterprise: Secure Agent Runtime for Engineering and Research Automation.**

Expanded description:

> CheetahClaws Enterprise helps teams run, govern, audit, and reproduce autonomous agents inside engineering and research workflows. It supports local-first deployment, multi-model execution, approval workflows, trace replay, and enterprise integrations.

---

### 8.2 Industrial MVP

The first industrial MVP should include five high-value capabilities:

1. **GitHub/GitLab issue resolver**
2. **CI failure investigator**
3. **Research experiment summarizer**
4. **Multi-model agent benchmark runner**
5. **Trace + audit + approval dashboard**

These are concrete, valuable, and measurable.

The MVP should avoid trying to be a full general-purpose enterprise assistant at the beginning.

---

### 8.3 Enterprise Deployment

Industrial users will need deployment options that fit their security and infrastructure requirements.

Required capabilities:

- Docker Compose deployment
- Kubernetes Helm chart
- VPC deployment
- on-prem deployment
- air-gapped deployment mode
- local model gateway
- OpenAI-compatible endpoint support
- config management
- workspace isolation
- data retention controls
- backup and restore
- upgrade path
- health check
- support bundle generation

Priority:

- [ ] Add Docker deployment.
- [ ] Add Helm chart.
- [ ] Add local model gateway documentation.
- [ ] Add private deployment guide.
- [ ] Add workspace isolation.
- [ ] Add configuration migration.
- [ ] Add health check command.
- [ ] Add backup/restore guide.

---

### 8.4 Identity, Access Control, and Governance

Industrial users need explicit identity and governance.

Required capabilities:

- SSO
- SAML/OIDC
- RBAC
- workspace-level permissions
- project-level permissions
- tool-level permissions
- model-level permissions
- approval policy
- audit trail
- admin console
- policy templates
- data access controls

Example roles:

| Role | Permissions |
|---|---|
| Viewer | View runs, traces, and reports. |
| Developer | Run approved workflows in assigned workspaces. |
| Approver | Approve file edits, shell commands, PRs, or external actions. |
| Admin | Configure policies, integrations, models, and workspaces. |
| Security Admin | Review audit logs, policy violations, and incident exports. |

Priority:

- [ ] Add workspace identity model.
- [ ] Add RBAC design.
- [ ] Add approval roles.
- [ ] Add policy configuration.
- [ ] Add admin audit view.
- [ ] Add SSO integration plan.
- [ ] Add enterprise policy templates.

---

### 8.5 Integration Hub

CheetahClaws should integrate with existing workflows instead of asking users to change everything.

Early integrations:

- GitHub
- GitLab
- Slack
- Jira
- Linear
- CI logs
- Google Drive
- Notion
- Confluence

Later integrations:

- Microsoft Teams
- Datadog
- Grafana
- PagerDuty
- Snowflake
- Postgres
- S3
- Kubernetes
- internal APIs
- customer-specific MCP servers

Priority:

- [ ] GitHub issue and PR integration.
- [ ] GitLab issue and MR integration.
- [ ] Slack notification and approval integration.
- [ ] Jira/Linear ticket integration.
- [ ] CI log ingestion.
- [ ] Google Drive/Notion document ingestion.
- [ ] Integration permission model.
- [ ] Integration audit logs.

---

### 8.6 Workflow Templates for Real Users

Industrial users want useful workflows, not only primitives.

Initial workflow templates:

| Workflow | Description |
|---|---|
| Fix CI Failure | Inspect CI logs, locate failure, propose fix, run tests, summarize patch. |
| Resolve GitHub Issue | Read issue, inspect repo, implement fix, generate PR summary. |
| Review PR | Review changed files, identify risks, suggest fixes, generate review comments. |
| Update Documentation | Detect API/code changes and update docs. |
| Generate Release Notes | Summarize merged PRs and commits into release notes. |
| Run Benchmark | Run model/code benchmark and generate report. |
| Summarize Experiments | Parse logs, extract metrics, generate tables and summary. |
| Migrate API Usage | Find deprecated API usage, update code, run tests. |
| Investigate Incident | Inspect logs/metrics/tickets and produce incident summary. |
| Research Report | Read papers/docs, extract key points, generate report draft. |

Priority:

- [ ] Build 3 stable engineering templates.
- [ ] Build 2 stable research templates.
- [ ] Add workflow template tests.
- [ ] Add workflow template documentation.
- [ ] Add template-level success metrics.
- [ ] Add template-level permission defaults.

---

### 8.7 Reliability, Evaluation, and Guardrails

Industrial product quality depends on measurable reliability.

Required metrics:

- task success rate
- test pass rate
- failure rate
- unsafe action rate
- command rejection rate
- approval rejection rate
- rollback rate
- human intervention rate
- cost per task
- latency per task
- token usage per task
- number of retries
- number of tool failures
- time saved estimate
- acceptance rate of generated PRs or reports

Guardrails:

- dangerous command blocking
- secret redaction
- model output validation
- tool result validation
- policy-based approval
- dry-run mode
- scoped workspace permissions
- external network control
- artifact review before publish
- rollback and restore

Priority:

- [ ] Define reliability metrics.
- [ ] Add workflow-level evaluation.
- [ ] Add model comparison dashboard.
- [ ] Add risk dashboard.
- [ ] Add ROI dashboard.
- [ ] Add guardrail test suite.
- [ ] Add regression tests for workflows.
- [ ] Add incident report generator.

---

### 8.8 Human-in-the-Loop Control

The first industrial product should emphasize controlled autonomy.

Required capabilities:

- approval queue
- approve/reject/edit actions
- diff-before-apply
- dry-run mode
- policy-based approval
- two-person approval for high-risk operations
- rollback button
- comments on agent actions
- human feedback memory
- escalation to human owner

Priority:

- [ ] Add approval queue.
- [ ] Add diff-before-apply.
- [ ] Add approve/reject/edit UI.
- [ ] Add policy-based approval rules.
- [ ] Add high-risk action escalation.
- [ ] Add human feedback logging.
- [ ] Add approval metrics.

---

### 8.9 Admin Console and Dashboards

Industrial users need visibility at team and organization level.

Dashboards:

- run dashboard
- workflow dashboard
- model performance dashboard
- cost dashboard
- risk dashboard
- ROI dashboard
- audit dashboard
- integration status dashboard
- failure analysis dashboard

Admin features:

- workspace management
- model configuration
- integration configuration
- permission policy
- approval policy
- secret management
- retention policy
- export controls

Priority:

- [ ] Add local web dashboard for traces.
- [ ] Add admin configuration page.
- [ ] Add model and integration status.
- [ ] Add workflow success dashboard.
- [ ] Add cost and latency dashboard.
- [ ] Add audit and approval dashboard.
- [ ] Add exportable reports.

---

### 8.10 Packaging and Supportability

Industrial users need maintainable software, not only code.

Required capabilities:

- stable release process
- versioned configs
- config migration
- upgrade guide
- deployment guide
- troubleshooting guide
- logs collection
- health checks
- backup/restore
- support bundle
- compatibility matrix
- example deployments
- security documentation

Priority:

- [ ] Add release checklist.
- [ ] Add config versioning.
- [ ] Add upgrade path.
- [ ] Add troubleshooting guide.
- [ ] Add health check command.
- [ ] Add support bundle command.
- [ ] Add compatibility matrix.
- [ ] Add security hardening guide.

---

## 9. Flagship Workflows

To avoid becoming a collection of disconnected features, CheetahClaws should focus on a small number of high-quality workflows.

---

### Workflow 1: Autonomous Coding Issue Resolver

Goal: given a GitHub issue or local bug report, CheetahClaws should inspect the repo, plan a fix, edit code, run tests, diagnose failures, and summarize the final patch.

Example:

```bash
cheetah solve-issue https://github.com/user/repo/issues/123
cheetah fix "pytest fails in tests/test_parser.py"
cheetah implement "add OAuth login support"
```

Expected artifacts:

- plan
- code diff
- test log
- failure diagnosis
- final summary
- optional PR description
- trace report

Milestones:

- [ ] Local issue resolver.
- [ ] Test-driven fix loop.
- [ ] Git diff summary.
- [ ] PR description generation.
- [ ] GitHub issue integration.
- [ ] GitLab issue integration.
- [ ] Benchmark on small repo tasks.
- [ ] Approval-before-PR mode.
- [ ] Trace and replay support.

---

### Workflow 2: CI Failure Investigator

Goal: reduce time spent debugging CI failures.

Example:

```bash
cheetah investigate-ci --run https://github.com/user/repo/actions/runs/123
cheetah fix-ci logs/failed_run.txt
```

Expected artifacts:

- failure summary
- likely root cause
- relevant files
- proposed fix
- test commands
- patch
- risk assessment

Milestones:

- [ ] CI log parser.
- [ ] GitHub Actions integration.
- [ ] GitLab CI integration.
- [ ] Local reproduction command generation.
- [ ] Failure classification.
- [ ] Patch proposal.
- [ ] Test rerun loop.
- [ ] PR summary.

---

### Workflow 3: Research Engineering Assistant

Goal: help researchers go from paper/project idea to code, experiments, results, and paper writing.

Example:

```bash
cheetah research "long-context privacy personalization benchmark"
cheetah read-paper paper.pdf
cheetah implement-paper arxiv_id
cheetah run-exp configs/*.yaml
cheetah make-latex-table results/*.jsonl
```

Expected artifacts:

- paper summary
- related work notes
- implementation plan
- experiment config
- scripts
- result tables
- LaTeX snippets
- BibTeX entries
- reproducible trace

Milestones:

- [ ] Paper reading skill.
- [ ] Related work skill.
- [ ] Experiment planning skill.
- [ ] LaTeX table/figure generation.
- [ ] Result summarization.
- [ ] Rebuttal assistant.
- [ ] Reproducible experiment trace.
- [ ] Benchmark report generation.
- [ ] Integration with local experiment logs.

---

### Workflow 4: Multi-Model Agent Benchmark

Goal: compare how different models perform under the same agent workflow.

Example:

```bash
cheetah benchmark --suite mini-swe --models claude,gpt,qwen-vllm,deepseek
cheetah benchmark --suite repo-debugging --models local-qwen,local-llama
cheetah report runs/benchmark-001
```

Metrics:

- task success rate
- test pass rate
- tool-call success rate
- edit success rate
- command failure rate
- hallucinated command rate
- unsafe command rate
- cost
- latency
- token usage
- context length usage
- number of retries
- human approvals required

Milestones:

- [ ] Define benchmark task format.
- [ ] Add benchmark runner.
- [ ] Add model comparison report.
- [ ] Add small built-in benchmark suite.
- [ ] Add local model benchmark templates.
- [ ] Add public leaderboard option.
- [ ] Add reproducibility checklist.
- [ ] Add trace export for benchmark runs.

---

### Workflow 5: Secure Internal Task Agent

Goal: support enterprise internal workflows with controlled tool/API access.

Example:

```bash
cheetah internal-task "summarize open Jira tickets for release 1.4"
cheetah internal-task "draft weekly engineering update"
cheetah internal-task "collect experiment results from shared folder"
```

Expected artifacts:

- retrieved sources
- action plan
- generated report
- approval record
- audit trail
- final artifact

Milestones:

- [ ] Slack approval integration.
- [ ] Jira/Linear integration.
- [ ] Google Drive/Notion integration.
- [ ] RBAC-aware tool access.
- [ ] Audit export.
- [ ] Admin policy configuration.
- [ ] Enterprise deployment template.

---

## 10. Development Phases

---

### Phase 0: Roadmap, Cleanup, and Positioning

Goal: clarify the project direction and reduce confusion.

Tasks:

- [ ] Add this `ROADMAP.md`.
- [ ] Update README positioning from personal assistant / Claude Code clone to Agent OS / agent runtime.
- [ ] Add a clear product statement for industrial users.
- [ ] Move non-core demos, such as trading, into `examples/`.
- [ ] Add `SECURITY.md`.
- [ ] Add `CONTRIBUTING.md` with roadmap-aligned contribution areas.
- [ ] Add architecture diagram.
- [ ] Add issue labels:
  - `runtime`
  - `model-provider`
  - `tool-runtime`
  - `memory`
  - `workflow`
  - `trace`
  - `audit`
  - `skill`
  - `security`
  - `benchmark`
  - `enterprise`
  - `integration`
  - `good-first-issue`

Success criteria:

- New contributors can understand the project direction within 3 minutes.
- README clearly explains what CheetahClaws is and is not.
- Roadmap provides concrete contribution paths.
- Industrial users can identify at least one concrete workflow that solves their problem.

---

### Phase 1: Stable Agent Runtime Core

Goal: stabilize the core runtime primitives.

Tasks:

- [ ] Define `AgentRuntime`.
- [ ] Define `ModelRuntime`.
- [ ] Define `ToolRuntime`.
- [ ] Define `TaskState`.
- [ ] Define `TraceEvent`.
- [ ] Normalize tool-call handling across providers.
- [ ] Add structured run logs.
- [ ] Add model/provider capability registry.
- [ ] Add basic cost/latency/token tracking.
- [ ] Add local configuration templates.
- [ ] Add OpenAI-compatible endpoint support.
- [ ] Add runtime test suite.

Success criteria:

- Same task can run across at least 5 providers.
- Each run produces a structured trace.
- Model/provider differences are abstracted behind a stable interface.
- Local models can run through the same workflow interface.

---

### Phase 2: Safe Tool Runtime

Goal: make CheetahClaws safe enough for autonomous workflows.

Tasks:

- [ ] Add permission modes.
- [ ] Add approval gates.
- [ ] Add command risk classifier.
- [ ] Add command allowlist/blocklist.
- [ ] Add secret redaction.
- [ ] Add file edit rollback.
- [ ] Add workspace sandbox.
- [ ] Add audit log.
- [ ] Add security documentation.
- [ ] Add dry-run mode.
- [ ] Add policy test suite.

Success criteria:

- Users can choose between read-only, approval-based, and autonomous modes.
- Dangerous operations are blocked or require approval.
- Every file edit and shell command is auditable.
- A failed or rejected tool action does not corrupt the workspace.

---

### Phase 3: Trace, Audit, and Replay

Goal: make agent behavior transparent, auditable, and reproducible early.

Tasks:

- [ ] Add trace event schema.
- [ ] Add terminal trace view.
- [ ] Add Markdown/JSON trace export.
- [ ] Add HTML trace report.
- [ ] Add file diff timeline.
- [ ] Add tool-call timeline.
- [ ] Add cost and latency summary.
- [ ] Add context snapshot inspection.
- [ ] Add run comparison.
- [ ] Add approval timeline.
- [ ] Add replay from checkpoint.

Success criteria:

- Users can understand why an agent succeeded or failed.
- Two runs with different models can be compared.
- Researchers can export traces for analysis.
- Industrial users can audit actions, approvals, and file changes.

---

### Phase 4: Durable Workflows

Goal: support long-running, recoverable agent tasks.

Tasks:

- [ ] Add task graph format.
- [ ] Add checkpoint/resume.
- [ ] Add task retry.
- [ ] Add task fork.
- [ ] Add task replay.
- [ ] Add task cancellation.
- [ ] Add parallel worker support.
- [ ] Add human approval nodes.
- [ ] Add CLI task management commands.
- [ ] Add workflow template format.

Success criteria:

- A failed task can resume from a checkpoint.
- Users can inspect and replay intermediate steps.
- Multi-step workflows are represented as explicit task graphs.
- Human approval can be inserted into workflows.

---

### Phase 5: Engineering Workflow Templates

Goal: provide immediate product value for developers and engineering teams.

Tasks:

- [ ] Build coding issue resolver.
- [ ] Build CI failure investigator.
- [ ] Build test-driven bug fixing workflow.
- [ ] Build PR review workflow.
- [ ] Build documentation update workflow.
- [ ] Build release notes workflow.
- [ ] Add GitHub/GitLab integration.
- [ ] Add workflow-level success metrics.

Success criteria:

- Users can solve real repo issues.
- Users can investigate CI failures with less manual effort.
- Workflows produce patches, tests, summaries, and traces.
- Generated PRs are reviewable and auditable.

---

### Phase 6: Research and Experiment Ops Workflows

Goal: provide immediate product value for research teams.

Tasks:

- [ ] Build paper reading workflow.
- [ ] Build related work workflow.
- [ ] Build experiment planning workflow.
- [ ] Build experiment log summarization.
- [ ] Build LaTeX table/figure generation workflow.
- [ ] Build benchmark report workflow.
- [ ] Build rebuttal assistant.
- [ ] Add reproducible experiment trace.

Success criteria:

- Researchers can use CheetahClaws for paper-to-code and experiment workflows.
- Experiment outputs can be transformed into tables and reports.
- Research workflows are reproducible and traceable.

---

### Phase 7: Multi-Model Benchmark Platform

Goal: make CheetahClaws a standard tool for evaluating agentic model ability.

Tasks:

- [ ] Define benchmark task format.
- [ ] Add benchmark runner.
- [ ] Add model comparison report.
- [ ] Add small built-in benchmark suite.
- [ ] Add local model benchmark templates.
- [ ] Add public leaderboard option.
- [ ] Add reproducibility checklist.
- [ ] Add benchmark trace export.
- [ ] Add failure taxonomy.

Success criteria:

- Users can compare models under the same workflow.
- Local models and proprietary models can be evaluated fairly.
- Reports include success, failure, cost, latency, and safety metrics.

---

### Phase 8: Enterprise Deployment and Integrations

Goal: make CheetahClaws deployable in real team and enterprise environments.

Tasks:

- [ ] Add Docker Compose deployment.
- [ ] Add Kubernetes Helm chart.
- [ ] Add VPC/on-prem deployment guide.
- [ ] Add local model gateway.
- [ ] Add workspace isolation.
- [ ] Add SSO/RBAC design.
- [ ] Add Slack approval integration.
- [ ] Add Jira/Linear integration.
- [ ] Add GitHub/GitLab enterprise integration.
- [ ] Add admin dashboard.
- [ ] Add health check and support bundle.

Success criteria:

- Teams can deploy CheetahClaws in private infrastructure.
- Admins can configure models, integrations, permissions, and policies.
- Workflows can be approved, audited, and monitored.

---

### Phase 9: Skill and Plugin Ecosystem

Goal: enable community and team extension.

Tasks:

- [ ] Define skill package format.
- [ ] Add skill install/list/run/test commands.
- [ ] Add official skill packs.
- [ ] Add skill permission metadata.
- [ ] Add security scan for skills.
- [ ] Add plugin registry design.
- [ ] Add community skill contribution guide.
- [ ] Add enterprise-approved skill registry.
- [ ] Add skill versioning.

Success criteria:

- Users can install and share reusable skills.
- Skills declare permissions and dependencies.
- Community contributions are easy to review.
- Enterprises can approve and manage internal skills.

---

### Phase 10: Team/Lab/Enterprise Agent OS

Goal: make CheetahClaws useful as shared agent infrastructure.

Tasks:

- [ ] Add multi-user workspace support.
- [ ] Add project-level memory.
- [ ] Add shared trace dashboard.
- [ ] Add team policy config.
- [ ] Add role-based permissions.
- [ ] Add remote worker nodes.
- [ ] Add lab/team deployment templates.
- [ ] Add optional web UI for monitoring.
- [ ] Add ROI dashboard.
- [ ] Add risk dashboard.

Success criteria:

- A lab or team can run CheetahClaws as shared agent infrastructure.
- Workflows, traces, and artifacts can be shared and reviewed.
- Permissions are configurable at project/team level.
- Teams can measure value, risk, reliability, and cost.

---

## 11. Success Metrics

CheetahClaws should be evaluated by real usefulness, not only features.

### Technical metrics

- model providers supported
- local model compatibility
- tool-call success rate
- workflow completion rate
- checkpoint recovery success
- replay success
- trace completeness
- benchmark reproducibility

### Product metrics

- time saved per task
- cost per task
- task success rate
- human intervention rate
- approval rejection rate
- PR acceptance rate
- test pass rate
- issue resolution time
- experiment iteration speed
- documentation update time

### Safety metrics

- unsafe action rate
- blocked risky command count
- secret redaction success
- rollback success rate
- policy violation rate
- audit completeness
- incident review time
- external network access violations

### Community metrics

- number of contributors
- number of official skills
- number of community skills
- number of supported workflow templates
- number of benchmark tasks
- issue response time
- documentation coverage
- examples that work end-to-end

---

## 12. Contribution Priorities

The community should prioritize contributions in this order:

1. **Runtime stability**
2. **Safety and permissions**
3. **Trace, audit, and observability**
4. **Durable workflows**
5. **Engineering workflow templates**
6. **Research workflow templates**
7. **Model provider support**
8. **Multi-model benchmark platform**
9. **Enterprise deployment**
10. **Integrations**
11. **Skill/plugin ecosystem**
12. **UI and remote bridges**
13. **Extra demos**

Avoid adding many unrelated demos before the runtime is stable and useful workflows exist.

---

## 13. Suggested GitHub Issues

### Good first issues

- Add `SECURITY.md`.
- Add roadmap badges to README.
- Add issue labels.
- Add provider capability table.
- Add simple trace JSON export.
- Add command allowlist/blocklist config.
- Add file edit rollback test.
- Add local vLLM configuration examples.
- Add skill template.
- Move non-core demos into `examples/`.
- Add Docker quickstart.
- Add workflow template documentation.
- Add README section for industrial use cases.
- Add simple CI log parser.

### Intermediate issues

- Implement `ModelRuntime` abstraction.
- Implement `ToolRuntime` permission layer.
- Implement structured `TraceEvent`.
- Implement checkpoint/resume for task runs.
- Implement HTML trace report.
- Implement benchmark task format.
- Implement coding issue resolver workflow.
- Implement CI failure investigator workflow.
- Implement GitHub issue integration.
- Implement approval queue.
- Implement cost and latency dashboard.
- Implement dry-run mode.

### Advanced issues

- Implement workflow scheduler.
- Implement multi-agent worker pool.
- Implement context constructor.
- Implement persistent memory.
- Implement replay from checkpoint.
- Implement sandbox execution.
- Implement skill security scanner.
- Implement multi-model benchmark runner.
- Implement RBAC and policy engine.
- Implement enterprise deployment templates.
- Implement trace dashboard.
- Implement integration hub.
- Implement ROI dashboard.

---

## 14. Recommended Repository Structure

A possible future structure:

```text
cheetahclaws/
  runtime/
    agent_runtime.py
    model_runtime.py
    tool_runtime.py
    workflow_runtime.py
    permission.py
    trace.py
    audit.py
  models/
    anthropic.py
    openai.py
    gemini.py
    qwen.py
    deepseek.py
    ollama.py
    vllm.py
    lmstudio.py
  tools/
    bash.py
    file_edit.py
    git.py
    python_exec.py
    notebook.py
    latex.py
    mcp.py
    ci_logs.py
    issue_tracker.py
  memory/
    working.py
    episodic.py
    semantic.py
    procedural.py
    artifacts.py
    team.py
  workflows/
    coding_issue_resolver/
    ci_failure_investigator/
    pr_review/
    research_assistant/
    experiment_ops/
    benchmark_runner/
    internal_task_agent/
  skills/
    coding/
    research/
    latex/
    vllm/
    safety/
    documentation/
    release_notes/
  security/
    sandbox.py
    secret_redaction.py
    command_policy.py
    approval.py
    rbac.py
  tracing/
    events.py
    exporter.py
    html_report.py
    replay.py
    comparison.py
  integrations/
    github/
    gitlab/
    slack/
    jira/
    linear/
    google_drive/
    notion/
    ci/
  enterprise/
    deployment/
    admin/
    policies/
    dashboards/
  benchmarks/
    tasks/
    runners/
    reports/
    leaderboards/
  examples/
    trading_agent/
    telegram_bridge/
    slack_bridge/
    voice_assistant/
  docs/
    architecture.md
    security.md
    deployment.md
    skills.md
    benchmarks.md
    enterprise.md
    workflows.md
```

This structure is only a proposal. The actual implementation should evolve incrementally.

---

## 15. Long-Term Direction

The high-level roadmap is:

```text
Phase 0: clarify positioning and roadmap
Phase 1: stabilize agent runtime core
Phase 2: build safe tool runtime
Phase 3: add trace, audit, and replay
Phase 4: add durable workflows
Phase 5: ship engineering workflow templates
Phase 6: ship research and experiment workflows
Phase 7: build multi-model benchmark platform
Phase 8: add enterprise deployment and integrations
Phase 9: build skill/plugin ecosystem
Phase 10: support teams, labs, and enterprise Agent OS
```

The long-term goal is:

> **CheetahClaws should become the Python-native, local-first Agent OS for researchers, developers, and engineering teams: a runtime that manages models, tools, memory, workflows, permissions, traces, and human approvals for secure, reproducible autonomous agents.**

---

## 16. Call for Contributors

We welcome contributions in the following areas:

- model provider integrations
- local model support
- tool runtime and permissions
- sandboxing and safety
- memory and context management
- workflow scheduling
- tracing and observability
- audit and approval workflows
- autonomous coding workflows
- CI failure investigation
- research automation workflows
- benchmark design
- enterprise deployment
- integration hub
- skill/plugin ecosystem
- documentation and examples

If you are unsure where to start, pick a `good-first-issue` or help improve the documentation, tests, and examples.

CheetahClaws is not just an assistant.  
It is becoming an open, hackable infrastructure layer for the next generation of autonomous agents.

The north star is simple:

> **Build agents that solve real problems, run safely, produce inspectable work, and can be trusted by developers, researchers, and teams.**
