# CheetahClaws Roadmap

> **CheetahClaws is evolving from a Python-native AI assistant into a local-first, multi-model Agent OS for autonomous coding, research workflows, and reproducible agent infrastructure.**

This document outlines the long-term direction, near-term milestones, and contribution priorities for the CheetahClaws community.

---

## 1. Vision

CheetahClaws aims to become a **Python-native Agent OS / Agent Runtime** for researchers, developers, and advanced users who want to build, run, inspect, control, and reproduce autonomous agents across different models and environments.

Instead of being only a coding assistant or a Claude Code reimplementation, CheetahClaws should provide the infrastructure layer for agentic workflows:

- unified model runtime
- safe tool execution
- memory and context management
- long-horizon workflow scheduling
- checkpoint, resume, rollback, and replay
- observability and trace inspection
- skill/plugin ecosystem
- local-first and multi-model experimentation

In product language, this is an **Agent OS**.  
In systems language, this is a **durable agent runtime infrastructure layer**.

---

## 2. Core Positioning

### What CheetahClaws is

CheetahClaws is:

- **Python-native**: easy to read, modify, extend, and research.
- **Local-first**: supports local models, local tools, local files, and local workflows.
- **Multi-model**: supports proprietary and open models through unified interfaces.
- **Hackable**: users can modify the agent loop, tools, memory, skills, and workflow logic.
- **Reproducible**: agent runs should be traceable, comparable, replayable, and debuggable.
- **Safety-aware**: tool execution, file editing, shell commands, and network access should be governed by explicit permissions.

### What CheetahClaws is not

CheetahClaws should not primarily be:

- only a Claude Code clone
- only a chatbot
- only a trading bot
- only a personal assistant
- only a collection of unrelated demos
- a closed, model-specific agent product

The project should focus on **developer and research agent infrastructure** first.

---

## 3. Design Philosophy

CheetahClaws should follow these principles:

1. **Runtime before app**  
   Build stable primitives for running agents before adding many user-facing demos.

2. **Workflows before prompts**  
   Useful agents need durable workflows, task graphs, checkpoints, tools, tests, and recovery mechanisms.

3. **Local-first by default**  
   Users should be able to run CheetahClaws with local models, local files, and local tools whenever possible.

4. **Multi-model by design**  
   The same workflow should be runnable with Claude, GPT, Gemini, Qwen, DeepSeek, Kimi, Zhipu, Ollama, vLLM, LM Studio, and other OpenAI-compatible endpoints.

5. **Safety is infrastructure**  
   Permission control, sandboxing, audit logs, and secret redaction should be core runtime features, not optional add-ons.

6. **Everything should be inspectable**  
   Agent traces, tool calls, file diffs, command logs, cost, latency, context snapshots, and memory changes should be visible.

7. **Research-friendly engineering**  
   CheetahClaws should make it easy to benchmark agents, compare models, reproduce results, and study failure modes.

---

## 4. Agent OS Primitives

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

These primitives should guide the architecture and roadmap.

---

## 5. Architecture Roadmap

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

Suggested permission modes:

| Mode | Description |
|---|---|
| `read-only` | Agent can inspect files but cannot modify or execute risky commands. |
| `approve-edits` | Agent must ask before modifying files. |
| `approve-bash` | Agent must ask before running shell commands. |
| `workspace-write` | Agent can edit files inside the workspace. |
| `sandboxed` | Agent runs commands only inside a controlled sandbox. |
| `full-auto` | Agent can act autonomously under configured policy. |

Priority:

- [ ] Add `SECURITY.md`.
- [ ] Add tool permission policy config.
- [ ] Add dangerous command detection.
- [ ] Add secret redaction.
- [ ] Add file edit rollback.
- [ ] Add per-tool audit logs.
- [ ] Add sandbox execution mode.

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

Context construction should answer:

- Which files are relevant?
- Which memories are relevant?
- Which past traces are useful?
- What should be compressed?
- What should be excluded for privacy or safety?
- What should be pinned into context?
- What should be retrieved only when needed?

Priority:

- [ ] Define memory interfaces.
- [ ] Add task-local working memory.
- [ ] Add persistent project memory.
- [ ] Add run trace memory.
- [ ] Add context snapshot export.
- [ ] Add memory diff viewer.
- [ ] Add privacy-aware context filtering.

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

---

### Layer 5: Observability, Trace, and Replay

Goal: make every agent run inspectable and debuggable.

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
- final artifacts

Core features:

- terminal trace summary
- HTML trace report
- web trace viewer
- run comparison
- replay from checkpoint
- failure diagnosis
- export to JSON/Markdown
- benchmark report generation

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

---

### Layer 6: Skill and Plugin Ecosystem

Goal: let users package reusable workflows, tools, prompts, and policies.

A CheetahClaws skill should include:

- instructions
- tool requirements
- workflow steps
- examples
- tests
- permission requirements
- model requirements
- expected artifacts

Example skill packs:

- `coding`
- `repo-debugging`
- `github-issue-resolver`
- `research`
- `paper-reading`
- `latex-writing`
- `benchmarking`
- `vllm`
- `notebook`
- `safety-redteam`
- `agent-evaluation`

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

---

## 6. Flagship Workflows

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

Milestones:

- [ ] Local issue resolver.
- [ ] Test-driven fix loop.
- [ ] Git diff summary.
- [ ] PR description generation.
- [ ] GitHub issue integration.
- [ ] Benchmark on small repo tasks.

---

### Workflow 2: Research Engineering Assistant

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

Milestones:

- [ ] Paper reading skill.
- [ ] Related work skill.
- [ ] Experiment planning skill.
- [ ] LaTeX table/figure generation.
- [ ] Result summarization.
- [ ] Rebuttal assistant.
- [ ] Reproducible experiment trace.

---

### Workflow 3: Multi-Model Agent Benchmark

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

---

## 7. Development Phases

---

### Phase 0: Roadmap, Cleanup, and Positioning

Goal: clarify the project direction and reduce confusion.

Tasks:

- [ ] Add this `ROADMAP.md`.
- [ ] Update README positioning from personal assistant / Claude Code clone to Agent OS / agent runtime.
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
  - `skill`
  - `security`
  - `benchmark`
  - `good-first-issue`

Success criteria:

- New contributors can understand the project direction within 3 minutes.
- README clearly explains what CheetahClaws is and is not.
- Roadmap provides concrete contribution paths.

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

Success criteria:

- Same task can run across at least 5 providers.
- Each run produces a structured trace.
- Model/provider differences are abstracted behind a stable interface.

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

Success criteria:

- Users can choose between read-only, approval-based, and autonomous modes.
- Dangerous operations are blocked or require approval.
- Every file edit and shell command is auditable.

---

### Phase 3: Durable Workflows

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

Success criteria:

- A failed task can resume from a checkpoint.
- Users can inspect and replay intermediate steps.
- Multi-step workflows are represented as explicit task graphs.

---

### Phase 4: Observability and Trace Viewer

Goal: make agent behavior transparent and debuggable.

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

Success criteria:

- Users can understand why an agent succeeded or failed.
- Two runs with different models can be compared.
- Researchers can export traces for analysis.

---

### Phase 5: Flagship Workflows

Goal: provide highly useful workflows that demonstrate the Agent OS.

Tasks:

- [ ] Build coding issue resolver.
- [ ] Build test-driven bug fixing workflow.
- [ ] Build research paper reading workflow.
- [ ] Build experiment automation workflow.
- [ ] Build LaTeX table/figure generation workflow.
- [ ] Build multi-model benchmark workflow.

Success criteria:

- Users can solve real repo issues.
- Researchers can use CheetahClaws for paper-to-code and experiment workflows.
- Benchmarks produce reproducible reports.

---

### Phase 6: Skill and Plugin Ecosystem

Goal: enable community extension.

Tasks:

- [ ] Define skill package format.
- [ ] Add skill install/list/run/test commands.
- [ ] Add official skill packs.
- [ ] Add skill permission metadata.
- [ ] Add security scan for skills.
- [ ] Add plugin registry design.
- [ ] Add community skill contribution guide.

Success criteria:

- Users can install and share reusable skills.
- Skills declare permissions and dependencies.
- Community contributions are easy to review.

---

### Phase 7: Agent OS for Teams and Labs

Goal: make CheetahClaws useful for research labs, teams, and shared infrastructure.

Tasks:

- [ ] Add multi-user workspace support.
- [ ] Add project-level memory.
- [ ] Add shared trace dashboard.
- [ ] Add team policy config.
- [ ] Add role-based permissions.
- [ ] Add remote worker nodes.
- [ ] Add lab/team deployment templates.
- [ ] Add optional web UI for monitoring.

Success criteria:

- A lab or team can run CheetahClaws as shared agent infrastructure.
- Workflows, traces, and artifacts can be shared and reviewed.
- Permissions are configurable at project/team level.

---

## 8. Contribution Priorities

The community should prioritize contributions in this order:

1. **Runtime stability**
2. **Safety and permissions**
3. **Trace and observability**
4. **Durable workflows**
5. **Model provider support**
6. **Flagship workflows**
7. **Skill/plugin ecosystem**
8. **UI and remote bridges**
9. **Extra demos**

Avoid adding many unrelated demos before the runtime is stable.

---

## 9. Suggested GitHub Issues

Good first issues:

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

Intermediate issues:

- Implement `ModelRuntime` abstraction.
- Implement `ToolRuntime` permission layer.
- Implement structured `TraceEvent`.
- Implement checkpoint/resume for task runs.
- Implement HTML trace report.
- Implement benchmark task format.
- Implement coding issue resolver workflow.

Advanced issues:

- Implement workflow scheduler.
- Implement multi-agent worker pool.
- Implement context constructor.
- Implement persistent memory.
- Implement replay from checkpoint.
- Implement sandbox execution.
- Implement skill security scanner.
- Implement multi-model benchmark runner.

---

## 10. Recommended Repository Structure

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
  models/
    anthropic.py
    openai.py
    gemini.py
    qwen.py
    deepseek.py
    ollama.py
    vllm.py
  tools/
    bash.py
    file_edit.py
    git.py
    python_exec.py
    notebook.py
    latex.py
    mcp.py
  memory/
    working.py
    episodic.py
    semantic.py
    procedural.py
    artifacts.py
  workflows/
    coding_issue_resolver/
    research_assistant/
    benchmark_runner/
  skills/
    coding/
    research/
    latex/
    vllm/
    safety/
  security/
    sandbox.py
    secret_redaction.py
    command_policy.py
  tracing/
    events.py
    exporter.py
    html_report.py
    replay.py
  benchmarks/
    tasks/
    runners/
    reports/
  examples/
    trading_agent/
    telegram_bridge/
    slack_bridge/
    voice_assistant/
  docs/
    architecture.md
    security.md
    skills.md
    benchmarks.md
```

This structure is only a proposal. The actual implementation should evolve incrementally.

---

## 11. Roadmap Summary

The high-level roadmap is:

```text
Phase 0: clarify positioning and roadmap
Phase 1: stabilize agent runtime core
Phase 2: build safe tool runtime
Phase 3: add durable workflows
Phase 4: add trace, observability, and replay
Phase 5: ship flagship coding and research workflows
Phase 6: build skill/plugin ecosystem
Phase 7: support teams and labs
```

The long-term goal is:

> **CheetahClaws should become the Python-native, local-first Agent OS for researchers and advanced developers: a runtime that manages models, tools, memory, workflows, permissions, and traces for reproducible autonomous agents.**

---

## 12. Call for Contributors

We welcome contributions in the following areas:

- model provider integrations
- local model support
- tool runtime and permissions
- sandboxing and safety
- memory and context management
- workflow scheduling
- tracing and observability
- autonomous coding workflows
- research automation workflows
- benchmark design
- skill/plugin ecosystem
- documentation and examples

If you are unsure where to start, pick a `good-first-issue` or help improve the documentation, tests, and examples.

CheetahClaws is not just an assistant.  
It is becoming an open, hackable infrastructure layer for the next generation of autonomous agents.
