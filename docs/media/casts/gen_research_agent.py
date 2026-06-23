"""Asciinema v2 cast: /agent research_assistant — autonomous background loop.

Scenario: launch the research_assistant agent template on a topic; show
three iteration cycles with summaries pushed to the bridge, then the
stagnation-stop guard kicking in to save tokens.

Run: python3 gen_research_agent.py > research_agent.cast
"""
import json
import random
import sys


HEADER = {
    "version": 2,
    "width": 110,
    "height": 32,
    "timestamp": 1747262400,
    "env": {"SHELL": "/bin/zsh", "TERM": "xterm-256color"},
    "title": "CheetahClaws /agent — autonomous research_assistant loop",
    "idle_time_limit": 1.5,
}

CYAN = "[36m"
GREEN = "[32m"
YELL = "[33m"
MAG = "[35m"
DIM = "[2m"
BOLD = "[1m"
GRAY = "[90m"
RED = "[31m"
BLUE = "[34m"
RST = "[0m"

events = []
t = 0.0


def out(delay, text):
    global t
    t += delay
    events.append([round(t, 3), "o", text])


def type_string(s, base=0.04, jitter=0.02):
    rng = random.Random(31)
    for ch in s:
        out(base + rng.random() * jitter, ch)


# Scene 1 — launch + /agent wizard
out(0.0, f"{GREEN}~{RST} {CYAN}❯{RST} cheetahclaws\r\n")
out(0.3, f"{DIM}[CheetahClaws v3.5.79 · claude-sonnet-4-6]{RST}\r\n\r\n")
out(0.2, f"{BOLD}{CYAN}[~] »{RST} ")
out(0.4, "")
type_string("/agent")
out(0.4, "\r\n\r\n")

out(0.3, f"{BOLD}Pick an agent template:{RST}\r\n")
out(0.2, f"  1. {CYAN}research_assistant{RST}  {DIM}— daily literature & trend digest{RST}\r\n")
out(0.15, f"  2. {CYAN}auto_bug_fixer{RST}      {DIM}— scan repo, propose fixes, run tests{RST}\r\n")
out(0.15, f"  3. {CYAN}paper_writer{RST}        {DIM}— draft & polish a paper section by section{RST}\r\n")
out(0.15, f"  4. {CYAN}auto_coder{RST}          {DIM}— implement TODOs from a backlog file{RST}\r\n")
out(0.15, f"  {DIM}(or drop a .md into ~/.cheetahclaws/agent_templates/ for a custom one){RST}\r\n\r\n")
out(0.3, f"{BOLD}Choose [1-4]:{RST} ")
out(0.5, "")
type_string("1")
out(0.4, "\r\n")
out(0.3, f"{BOLD}Topic for research_assistant:{RST} ")
out(0.4, "")
type_string("Multi-agent debate vs single-model — papers from the last 30 days")
out(0.4, "\r\n\r\n")

out(0.4, f"{GREEN}✓{RST}  Agent {BOLD}research_assistant_8f3a2c{RST} started — loop every 4 hours · push to Telegram\r\n")
out(0.2, f"{DIM}    Output dir: ~/.cheetahclaws/agents/research_assistant_8f3a2c/output/{RST}\r\n\r\n")

# Scene 2 — iteration 1
def iteration(n, ts, color, summary_lines, stagnation=False):
    badge = YELL if stagnation else GREEN
    out(0.5, f"  {color}─── Iteration #{n} ─── {DIM}{ts}{RST}\r\n")
    for line in summary_lines:
        out(0.25, f"    {line}\r\n")
    out(0.3, f"    {DIM}→ pushed iteration summary to Telegram chat 458291205{RST}\r\n\r\n")

iteration(1, "11:00 PT", CYAN, [
    f"{YELL}[Read]{RST} ~/.cheetahclaws/agents/.../state.json  {DIM}(first run, empty){RST}",
    f"{YELL}[research]{RST} fanned out across 20 sources for the last 24h",
    f"{GREEN}● Found 17 new papers, 3 high-signal:{RST}",
    f"    {DIM}•{RST} \"AdvDebate: …\" (arXiv 2605.04123) — adversarial multi-agent debate",
    f"    {DIM}•{RST} \"OneShot or N: …\" (arXiv 2605.04588) — single-model can rival debate",
    f"    {DIM}•{RST} \"Skeptic Loop: …\" (Reddit + GitHub) — open-source debate framework",
    f"{YELL}[Write]{RST} digest_day_1.md saved to output/",
])

iteration(2, "15:00 PT", MAG, [
    f"{YELL}[Read]{RST} state.json  {DIM}(last digest: digest_day_1.md){RST}",
    f"{YELL}[research]{RST} new since 11:00 → 4 papers, 1 high-signal",
    f"{GREEN}● Notable:{RST} \"Beyond Debate: …\" (NeurIPS workshop preprint)",
    f"    {DIM}— suggests debate gains shrink as base model gets larger{RST}",
    f"{YELL}[Write]{RST} digest_day_1.md (appended)",
])

iteration(3, "19:00 PT", BLUE, [
    f"{YELL}[research]{RST} new since 15:00 → 0 papers (quiet window)",
    f"{DIM}● No new high-signal items. Reused yesterday's analysis.{RST}",
    f"{YELL}[Write]{RST} digest_day_1.md (timestamp updated)",
])

# Scene 3 — stagnation-stop kicks in
out(0.5, f"  {YELL}─── Iteration #4 ─── {DIM}23:00 PT{RST}\r\n")
out(0.3, f"    {YELL}[research]{RST} 0 new papers · summary identical to #3\r\n")
out(0.3, f"    {RED}● Stagnation-stop:{RST} same summary for 3 iterations in a row.\r\n")
out(0.2, f"    {DIM}      threshold: auto_agent_dup_summary_limit = 3 (set 0 to disable){RST}\r\n")
out(0.3, f"    {YELL}● Loop paused.{RST} Next attempt at 09:00 PT (manual or /agent resume).\r\n\r\n")

# Scene 4 — output summary
out(0.4, f"{BOLD}Output (so far):{RST}\r\n")
out(0.25, f"  ~/.cheetahclaws/agents/research_assistant_8f3a2c/output/\r\n")
out(0.2, f"    ├── {BOLD}digest_day_1.md{RST}   {DIM}(2.4 KB, 4 papers analysed){RST}\r\n")
out(0.2, f"    ├── state.json         {DIM}(loop bookkeeping){RST}\r\n")
out(0.2, f"    └── notes.md           {DIM}(running scratchpad){RST}\r\n\r\n")

out(0.4, f"{DIM}Three iterations · 38k tokens · $0.31. Saved ~$0.90 in API spend by stopping.{RST}\r\n\r\n")
out(0.4, f"{BOLD}{CYAN}[~] »{RST} ")
out(0.8, "")

sys.stdout.write(json.dumps(HEADER) + "\n")
for ev in events:
    sys.stdout.write(json.dumps(ev) + "\n")
