"""Asciinema v2 cast: 20-source research pipeline.

Scenario: /research "LLM agents 2026" fans out across arXiv, HuggingFace,
Semantic Scholar, HackerNews, GitHub, Reddit, 知乎, B站, 微博, 小红书.
Shows live source completion, entity heat table, citation pull.

Run: python3 gen_research.py > research.cast
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
    "title": "CheetahClaws /research — parallel fan-out across 20 sources",
    "idle_time_limit": 1.5,
}

CYAN  = "[36m"
GREEN = "[32m"
YELL  = "[33m"
MAG   = "[35m"
DIM   = "[2m"
BOLD  = "[1m"
GRAY  = "[90m"
RED   = "[31m"
BLUE  = "[34m"
RST   = "[0m"

events = []
t = 0.0


def out(delay, text):
    global t
    t += delay
    events.append([round(t, 3), "o", text])


def type_string(s, base=0.04, jitter=0.02):
    rng = random.Random(13)
    for ch in s:
        out(base + rng.random() * jitter, ch)


# Scene 1 — launch + /research
out(0.0, f"{GREEN}~{RST} {CYAN}❯{RST} ")
out(0.6, "")
type_string("cheetahclaws")
out(0.4, "\r\n")
out(0.3, f"{DIM}[CheetahClaws v3.5.79 · claude-sonnet-4-6]{RST}\r\n\r\n")
out(0.2, f"{BOLD}{CYAN}[~] »{RST} ")
out(0.5, "")
type_string("/research \"LLM agents 2026 trends\" --range 6m --expand")
out(0.4, "\r\n\r\n")

# Scene 2 — query expansion
out(0.5, f"{DIM}● Expanding query into 4 sibling sub-queries…{RST}\r\n")
sub_qs = [
    "autonomous coding agents benchmarks 2026",
    "multi-agent debate / reviewer-author loops",
    "agentic tool use plus MCP / function calling",
    "open-weight models for agent workflows",
]
for q in sub_qs:
    out(0.25, f"  {DIM}↳{RST} {q}\r\n")
out(0.4, "\r\n")

# Scene 3 — fan-out, live source completions
out(0.4, f"{BOLD}Fanning out across 20 sources in parallel…{RST}\r\n\r\n")

sources = [
    ("arXiv",          "342",  GREEN, 0.35),
    ("Semantic Scholar","218", GREEN, 0.25),
    ("HuggingFace Papers","176",GREEN, 0.20),
    ("OpenAlex",       "412",  GREEN, 0.30),
    ("alphaXiv",       " 84",  GREEN, 0.22),
    ("HackerNews",     "511",  GREEN, 0.20),
    ("GitHub",         "298",  GREEN, 0.28),
    ("Reddit r/MachineLearning","147", GREEN, 0.18),
    ("StackOverflow",  " 62",  GREEN, 0.15),
    ("Google News",    "203",  GREEN, 0.25),
    ("Polymarket",     "  9",  GREEN, 0.18),
    ("SEC EDGAR",      " 14",  GREEN, 0.20),
    ("Twitter / X",    "1.2k", YELL,  0.30),
    ("Brave Search",   "188",  GREEN, 0.18),
    ("Tavily",         "151",  GREEN, 0.20),
    ("Google Scholar", "224",  GREEN, 0.25),
    ("知乎 Zhihu",     "186",  GREEN, 0.22),
    ("B站 Bilibili",   "298",  GREEN, 0.25),
    ("微博 Weibo",     "412",  YELL,  0.30),
    ("小红书 Xiaohongshu","127",GREEN, 0.20),
]
for name, count, color, delay in sources:
    out(delay, f"  {color}✓{RST}  {BOLD}{name:<28}{RST} {DIM}→{RST} {count:>5} hits\r\n")

out(0.4, "\r\n")

# Scene 4 — entity heat table
out(0.6, f"{BOLD}Top entities by cross-platform attention:{RST}\r\n\r\n")
out(0.3, f"  {DIM}entity              arXiv   HF   GH   HN   微博   Zhihu   total{RST}\r\n")
out(0.2, f"  {DIM}{'─'*70}{RST}\r\n")
rows = [
    ("Claude 4.6",      "127", " 84", " 32", "298", " 412", " 186", "  1,139"),
    ("DeepSeek V4",     "108", "112", "147", "176", " 287", " 298", "  1,128"),
    ("Qwen3-Coder",     " 87", " 92", "211", " 88", " 154", " 287", "    919"),
    ("MCP Protocol",    " 42", " 28", "188", "247", "  64", "  72", "    641"),
    ("Llama 4",         " 96", "118", " 94", "203", " 167", "  88", "    766"),
]
for ent, *vals in rows:
    cells = "  ".join(f"{v:>5}" for v in vals[:-1])
    total = vals[-1]
    out(0.25, f"  {BOLD}{ent:<18}{RST} {DIM}{cells}{RST}    {GREEN}{total}{RST}\r\n")

out(0.5, "\r\n")

# Scene 5 — citations verified
out(0.5, f"{DIM}● Verifying citations against arXiv / Semantic Scholar / CrossRef…{RST}\r\n")
out(0.4, f"  {GREEN}✓{RST}  47 papers, {GREEN}45 verified{RST}, {RED}2 flagged for hallucination{RST}\r\n\r\n")

# Scene 6 — report saved
out(0.4, f"{GREEN}✓{RST}  Brief saved → {CYAN}~/.cheetahclaws/research_reports/llm-agents-2026-trends-{t:.0f}.md{RST}\r\n")
out(0.2, f"   {DIM}3,124 words · 47 citations · cross-platform heat table · 12-month trend sparkline{RST}\r\n\r\n")

# Scene 7 — follow-up
out(0.5, f"{BOLD}{CYAN}[~] »{RST} ")
out(0.5, "")
type_string("/reports open")
out(0.4, "\r\n")
out(0.4, f"{DIM}Opening llm-agents-2026-trends-{t:.0f}.md in your editor…{RST}\r\n\r\n")
out(0.5, f"{BOLD}{CYAN}[~] »{RST} ")
out(0.7, "")

sys.stdout.write(json.dumps(HEADER) + "\n")
for ev in events:
    sys.stdout.write(json.dumps(ev) + "\n")
