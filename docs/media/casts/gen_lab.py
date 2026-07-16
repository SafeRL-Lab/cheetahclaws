"""Asciinema v2 cast: /lab autonomous multi-agent paper writing.

Scenario: /lab start on iris classification comparison. Show the 9 stages
advancing with agent messages and reviewer iteration. End with the
deliverable file tree.

Run: python3 gen_lab.py > lab.cast
"""
import json
import random
import sys


HEADER = {
    "version": 2,
    "width": 110,
    "height": 34,
    "timestamp": 1747262400,
    "env": {"SHELL": "/bin/zsh", "TERM": "xterm-256color"},
    "title": "CheetahClaws /lab — 9 agents drive a paper from question to PDF",
    "idle_time_limit": 1.3,
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
    rng = random.Random(23)
    for ch in s:
        out(base + rng.random() * jitter, ch)


# Scene 1 — launch + /lab start
out(0.0, f"{GREEN}~{RST} {CYAN}❯{RST} cheetahclaws\r\n")
out(0.3, f"{DIM}[CheetahClaws v3.5.79 · claude-sonnet-4-6][/lab engine v0]{RST}\r\n\r\n")
out(0.2, f"{BOLD}{CYAN}[~] »{RST} ")
out(0.5, "")
type_string("/lab start \"Compare logistic regression vs random forest on iris, k-fold CV\"")
out(0.4, "\r\n")
out(0.4, f"{GREEN}✓{RST}  Lab run {BOLD}lab_a3b1c8e9f012{RST} launched. Budget: 60 min · 200k tokens.\r\n\r\n")

# Scene 2 — stage pipeline
def stage(name, agent_color, agent_name, msg, ok=True):
    badge_color = GREEN if ok else YELL
    out(0.5, f"  {badge_color}[{name}]{RST}\r\n")
    out(0.3, f"    {agent_color}● {agent_name}{RST}: {msg}\r\n")

stage("QUESTIONING", BLUE, "PI",
      "Picking Q2: 'Does RF outperform logistic regression on iris under 5-fold CV?'")
out(0.1, f"    {DIM}● Lay Reader: question is concrete and testable.{RST}\r\n\r\n")

stage("SURVEY", MAG, "Surveyor",
      "12 papers retrieved; baselines on iris well-characterised since 1936.")
out(0.1, "\r\n")

stage("OUTLINE", BLUE, "Designer",
      "5-section outline: intro, related, method, results, threats.")
out(0.1, f"    {DIM}● Reviewer×3 critique → 2 pass, 1 asks for ablation; PI signs off.{RST}\r\n\r\n")

stage("CODE_DRAFT", YELL, "Engineer",
      "scripted iris loader, GridSearchCV for both models, 5-fold stratified.")
out(0.1, "\r\n")

stage("EXPERIMENT", YELL, "Engineer",
      "Running sandboxed subprocess…")
out(0.4, f"      {DIM}stdout: Best LR C=10  acc=0.967 ± 0.025{RST}\r\n")
out(0.3, f"      {DIM}stdout: Best RF n=50 acc=0.967 ± 0.033{RST}\r\n")
out(0.2, f"      {DIM}saved figure_1.png (boxplot), results.csv{RST}\r\n\r\n")

stage("ANALYSIS", YELL, "Engineer",
      "Models tie on accuracy; RF has higher variance. Recommend LR for tabular small-n.")
out(0.1, "\r\n")

stage("DRAFTING", CYAN, "Drafter",
      "Composed 2,840-word draft with inline [1]–[12] citations.")
out(0.1, "\r\n")

# Reviewer loop
out(0.4, f"  {GREEN}[REVIEW LOOP]{RST}\r\n")
out(0.3, f"    {RED}● Reviewer #1{RST}: 'Section 3.2 doesn't address class imbalance — minor revision.'\r\n")
out(0.3, f"    {RED}● Reviewer #2{RST}: 'Threats section thin. Add overfitting note.'\r\n")
out(0.3, f"    {GREEN}● Reviewer #3{RST}: 'Accept.'\r\n")
out(0.4, f"    {CYAN}● Drafter{RST}: revised §3.2 + §6, rebuilt bib.\r\n")
out(0.3, f"    {GREEN}● Reviewer×3{RST}: {BOLD}2/3 accept on round 2{RST} → PI signs off.\r\n\r\n")

stage("CITATION VERIFY", GREEN, "Citation Checker",
      "12/12 references verified against arXiv / Semantic Scholar / CrossRef.")
out(0.1, "\r\n")

# Final
out(0.4, f"  {GREEN}[FINALISE]{RST}  Bundle ready.\r\n\r\n")
out(0.4, f"{GREEN}✓{RST}  Output at {CYAN}~/.cheetahclaws/research_papers/lab_a3b1c8e9f012/{RST}\r\n\r\n")
out(0.3, f"    ├── {BOLD}report.md{RST}              {DIM}(2,940 words, 12 refs){RST}\r\n")
out(0.15, f"    ├── references.bib       {DIM}(verified BibTeX){RST}\r\n")
out(0.15, f"    ├── citations_verified.json\r\n")
out(0.15, f"    └── workspace/\r\n")
out(0.15, f"        ├── experiment.py     {DIM}(83 lines){RST}\r\n")
out(0.15, f"        ├── figure_1.png      {DIM}(boxplot){RST}\r\n")
out(0.15, f"        └── results.csv       {DIM}(5 folds × 2 models){RST}\r\n\r\n")

out(0.5, f"{DIM}Total: 22 min · 142k tokens · $1.40 in API cost{RST}\r\n\r\n")
out(0.4, f"{BOLD}{CYAN}[~] »{RST} ")
out(0.8, "")

sys.stdout.write(json.dumps(HEADER) + "\n")
for ev in events:
    sys.stdout.write(json.dumps(ev) + "\n")
