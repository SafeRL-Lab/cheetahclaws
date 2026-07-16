"""Asciinema v2 cast: code review workflow.

Scenario: ask CheetahClaws to find a performance bug in a Python script,
switch to a local Ollama model, then apply the fix.

Run: python3 gen_code_review.py > code_review.cast
"""
import json
import random
import sys


HEADER = {
    "version": 2,
    "width": 100,
    "height": 28,
    "timestamp": 1747262400,
    "env": {"SHELL": "/bin/zsh", "TERM": "xterm-256color"},
    "title": "CheetahClaws — find a perf bug, switch to local Ollama, apply the fix",
    "idle_time_limit": 1.2,
}

CYAN  = "[36m"
GREEN = "[32m"
YELL  = "[33m"
MAG   = "[35m"
DIM   = "[2m"
BOLD  = "[1m"
GRAY  = "[90m"
RED   = "[31m"
RST   = "[0m"

events = []
t = 0.0


def out(delay, text):
    global t
    t += delay
    events.append([round(t, 3), "o", text])


def type_string(s, base=0.04, jitter=0.02):
    rng = random.Random(11)
    for ch in s:
        out(base + rng.random() * jitter, ch)


# Scene 1 — launch
out(0.0, f"{GREEN}~/projects/parser{RST} {CYAN}❯{RST} ")
out(0.7, "")
type_string("cheetahclaws")
out(0.4, "\r\n")
out(0.3, f"{DIM}[CheetahClaws v3.5.79 · claude-sonnet-4-6 · auto mode]{RST}\r\n")
out(0.2, f"{DIM}Type /help for commands, /model to switch, !cmd for shell, Ctrl+C to quit{RST}\r\n\r\n")
out(0.2, f"{BOLD}{CYAN}[project] »{RST} ")

# Scene 2 — prompt
out(0.6, "")
type_string("Why is parse_logs() so slow on the 1GB sample file? Profile and fix it.")
out(0.5, "\r\n\r\n")

# Scene 3 — tool calls
out(0.5, f"{DIM}● Reading the function and tracing hot spots…{RST}\r\n")
out(0.4, f"{YELL}[Read]{RST} src/parser.py {DIM}(187 lines){RST}\r\n")
out(0.4, f"{YELL}[Grep]{RST} 'for .* in' src/parser.py {DIM}(7 matches){RST}\r\n")
out(0.4, f"{YELL}[Bash]{RST} python -X importtime -c 'from parser import parse_logs; parse_logs(\"sample.log\")'\r\n")
out(0.5, f"  {DIM}real  18.42s   user  17.99s   sys  0.41s{RST}\r\n\r\n")

# Scene 4 — diagnosis
out(0.6, f"{BOLD}Found the hot path:{RST}\r\n\r\n")
out(0.4, f"  {RED}line 84{RST}  {DIM}re.compile(PATTERN){RST}     called inside the loop\r\n")
out(0.3, f"  {RED}line 97{RST}  {DIM}result += line{RST}          string concatenation in O(n²)\r\n")
out(0.3, f"  {RED}line 112{RST} {DIM}open(path).read(){RST}       loads the whole 1GB into RAM\r\n\r\n")
out(0.5, f"{BOLD}3 fixes will bring it under 2 seconds.{RST} I'll show them before applying.\r\n\r\n")

# Scene 5 — switch to local model
out(0.5, f"{BOLD}{CYAN}[project] »{RST} ")
out(0.4, "")
type_string("/model ollama/qwen2.5-coder")
out(0.4, "\r\n")
out(0.4, f"{GREEN}✓{RST} Switched to {BOLD}ollama/qwen2.5-coder{RST} {DIM}(local, $0/call){RST}\r\n\r\n")

# Scene 6 — apply fix
out(0.3, f"{BOLD}{CYAN}[project] »{RST} ")
out(0.5, "")
type_string("apply all three fixes")
out(0.4, "\r\n\r\n")

out(0.5, f"{YELL}[Edit]{RST} src/parser.py\r\n")
out(0.4, f"  {RED}- PATTERN = r\"\\\\[(\\\\d{{4}}-\\\\d{{2}}-\\\\d{{2}})\\\\] (.*)\"{RST}\r\n")
out(0.3, f"  {RED}- def parse_logs(path):{RST}\r\n")
out(0.3, f"  {RED}-     for line in open(path).read().splitlines():{RST}\r\n")
out(0.3, f"  {RED}-         m = re.compile(PATTERN).match(line){RST}\r\n")
out(0.3, f"  {GREEN}+ _PATTERN_RE = re.compile(r\"\\\\[(\\\\d{{4}}-\\\\d{{2}}-\\\\d{{2}})\\\\] (.*)\"){RST}\r\n")
out(0.3, f"  {GREEN}+ def parse_logs(path):{RST}\r\n")
out(0.3, f"  {GREEN}+     out = []{RST}\r\n")
out(0.3, f"  {GREEN}+     with open(path) as fh:{RST}\r\n")
out(0.3, f"  {GREEN}+         for line in fh:{RST}\r\n")
out(0.3, f"  {GREEN}+             m = _PATTERN_RE.match(line){RST}\r\n\r\n")
out(0.4, f"{GREEN}✓{RST} 1 file changed, 9 insertions(+), 4 deletions(-).\r\n\r\n")

# Scene 7 — re-bench
out(0.5, f"{YELL}[Bash]{RST} python -c 'from parser import parse_logs; parse_logs(\"sample.log\")'\r\n")
out(0.6, f"  {DIM}real  1.62s    user  1.51s    sys  0.09s{RST}   {GREEN}↓ 11×{RST}\r\n\r\n")
out(0.5, f"{BOLD}{CYAN}[project] »{RST} ")
out(0.8, "")

sys.stdout.write(json.dumps(HEADER) + "\n")
for ev in events:
    sys.stdout.write(json.dumps(ev) + "\n")
