"""Asciinema v2 cast: /brainstorm multi-persona adversarial debate.

Scenario: ask 5 personas to debate whether to migrate the order service
to event sourcing. They argue, push back, then converge on a todo list.

Run: python3 gen_brainstorm.py > brainstorm.cast
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
    "title": "CheetahClaws /brainstorm — 5 personas debate event sourcing",
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
    rng = random.Random(17)
    for ch in s:
        out(base + rng.random() * jitter, ch)


# Scene 1 — launch + /brainstorm
out(0.0, f"{GREEN}~/projects/checkout{RST} {CYAN}❯{RST} cheetahclaws\r\n")
out(0.3, f"{DIM}[CheetahClaws v3.5.79 · claude-sonnet-4-6]{RST}\r\n\r\n")
out(0.2, f"{BOLD}{CYAN}[checkout] »{RST} ")
out(0.5, "")
type_string("/brainstorm \"Should we migrate the order service from CRUD to event sourcing?\"")
out(0.4, "\r\n\r\n")

out(0.4, f"{DIM}● Spawning 5 expert personas in parallel…{RST}\r\n\r\n")

# Scene 2 — persona round 1
personas = [
    (BLUE,   "ARCHITECT",  "Event sourcing maps cleanly to checkout's domain — every state change is\r\n             already an event (cart-added, payment-authorised, fulfilled). "
                           "We get audit\r\n             trail, time-travel debugging, and replay for free."),
    (RED,    "SKEPTIC",    "Three months of dev time to rewrite something that works fine. Where's the\r\n             user-facing win? You're trading a known boring solution for a fashionable\r\n             one we'll regret in 18 months when the lead architect leaves."),
    (MAG,    "PRAGMATIST", "Halfway compromise: keep CRUD as the source of truth, emit an outbox event\r\n             on each write. We get the audit log without the rewrite. Doors stay open\r\n             for full ES later."),
    (YELL,   "DBA",        "Two real issues with the proposal: (1) Postgres event-store throughput on the\r\n             checkout volume — we'd need partitioning by tenant; (2) projections lag\r\n             will surface as stale reads during Black Friday."),
    (GREEN,  "PM",         "What problem are we actually solving? If it's audit/compliance — outbox\r\n             pattern is enough. If it's the analytics team rebuilding cart funnels\r\n             every quarter — yes, ES pays off."),
]

for color, name, msg in personas:
    out(0.5, f"  {color}● [{name}]{RST}\r\n")
    for line in msg.split("\n"):
        out(0.2, f"             {line}\r\n")

out(0.5, "\r\n")

# Scene 3 — debate round 2 (rebuttals)
out(0.3, f"{DIM}● Round 2: rebuttals…{RST}\r\n\r\n")

rebuttals = [
    (RED, "SKEPTIC", "→ ARCHITECT: 'time-travel debugging for free' has never been free in production.\r\n             Snapshot management alone is a quarter of work."),
    (BLUE, "ARCHITECT", "→ SKEPTIC: agreed on snapshots — but DBA's outbox path is half-measure.\r\n             Once we duplicate state we own two truths."),
    (MAG, "PRAGMATIST", "→ Both: outbox is not a half-measure if we treat it as a stepping stone with\r\n             a six-month review gate."),
]
for color, name, msg in rebuttals:
    out(0.5, f"  {color}● [{name}]{RST}\r\n")
    for line in msg.split("\n"):
        out(0.2, f"             {line}\r\n")
out(0.4, "\r\n")

# Scene 4 — synthesis
out(0.5, f"{BOLD}─── Synthesis ────────────────────────────────────────────────────────{RST}\r\n\r\n")
out(0.3, f"  {GREEN}{BOLD}Consensus:{RST} no full migration this quarter. Ship outbox + replayer first.\r\n\r\n")
out(0.3, f"  {BOLD}Decision pivots on PM's framing:{RST}\r\n")
out(0.25, f"    • audit/compliance only → {GREEN}stay CRUD + outbox events{RST}\r\n")
out(0.25, f"    • analytics rebuild every quarter → {YELL}plan full ES, but in Q3{RST}\r\n\r\n")

# Scene 5 — todo_list output
out(0.5, f"{YELL}[Write]{RST} brainstorm_outputs/todo_list.txt\r\n")
out(0.3, f"  {GREEN}1.{RST} {DIM}[ ]{RST} Add outbox table + transactional event publisher (1 week)\r\n")
out(0.2, f"  {GREEN}2.{RST} {DIM}[ ]{RST} Wire Kafka consumer → analytics warehouse (3 days)\r\n")
out(0.2, f"  {GREEN}3.{RST} {DIM}[ ]{RST} Benchmark projection lag with prod-shaped Black-Friday replay\r\n")
out(0.2, f"  {GREEN}4.{RST} {DIM}[ ]{RST} Schedule Q3 ES decision review (compliance + analytics inputs)\r\n\r\n")

out(0.4, f"{GREEN}✓{RST}  4 tasks ready. Run {CYAN}/worker{RST} to auto-implement them.\r\n\r\n")
out(0.4, f"{BOLD}{CYAN}[checkout] »{RST} ")
out(0.8, "")

sys.stdout.write(json.dumps(HEADER) + "\n")
for ev in events:
    sys.stdout.write(json.dumps(ev) + "\n")
