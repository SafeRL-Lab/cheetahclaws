"""Asciinema v2 cast: Telegram bridge remote control.

Scenario: start the Telegram bridge, then show two chat round-trips
from the phone — checking server load, then queuing a job — followed
by `!jobs` to inspect the queue.

Run: python3 gen_telegram.py > telegram.cast
"""
import json
import random
import sys


HEADER = {
    "version": 2,
    "width": 105,
    "height": 32,
    "timestamp": 1747262400,
    "env": {"SHELL": "/bin/zsh", "TERM": "xterm-256color"},
    "title": "CheetahClaws Telegram bridge — control the agent from your phone",
    "idle_time_limit": 1.4,
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
    rng = random.Random(29)
    for ch in s:
        out(base + rng.random() * jitter, ch)


# Scene 1 — start the bridge
out(0.0, f"{GREEN}~{RST} {CYAN}❯{RST} cheetahclaws\r\n")
out(0.3, f"{DIM}[CheetahClaws v3.5.79 · claude-sonnet-4-6]{RST}\r\n\r\n")
out(0.2, f"{BOLD}{CYAN}[~] »{RST} ")
out(0.5, "")
type_string("/telegram 7890:AAEx_REDACTED 458291205")
out(0.4, "\r\n")
out(0.4, f"{GREEN}✓{RST}  Telegram bridge online — bot @{BOLD}cheetah_personal_bot{RST}, chat 458291205\r\n")
out(0.2, f"{DIM}    Listening for messages. Typing indicator + slash passthrough enabled.{RST}\r\n\r\n")

# Scene 2 — phone message 1: server status
out(0.6, f"{BLUE}┌─ Telegram ────────────────────────────────────────────────────────┐{RST}\r\n")
out(0.3, f"{BLUE}│{RST}  {DIM}11:42{RST}  {BOLD}You{RST}: What's the CPU load on the server right now?\r\n")
out(0.6, f"{BLUE}│{RST}         {DIM}🐆 typing…{RST}\r\n")
out(0.3, f"{BLUE}└───────────────────────────────────────────────────────────────────┘{RST}\r\n\r\n")

out(0.4, f"{YELL}[Bash]{RST} uptime\r\n")
out(0.3, f"  {DIM}11:42:18 up 14 days, load average: 0.41, 0.55, 0.62{RST}\r\n\r\n")

out(0.4, f"{BLUE}┌─ Telegram ────────────────────────────────────────────────────────┐{RST}\r\n")
out(0.2, f"{BLUE}│{RST}  {DIM}11:42{RST}  🐆 CPU is {GREEN}quiet{RST}: 0.41 / 0.55 / 0.62 (1m / 5m / 15m).\r\n")
out(0.2, f"{BLUE}│{RST}         Server has been up 14 days. Want me to check memory or disk?\r\n")
out(0.3, f"{BLUE}└───────────────────────────────────────────────────────────────────┘{RST}\r\n\r\n")

# Scene 3 — phone message 2: queue a job while AI is busy
out(0.5, f"{BLUE}┌─ Telegram ────────────────────────────────────────────────────────┐{RST}\r\n")
out(0.3, f"{BLUE}│{RST}  {DIM}11:43{RST}  {BOLD}You{RST}: Re-run the nightly backup and tell me when it's done\r\n")
out(0.3, f"{BLUE}└───────────────────────────────────────────────────────────────────┘{RST}\r\n\r\n")

out(0.4, f"{YELL}[Bash]{RST} bash /opt/scripts/nightly_backup.sh   {DIM}(long-running, queued as job #2){RST}\r\n")
out(0.3, f"{BLUE}│{RST}  {DIM}11:43{RST}  🐆 Queued as job #2. I'll ping you when it finishes.\r\n\r\n")

# Scene 4 — !jobs inspect queue
out(0.5, f"{BLUE}┌─ Telegram ────────────────────────────────────────────────────────┐{RST}\r\n")
out(0.3, f"{BLUE}│{RST}  {DIM}11:43{RST}  {BOLD}You{RST}: !jobs\r\n")
out(0.3, f"{BLUE}│{RST}  {DIM}11:43{RST}  🐆 Job queue:\r\n")
out(0.25, f"{BLUE}│{RST}         {GREEN}#1{RST} {DIM}(done    11:42){RST}  uptime check\r\n")
out(0.2, f"{BLUE}│{RST}         {YELL}#2{RST} {DIM}(running 11:43){RST}  nightly_backup.sh           [████░░░░░░] 41%\r\n")
out(0.2, f"{BLUE}│{RST}         {DIM}                     `!cancel 2` to stop · `!job 2` for details{RST}\r\n")
out(0.3, f"{BLUE}└───────────────────────────────────────────────────────────────────┘{RST}\r\n\r\n")

# Scene 5 — job finishes, bot pushes notification
out(0.6, f"{BLUE}┌─ Telegram ────────────────────────────────────────────────────────┐{RST}\r\n")
out(0.3, f"{BLUE}│{RST}  {DIM}11:51{RST}  🐆 Job #2 done. {GREEN}Backup OK{RST} — 4.2 GB → s3://prod-backups/2026-05-10/\r\n")
out(0.2, f"{BLUE}│{RST}         {DIM}Took 7m 51s. Logs at ~/.cheetahclaws/jobs/2/stdout.txt{RST}\r\n")
out(0.3, f"{BLUE}└───────────────────────────────────────────────────────────────────┘{RST}\r\n\r\n")

out(0.4, f"{DIM}Also available: /wechat (微信), /slack — same job queue & passthrough.{RST}\r\n\r\n")
out(0.5, f"{BOLD}{CYAN}[~] »{RST} ")
out(0.8, "")

sys.stdout.write(json.dumps(HEADER) + "\n")
for ev in events:
    sys.stdout.write(json.dumps(ev) + "\n")
