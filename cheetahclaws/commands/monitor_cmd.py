"""
commands/monitor_cmd.py — AI Monitor + Decision Assistant commands.

Commands:
  /subscribe <topic> [schedule] [--telegram] [--slack]
      Subscribe to a topic with optional schedule and channel flags.
      Examples:
        /subscribe ai_research
        /subscribe stock_TSLA daily
        /subscribe crypto_BTC 6h --telegram
        /subscribe world_news --slack
        /subscribe custom:quantum computing weekly

  /subscriptions
      List all active subscriptions.

  /unsubscribe <topic>
      Remove a subscription.

  /monitor run [topic]
      Run all (or one) subscription(s) now, print reports.

  /monitor start
      Start the background scheduler daemon.

  /monitor stop
      Stop the background scheduler daemon.

  /monitor status
      Show scheduler status and subscription overview.

  /monitor set telegram <token> <chat_id>
      Configure Telegram delivery for monitor reports.

  /monitor set slack <token> <channel_id>
      Configure Slack delivery for monitor reports.

  /monitor topics
      List available built-in topics.
"""
from __future__ import annotations

from cheetahclaws.ui.render import clr, info, ok, warn, err

_BUILTIN_TOPICS = {
    "ai_research":  "Latest arxiv papers (cs.AI, cs.LG, cs.CL)",
    "world_news":   "Top world news (Reuters, BBC, Guardian, AP)",
    "stock_<TICKER>":  "Stock price & data — e.g. stock_TSLA, stock_AAPL",
    "crypto_<SYMBOL>": "Crypto market data — e.g. crypto_BTC, crypto_ETH",
    "custom:<QUERY>":  "Custom search query — e.g. custom:quantum computing",
    "research:<QUERY>": (
        "Full /research pipeline (17 sources, heat table, sparkline) — "
        "e.g. research:transformer efficiency, or with a window: "
        "research:30d:RLHF  (range presets: 3d, 7d, 30d, 90d, 6m, 1y)"
    ),
}

_VALID_SCHEDULES = {"15m", "30m", "1h", "2h", "6h", "12h", "daily", "weekly"}


def _parse_subscribe_args(args: str):
    """Parse '/subscribe <topic> [schedule] [--telegram] [--slack]'.

    Topic may contain spaces (e.g. ``research:7d:Agent OS Benchmark``).
    The previous parser treated the FIRST whitespace-separated token as
    the topic and dropped the rest, which truncated multi-word topics
    coming from the SSJ trend-track menu.

    The new rule: walk left-to-right, peel off ``--flag`` tokens into
    ``channels``. Among the remaining tokens, if the LAST one is a
    recognised schedule (in ``_VALID_SCHEDULES``), that's the schedule
    and everything before it joined by single spaces is the topic.
    Otherwise the entire non-flag remainder is the topic and schedule
    keeps its default. This correctly handles all of:

        ai_research                           → ai_research, daily
        ai_research weekly                    → ai_research, weekly
        custom:quantum computing weekly       → custom:quantum computing, weekly
        research:7d:Agent OS Benchmark daily  → research:7d:Agent OS Benchmark, daily
        research:7d:Agent OS Benchmark        → research:7d:Agent OS Benchmark, daily
    """
    parts = args.split()
    schedule = "daily"
    channels: list[str] = []
    topic_tokens: list[str] = []

    for p in parts:
        if p.startswith("--"):
            flag = p[2:]
            if flag in ("telegram", "slack", "console"):
                channels.append(flag)
        else:
            topic_tokens.append(p)

    # If the last non-flag token is a schedule, peel it off.
    if topic_tokens and topic_tokens[-1].lower() in _VALID_SCHEDULES:
        schedule = topic_tokens.pop().lower()

    topic = " ".join(topic_tokens) if topic_tokens else None
    return topic, schedule, channels


def cmd_subscribe(args: str, state, config) -> bool:
    """Subscribe to a monitoring topic."""
    from cheetahclaws.monitor.store import add_subscription, get_subscription

    if not args.strip():
        info("Usage: /subscribe <topic> [schedule] [--telegram] [--slack]")
        info("")
        info("Available topics:")
        for t, desc in _BUILTIN_TOPICS.items():
            print(f"  {clr(t, 'cyan'):<30} {desc}")
        info("")
        info("Schedules: 15m, 30m, 1h, 2h, 6h, 12h, daily, weekly")
        return True

    topic, schedule, channels = _parse_subscribe_args(args.strip())
    if not topic:
        err("No topic specified. Example: /subscribe ai_research daily")
        return True

    existing = get_subscription(topic)
    sub = add_subscription(topic, schedule=schedule, channels=channels or None)

    action = "Updated" if existing else "Subscribed"
    ok(f"{action}: {clr(topic, 'cyan')} | schedule: {schedule} | "
       f"channels: {', '.join(channels) if channels else 'auto'}")
    info(f"  ID: {sub['id']}  |  Use '/monitor run {topic}' to run now")
    return True


def cmd_subscriptions(args: str, state, config) -> bool:
    """List all active subscriptions."""
    from cheetahclaws.monitor.store import list_subscriptions
    from cheetahclaws.monitor.scheduler import is_running

    subs = list_subscriptions()
    if not subs:
        info("No subscriptions yet. Use /subscribe <topic> to add one.")
        return True

    status_str = clr("RUNNING", "green") if is_running() else clr("stopped", "dim")
    info(f"Subscriptions ({len(subs)})  [scheduler: {status_str}]")
    print()
    for s in subs:
        topic = s["topic"]
        schedule = s.get("schedule", "daily")
        channels = s.get("channels") or []
        last_run = s.get("last_run", "never")
        if last_run and last_run != "never":
            # Pretty-print relative time
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(last_run)
                delta = datetime.now() - dt
                if delta.total_seconds() < 3600:
                    last_run = f"{int(delta.total_seconds()//60)}m ago"
                elif delta.total_seconds() < 86400:
                    last_run = f"{int(delta.total_seconds()//3600)}h ago"
                else:
                    last_run = f"{int(delta.days)}d ago"
            except Exception:
                pass
        ch_str = ", ".join(channels) if channels else "auto"
        print(
            f"  {clr(topic, 'cyan'):<28} "
            f"every {clr(schedule, 'yellow'):<8} "
            f"channels: {ch_str:<16} "
            f"last: {last_run}"
        )
    print()
    info("Commands: /monitor run | /monitor start | /monitor stop | /unsubscribe <topic>")
    return True


def cmd_unsubscribe(args: str, state, config) -> bool:
    """Remove a subscription."""
    from cheetahclaws.monitor.store import remove_subscription

    topic = args.strip()
    if not topic:
        err("Usage: /unsubscribe <topic>")
        return True

    if remove_subscription(topic):
        ok(f"Unsubscribed: {topic}")
    else:
        err(f"No subscription found for: {topic}")
    return True


def cmd_monitor(args: str, state, config) -> bool:
    """Monitor management. No args → interactive setup wizard."""
    from cheetahclaws.monitor import scheduler as _sched
    from cheetahclaws.monitor.store import list_subscriptions

    parts = args.strip().split(None, 1)
    sub = parts[0].lower() if parts else ""
    rest = parts[1].strip() if len(parts) > 1 else ""

    if sub == "" or sub == "setup":
        _cmd_monitor_wizard(config)

    elif sub == "run":
        _cmd_monitor_run(rest, config)

    elif sub == "start":
        # F-3: when a daemon is running, the scheduler is *its* job.
        # Spinning up another loop in REPL would race the daemon's
        # writes to monitor_subscriptions.last_run_at and double-fire
        # subscriptions.  Detect-and-skip; subscriptions added/removed
        # in REPL are still picked up by the daemon scheduler on its
        # next 60 s poll because both processes read the same SQLite.
        try:
            from cheetahclaws.daemon import discovery as _disc
            _live = _disc.locate()
        except Exception:
            _live = None
        if _live is not None:
            info(f"Scheduler is owned by the running daemon "
                 f"(pid={_live.get('pid', '?')}); /monitor start is a no-op.")
        elif _sched.is_running():
            ok("Scheduler is already running.")
        else:
            _sched.start(config, on_report=None)
            ok("Monitor scheduler started.")
            info("Reports will run on schedule and be pushed to configured channels.")
            info("Use /monitor stop to stop, /monitor status to check.")

    elif sub == "stop":
        try:
            from cheetahclaws.daemon import discovery as _disc
            _live = _disc.locate()
        except Exception:
            _live = None
        if _live is not None:
            info(f"Scheduler is owned by the running daemon "
                 f"(pid={_live.get('pid', '?')}); /monitor stop is a no-op. "
                 f"Run `cheetahclaws daemon stop` to stop the daemon itself.")
        elif _sched.stop():
            ok("Monitor scheduler stopped.")
        else:
            info("Scheduler was not running.")

    elif sub == "status":
        _cmd_monitor_status(config)

    elif sub == "set":
        _cmd_monitor_set(rest, config)

    elif sub == "topics":
        info("Built-in topics:")
        for t, desc in _BUILTIN_TOPICS.items():
            print(f"  {clr(t, 'cyan'):<30} {desc}")

    else:
        err(f"Unknown subcommand: {sub}")
        info("Usage: /monitor  (wizard) | run [topic] | start | stop | status | set telegram/slack")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Interactive setup wizard
# ─────────────────────────────────────────────────────────────────────────────

_WIZARD_TOPICS = [
    ("ai_research",  "AI Research",   "Latest arxiv papers  (cs.AI · cs.LG · cs.CL)"),
    ("world_news",   "World News",    "Top headlines        (Reuters · BBC · AP)"),
    ("stock_",       "Stock",         "Stock price & data   (enter ticker, e.g. TSLA)"),
    ("crypto_",      "Crypto",        "Crypto market data   (BTC · ETH · SOL · ...)"),
    ("custom:",      "Custom query",  "Monitor any topic    (e.g. 'quantum computing')"),
    ("research:",    "Trend tracker", "Full 17-source /research brief with heat table (weekly trend mode)"),
]

_WIZARD_SCHEDULES = [
    ("1h",     "Every hour"),
    ("6h",     "Every 6 hours"),
    ("12h",    "Every 12 hours"),
    ("daily",  "Once a day"),
    ("weekly", "Once a week"),
]


def _wizard_ask(prompt: str, config: dict, menu_ctx: str = "") -> str:
    try:
        from cheetahclaws.tools import ask_input_interactive
        return ask_input_interactive(clr(prompt, "cyan"), config, menu_ctx).strip()
    except Exception:
        return input(prompt).strip()


def _cmd_monitor_wizard(config: dict) -> None:
    """Full interactive setup wizard — zero prior knowledge required."""
    from cheetahclaws.monitor.store import list_subscriptions, add_subscription, remove_subscription
    from cheetahclaws.monitor import scheduler as _sched
    from cheetahclaws.config import save_config

    _BORDER = clr("─" * 52, "dim")

    def _header(title: str):
        print()
        print(clr("╭─ ", "dim") + clr(title, "bold") + clr(" " + "─" * max(0, 48 - len(title)), "dim"))

    def _footer():
        print(clr("╰" + "─" * 51, "dim"))

    # ── Welcome ───────────────────────────────────────────────────────────
    _header("AI Monitor + Decision Assistant")
    print(clr("│", "dim") + "  Monitor anything 24/7. AI summarizes & pushes to you.")
    print(clr("│", "dim"))

    subs = list_subscriptions()
    if subs:
        print(clr("│", "dim") + f"  You have {len(subs)} active subscription(s):")
        for s in subs:
            ch = ", ".join(s.get("channels") or []) or "console"
            print(clr("│", "dim") + f"    • {clr(s['topic'], 'cyan')}  every {s.get('schedule','daily')}  → {ch}")
        print(clr("│", "dim"))

    print(clr("│", "dim") + clr("  What do you want to do?", "bold"))
    print(clr("│", "dim"))
    print(clr("│", "dim") + f"  {clr('1.', 'bold')}  Add a new subscription")
    print(clr("│", "dim") + f"  {clr('2.', 'bold')}  Run all subscriptions now  (preview reports)")
    print(clr("│", "dim") + f"  {clr('3.', 'bold')}  {'Stop' if _sched.is_running() else 'Start'} background scheduler")
    if subs:
        print(clr("│", "dim") + f"  {clr('4.', 'bold')}  Remove a subscription")
    print(clr("│", "dim") + f"  {clr('5.', 'bold')}  Configure push notifications  (Telegram / Slack)")
    print(clr("│", "dim") + f"  {clr('0.', 'bold')}  Exit")
    _footer()

    choice = _wizard_ask("  » ", config)

    if choice == "0" or choice.lower() in ("q", "exit"):
        return

    if choice == "1":
        _wizard_add_subscription(config)
    elif choice == "2":
        _cmd_monitor_run("", config)
    elif choice == "3":
        if _sched.is_running():
            _sched.stop()
            ok("Scheduler stopped.")
        else:
            _sched.start(config, on_report=None)
            ok("Scheduler started — running subscriptions on their schedules.")
    elif choice == "4" and subs:
        _wizard_remove_subscription(config, subs)
    elif choice == "5":
        _wizard_configure_notifications(config)
    else:
        # Treat unknown input as topic shortcut
        _wizard_add_subscription(config, prefill=choice)


def _wizard_add_subscription(config: dict, prefill: str = "") -> None:
    """Step-by-step: pick topic → schedule → channel → confirm."""
    from cheetahclaws.monitor.store import add_subscription
    from cheetahclaws.monitor import scheduler as _sched

    def _header(title: str):
        print()
        print(clr("╭─ ", "dim") + clr(title, "bold") + clr(" " + "─" * max(0, 48 - len(title)), "dim"))

    def _footer():
        print(clr("╰" + "─" * 51, "dim"))

    # ── Step 1: Pick topic ────────────────────────────────────────────────
    _header("Step 1 / 3 — What to monitor?")
    for i, (_, label, desc) in enumerate(_WIZARD_TOPICS, 1):
        print(clr("│", "dim") + f"  {clr(str(i)+'.', 'bold')}  {clr(label, 'cyan'):<16} {desc}")
    _footer()

    raw = _wizard_ask(f"  » Choose [1-{len(_WIZARD_TOPICS)}]: ", config) if not prefill else prefill

    topic = None
    if raw.isdigit() and 1 <= int(raw) <= len(_WIZARD_TOPICS):
        key, label, _ = _WIZARD_TOPICS[int(raw) - 1]
        if key.endswith("_"):
            ticker = _wizard_ask(f"  Enter {label} symbol (e.g. TSLA, AAPL): ", config).upper()
            if not ticker:
                warn("Cancelled.")
                return
            topic = key + ticker
        elif key.endswith(":"):
            query = _wizard_ask(f"  Enter your search topic (e.g. quantum computing): ", config)
            if not query:
                warn("Cancelled.")
                return
            topic = key + query
        else:
            topic = key
    else:
        # Accept a direct topic name as input
        raw = raw.strip()
        if raw:
            topic = raw
        else:
            warn("Cancelled.")
            return

    # ── Step 2: Pick schedule ─────────────────────────────────────────────
    _header("Step 2 / 3 — How often?")
    for i, (_, label) in enumerate(_WIZARD_SCHEDULES, 1):
        marker = clr(" ◀ default", "dim") if i == 4 else ""
        print(clr("│", "dim") + f"  {clr(str(i)+'.', 'bold')}  {label}{marker}")
    _footer()

    sched_raw = _wizard_ask("  » Choose [1-5] or press Enter for daily: ", config)
    if sched_raw.isdigit() and 1 <= int(sched_raw) <= len(_WIZARD_SCHEDULES):
        schedule = _WIZARD_SCHEDULES[int(sched_raw) - 1][0]
    elif sched_raw.lower() in _VALID_SCHEDULES:
        schedule = sched_raw.lower()
    else:
        schedule = "daily"

    # ── Step 3: Push channel ──────────────────────────────────────────────
    from cheetahclaws.monitor.notifier import auto_channels
    auto_ch = auto_channels(config)

    _header("Step 3 / 3 — Where to send reports?")
    options = []
    print(clr("│", "dim") + f"  {clr('1.', 'bold')}  Console only (print here in terminal)")
    options.append("console")
    if "telegram" in auto_ch:
        print(clr("│", "dim") + f"  {clr('2.', 'bold')}  {clr('Telegram', 'cyan')}  (already configured)")
        options.append("telegram")
    else:
        print(clr("│", "dim") + f"  {clr('2.', 'bold')}  Telegram  {clr('(not configured — choose 4 to set up)', 'dim')}")
        options.append(None)
    if "slack" in auto_ch:
        print(clr("│", "dim") + f"  {clr('3.', 'bold')}  {clr('Slack', 'cyan')}    (already configured)")
        options.append("slack")
    else:
        print(clr("│", "dim") + f"  {clr('3.', 'bold')}  Slack     {clr('(not configured — choose 4 to set up)', 'dim')}")
        options.append(None)
    print(clr("│", "dim") + f"  {clr('4.', 'bold')}  Set up Telegram / Slack now")
    _footer()

    ch_raw = _wizard_ask("  » Choose [1-4] or Enter for console: ", config)
    channels: list[str] = []
    if ch_raw == "4":
        _wizard_configure_notifications(config)
        channels = auto_channels(config) or ["console"]
    elif ch_raw.isdigit() and 1 <= int(ch_raw) <= 3:
        chosen = options[int(ch_raw) - 1]
        if chosen:
            channels = [chosen]
        else:
            warn("That channel is not configured. Defaulting to console.")
            channels = ["console"]
    else:
        channels = ["console"]

    # ── Save + confirm ────────────────────────────────────────────────────
    sub = add_subscription(topic, schedule=schedule, channels=channels)
    ch_display = ", ".join(channels)

    print()
    ok(f"Subscribed!")
    print(f"  Topic:    {clr(topic, 'cyan')}")
    print(f"  Schedule: every {clr(schedule, 'yellow')}")
    print(f"  Delivery: {ch_display}")
    print()

    # Offer to run now
    run_now = _wizard_ask("  Run now to preview the first report? [Y/n]: ", config).lower()
    if run_now != "n":
        from cheetahclaws.monitor.scheduler import run_one
        info(f"Fetching data for {topic} ...")
        report = run_one(topic, config, force=True)
        print()
        print(report)
        print()

    # Offer to start scheduler if not running
    from cheetahclaws.monitor import scheduler as _sched
    if not _sched.is_running():
        start_sched = _wizard_ask("  Start background scheduler so reports run automatically? [Y/n]: ", config).lower()
        if start_sched != "n":
            _sched.start(config, on_report=None)
            ok("Scheduler started. Reports will run automatically.")
            info("Use /monitor stop to stop, /monitor status to check.")


def _wizard_remove_subscription(config: dict, subs: list) -> None:
    """Pick a subscription to remove."""
    print()
    print(clr("  Active subscriptions:", "bold"))
    for i, s in enumerate(subs, 1):
        print(f"  {clr(str(i)+'.', 'bold')}  {clr(s['topic'], 'cyan')}")
    print(f"  {clr('0.', 'bold')}  Cancel")

    raw = _wizard_ask("  » Which to remove: ", config)
    if raw.isdigit() and 1 <= int(raw) <= len(subs):
        from cheetahclaws.monitor.store import remove_subscription
        topic = subs[int(raw) - 1]["topic"]
        remove_subscription(topic)
        ok(f"Removed: {topic}")
    else:
        info("Cancelled.")


def _wizard_configure_notifications(config: dict) -> None:
    """Walk through Telegram / Slack setup."""
    from cheetahclaws.config import save_config

    print()
    print(clr("  Push notification setup:", "bold"))
    print(f"  {clr('1.', 'bold')}  Telegram")
    print(f"  {clr('2.', 'bold')}  Slack")
    print(f"  {clr('0.', 'bold')}  Cancel")

    ch = _wizard_ask("  » Choose: ", config)

    if ch == "1":
        print()
        print(clr("  Telegram setup", "bold"))
        print("  1. Create a bot: message @BotFather on Telegram → /newbot")
        print("  2. Get the bot token (looks like 123456:ABC-DEF...)")
        print("  3. Start a chat with your bot, then get your chat ID from @userinfobot")
        print()
        token = _wizard_ask("  Paste your bot token: ", config)
        if not token:
            info("Cancelled.")
            return
        chat_id = _wizard_ask("  Paste your chat ID: ", config)
        if not chat_id:
            info("Cancelled.")
            return
        config["monitor_telegram_token"] = token
        config["monitor_telegram_chat_id"] = chat_id
        save_config(config)

        # Send test message
        try:
            from cheetahclaws.bridges.telegram import _tg_send
            _tg_send(token, int(chat_id),
                     "✅ CheetahClaws Monitor connected! You'll receive AI reports here.")
            ok("Telegram configured and test message sent!")
        except Exception as e:
            warn(f"Saved, but test message failed: {e}")
            warn("Check token and chat_id are correct.")

    elif ch == "2":
        print()
        print(clr("  Slack setup", "bold"))
        print("  1. Create a Slack App at api.slack.com/apps")
        print("  2. Add 'chat:write' permission → install to workspace")
        print("  3. Copy the Bot OAuth Token (xoxb-...)")
        print("  4. Get the channel ID (right-click channel → Copy link, the ID is at the end)")
        print()
        token = _wizard_ask("  Paste your Slack bot token (xoxb-...): ", config)
        if not token:
            info("Cancelled.")
            return
        channel = _wizard_ask("  Paste your channel ID (e.g. C1234567890): ", config)
        if not channel:
            info("Cancelled.")
            return
        config["monitor_slack_token"] = token
        config["monitor_slack_channel"] = channel
        save_config(config)

        try:
            from cheetahclaws.bridges.slack import _slack_send
            _slack_send(token, channel, "✅ CheetahClaws Monitor connected! You'll receive AI reports here.")
            ok("Slack configured and test message sent!")
        except Exception as e:
            warn(f"Saved, but test message failed: {e}")
            warn("Check token and channel ID are correct.")
    else:
        info("Cancelled.")


def _cmd_monitor_run(topic_arg: str, config: dict) -> None:
    from cheetahclaws.monitor.store import list_subscriptions
    from cheetahclaws.monitor.scheduler import run_one
    from cheetahclaws.monitor.fetchers import fetch
    from cheetahclaws.monitor.summarizer import summarize
    from cheetahclaws.monitor.notifier import auto_channels, deliver

    if topic_arg:
        # Run a specific topic (even if not subscribed yet, for ad-hoc use)
        info(f"Running monitor for: {topic_arg} ...")
        report = run_one(topic_arg, config, force=True)
        if not any(c in (auto_channels(config) or []) for c in ["telegram", "slack"]):
            # Print to console since no channels
            print()
            print(report)
        else:
            print()
            print(report)
    else:
        subs = list_subscriptions()
        if not subs:
            info("No subscriptions. Add one with /subscribe <topic>")
            return
        info(f"Running {len(subs)} subscription(s) now...")
        for sub in subs:
            topic = sub["topic"]
            info(f"  Fetching {topic}...")
            report = run_one(topic, config, force=True)
            print()
            print(report)
            print()


def _cmd_monitor_status(config: dict) -> None:
    from cheetahclaws.monitor import scheduler as _sched
    from cheetahclaws.monitor.store import list_subscriptions

    running = _sched.is_running()
    subs = list_subscriptions()

    status_str = clr("RUNNING", "green") if running else clr("stopped", "dim")
    print(f"  Scheduler:     {status_str}")
    print(f"  Subscriptions: {len(subs)}")

    tg_token = config.get("monitor_telegram_token") or config.get("_tg_token")
    tg_chat = config.get("monitor_telegram_chat_id") or config.get("_tg_chat_id")
    sl_token = config.get("monitor_slack_token") or config.get("_slack_token")
    sl_chan = config.get("monitor_slack_channel") or config.get("_slack_channel")

    tg_ok = bool(tg_token and tg_chat)
    sl_ok = bool(sl_token and sl_chan)
    print(f"  Telegram:      {clr('configured', 'green') if tg_ok else clr('not set', 'dim')}")
    print(f"  Slack:         {clr('configured', 'green') if sl_ok else clr('not set', 'dim')}")

    if subs:
        print()
        for s in subs:
            topic = s["topic"]
            schedule = s.get("schedule", "daily")
            last = s.get("last_run", "never")
            if last and last != "never":
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(last)
                    delta = datetime.now() - dt
                    if delta.total_seconds() < 3600:
                        last = f"{int(delta.total_seconds()//60)}m ago"
                    elif delta.total_seconds() < 86400:
                        last = f"{int(delta.total_seconds()//3600)}h ago"
                    else:
                        last = f"{int(delta.days)}d ago"
                except Exception:
                    pass
            print(f"    {clr(topic, 'cyan'):<28} every {schedule:<8} last: {last}")

    if not running and subs:
        print()
        info("Start the scheduler with: /monitor start")


def _cmd_monitor_set(args: str, config: dict) -> None:
    from cheetahclaws.config import save_config

    parts = args.split()
    if not parts:
        info("Usage: /monitor set telegram <token> <chat_id>")
        info("       /monitor set slack <token> <channel_id>")
        return

    channel = parts[0].lower()
    if channel == "telegram":
        if len(parts) < 3:
            err("Usage: /monitor set telegram <bot_token> <chat_id>")
            return
        config["monitor_telegram_token"] = parts[1]
        config["monitor_telegram_chat_id"] = parts[2]
        save_config(config)
        ok(f"Telegram configured: token={parts[1][:8]}... chat_id={parts[2]}")

    elif channel == "slack":
        if len(parts) < 3:
            err("Usage: /monitor set slack <bot_token> <channel_id>")
            return
        config["monitor_slack_token"] = parts[1]
        config["monitor_slack_channel"] = parts[2]
        save_config(config)
        ok(f"Slack configured: token={parts[1][:8]}... channel={parts[2]}")

    else:
        err(f"Unknown channel: {channel}. Use 'telegram' or 'slack'.")
