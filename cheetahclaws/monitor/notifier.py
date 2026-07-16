"""
monitor/notifier.py — Deliver reports via configured channels.

Channels: telegram, slack, console (always)

Config keys (in ~/.cheetahclaws/config.json):
  monitor_telegram_token   — Telegram bot token
  monitor_telegram_chat_id — Telegram chat ID (int or str)
  monitor_slack_token      — Slack bot token
  monitor_slack_channel    — Slack channel ID
"""
from __future__ import annotations

import sys


def _send_telegram(text: str, config: dict) -> str | None:
    """Send text via Telegram. Returns error string or None on success."""
    token = config.get("monitor_telegram_token") or config.get("_tg_token")
    chat_id = config.get("monitor_telegram_chat_id") or config.get("_tg_chat_id")
    if not token or not chat_id:
        return "Telegram not configured (set monitor_telegram_token + monitor_telegram_chat_id)"
    try:
        from cheetahclaws.bridges.telegram import _tg_send
        _tg_send(token, int(chat_id), text)
        return None
    except Exception as e:
        return f"Telegram send failed: {e}"


def _send_slack(text: str, config: dict) -> str | None:
    """Send text via Slack. Returns error string or None on success."""
    token = config.get("monitor_slack_token") or config.get("_slack_token")
    channel = config.get("monitor_slack_channel") or config.get("_slack_channel")
    if not token or not channel:
        return "Slack not configured (set monitor_slack_token + monitor_slack_channel)"
    try:
        from cheetahclaws.bridges.slack import _slack_send
        _slack_send(token, channel, text)
        return None
    except Exception as e:
        return f"Slack send failed: {e}"


def deliver(report: str, channels: list[str], config: dict) -> dict[str, str | None]:
    """
    Deliver a report to all requested channels.
    Returns {channel: error_or_None}.
    """
    results: dict[str, str | None] = {}

    for ch in channels:
        if ch == "telegram":
            results["telegram"] = _send_telegram(report, config)
        elif ch == "slack":
            results["slack"] = _send_slack(report, config)
        elif ch == "console":
            print(report)
            print()
            results["console"] = None
        else:
            results[ch] = f"Unknown channel: {ch}"

    return results


def auto_channels(config: dict) -> list[str]:
    """Infer which channels are configured."""
    channels = []
    if (config.get("monitor_telegram_token") or config.get("_tg_token")) and \
       (config.get("monitor_telegram_chat_id") or config.get("_tg_chat_id")):
        channels.append("telegram")
    if (config.get("monitor_slack_token") or config.get("_slack_token")) and \
       (config.get("monitor_slack_channel") or config.get("_slack_channel")):
        channels.append("slack")
    return channels
