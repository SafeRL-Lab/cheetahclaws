"""Email tools — read (IMAP) and send (SMTP) emails.

Uses Python stdlib (imaplib, smtplib, email) — no external dependencies.
Requires config keys: email_address, email_password, email_imap_host, email_smtp_host.
"""
from __future__ import annotations

import email
import email.utils
import imaplib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from cheetahclaws.tool_registry import ToolDef, register_tool

_CONFIG_HINT = (
    "Email not configured. Set these in the REPL:\n"
    "  /config email_address=you@example.com\n"
    "  /config email_password=your-app-password\n"
    "  /config email_imap_host=imap.gmail.com\n"
    "  /config email_smtp_host=smtp.gmail.com\n"
    "\nFor Gmail, use an App Password (not your regular password):\n"
    "  https://myaccount.google.com/apppasswords"
)


def _get_email_config(config: dict) -> tuple[str, str, str, str] | str:
    """Return (address, password, imap_host, smtp_host) or error string."""
    addr = config.get("email_address", "")
    pwd = config.get("email_password", "")
    imap = config.get("email_imap_host", "")
    smtp = config.get("email_smtp_host", "")
    if not all([addr, pwd, imap]):
        return _CONFIG_HINT
    if not smtp:
        smtp = imap.replace("imap", "smtp")
    return addr, pwd, imap, smtp


def _read_emails(params: dict, config: dict) -> str:
    """Read recent emails from inbox via IMAP."""
    creds = _get_email_config(config)
    if isinstance(creds, str):
        return creds
    addr, pwd, imap_host, _ = creds

    folder = params.get("folder", "INBOX")
    limit = min(params.get("limit", 5), 20)
    search = params.get("search", "ALL")

    try:
        conn = imaplib.IMAP4_SSL(imap_host, timeout=15)
        conn.login(addr, pwd)
        conn.select(folder, readonly=True)

        # Search
        if search and search != "ALL":
            # Support common search patterns
            if "@" in search:
                criteria = f'(FROM "{search}")'
            else:
                criteria = f'(SUBJECT "{search}")'
        else:
            criteria = "ALL"

        _, msg_nums = conn.search(None, criteria)
        ids = msg_nums[0].split()

        if not ids:
            conn.logout()
            return f"No emails found in {folder} matching: {search}"

        # Get the last N emails
        selected = ids[-limit:]
        results = []

        for mid in reversed(selected):
            _, data = conn.fetch(mid, "(RFC822)")
            raw = data[0][1]
            msg = email.message_from_bytes(raw)

            subject = _decode_header(msg.get("Subject", "(no subject)"))
            sender = _decode_header(msg.get("From", "unknown"))
            date = msg.get("Date", "")
            to = _decode_header(msg.get("To", ""))

            # Extract body
            body = _get_body(msg)
            body_preview = body[:500] if body else "(no body)"

            results.append(
                f"--- Email #{len(results)+1} ---\n"
                f"From: {sender}\n"
                f"To: {to}\n"
                f"Date: {date}\n"
                f"Subject: {subject}\n"
                f"\n{body_preview}\n"
            )

        conn.logout()
        header = f"Found {len(ids)} email(s) in {folder}, showing latest {len(results)}:\n\n"
        return header + "\n".join(results)

    except imaplib.IMAP4.error as e:
        return f"IMAP error: {e}\nCheck your email_address, email_password, and email_imap_host."
    except Exception as e:
        return f"Email error: {type(e).__name__}: {e}"


def _send_email(params: dict, config: dict) -> str:
    """Send an email via SMTP."""
    creds = _get_email_config(config)
    if isinstance(creds, str):
        return creds
    addr, pwd, _, smtp_host = creds

    to = params["to"]
    subject = params["subject"]
    body = params["body"]
    reply_to = params.get("reply_to")

    try:
        msg = MIMEMultipart()
        msg["From"] = addr
        msg["To"] = to
        msg["Subject"] = subject
        if reply_to:
            msg["In-Reply-To"] = reply_to
            msg["References"] = reply_to
        msg.attach(MIMEText(body, "plain", "utf-8"))

        smtp_port = int(config.get("email_smtp_port", 587))
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.starttls()
            server.login(addr, pwd)
            server.send_message(msg)

        return f"Email sent to {to}\nSubject: {subject}"

    except smtplib.SMTPAuthenticationError:
        return "SMTP authentication failed. Check email_password (use App Password for Gmail)."
    except Exception as e:
        return f"Send error: {type(e).__name__}: {e}"


def _decode_header(raw: str) -> str:
    """Decode RFC2047 encoded header."""
    parts = email.header.decode_header(raw)
    decoded = []
    for data, charset in parts:
        if isinstance(data, bytes):
            decoded.append(data.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(data)
    return " ".join(decoded)


def _get_body(msg) -> str:
    """Extract plain text body from email message."""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace")
        # Fallback to first text/html
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                html = payload.decode(charset, errors="replace")
                import re
                return re.sub(r'<[^>]+>', '', html).strip()
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")
    return ""


# ── Register ─────────────────────────────────────────────────────────────

register_tool(ToolDef(
    name="ReadEmail",
    schema={
        "name": "ReadEmail",
        "description": (
            "Read recent emails from the user's inbox via IMAP. "
            "Can search by sender email or subject keyword. "
            "Returns subject, sender, date, and body preview for each email."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder": {
                    "type": "string",
                    "description": "IMAP folder (default: INBOX)",
                    "default": "INBOX",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max emails to return (default: 5, max: 20)",
                    "default": 5,
                },
                "search": {
                    "type": "string",
                    "description": "Search query — email address (searches From) or keyword (searches Subject). Default: ALL",
                    "default": "ALL",
                },
            },
        },
    },
    func=_read_emails,
    read_only=True,
    concurrent_safe=True,
))

register_tool(ToolDef(
    name="SendEmail",
    schema={
        "name": "SendEmail",
        "description": (
            "Send an email via SMTP. Requires email configuration in /config. "
            "Always confirm with the user before sending."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address",
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line",
                },
                "body": {
                    "type": "string",
                    "description": "Email body (plain text)",
                },
                "reply_to": {
                    "type": "string",
                    "description": "Optional Message-ID to reply to (for threading)",
                },
            },
            "required": ["to", "subject", "body"],
        },
    },
    func=_send_email,
    read_only=False,
    concurrent_safe=False,
))
