"""Auth primitives for the web UI: bcrypt password hashing + JWT cookies.

We issue a stateless JWT signed with a server-side secret. The secret is
persisted to ~/.cheetahclaws/web_secret so restarts don't invalidate sessions.

Password hashing uses the `bcrypt` package directly. We previously routed
through passlib, but passlib 1.7.4 is unmaintained and crashes on
`bcrypt>=4.1` (it reads the removed `bcrypt.__about__.__version__`).
Existing `$2b$...` hashes in the DB remain compatible — bcrypt verifies
them natively.
"""
from __future__ import annotations

import os
import secrets
import time
from pathlib import Path
from typing import Optional

try:
    import bcrypt
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "bcrypt is required for the web UI. "
        "Install it with: pip install 'cheetahclaws[web]'"
    ) from exc

try:
    import jwt  # PyJWT
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyJWT is required for the web UI. "
        "Install it with: pip install 'cheetahclaws[web]'"
    ) from exc


# bcrypt caps the input at 72 bytes; longer passwords silently truncate.
# That matches what passlib did, so existing hashes remain verifiable.
_BCRYPT_MAX_BYTES = 72

JWT_ALG = "HS256"
JWT_TTL_SECONDS = 7 * 24 * 3600  # 7 days

_SECRET_PATH = Path.home() / ".cheetahclaws" / "web_secret"


def _load_or_create_secret() -> str:
    """Return a stable 32-byte URL-safe secret, creating it on first use."""
    env = os.environ.get("CHEETAHCLAWS_WEB_SECRET")
    if env:
        return env
    try:
        if _SECRET_PATH.exists():
            return _SECRET_PATH.read_text().strip()
    except OSError:
        pass
    _SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_urlsafe(32)
    _SECRET_PATH.write_text(secret)
    try:
        os.chmod(_SECRET_PATH, 0o600)
    except OSError:
        pass
    return secret


_JWT_SECRET: Optional[str] = None


def _secret() -> str:
    global _JWT_SECRET
    if _JWT_SECRET is None:
        _JWT_SECRET = _load_or_create_secret()
    return _JWT_SECRET


# ── Password hashing ─────────────────────────────────────────────────────

def _pw_bytes(password: str) -> bytes:
    return password.encode("utf-8")[:_BCRYPT_MAX_BYTES]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_pw_bytes(password), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, hashed: str) -> bool:
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(_pw_bytes(password), hashed.encode("ascii"))
    except (ValueError, TypeError):
        return False


# ── JWT ──────────────────────────────────────────────────────────────────

def issue_token(user_id: int, username: str) -> str:
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "usr": username,
        "iat": now,
        "exp": now + JWT_TTL_SECONDS,
    }
    return jwt.encode(payload, _secret(), algorithm=JWT_ALG)


def decode_token(token: str) -> Optional[dict]:
    """Verify a JWT and return its payload, or None if invalid/expired."""
    try:
        return jwt.decode(token, _secret(), algorithms=[JWT_ALG])
    except jwt.PyJWTError:
        return None


def extract_token_from_cookie(cookie_header: str) -> str:
    """Pull the cctoken value out of a Cookie header."""
    if not cookie_header:
        return ""
    for part in cookie_header.split(";"):
        part = part.strip()
        if part.startswith("cctoken="):
            from urllib.parse import unquote
            return unquote(part[len("cctoken="):])
    return ""


def auth_user_id(cookie_header: str) -> Optional[int]:
    """Return the authenticated user id from a cookie header, or None."""
    token = extract_token_from_cookie(cookie_header)
    if not token:
        return None
    payload = decode_token(token)
    if not payload:
        return None
    try:
        return int(payload.get("sub", ""))
    except (TypeError, ValueError):
        return None


COOKIE_NAME = "ccjwt"


def build_cookie(token: str, max_age: int = JWT_TTL_SECONDS) -> str:
    """Build a Set-Cookie header line for the chat-UI JWT."""
    return (
        f"Set-Cookie: {COOKIE_NAME}={token}; "
        f"Path=/; HttpOnly; SameSite=Strict; Max-Age={max_age}\r\n"
    )


def clear_cookie() -> str:
    return (
        f"Set-Cookie: {COOKIE_NAME}=; Path=/; HttpOnly; SameSite=Strict; "
        f"Max-Age=0\r\n"
    )
