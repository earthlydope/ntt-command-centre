"""
The password on the door.

The customer asked for one password on the whole platform, and this module is
all of it: a login endpoint that turns the password into a bearer token, a
verifier, and the ASGI middleware that refuses every other request until that
token is presented.

What it is NOT. This is a shared demo password, not identity. Knowing it says
you are allowed into the demo, not who you are, so nothing downstream reads
anything from the token: row-level security still resolves from the persona
and identity exactly as it did before (`api/main.py`), and a token cannot
widen a scope. When the product moves behind Entra, this middleware is what
gets replaced and the endpoints behind it do not change.

The token is bearer-only. There is no cookie, so the browser never sends it
on its own and there is nothing for a cross-site page to ride on; the client
attaches it to each request explicitly. Its shape is
`urlsafe-base64("<expiry>.<signature>")`, where the signature is an HMAC over
the expiry under a secret derived from the password. That makes a token
self-describing — no session table, nothing to survive a restart — and it
means every outstanding token is void the moment the password is rotated
through `NTT_ACCESS_PASSWORD`, which is exactly what a rotation is for.

Two development switches turn the gate off: `NTT_ACCESS_DISABLED=1` in the
environment, or the presence of `.access-disabled` beside `api/`. The file is
gitignored and checked on every request, so it cannot ship and it takes
effect the instant it appears or disappears — no restart, no reload.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from datetime import datetime, timezone
from urllib.parse import parse_qs

from fastapi import APIRouter, Body, HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import (
    ACCESS_DISABLED,
    ACCESS_DISABLED_FILE,
    ACCESS_PASSWORD,
    ACCESS_SECRET,
)

#: A month. Long enough that the customer types the password once per device
#: for the life of the demo; short enough that a forgotten laptop ages out.
TOKEN_TTL_S = 30 * 24 * 60 * 60

#: Reachable without a token. The health probe, because Cloud Run's checker
#: does not carry one; the login itself, obviously; and the OpenAPI pages,
#: because the contract is meant to be readable. Nothing under /api/ that
#: returns a figure is in this set.
EXEMPT_PATHS = frozenset({"/healthz", "/api/health", "/api/auth/login", "/docs", "/openapi.json"})


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #


def _secret() -> bytes:
    """
    The signing key. An explicit `NTT_ACCESS_SECRET` wins; otherwise the key
    is a hash of the password, so it is stable across reloads (tokens keep
    working) yet changes with the password (tokens stop working).
    """
    if ACCESS_SECRET:
        return ACCESS_SECRET.encode("utf-8")
    return hashlib.sha256(b"ntt-access-gate:" + ACCESS_PASSWORD.encode("utf-8")).digest()


def _sign(exp: int) -> str:
    return hmac.new(_secret(), str(exp).encode("ascii"), hashlib.sha256).hexdigest()


def check_password(candidate: str) -> bool:
    """Constant-time comparison, so a wrong guess takes as long as a right one."""
    return hmac.compare_digest(candidate.encode("utf-8"), ACCESS_PASSWORD.encode("utf-8"))


def issue(now: float | None = None) -> tuple[str, int]:
    """Mint a token. Returns it with its expiry as a unix timestamp."""
    exp = int(time.time() if now is None else now) + TOKEN_TTL_S
    raw = f"{exp}.{_sign(exp)}".encode("ascii")
    return base64.urlsafe_b64encode(raw).decode("ascii"), exp


def verify(token: str) -> bool:
    """True only for a token this process (or one sharing its secret) minted
    and whose expiry has not passed. Malformed input is simply False."""
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii")).decode("ascii")
        exp_text, sig = raw.split(".", 1)
        exp = int(exp_text)
    except ValueError:
        # Covers bad base64 (binascii.Error), a missing dot, a non-integer
        # expiry and non-ASCII input alike; none of them is a token.
        return False
    if exp <= time.time():
        return False
    return hmac.compare_digest(sig, _sign(exp))


def enabled() -> bool:
    """Whether the gate is on. Evaluated per request so the marker file works
    without a restart; a stat is far cheaper than any handler behind it."""
    return not ACCESS_DISABLED and not ACCESS_DISABLED_FILE.exists()


# --------------------------------------------------------------------------- #
# The endpoint
# --------------------------------------------------------------------------- #

router = APIRouter()


@router.post("/api/auth/login")
def login(body: dict = Body(...)) -> dict:
    """Exchange the password for a bearer token. A wrong password is a 401
    with no hint about how wrong; the client says so in its own words."""
    password = body.get("password", "") if isinstance(body, dict) else ""
    if not isinstance(password, str) or not check_password(password):
        raise HTTPException(status_code=401, detail="wrong password")
    token, exp = issue()
    return {
        "token": token,
        "expiresAt": datetime.fromtimestamp(exp, tz=timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# The middleware
# --------------------------------------------------------------------------- #


def _token_from_scope(scope: Scope) -> str | None:
    """The bearer header first; `?token=` second, for the rare direct link to
    an endpoint from somewhere a header cannot be set."""
    for name, value in scope.get("headers", []):
        if name == b"authorization":
            parts = value.decode("latin-1").split(None, 1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1].strip()
            break
    found = parse_qs(scope.get("query_string", b"").decode("latin-1")).get("token")
    return found[0] if found else None


class AccessGate:
    """
    Pure ASGI, deliberately: no BaseHTTPMiddleware, so nothing here buffers a
    body or wraps the response, and the cost of the gate on a permitted
    request is one header scan.

    It sits INSIDE CORSMiddleware (`api/main.py` adds it first, CORS last, and
    Starlette's last-added middleware is the outermost), so a 401 still
    carries the CORS headers the browser needs to read it. Without that, a
    locked-out page would see an opaque network error instead of the status
    that tells it to show the gate.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not enabled() or self._exempt(scope):
            await self.app(scope, receive, send)
            return
        token = _token_from_scope(scope)
        if token and verify(token):
            await self.app(scope, receive, send)
            return
        refusal = JSONResponse(
            {"detail": "locked"},
            status_code=401,
            headers={"www-authenticate": "Bearer"},
        )
        await refusal(scope, receive, send)

    @staticmethod
    def _exempt(scope: Scope) -> bool:
        # OPTIONS is exempt so a preflight that reaches this layer is answered,
        # though in practice CORSMiddleware has already handled it upstream.
        return scope.get("method") == "OPTIONS" or scope.get("path") in EXEMPT_PATHS
