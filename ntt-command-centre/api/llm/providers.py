"""
LLM transport: Claude first (when a key is configured), then Gemini, then
OpenRouter, then the computed template.

Claude goes first because it is the one provider with a real quota behind it.
On the 16 Sep call the team agreed to put the Claude API key on this product
precisely because the free tiers exhaust mid-demo; the two findings below about
Gemini and OpenRouter are why the free providers stay as fallbacks rather than
the main path. The Claude call uses the official SDK with a JSON-schema output
format, so the same schema the other providers are held to is enforced by the
API itself rather than by a prompt asking nicely.

Three findings from probing the supplied keys directly, each of which changes
the code rather than merely informing it:

1. **`thinkingConfig.thinkingBudget: 0` is mandatory on `gemini-3.6-flash`.**
   Thinking is on by default and its tokens are billed against
   `maxOutputTokens`. Measured: a 60-token cap returned `finishReason:
   MAX_TOKENS` with `thoughtsTokenCount: 56` and an EMPTY body. With the budget
   zeroed, the same schema-constrained call returned clean JSON in 2.4s. Without
   this line the product silently renders blank cards.

2. **A hard per-attempt timeout is not optional.** One key hung for a full 60s
   and returned no HTTP status, then succeeded in under two seconds on retry.

3. **OpenRouter's free models fail in a way that looks like success**: some
   return `content: null` with `finish_reason: "length"` because reasoning
   tokens consumed the budget. A null body is treated as an error here, not as
   an empty answer, and `max_tokens` is set generously.

The chain has a WALL-CLOCK deadline, not a per-attempt one. A user waiting on a
card does not care which provider is slow; if the whole chain cannot answer in
time the deterministic template renders instead and the card says it did.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import anthropic

from ..config import (
    ANTHROPIC_KEY,
    ANTHROPIC_MODEL,
    GEMINI_KEYS,
    GEMINI_MODEL,
    LLM_ENABLED,
    LLM_TIMEOUT_S,
    OPENROUTER_KEY,
    OPENROUTER_MODEL,
)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

#: OpenRouter's documented free model has been observed routing to a provider
#: returning 404. The `models` array lets OpenRouter fail over itself.
OPENROUTER_FALLBACKS = ["z-ai/glm-5.2:free", "google/gemma-4-31b-it:free",
                        "nvidia/nemotron-3-super-120b-a12b:free"]


@dataclass
class LLMResult:
    obj: dict
    provider: str
    model: str
    latency_ms: int
    degraded: bool = False
    attempts: list[dict] = field(default_factory=list)


class LLMUnavailable(RuntimeError):
    """Every provider failed or the deadline expired. Caller renders template."""


class RateLimited(RuntimeError):
    """The provider refused for now and said when to come back."""

    def __init__(self, retry_after: float):
        super().__init__(f"rate limited, retry in {retry_after:.0f}s")
        self.retry_after = retry_after


def _retry_after(body: bytes) -> float:
    """
    Seconds until the quota window reopens, from the provider's own response.

    Gemini returns a `RetryInfo` detail with a `retryDelay` like "2s" on a 429.
    Honouring it is the difference between a working feature and a permanently
    degraded one: the free tier is a per-MINUTE allowance, so the wait is
    usually a couple of seconds, not a quota that has been spent.
    """
    try:
        err = json.loads(body.decode()).get("error", {})
        for d in err.get("details", []):
            delay = d.get("retryDelay")
            if isinstance(delay, str) and delay.endswith("s"):
                return float(delay[:-1] or 0)
    except (ValueError, AttributeError, TypeError):
        pass
    return 0.0


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={
        "content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RateLimited(_retry_after(e.read())) from e
        raise


def _to_gemini_schema(schema: dict) -> dict:
    """JSON Schema -> Gemini's uppercase dialect."""
    t = schema.get("type", "object")
    out: dict = {"type": {"object": "OBJECT", "array": "ARRAY", "string": "STRING",
                          "number": "NUMBER", "integer": "INTEGER",
                          "boolean": "BOOLEAN"}.get(t, "STRING")}
    if t == "object":
        out["properties"] = {k: _to_gemini_schema(v)
                             for k, v in schema.get("properties", {}).items()}
        if schema.get("required"):
            out["required"] = schema["required"]
    elif t == "array":
        out["items"] = _to_gemini_schema(schema.get("items", {"type": "string"}))
    if schema.get("enum"):
        out["enum"] = [str(e) for e in schema["enum"]]
    return out


def _to_anthropic_schema(schema: dict) -> dict:
    """
    JSON Schema -> the subset the structured-output format accepts.

    Every object is closed (`additionalProperties: false`) so a stray field the
    grounding layer would never read cannot appear, and `required` is carried
    through untouched: the planner schema deliberately leaves most fields
    optional (see the note on `refuseReason` in service.py).
    """
    out = dict(schema)
    t = schema.get("type")
    if t == "object":
        out["properties"] = {k: _to_anthropic_schema(v)
                             for k, v in schema.get("properties", {}).items()}
        out["additionalProperties"] = False
    elif t == "array":
        out["items"] = _to_anthropic_schema(schema.get("items", {"type": "string"}))
    return out


_anthropic_client: anthropic.Anthropic | None = None


def _anthropic(system: str, user: str, schema: dict, max_tokens: int,
               timeout: float) -> dict:
    """
    One structured call to Claude.

    Two things about the request shape are deliberate. `max_tokens` is set far
    above the caller's figure because adaptive thinking is on and its tokens
    count against the cap — a 500-token ceiling would be spent before the JSON
    started, which is exactly the Gemini failure in finding 1. Depth is
    controlled with `effort: "low"` instead: these are short, schema-bound
    narrations over pre-formatted figures, not reasoning tasks, and low effort
    keeps them inside the chain's deadline. No sampling parameters are sent —
    this model family rejects them.

    Server-side refusal fallbacks are requested as the API recommends; if the
    account or SDK does not accept that beta the call is retried without it.
    """
    global _anthropic_client
    if _anthropic_client is None:
        # Retries are the chain's job, not the SDK's: a 429 here should move on
        # to the next provider, not wait inside this attempt.
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY, max_retries=0)
    client = _anthropic_client.with_options(timeout=timeout)
    request = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max(16000, max_tokens * 8),
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": _to_anthropic_schema(schema)},
        },
    }
    try:
        r = client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request)
    except anthropic.BadRequestError as e:
        low = str(e).lower()
        if "fallback" not in low and "beta" not in low:
            raise
        r = client.messages.create(**request)

    if r.stop_reason == "refusal":
        raise LLMUnavailable("claude declined this request")
    text = next((b.text for b in r.content if b.type == "text"), None)
    if not text:
        raise LLMUnavailable(f"claude returned no text (stop_reason={r.stop_reason})")
    return json.loads(text)


def _gemini(system: str, user: str, schema: dict, max_tokens: int,
            temperature: float, key: str, timeout: float) -> dict:
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
            "responseMimeType": "application/json",
            "responseSchema": _to_gemini_schema(schema),
            # See finding 1 in the module docstring. Without this the body is empty.
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    d = _post(GEMINI_URL.format(model=GEMINI_MODEL, key=key), payload, {}, timeout)
    cand = (d.get("candidates") or [{}])[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = next((p.get("text") for p in parts if p.get("text")), None)
    if not text:
        raise LLMUnavailable(
            f"gemini returned no text (finishReason={cand.get('finishReason')})")
    return json.loads(text)


def _openrouter(system: str, user: str, schema: dict, max_tokens: int,
                temperature: float, timeout: float) -> dict:
    payload = {
        "model": OPENROUTER_MODEL,
        "models": OPENROUTER_FALLBACKS,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        # Generous, because reasoning tokens are consumed even when excluded.
        "max_tokens": max(max_tokens * 3, 1500),
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        # `enabled: false` is rejected outright by this endpoint; `exclude`
        # keeps the reasoning out of the payload without disabling it.
        "reasoning": {"exclude": True},
    }
    d = _post(OPENROUTER_URL, payload, {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "X-Title": "NTT Deal Intelligence",
    }, timeout)
    if d.get("error"):
        raise LLMUnavailable(f"openrouter: {d['error'].get('message')}")
    msg = ((d.get("choices") or [{}])[0].get("message") or {})
    content = msg.get("content")
    if not content:
        raise LLMUnavailable("openrouter returned a null body (reasoning ate the budget)")
    return json.loads(content)


def complete(system: str, user: str, schema: dict, *, max_tokens: int = 900,
             temperature: float = 0.2, deadline_s: float | None = None) -> LLMResult:
    """
    Run the chain until one provider answers, or raise `LLMUnavailable`.

    Keys are rotated per call rather than per failure, so a rate limit on one
    key does not pin every request onto the other one.
    """
    if not LLM_ENABLED:
        raise LLMUnavailable("LLM disabled by configuration")

    deadline = time.monotonic() + (deadline_s or LLM_TIMEOUT_S)
    attempts: list[dict] = []

    def left() -> float:
        return deadline - time.monotonic()

    # Claude first: one attempt, no waiting. A 429 or a network fault simply
    # hands the call to the free providers below within the same deadline.
    if ANTHROPIC_KEY and left() > 1.0:
        t0 = time.monotonic()
        try:
            obj = _anthropic(system, user, schema, max_tokens, min(left(), 30.0))
            ms = int((time.monotonic() - t0) * 1000)
            attempts.append({"provider": "anthropic", "ms": ms, "outcome": "ok"})
            return LLMResult(obj, "anthropic", ANTHROPIC_MODEL, ms, attempts=attempts)
        except anthropic.RateLimitError as e:
            wait = e.response.headers.get("retry-after", "?")
            attempts.append({"provider": "anthropic",
                             "ms": int((time.monotonic() - t0) * 1000),
                             "outcome": f"429, retry in {wait}s"})
        except (anthropic.APIError, ValueError, KeyError, LLMUnavailable,
                TimeoutError, OSError) as e:
            attempts.append({"provider": "anthropic",
                             "ms": int((time.monotonic() - t0) * 1000),
                             "outcome": f"{type(e).__name__}: {str(e)[:120]}"})

    # Each Gemini key gets one immediate try; a key that was rate-limited is
    # revisited once at the end, after the delay it asked for, if the deadline
    # allows.
    deferred: list[tuple[int, str, float]] = []

    for i, key in enumerate(GEMINI_KEYS):
        if left() <= 1.0:
            break
        t0 = time.monotonic()
        try:
            obj = _gemini(system, user, schema, max_tokens, temperature, key,
                          min(left(), 20.0))
            ms = int((time.monotonic() - t0) * 1000)
            attempts.append({"provider": "gemini", "key": i, "ms": ms, "outcome": "ok"})
            return LLMResult(obj, "gemini", GEMINI_MODEL, ms, attempts=attempts)
        except RateLimited as e:
            attempts.append({"provider": "gemini", "key": i,
                             "ms": int((time.monotonic() - t0) * 1000),
                             "outcome": f"429, retry in {e.retry_after:.0f}s"})
            deferred.append((i, key, e.retry_after))
        except (urllib.error.URLError, OSError, ValueError, KeyError,
                LLMUnavailable, TimeoutError) as e:
            attempts.append({"provider": "gemini", "key": i,
                             "ms": int((time.monotonic() - t0) * 1000),
                             "outcome": f"{type(e).__name__}: {str(e)[:120]}"})

    # Come back to the quickest rate-limited key, if waiting still leaves time
    # to make the call and return inside the deadline.
    for i, key, wait in sorted(deferred, key=lambda d: d[2]):
        if wait <= 0 or wait + 3.0 > left():
            continue
        time.sleep(wait)
        t0 = time.monotonic()
        try:
            obj = _gemini(system, user, schema, max_tokens, temperature, key,
                          min(left(), 20.0))
            ms = int((time.monotonic() - t0) * 1000)
            attempts.append({"provider": "gemini", "key": i, "ms": ms,
                             "outcome": "ok after waiting"})
            return LLMResult(obj, "gemini", GEMINI_MODEL, ms, attempts=attempts)
        except (RateLimited, urllib.error.URLError, OSError, ValueError, KeyError,
                LLMUnavailable, TimeoutError) as e:
            attempts.append({"provider": "gemini", "key": i,
                             "ms": int((time.monotonic() - t0) * 1000),
                             "outcome": f"retry: {type(e).__name__}: {str(e)[:90]}"})
            break

    if OPENROUTER_KEY and left() > 1.0:
        t0 = time.monotonic()
        try:
            obj = _openrouter(system, user, schema, max_tokens, temperature,
                              min(left(), 25.0))
            ms = int((time.monotonic() - t0) * 1000)
            attempts.append({"provider": "openrouter", "ms": ms, "outcome": "ok"})
            return LLMResult(obj, "openrouter", OPENROUTER_MODEL, ms, attempts=attempts)
        except (RateLimited, urllib.error.URLError, OSError, ValueError, KeyError,
                LLMUnavailable, TimeoutError) as e:
            attempts.append({"provider": "openrouter",
                             "ms": int((time.monotonic() - t0) * 1000),
                             "outcome": f"{type(e).__name__}: {str(e)[:120]}"})

    # Every provider failed. Summarise WHY in one human sentence — the raw
    # attempt list is diagnostics and is carried separately, because it used to
    # be rendered verbatim into the page and a user does not need to read an
    # HTTP stack trace to understand that a sentence was computed rather than
    # written.
    outcomes = [a["outcome"] for a in attempts]
    if outcomes and all("429" in o or "rate limit" in o.lower() for o in outcomes):
        # Gemini's 429 carries RetryInfo of about a minute — this is a
        # per-minute allowance being hit, not a quota being spent, so the
        # message says so rather than implying the key is dead.
        why = "the language service is busy; it frees up within a minute"
    elif any("Timeout" in o or "timed out" in o for o in outcomes):
        why = "no language provider answered in time"
    elif not outcomes:
        why = "the language layer is switched off"
    else:
        why = "no language provider was reachable"
    err = LLMUnavailable(why)
    err.attempts = attempts  # type: ignore[attr-defined]
    raise err


def health() -> dict:
    # `enabled` answers whether a model can actually be reached, not whether
    # the switch is on. With the flag up and no key configured every call
    # falls straight through to the computed template, and a probe that said
    # `enabled: true` there had monitoring believe the AI was live while every
    # card was rendering degraded. The configured providers are listed in the
    # order the chain tries them.
    configured = [name for name, ok in (("anthropic", bool(ANTHROPIC_KEY)),
                                        ("gemini", bool(GEMINI_KEYS)),
                                        ("openrouter", bool(OPENROUTER_KEY))) if ok]
    return {
        "enabled": bool(LLM_ENABLED and configured),
        "flag": LLM_ENABLED,
        "providersConfigured": configured,
        "anthropicConfigured": bool(ANTHROPIC_KEY),
        "anthropicModel": ANTHROPIC_MODEL,
        "geminiKeys": len(GEMINI_KEYS),
        "geminiModel": GEMINI_MODEL,
        "openrouterModel": OPENROUTER_MODEL,
        "openrouterConfigured": bool(OPENROUTER_KEY),
        "timeoutSeconds": LLM_TIMEOUT_S,
    }
