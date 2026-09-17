"""
Deployment-facing configuration. Everything environment-dependent lives here and
nowhere else, so the Cloud Run / Vercel split is a matter of environment
variables rather than code edits.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NTT_DATA_DIR", ROOT / "data" / "source"))

#: Every figure in the product is computed "as of" this date. It follows the
#: calendar — today, read once when the process starts, so every stall age,
#: past-due count and quarter-to-date figure is consistent for the life of the
#: process and moves on when the process restarts on a new day. The customer
#: asked for the as-of date on screen to be the real date; the movement log
#: ends 2026-09-14, so a deal silent since then simply reads as silent for one
#: more day each morning, which is the truth. NTT_AS_OF pins it — the
#: regression harness sets 2026-09-15 so its figures are stable, and a rehearsal
#: can pin the demo day the same way.
AS_OF = date.fromisoformat(os.environ.get("NTT_AS_OF") or date.today().isoformat())

#: NTT's fiscal year starts in April: FY26 runs Apr 2026 - Mar 2027.
FY_START_MONTH = 4

OPPORTUNITIES_CSV = DATA_DIR / "opportunities.csv"
MOVEMENT_CSV = DATA_DIR / "movement.csv"
ANOMALIES_CSV = DATA_DIR / "anomalies.csv"
DISTRIBUTION_CSV = DATA_DIR / "distribution_comparison.csv"
#: The data-science cross-sell export. Arrived with the functional SOW; its
#: granularity is LOB / ServiceCategory, one level finer than the whitespace
#: this layer computes natively, so the two corroborate rather than duplicate.
CROSS_SELL_CSV = DATA_DIR / "cross_sell.csv"

# --------------------------------------------------------------------------- #
# LLM providers
# --------------------------------------------------------------------------- #
# Keys come from the environment and nowhere else. The repository is public, so
# no literal may live here; locally they are read from a gitignored `.env`
# beside this package (loaded below, without a dependency), and on Cloud Run
# they arrive as environment variables. With no key at all the product still
# runs — every AI surface falls back to its computed template and says so.


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from KEY=VALUE lines. Existing variables win, so a
    deployment that sets a name explicitly is never overridden by a stray file."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

# Claude is the first provider in the chain when a key is present: it is the
# paid, quota-backed one, so the free-tier per-minute 429s that keep knocking
# Gemini and OpenRouter out during a demo do not reach the page. NTT_ANTHROPIC_KEY
# is the product's own name; ANTHROPIC_API_KEY is honoured so a developer's
# shell works without a second variable.
ANTHROPIC_KEY = (os.environ.get("NTT_ANTHROPIC_KEY")
                 or os.environ.get("ANTHROPIC_API_KEY", "")).strip()
ANTHROPIC_MODEL = os.environ.get("NTT_ANTHROPIC_MODEL", "claude-opus-5")

GEMINI_KEYS = [
    k.strip()
    for k in os.environ.get("NTT_GEMINI_KEYS", "").split(",")
    if k.strip()
]
GEMINI_MODEL = os.environ.get("NTT_GEMINI_MODEL", "gemini-3.6-flash")

OPENROUTER_KEY = os.environ.get("NTT_OPENROUTER_KEY", "")
OPENROUTER_MODEL = os.environ.get("NTT_OPENROUTER_MODEL", "z-ai/glm-5.2:free")

LLM_TIMEOUT_S = float(os.environ.get("NTT_LLM_TIMEOUT", "35"))
LLM_ENABLED = os.environ.get("NTT_LLM_ENABLED", "1") not in ("0", "false", "False")

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "NTT_ALLOWED_ORIGINS",
        "http://localhost:5178,http://127.0.0.1:5178,"
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:4173,http://127.0.0.1:4173",
    ).split(",")
    if o.strip()
]

# --------------------------------------------------------------------------- #
# The access gate
# --------------------------------------------------------------------------- #
# One shared password on the whole platform, asked for by the customer for the
# demo. It is a door, not an identity: it says nothing about WHO is inside, so
# row-level security still comes from the persona (see api/main.py). Rotate it
# by setting NTT_ACCESS_PASSWORD on the deployment; the literal below is the
# password the customer specified and exists so a clean checkout runs.
ACCESS_PASSWORD = os.environ.get("NTT_ACCESS_PASSWORD", "ntt@2026")

#: The key that signs access tokens. Left empty, api/auth.py derives a stable
#: one from the password with a hash, so tokens survive a process restart but
#: every token dies the moment the password is rotated — which is the
#: behaviour a rotation is for. Set it explicitly to keep tokens valid across
#: a password change.
ACCESS_SECRET = os.environ.get("NTT_ACCESS_SECRET", "")

#: Two ways to turn the gate off, both for development only. The environment
#: variable is for a shell or a container; the marker file is for a working
#: tree where several tools are calling the API at once and none of them
#: carries a token. The file is checked per request so creating or deleting it
#: takes effect immediately, and it is gitignored so it cannot ship.
ACCESS_DISABLED = os.environ.get("NTT_ACCESS_DISABLED", "") in ("1", "true", "True")
ACCESS_DISABLED_FILE = ROOT / ".access-disabled"
