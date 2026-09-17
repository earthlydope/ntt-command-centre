#!/usr/bin/env bash
#
# Screenshots of every lens, in both themes, straight out of the running app.
#
# Both halves must already be up (./run.sh). Every view is addressed by URL —
# lens, filters, measure, persona, theme and even the Ask overlay are all in the
# query string — so a screenshot is reproducible and nothing has to be clicked
# to take one. That property is not for the screenshots; it is what lets an
# Outlook card link back into an exact slice.
#
set -euo pipefail
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/screenshots"
WEB="${NTT_WEB_URL:-http://localhost:5178}"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
SIZE="${SIZE:-1600,2200}"
mkdir -p "$OUT"

shot() { # shot <file> <query>
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars \
    --window-size="$SIZE" --virtual-time-budget=9000 \
    --screenshot="$OUT/$1.png" "$WEB/$2" >/dev/null 2>&1
  echo "  $1.png"
}

for theme in dark light; do
  for lens in home context outcomes drivers actions; do
    shot "${lens}-${theme}" "?lens=${lens}&theme=${theme}"
  done
done

# The cross-filter beat: the same lens, scoped by a chart-mark click to a single
# stage, with a recomputed narrative, KPI row and action list.
shot "home-filtered-stage-identification-dark" "?stage=Identification&theme=dark"
shot "drivers-filtered-lob-security-dark" "?lens=drivers&lob=Security&theme=dark"

# The Ask AI Expert overlay and the explainability detail, both in-page.
shot "ask-overlay-dark" "?ask=past_due&theme=dark"
shot "ask-overlay-light" "?ask=coverage&theme=light"
shot "insight-past-due-dark" "?insight=past_due&stage=Identification&theme=dark"

# Row-level security: the same page as a principal who cannot see the whole file.
shot "home-persona-networking-dark" "?persona=lob_lead_networking&theme=dark"

echo "→ $OUT"
