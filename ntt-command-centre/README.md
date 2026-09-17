# NTT DATA — Deal Intelligence

The insight experience layer over NTT Global's North America sales pipeline, built by Syren Cloud.
React + TypeScript + Vite with D3 for every visualisation; FastAPI + pandas for the semantic layer.

```bash
./run.sh            # both halves — API on :8808, app on http://localhost:5178
./run.sh verify     # the regression harness; exit code is the failure count
```

---

## What it is

Three datasets from the client, three personas, and three use cases the data-science team defined on
the 16 Sep call. Every figure on screen is computed by `api/semantic/` from those files at request
time. **There is no mock and no fallback number**: if the API is down the page says so rather than
showing something that looks real.

### The data

| File | Rows | What it is |
|---|---|---|
| `NA_Synthetic_SFDC_Opportunities 1.xlsx` | 3,034 | The fact table, at **opportunity-line** grain. 2,050 opportunities, 363 accounts, 70 reps, $58.85M ACV revenue, $9.55M ACV GP. |
| `NA_Synthetic_SFDC_Opportunity_Movement.csv` | 36,631 | The field-level change log. Every behavioural finding in the product comes from here. |
| `Client_Anomaly_Report.csv` | 485 | The data-science team's detections across 7 categories and 23 types. |
| `NA_SFDC_Deal_Closure_Model_v2_SHAP_Benchmarks.xlsx` | 14,872 | Their closure model: calibrated P(win), risk buckets, a per-deal SHAP driver, and 9 benchmarked features. Covers **all** of this book's open deals. |

`AS_OF` is **today**, read once when the process starts, so every figure is consistent for the
life of the process and moves on with the calendar. `NTT_AS_OF` pins it: the regression harness
pins 2026-09-15 so its figures are stable, and a rehearsal can pin the demo day the same way.

---

## The three use cases

**1 · Anomaly detection** — `api/semantic/anomalies.py`

The 485 DS findings are ingested, enriched against the fact table (LOB, portfolio, owner, stage,
value), triaged by severity × money, and routed to the persona who can actually act on them. On top
of that, **11 anomaly types are recomputed here from the movement log** so the capability is
demonstrably live rather than a CSV replay. Where both detectors independently flag the same thing
the card says so — 186 findings are corroborated that way, including three rep-level patterns
(Brian Thompson's pipeline concentration, Paul Martinez's value shrinkage, Rebecca Robinson's
renewal mix) that this layer found before the DS file was read. Every card drills to the actual
change-log rows that produced it.

**2 · Deal closure likelihood** — `api/semantic/ds_model.py` + `predict.py`

The DS team's model is the headline probability, with its per-deal SHAP driver and its 9
direction-aware GREEN/RED feature benchmarks. Four of those nine carry |r| < 0.02 with the outcome
and the workbook says their colour is near-noise; the UI renders those muted and prints the caveat
rather than passing noise off as a verdict.

**3 · Cross-sell / upsell** — `api/semantic/accounts.py`

"This company only bought X; similar companies bought X, Y and Z." Computed natively, so it works
now rather than waiting on the fourth DS file. The finding is the **asymmetry**: accounts holding
Security almost always also hold Networking, while Networking accounts mostly do not hold Security —
that direction is which play to run. Each candidate is sized against **size-matched** comparables
(same total-GP quartile), median not mean, and the method is printed on the card.

---

## The honest read on the model

> Test AUC **0.5952** against 0.50 for chance.

This is stated on screen, not buried, and it is the most important thing in the product.

`ForecastCategory` and `Confidence` **encode the outcome** in this extract — P(Won | Closed) = 1.000,
P(Won | Omitted) = 0.000, every deal at confidence ≤ 0.19 lost and every deal ≥ 0.97 won. Strip them
and nothing else is strong: no remaining feature exceeds |r| = 0.06, every closed deal reaches
Finalist so the stage path does not discriminate, and segment win rates span only 31–44%.

This layer reproduced that independently before the DS workbook arrived (leakage-free holdout AUC
0.5628) and both numbers are shown side by side. Two teams reaching the same weak result by
different routes is a finding worth displaying: **the ceiling is in the data, not in either method.**

So closure probability is presented as a ranking signal, and the surface the product acts on is
**Deal Risk** — a transparent sum of observable facts (days silent, days past close, close-date slip,
value shrinkage, skipped stages, owner's peer-relative pattern), each of which can be defended to
the rep whose deal it flags. That is also the direct answer to the client's "if a deal is at higher
risk, what is the variable driving it".

---

## Three personas, fourteen pages

Persona changes the **row set**, not the emphasis. The predicate is applied in `slice_frame` before
any aggregation — filtering after would leak the denominator and quietly make every percentage wrong.

| | Scope | Pages | Unique capability |
|---|---|---|---|
| **Account Executive** | own deals | My Day · My Deals · My Accounts · My Track Record | Deal timeline reconstruction with silence gaps drawn to scale |
| **Sales Manager** | a pod of ~11 reps | Pod Pulse · Rep Benchmark · Process · Calibration · Pod Whitespace | Rep behaviour fingerprint against the all-rep percentile distribution |
| **Executive** | everything | Executive Brief · Performance · Structure · Risks · Decisions | The board narrative, with the data caveats volunteered rather than caught |

A persona cannot reach another's page — a deep link redirects to its own home. Actions carry
role-appropriate verbs: an AE gets *call this account today*, a manager *coach this rep*, an
executive *reallocate coverage to this LOB*.

**Pods are derived and the product says so.** The extract has no manager or team column. The 70 reps
are snake-drafted by owned GP into 6 pods so every pod holds a comparable mix; the rule is printed
in the persona switcher rather than the roster being passed off as a source field.

---

## The shell

The chrome is the deployed Cursor build's, rebuilt on this app's tokens — the parts of it that were
right were right, and re-inventing them would have been vanity.

| | |
|---|---|
| **Grouped sidebar** | Not a tab rail. Fourteen pages across three profiles is a product, so pages are grouped ("Today / My book / Me", "The read / The numbers / The decisions") and each item carries its page's stated question as its second line — *Calibration* alone does not tell you whether it is the page you want. Built from `meta.pages`, so a profile cannot see the name of someone else's page. |
| **Filter deck** | Every validated dimension gets a control, per persona — an AE has no rep selector because there is one rep in their scope. Selecting dispatches to the reducer and refetches; **nothing is filtered on the client**, because a client-side filter gives a narrowed numerator over an unfiltered denominator and every percentage on the page goes quietly wrong. |
| **KPI tiles** | Icon, value, sub, and a sparkline **only where a real series exists**. Open pipeline is reconstructed honestly — at each month end a deal was open if it had been created and had not yet closed — and snapshot measures with no history get no line at all, because a flat stroke under a number implies a trend was checked. Past-due history is deliberately *not* offered: the committed close date moved over each deal's life and the fact table keeps only the final value. |
| **Findings bell** | Badged with what needs a decision now, not an unread count. There is nothing to mark as read. |
| **Ask** | In the header and as a floating action, because it must be reachable from every page and the header collapses on narrow screens. |

Below 1100px the rail becomes a drawer and the shell a single column; below 720px the filter deck
scrolls sideways as one row rather than eating the fold. The layout is checked at 375px.

---

## The AI cannot repeat the charts

The client's constraint was explicit: *the AI features must be unique and must not feel redundant
with the charts.* That is enforced mechanically, not asked for in a prompt.

Every `ChartSpec` declares `says: string[]` — claim keys like `gp.open.by:lob`. The view payload
unions them into `chartsSay`. Every AI sentence carries a **lens**:

| lens | licence |
|---|---|
| `cause` `norm` `delta` `action` `answer` | always — a chart cannot show mechanism, a peer norm, a change, or a prescription |
| `state` | **only if its claim is not already on screen** |

`grounding.strip_redundant()` drops the rest server-side, and the server recomputes `chartsSay`
itself and unions it with the client's, so a stale client cannot unlock a redundant answer.

**Numbers are equally mechanical.** The model receives only pre-formatted strings under English
labels — never a raw float, so it cannot do arithmetic — and `grounding.verify()` rejects any
sentence containing a numeric token the server did not supply. In testing it has caught real
hallucinations (`3.0x` coverage, `77%`) while passing every figure the server actually handed over.

### Ad-hoc questions

Free text → the model emits a constrained **query plan** (never SQL, never a column name outside the
catalog) → the server validates every field against the live registries and executes a fixed
dispatch over the semantic layer → a chart is chosen from the result's shape → a second model call
narrates it, with its own chart added to the forbidden list so it cannot describe the bars it just
drew. User text never reaches pandas; there is no `eval`; and a plan cannot widen the caller's scope.

`POST /api/v1/query` runs the same path with no model in the loop, so the contract is testable
independently of whether a language model is available.

**Providers:** Claude (`claude-opus-5`, when `NTT_ANTHROPIC_KEY` is set — the paid, quota-backed
provider goes first) → Gemini `gemini-3.6-flash` (two keys, rotated) → OpenRouter → a deterministic
computed answer. Every suggested question, and every question typed beside a chart, has a computed
answer (`api/semantic/answers.py`, `api/llm/chart_ask.py`) so the product never says "cannot answer"
when the free tiers are exhausted. Three things the transport had to learn the hard way:

- `thinkingConfig.thinkingBudget: 0` is **mandatory**. Without it, thinking tokens are billed
  against `maxOutputTokens` and the body comes back empty — a 60-token cap returned
  `finishReason: MAX_TOKENS` with `thoughtsTokenCount: 56` and nothing else.
- A 429 carries `RetryInfo.retryDelay`, usually a couple of seconds, because **the free tier is a
  per-minute allowance rather than a spent quota**. The chain reads that delay and comes back for
  the quickest rate-limited key when the deadline can absorb the wait.
- OpenRouter's free models can return `content: null` with `finish_reason: "length"` when reasoning
  tokens eat the budget. That is treated as an error, not as an empty answer.

The generated surfaces are cached for the process lifetime rather than minutes: the cache key is
persona + identity + page + filters, and `AS_OF` is pinned, so the inputs behind a key cannot change
while the process lives. Re-asking for a brief nobody's inputs have changed is what exhausts a
per-minute allowance.

When every provider is unavailable the computed template renders instead — written in complement
lenses, so it reads as finished product — and the response carries `degraded: true` with a one-line
human reason. The raw provider attempts stay in the payload for debugging and never reach the page.

---

## The context layer

`GET /api/v1/catalog` — the machine-readable description of this business that the model is given
instead of the data. Entities, every column's business meaning, enumerated values, every measure with
its formula in words and in pandas, the joins, the business rules, and **nine caveats**, because a
model that is not told `ForecastCategory` is circular will produce confident nonsense. 4,714 tokens
against a 6,000 budget. It is published deliberately: another consumer reasoning over this business
reads the same description our own model does.

---

## The chart repository

Indexed by the **shape** of the data, resolved by one rule. The server resolves it and stamps
`repositoryKey`; the client re-runs the same rule and prints `shape → key` in every card footer, so
the claim is checkable on screen. Fourteen types are in use across the three personas.

| | serves | the question only it answers |
|---|---|---|
| `waterfall.bridge` | `bridge` | What *moved* the number — plan to position |
| `mekko.marimekko` | `categorical×measure×width` | Size, mix and margin in one picture |
| `sankey.flow` | `source×target×measure` | Where value actually flows, and how it ends |
| `treemap.nested` | `hierarchy×measure` | How concentrated we are (one account is 14.5%) |
| `gantt.timeline` | `entity×start×end` | Which deals have run out of road |
| `funnel.stage` | `cohort×stage` | Where the pod leaks — **cohort-anchored** |
| `bubble.scatter` | `x×y×size` | What to touch first |
| `combo.columnline` | `temporal×measure×measure` | Are we tracking month to month against plan |
| `bullet.target` | `target×actual` | Open pipeline against what is still left of the plan, per line |
| `heat.grid` `bar.categorical` `bar.stacked` `line.timeseries` `table.compact` | | the basic set |

The funnel is cohort-anchored because a naive stage funnel on this data **exceeds 100%** partway
down — deals are logged straight into the middle of the ladder. It is anchored on opportunities whose
history starts at Identification, and the module re-checks monotonicity at render time and refuses to
draw a funnel that widens.

---

## Data defects, surfaced rather than hidden

Lead with these; do not wait to be asked. They are on the Context surfaces and in the catalog.

- **`ForecastCategory` and `Confidence` encode the outcome.** Fine as filters, fatal as features.
- **FY26-Q3 has not happened.** Its zero attainment and red RAG are a calendar position. The mirror
  is that Q1 and Q2 read 149% and 170% of plan, because the annual plan was rolled evenly across
  three quarters while every win sits in Q1–Q2.
- **The plan does not foot at LOB grain.** Budget cells are denormalised onto opportunity rows, so a
  cell exists only where a line happens to sit there. 52 of 60 quarter cells and 136 of 180 month
  cells are present; the Q3 breakdown is 8.8% short of its headline and December 23.9% short. Every
  breakdown returns its own `residual` and the UI prints it — on the performance grid and on the
  actions-page bullets, in words: the rows add up to $317K of the $370K still to deliver, and the
  footnote says which cells are missing and how much surplus was floored.
- **Coverage has one definition.** `budget.forward_window` picks the window (the current quarter
  onward, or the one quarter a filter names) and `budget._coverage_cells` computes plan, won and
  open per LOB × portfolio over it; the tile, the grid, the bullets and the executive's coverage
  cards are all rolled up from those cells, so they cannot disagree about a line.
- **`value_at_stake` in the anomaly table is gross profit, not revenue** — ~6× off against a revenue
  chart.
- **Concentration.** One account is 14.5% of GP and the top five are 39%, so any share chart groups
  its tail and any "typical account" statement uses a median.
- **`AccountGroup` is blank ~65% of the time by design** — the client says the sibling-account de-dup
  is not always done. Shown as `(ungrouped)`, not as a defect.
- **Win rate does not vary meaningfully by rep.** Behaviour does. Coach the behaviour.

---

## Architecture

```
data/source/              the client's four files, canonicalised
        │
api/semantic/             THE SEMANTIC LAYER — every rule, definition and calculation
        │  loader · movement_features · measures · dimensions · budget · accounts
        │  anomalies · predict · ds_model · personas · charts · actions · narrative
        │  catalog (the LLM's context) · query (NL plan → validated execution) · views
        │
api/llm/                  providers · grounding · prompts · service
        │
api/main.py               THE API BOUNDARY — auth and RLS attach here and nowhere else
        │
web/src/                  THE FRONT END — draws payloads, computes nothing
           charts/        the D3 repository, indexed by data shape
           lenses/        one page template, fourteen pages
           state/         one reducer, mirrored into the URL
```

**The front end holds no arithmetic.** `GET /api/v1/measures` returns the raw measure dictionary with
no view model at all, which is the proof the semantic layer is not welded to this UI.

Nothing ever opens a window. Persona, identity, page, every filter, the measure toggle and the theme
live in the query string, so any view is reproducible from its URL.

---

## Deployment

Front end → Vercel (`npm run build`, output `web/dist`, root directory `web/`). API → GCP Cloud Run
(`uvicorn api.main:app`, built from the `Dockerfile` here). Everything environment-dependent is in
`api/config.py`:

| Variable | What |
|---|---|
| `NTT_ACCESS_PASSWORD` | The one shared password on the whole platform (default `ntt@2026`). `NTT_ACCESS_SECRET` signs the bearer tokens; `NTT_ACCESS_DISABLED=1` or a `.access-disabled` file beside this README turns the gate off for local scripts. |
| `NTT_ANTHROPIC_KEY` | Claude, first in the chain. `NTT_ANTHROPIC_MODEL` defaults to `claude-opus-5`. |
| `NTT_GEMINI_KEYS`, `NTT_OPENROUTER_KEY` | The free-tier fallbacks. |
| `NTT_ALLOWED_ORIGINS` | The front end's origins, comma-separated. |
| `NTT_AS_OF` | Pins the business date; unset, the product follows the calendar. |
| `NTT_DATA_DIR` | The four source files; the image ships them under `/app/data/source`. |

Set `VITE_API_BASE` on the front end when the two halves are not same-origin.

**No key is committed.** Locally they come from a gitignored `.env` beside this README; on Cloud Run
they are environment variables. The password is not identity — row-level security still comes from
the persona switch — it is the door.
