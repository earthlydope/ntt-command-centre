/**
 * How each rep works, next to the rest of the team.
 *
 * WHAT WAS WRONG. This table used to print three numbers in every cell — the
 * rep's rate, the team's rate and a z-score — so a cell read "38%peer 38%-0.01σ".
 * Six behaviours times eleven reps is 198 numbers on one screen, and the one
 * question a manager actually brings to it ("who is off the pattern, and on
 * what?") could not be answered without reading every one of them.
 *
 * WHAT IT DOES NOW. One number per cell — the rep's own rate, which is the
 * figure a manager will quote in the conversation. Underneath it, a short track
 * with a line down the middle: that line IS the team norm, and the dot is this
 * rep. Left of the line is below the norm, right is above, and the dot is
 * coloured by whether that is better or worse for this particular behaviour —
 * more wins is good, more stalling is not. So the table is read by SHAPE: the
 * coloured dots are the reps who are off the pattern, and the column they sit
 * in is what they are off on. No number has to be read to see it.
 *
 * The exact team norm and the distance from it are still there, on the cell's
 * tooltip and in a screen-reader line, because the manager who is about to
 * raise this with someone needs the evidence. They are just not on the face of
 * the table any more.
 *
 * SORTING IS BY DISTANCE FROM THE NORM, not by the rate, because "who is
 * furthest from what everyone else does" is the question the table answers. A
 * rep winning 62% and a rep winning 20% are both interesting; a rep winning 41%
 * on a 41% norm is not, whichever end of the list they would land on by rate.
 *
 * REPS WITH TOO FEW DEALS ARE SHOWN, NEVER RANKED. The server flags them
 * `thin`. A 100% shrink rate on two deals is a sample-size artefact, and letting
 * one sort to the top of a coaching list would burn the credibility of every
 * other flag on the page. They sit at the bottom of every sort, muted, and say
 * why in words.
 *
 * Nothing here computes a business figure. The rates, the norms and the
 * distances all arrive on the payload; the only arithmetic is turning a rate
 * into a percentage for display and a distance into a dot position.
 */
import { useId, useMemo, useState } from "react";
import { num, pct } from "../lib/format";

/**
 * One rep, exactly as the semantic layer sends it.
 *
 * Every behaviour arrives as three fields — `<m>` the rep's own rate, `<m>_peer`
 * the team norm, `<m>_z` the distance from that norm — plus `deals` and `thin`.
 * The index signature keeps the many fields this component does not read (open
 * deals, won gross profit, cycle length) without restating the whole row shape,
 * and nothing is assumed to be present: a missing figure renders an em dash.
 */
export interface RepRow {
  rep: string;
  deals?: number;
  thin?: boolean;
  [field: string]: string | number | boolean | null | undefined;
}

export interface RepTableProps {
  reps: RepRow[];
  /** Clicking a rep hands their name back — usually to scope the page to them. */
  onPickRep?: (rep: string) => void;
}

/**
 * The six behaviours, in plain words.
 *
 * `betterHigher` is the only thing that cannot be read off the data: winning
 * more is good, stalling more is not, and the colour of a dot depends on
 * knowing which. Everything else on this list is language.
 */
interface Behaviour {
  key: string;
  label: string;
  /** Two or three words under the heading, so the label needs no glossary. */
  help: string;
  /** The longer sentence, on the heading's tooltip. */
  means: string;
  betterHigher: boolean;
}

const BEHAVIOURS: Behaviour[] = [
  {
    key: "win_rate",
    label: "Win rate",
    help: "of deals closed",
    means: "How many of the deals this rep closed were won.",
    betterHigher: true,
  },
  {
    key: "shrink_rate",
    label: "Shrinks deals",
    help: "got smaller",
    means: "How often this rep's deals end up smaller than the first estimate.",
    betterHigher: false,
  },
  {
    key: "inflate_rate",
    label: "Oversizes deals",
    help: "grew far bigger",
    means: "How often this rep's deals grow well beyond the first estimate.",
    betterHigher: false,
  },
  {
    key: "regression_rate",
    label: "Changes forecast",
    help: "walked backwards",
    means: "How often this rep moves a deal backwards in the forecast after calling it.",
    betterHigher: false,
  },
  {
    key: "stall_rate",
    label: "Sits still",
    help: "open, not moving",
    means: "How much of this rep's open book has stopped moving.",
    betterHigher: false,
  },
  {
    key: "new_business_share",
    label: "New business",
    help: "not repeat work",
    means: "How much of this rep's book is new business rather than repeat work.",
    betterHigher: true,
  },
];

/** The width of the track, in distances from the norm. Beyond this a dot pins
 *  to the end rather than leaving the cell — the tooltip carries the true figure. */
const TRACK = 3;
/** Far enough from the norm to be worth a manager's attention. */
const OFF_PATTERN = 1;
/** Far enough that "much" is the honest word for it. */
const FAR = 2;
/** Below this the rep is, for reading purposes, exactly on the norm. */
const ON_THE_NOSE = 0.05;

type Tone = "better" | "worse" | "level" | "none";

interface Cell {
  rate: number | null;
  peer: number | null;
  distance: number | null;
  tone: Tone;
  far: boolean;
  offPattern: boolean;
  /** Where the dot sits on the track, 0–100, centre being the norm. */
  position: number;
  valueText: string;
  /** The evidence, for the tooltip and for a screen reader. */
  detail: string;
}

interface RankedRep {
  rep: string;
  thin: boolean;
  dealsText: string | null;
  cells: Cell[];
  /** The biggest distance from the norm on any behaviour. Drives the default sort. */
  worst: number | null;
}

/** A number the payload actually carries, or null. Never NaN, never a guess. */
function figure(row: RepRow, field: string): number | null {
  const v = row[field];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Rates arrive as fractions of one. Showing a fraction as a percentage is
 *  formatting, not arithmetic — the value is unchanged. */
const rateText = (v: number): string => pct(v * 100, 0);

const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

function verdictFor(distance: number, above: boolean, betterHigher: boolean): {
  tone: Tone;
  far: boolean;
  words: string;
} {
  if (distance < OFF_PATTERN) return { tone: "level", far: false, words: "In line with the team" };
  const better = above === betterHigher;
  const far = distance >= FAR;
  const words = `${far ? "Much " : ""}${better ? "better" : "worse"} than the team`;
  return { tone: better ? "better" : "worse", far, words };
}

function buildCell(row: RepRow, b: Behaviour, thin: boolean): Cell {
  const rate = figure(row, b.key);
  const peer = figure(row, `${b.key}_peer`);
  const z = figure(row, `${b.key}_z`);

  if (rate === null) {
    return {
      rate: null,
      peer,
      distance: null,
      tone: "none",
      far: false,
      offPattern: false,
      position: 50,
      valueText: "—",
      detail: `${b.label}: nothing recorded for this rep.`,
    };
  }

  const valueText = rateText(rate);
  const parts = [`${b.label}: ${valueText}.`];
  if (peer !== null) parts.push(`Team norm: ${rateText(peer)}.`);

  if (z === null) {
    if (!thin) parts.push("There is no comparison with the team for this one.");
    else parts.push("Too few deals to judge this rep.");
    return {
      rate,
      peer,
      distance: null,
      tone: "none",
      far: false,
      offPattern: false,
      position: 50,
      valueText,
      detail: parts.join(" "),
    };
  }

  const distance = Math.abs(z);
  const above = z > 0;
  const { tone, far, words } = verdictFor(distance, above, b.betterHigher);

  parts.push(
    distance < ON_THE_NOSE
      ? "Right on the team norm."
      : `${words} — ${distance.toFixed(1)} standard deviations ${above ? "above" : "below"} the norm.`,
  );
  if (thin) parts.push("Too few deals to judge this rep, so this is not counted against them.");

  return {
    rate,
    peer,
    distance,
    tone,
    far,
    offPattern: !thin && distance >= OFF_PATTERN,
    position: 50 + (clamp(z, -TRACK, TRACK) / TRACK) * 50,
    valueText,
    detail: parts.join(" "),
  };
}

export function RepTable({ reps, onPickRep }: RepTableProps) {
  const uid = useId();
  const [sortKey, setSortKey] = useState<string>("overall");
  const [dir, setDir] = useState<"asc" | "desc">("desc");

  const ranked = useMemo<RankedRep[]>(
    () =>
      reps.map((row) => {
        const thin = row.thin === true;
        const cells = BEHAVIOURS.map((b) => buildCell(row, b, thin));
        const distances = cells
          .map((c) => c.distance)
          .filter((d): d is number => d !== null);
        const deals = figure(row, "deals");
        return {
          rep: row.rep,
          thin,
          dealsText: deals === null ? null : `${num(Math.round(deals))} deals`,
          cells,
          worst: distances.length > 0 ? Math.max(...distances) : null,
        };
      }),
    [reps],
  );

  const rows = useMemo<RankedRep[]>(() => {
    const idx = BEHAVIOURS.findIndex((b) => b.key === sortKey);
    const out = [...ranked];
    out.sort((a, b) => {
      // A rep the data cannot judge is never ranked against one it can, in
      // either direction — they sit below the sort, not at either end of it.
      if (a.thin !== b.thin) return a.thin ? 1 : -1;

      if (sortKey === "rep") {
        const c = a.rep.localeCompare(b.rep);
        return dir === "asc" ? c : -c;
      }

      const av = sortKey === "overall" ? a.worst : idx >= 0 ? a.cells[idx].distance : null;
      const bv = sortKey === "overall" ? b.worst : idx >= 0 ? b.cells[idx].distance : null;
      if (av === null && bv === null) return a.rep.localeCompare(b.rep);
      if (av === null) return 1;
      if (bv === null) return -1;
      if (av !== bv) return dir === "desc" ? bv - av : av - bv;
      return a.rep.localeCompare(b.rep);
    });
    return out;
  }, [ranked, sortKey, dir]);

  function sortBy(key: string): void {
    if (key === sortKey) {
      setDir((d) => (d === "asc" ? "desc" : "asc"));
      return;
    }
    setSortKey(key);
    // A behaviour opens on its furthest-out reps, which is what you clicked it
    // for. A name opens A to Z.
    setDir(key === "rep" ? "asc" : "desc");
  }

  const active = BEHAVIOURS.find((b) => b.key === sortKey);
  const sortSentence =
    sortKey === "overall"
      ? "Sorted by the biggest gap from the team norm, on any behaviour."
      : sortKey === "rep"
        ? `Sorted by name, ${dir === "asc" ? "A to Z" : "Z to A"}.`
        : `Sorted by ${active ? active.label.toLowerCase() : sortKey}, ${
            dir === "desc" ? "furthest from the team norm first" : "closest to the team norm first"
          }.`;

  const titleId = `${uid}-title`;

  const ariaSortFor = (key: string): "ascending" | "descending" | "none" =>
    sortKey === key ? (dir === "asc" ? "ascending" : "descending") : "none";

  if (reps.length === 0) {
    return (
      <section className="rep-table" aria-labelledby={titleId}>
        <header className="rep-table__head">
          <h3 className="rep-table__title" id={titleId}>
            How each rep works, next to the team
          </h3>
        </header>
        <p className="rep-table__empty">There is nobody in this team to compare yet.</p>
      </section>
    );
  }

  return (
    <section className="rep-table" aria-labelledby={titleId}>
      <header className="rep-table__head">
        <h3 className="rep-table__title" id={titleId}>
          How each rep works, next to the team
        </h3>

        <ul className="rep-table__legend">
          <li className="rep-table__legend-item">
            <span
              className="rep-table__dot rep-table__dot--legend rep-table__dot--better"
              aria-hidden="true"
            />
            Better than the team
          </li>
          <li className="rep-table__legend-item">
            <span
              className="rep-table__dot rep-table__dot--legend rep-table__dot--level"
              aria-hidden="true"
            />
            In line
          </li>
          <li className="rep-table__legend-item">
            <span
              className="rep-table__dot rep-table__dot--legend rep-table__dot--worse"
              aria-hidden="true"
            />
            Worse than the team
          </li>
          {sortKey !== "overall" ? (
            <li className="rep-table__legend-item">
              <button
                type="button"
                className="rep-table__reset"
                onClick={() => {
                  setSortKey("overall");
                  setDir("desc");
                }}
              >
                Back to the biggest gaps
              </button>
            </li>
          ) : null}
        </ul>
      </header>

      {/* The sort is a keyboard action with no visible confirmation of its own,
          so it is spoken. */}
      <p className="rep-table__status" role="status" aria-live="polite">
        {sortSentence}
      </p>

      <div className="rep-table__scroll">
        <table className="rep-table__table">
          <caption className="rep-table__caption">
            Off the pattern means a rep sits a long way from what the rest of the team does — the
            line down the middle of every cell is the team norm, and the dot is this rep.
          </caption>

          <thead>
            <tr>
              <th
                scope="col"
                className="rep-table__col rep-table__col--rep"
                aria-sort={ariaSortFor("rep")}
              >
                <button
                  type="button"
                  className="rep-table__sort"
                  onClick={() => sortBy("rep")}
                  aria-label="Sort by rep name"
                >
                  <span className="rep-table__col-label">Rep</span>
                  <span className="rep-table__arrow" aria-hidden="true">
                    {sortKey === "rep" ? (dir === "asc" ? "▲" : "▼") : ""}
                  </span>
                </button>
              </th>

              {BEHAVIOURS.map((b) => (
                <th
                  scope="col"
                  className="rep-table__col"
                  key={b.key}
                  aria-sort={ariaSortFor(b.key)}
                >
                  <button
                    type="button"
                    className="rep-table__sort"
                    onClick={() => sortBy(b.key)}
                    title={b.means}
                    aria-label={`${b.label}. ${b.means} Sort by how far each rep is from the team norm on this.`}
                  >
                    <span className="rep-table__col-label">{b.label}</span>
                    <span className="rep-table__arrow" aria-hidden="true">
                      {sortKey === b.key ? (dir === "asc" ? "▲" : "▼") : ""}
                    </span>
                  </button>
                  <span className="rep-table__col-help">{b.help}</span>
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {rows.map((r) => {
              const rowClass = [
                "rep-table__row",
                onPickRep ? "rep-table__row--clickable" : "",
                r.thin ? "rep-table__row--thin" : "",
              ]
                .filter(Boolean)
                .join(" ");

              return (
                <tr
                  className={rowClass}
                  key={r.rep}
                  onClick={onPickRep ? () => onPickRep(r.rep) : undefined}
                >
                  <th scope="row" className="rep-table__rep">
                    {onPickRep ? (
                      <button
                        type="button"
                        className="rep-table__rep-name"
                        onClick={(e) => {
                          // The row is clickable too; without this the pick fires twice.
                          e.stopPropagation();
                          onPickRep(r.rep);
                        }}
                      >
                        {r.rep}
                      </button>
                    ) : (
                      <span className="rep-table__rep-name">{r.rep}</span>
                    )}
                    {r.dealsText ? (
                      <span className="rep-table__rep-meta">{r.dealsText}</span>
                    ) : null}
                    {r.thin ? (
                      <span className="rep-table__thin">too few deals to judge</span>
                    ) : null}
                  </th>

                  {r.cells.map((c, i) => {
                    const b = BEHAVIOURS[i];
                    const cellClass = [
                      "rep-table__cell",
                      c.offPattern ? "rep-table__cell--off" : "",
                      c.rate === null ? "rep-table__cell--missing" : "",
                    ]
                      .filter(Boolean)
                      .join(" ");

                    const dotClass = [
                      "rep-table__dot",
                      r.thin ? "rep-table__dot--thin" : `rep-table__dot--${c.tone}`,
                      c.far && !r.thin ? "rep-table__dot--far" : "",
                    ]
                      .filter(Boolean)
                      .join(" ");

                    return (
                      <td className={cellClass} key={b.key} title={`${r.rep} — ${c.detail}`}>
                        <span className="rep-table__value">{c.valueText}</span>
                        <span className="rep-table__gauge" aria-hidden="true">
                          <span className="rep-table__norm" />
                          {c.distance !== null ? (
                            <span className={dotClass} style={{ left: `${c.position}%` }} />
                          ) : null}
                        </span>
                        <span className="rep-table__sr">{c.detail}</span>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="rep-table__foot">
        A rep with too few deals is shown but never ranked — one odd deal out of a handful is noise,
        not a pattern. The exact team norm and the distance from it are on every cell; hover one, or
        read it with a screen reader.
      </p>
    </section>
  );
}
