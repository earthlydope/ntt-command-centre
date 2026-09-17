/**
 * The filter deck.
 *
 * Until this existed the only way to narrow the page was to click a chart mark,
 * which is a fine gesture and a terrible only-option: it means a question you
 * cannot already see the answer to cannot be asked. Every validated dimension
 * now has a control.
 *
 * Two things it does NOT do, on purpose:
 *
 * **It does not filter on the client.** Selecting a value dispatches to the
 * reducer, which puts it in the URL and refetches; the server narrows the frame
 * before it aggregates. A client-side filter would produce a numerator over an
 * unfiltered denominator and every percentage on the page would quietly be wrong.
 *
 * **It does not offer a dimension the persona cannot use.** An AE has no rep
 * selector — there is exactly one rep in their scope — and the values in each
 * list are the ones present in THEIR rows, served by `/api/meta`, so a control
 * can never offer a value that returns nothing.
 */
import type { DimKey, Measure, MetaPayload, PersonaKey } from "../api/types";

/** Which dimensions each profile may slice by, in the order they think in. */
const FOR_PERSONA: Record<PersonaKey, DimKey[]> = {
  ae: ["stage", "forecast", "lob", "portfolio", "account", "orderType"],
  manager: ["rep", "stage", "lob", "portfolio", "orderType", "quarter"],
  executive: ["lob", "portfolio", "industry", "country", "quarter", "orderType", "stage"],
};

/** Beyond this a <select> is the wrong control and the list is searched instead. */
const LONG_LIST = 25;

export function FilterBar({
  meta,
  persona,
  active,
  measure,
  onSet,
  onClear,
  onMeasure,
}: {
  meta: MetaPayload | null;
  persona: PersonaKey;
  active: Partial<Record<DimKey, string | null>>;
  measure: Measure;
  onSet: (dim: DimKey, value: string | null) => void;
  onClear: () => void;
  onMeasure: (m: Measure) => void;
}) {
  if (!meta) return <div className="fbar fbar--skeleton" aria-hidden="true" />;

  const byKey = new Map(meta.dimensions.map((d) => [d.key, d]));
  const dims = FOR_PERSONA[persona].map((k) => byKey.get(k)).filter(Boolean);
  const count = Object.values(active).filter(Boolean).length;

  return (
    <section className="fbar" aria-label="Filters">
      {dims.map((d) => {
        if (!d) return null;
        const value = active[d.key] ?? "";
        const long = d.values.length > LONG_LIST;
        const id = `f-${d.key}`;
        return (
          <span className={`fbar__field${value ? " fbar__field--on" : ""}`} key={d.key}>
            <label className="fbar__label" htmlFor={id}>
              {d.label}
            </label>
            <select
              id={id}
              className="fbar__select"
              value={value}
              // `countBasis` is on the control itself, because which grain a
              // dimension counts on is the single most common way to misread
              // this data and the answer belongs where the choice is made.
              title={`${d.description} ${d.basisNote}`}
              onChange={(e) => onSet(d.key, e.target.value || null)}
            >
              <option value="">
                {long ? `All ${d.values.length}` : "All"}
              </option>
              {d.values.map((v) => (
                <option value={v} key={v}>
                  {v}
                </option>
              ))}
            </select>
          </span>
        );
      })}

      <span className="fbar__spacer" />

      {count > 0 && (
        <button type="button" className="fbar__clear" onClick={onClear}>
          Clear {count}
        </button>
      )}

      {/* Profit is the default and revenue is the alternative view of the same
          rows, so the switch belongs with the other things that change what you
          are looking at — not in the header beside who you are. */}
      <span className="fbar__field">
        <span className="fbar__label">Show</span>
        <span className="seg" role="group" aria-label="Measure">
          {(["gp", "revenue"] as Measure[]).map((m) => (
            <button
              type="button"
              key={m}
              className={`segbtn${measure === m ? " on" : ""}`}
              aria-pressed={measure === m}
              onClick={() => onMeasure(m)}
            >
              {m === "gp" ? "Profit" : "Revenue"}
            </button>
          ))}
        </span>
      </span>
    </section>
  );
}
