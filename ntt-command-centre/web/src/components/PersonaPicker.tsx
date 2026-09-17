/**
 * Who am I looking at this with?
 *
 * This replaces the persona dropdown that used to live inside the header. The
 * old control asked two questions in sequence — pick a role, then pick a person
 * from a second list that appeared underneath — and in a demo that is two
 * gestures and a moment of hesitation in between. Here the two questions are
 * one: put the pointer (or the focus ring) on a role and the other two roles
 * fold away while that role's own people open in the space they leave. One
 * movement, one click, and you are looking at the right book.
 *
 * WHY EACH PERSON CARRIES A COUNT. The roster is 70 reps and a dozen of them
 * have nothing open. Landing on one of those is a screen full of nothing, which
 * reads as a broken product rather than an empty book, so every row states how
 * many open deals sit behind it BEFORE you commit to it. The number is the
 * server's, off the identity options; nothing here computes a business figure.
 *
 * THE MARKS are drawn here rather than shipped as images — the app takes no new
 * dependency for three icons. Each is a small isometric solid lit from the top
 * left: the top face at full strength, the left face a little shaded, the right
 * face darkest. Every face is `currentColor` at an opacity, never a colour
 * value, so one mark is correct on both themes and tints to whichever persona
 * it belongs to. The three read apart at 28px because their SILHOUETTES differ,
 * not their detail — Sales is a single tall block with a piece lifting off it,
 * Manager is three blocks of unequal height standing together, Executive is a
 * wide three-tier stack seen from above.
 *
 * KEYBOARD. Focus does exactly what hover does, and nothing is trapped. Tab
 * walks the three roles; landing on one opens it, which means focus never rests
 * on something that is folded away. Inside an open role the arrow keys walk the
 * people (one tab stop, not seventy), Up from the first person returns to the
 * role, and Escape closes and hands focus back to whatever opened this.
 */
import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type FocusEvent as ReactFocusEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
} from "react";
import type { IdentityOption, PersonaDef, PersonaKey } from "../api/types";

/* ------------------------------------------------------------------ marks */

/** One face of a solid. `o` is how much of the light it catches. */
interface Face {
  d: string;
  o: number;
}

/** Top-left light source: the up face full, the left face shaded, the right
 *  face deepest. Three tones is all an isometric solid needs to read as one. */
const LIT = 1;
const SHADE = 0.66;
const DEEP = 0.36;

/**
 * Sales — one block, one piece lifting off it. A rep works their own deals.
 * Tall and narrow, so it never reads as the other two at a glance.
 */
const SALES: Face[] = [
  { d: "M16 2.1 L19.8 4 L16 5.9 L12.2 4 Z", o: 0.8 },
  { d: "M16 9.5 L23 13 L16 16.5 L9 13 Z", o: LIT },
  { d: "M9 13 L16 16.5 L16 27 L9 23.5 Z", o: SHADE },
  { d: "M23 13 L16 16.5 L16 27 L23 23.5 Z", o: DEEP },
];

/**
 * Manager — three blocks of unequal height on one ground line. A team, and the
 * unevenness is the job: the coaching is in the difference between them.
 */
const MANAGER: Face[] = [
  { d: "M6 17 L10 19 L6 21 L2 19 Z", o: LIT },
  { d: "M2 19 L6 21 L6 27 L2 25 Z", o: SHADE },
  { d: "M10 19 L6 21 L6 27 L10 25 Z", o: DEEP },
  { d: "M16 11 L20 13 L16 15 L12 13 Z", o: LIT },
  { d: "M12 13 L16 15 L16 27 L12 25 Z", o: SHADE },
  { d: "M20 13 L16 15 L16 27 L20 25 Z", o: DEEP },
  { d: "M26 14 L30 16 L26 18 L22 16 Z", o: LIT },
  { d: "M22 16 L26 18 L26 27 L22 25 Z", o: SHADE },
  { d: "M30 16 L26 18 L26 27 L30 25 Z", o: DEEP },
];

/**
 * Executive — a wide three-tier stack, drawn bottom slab first so each tier
 * sits on the one below it. Broad and low: the whole region, rolled up.
 */
const EXECUTIVE: Face[] = [
  { d: "M16 11 L29 17.5 L16 24 L3 17.5 Z", o: LIT },
  { d: "M3 17.5 L16 24 L16 27 L3 20.5 Z", o: SHADE },
  { d: "M29 17.5 L16 24 L16 27 L29 20.5 Z", o: DEEP },
  { d: "M16 10.25 L24.5 14.5 L16 18.75 L7.5 14.5 Z", o: LIT },
  { d: "M7.5 14.5 L16 18.75 L16 21.75 L7.5 17.5 Z", o: SHADE },
  { d: "M24.5 14.5 L16 18.75 L16 21.75 L24.5 17.5 Z", o: DEEP },
  { d: "M16 9.25 L20.5 11.5 L16 13.75 L11.5 11.5 Z", o: LIT },
  { d: "M11.5 11.5 L16 13.75 L16 16.75 L11.5 14.5 Z", o: SHADE },
  { d: "M20.5 11.5 L16 13.75 L16 16.75 L20.5 14.5 Z", o: DEEP },
];

const MARKS: Record<PersonaKey, Face[]> = {
  ae: SALES,
  manager: MANAGER,
  executive: EXECUTIVE,
};

export interface PersonaMarkProps {
  persona: PersonaKey;
  /** Drawn size in CSS pixels. 28 is the size these were cut for. */
  size?: number;
  /**
   * An explicit tint. Left off, the mark takes `--persona`, which resolves to
   * whichever persona the shell is currently in; `"currentColor"` makes it
   * inherit from whatever wraps it. Either way a `--mark` custom property set
   * anywhere above it wins, so stylesheets keep the final say without this
   * file carrying a colour value of its own — the header tile uses exactly
   * that to knock the mark out against its solid persona fill.
   */
  color?: string;
}

/** The isometric mark for one persona. Decorative: the row says the name. */
export function PersonaMark({ persona, size = 28, color }: PersonaMarkProps) {
  const tint = `var(--mark, ${color ?? "var(--persona)"})`;
  return (
    <svg
      className="ppick__icon"
      data-persona={persona}
      viewBox="0 0 32 32"
      width={size}
      height={size}
      role="presentation"
      aria-hidden="true"
      focusable="false"
      style={{ color: tint }}
    >
      {MARKS[persona].map((f) => (
        <path key={f.d} d={f.d} fill="currentColor" opacity={f.o} />
      ))}
    </svg>
  );
}

/* ----------------------------------------------------------------- wording */

/** What the second question is called, per role. Plain words, no field names. */
const PICK_HEAD: Record<PersonaKey, string> = {
  ae: "Pick a rep",
  manager: "Pick a team",
  executive: "Pick a region",
};

const NOUN: Record<PersonaKey, { one: string; many: string }> = {
  ae: { one: "rep", many: "reps" },
  manager: { one: "team", many: "teams" },
  executive: { one: "region", many: "regions" },
};

/**
 * The line under a person's name.
 *
 * Open deals first, because that is the one that stops a demo landing on an
 * empty book. Everything here is a count the server sent; nothing is derived.
 */
function countLine(o: IdentityOption): string | null {
  if (typeof o.openDeals === "number") {
    return `${o.openDeals} open ${o.openDeals === 1 ? "deal" : "deals"}`;
  }
  if (typeof o.opportunities === "number") {
    return `${o.opportunities} ${o.opportunities === 1 ? "deal" : "deals"}`;
  }
  if (typeof o.reps === "number") {
    return `${o.reps} ${o.reps === 1 ? "rep" : "reps"}`;
  }
  return null;
}

/** Team size, shown beside the open-deal count where the server sends both. */
function teamLine(o: IdentityOption): string | null {
  if (typeof o.openDeals !== "number" || typeof o.reps !== "number") return null;
  return `${o.reps} ${o.reps === 1 ? "rep" : "reps"}`;
}

/* ----------------------------------------------------------------- picker */

export interface PersonaPickerProps {
  personas: PersonaDef[];
  identities: Record<PersonaKey, IdentityOption[]>;
  activePersona: PersonaKey;
  activeIdentity: string;
  /** Picked a role, or a role and a specific person within it. */
  onPick: (persona: PersonaKey, identity?: string) => void;
  onClose: () => void;
}

export function PersonaPicker({
  personas,
  identities,
  activePersona,
  activeIdentity,
  onPick,
  onClose,
}: PersonaPickerProps) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const openerRef = useRef<HTMLElement | null>(null);
  const rowRefs = useRef<Map<PersonaKey, HTMLButtonElement | null>>(new Map());
  const peopleRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const baseId = useId();

  // Hover and focus are kept apart so the two can disagree without the panel
  // flickering. Focus wins whenever it is inside, which is the only rule that
  // guarantees a keyboard user's focus ring is never sitting on a folded row.
  const [hovered, setHovered] = useState<PersonaKey | null>(null);
  const [focused, setFocused] = useState<PersonaKey | null>(null);
  const open: PersonaKey | null = focused ?? hovered;

  const list = useMemo<IdentityOption[]>(
    () => (open ? (identities[open] ?? []) : []),
    [open, identities],
  );

  // Which person in the open list is the single tab stop. It starts on the one
  // already in use, so tabbing in puts you where you are rather than at the top.
  //
  // The refs array is trimmed here, never cleared: the callback refs below have
  // already attached for this render by the time an effect runs, so wiping the
  // array would throw away every button that just mounted and leave the arrow
  // keys with nothing to move between. Trimming only drops the tail left over
  // from a longer list that has since closed.
  const [cursor, setCursor] = useState(0);
  useEffect(() => {
    peopleRefs.current.length = list.length;
    if (!open) {
      setCursor(0);
      return;
    }
    const i = open === activePersona ? list.findIndex((o) => o.id === activeIdentity) : -1;
    setCursor(i > 0 ? i : 0);
  }, [open, list, activePersona, activeIdentity]);

  // The panel's resting height, pinned as its floor. The people list mounts
  // on hover, and the three roles that fold away to make room for it are not
  // always taller than what replaces them: one region under Executive is a
  // shorter panel than three roles at rest, and a menu that changes height
  // under the pointer is the surest way to make hover feel like a fault. The
  // stylesheet carries a floor built from its own tokens for the first paint;
  // this measurement corrects it for whatever the text actually did — a role
  // line that wrapped, a wider type scale — and is taken only while all three
  // roles are showing and nothing is mid-transition, which is the moment this
  // mounts. Layout, not a business figure: nothing here is a number the page
  // shows.
  const [floor, setFloor] = useState<number | null>(null);
  useLayoutEffect(() => {
    if (personas.length === 0) return;
    const el = rootRef.current;
    if (el) setFloor(el.getBoundingClientRect().height);
  }, [personas.length]);

  // Focus moves in on arrival and back to whatever opened this on the way out,
  // so closing never drops a keyboard user at the top of the document.
  useEffect(() => {
    openerRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const t = window.setTimeout(() => rootRef.current?.focus({ preventScroll: true }), 0);
    return () => {
      window.clearTimeout(t);
      // Only when closing left focus nowhere. Removing a focused subtree drops
      // focus to the body; if instead the user closed this by clicking some
      // other control, focus is already where they put it and must stay there.
      const lost = document.activeElement === null || document.activeElement === document.body;
      const back = openerRef.current;
      if (lost && back && document.contains(back)) back.focus();
    };
  }, []);

  const choose = useCallback(
    (persona: PersonaKey, identity?: string) => {
      onPick(persona, identity);
      onClose();
    },
    [onPick, onClose],
  );

  const focusPerson = useCallback((i: number) => {
    setCursor(i);
    peopleRefs.current[i]?.focus();
  }, []);

  /* ---- pointer ---------------------------------------------------------- */

  // Opening on hover is deferred by a beat. The pointer arrives from the
  // trigger above, so it crosses the top row on its way to any other role; if
  // that row opened the instant it was crossed, the other two would fold away
  // under the pointer before it reached them. The delay is under the threshold
  // at which a rest on a row reads as a wait, and it means the panel only ever
  // reacts to where the pointer stops, not to where it passed. Because the
  // rows are hidden while one is open, leaving the whole panel is the way back
  // to all three, and that part is instant.
  const HOVER_INTENT_MS = 140;
  const hoverTimer = useRef<number | null>(null);
  const cancelHover = useCallback(() => {
    if (hoverTimer.current !== null) {
      window.clearTimeout(hoverTimer.current);
      hoverTimer.current = null;
    }
  }, []);
  useEffect(() => cancelHover, [cancelHover]);

  // If focus is parked on a different row when the pointer settles, focus
  // steps back to the container rather than being stranded on something about
  // to fold away. The element is read synchronously: React clears
  // `currentTarget` once the handler returns.
  const onEnterItem = useCallback(
    (key: PersonaKey, e: ReactMouseEvent<HTMLLIElement>) => {
      const here = e.currentTarget;
      cancelHover();
      hoverTimer.current = window.setTimeout(() => {
        hoverTimer.current = null;
        setHovered(key);
        const active = document.activeElement;
        if (
          active instanceof HTMLElement &&
          rootRef.current?.contains(active) &&
          !here.contains(active)
        ) {
          setFocused(null);
          rootRef.current.focus({ preventScroll: true });
        }
      }, HOVER_INTENT_MS);
    },
    [cancelHover],
  );

  const onLeavePicker = useCallback(() => {
    cancelHover();
    setHovered(null);
  }, [cancelHover]);

  /* ---- keyboard --------------------------------------------------------- */

  const onRootKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    }
  };

  // Leaving the picker altogether closes it. A null target is the window
  // losing focus, which is not the user going anywhere. Focus arriving on the
  // control that opened this is not a departure either: a mouse press on that
  // control focuses it before its click fires, and if the press closed the
  // panel here the click would open it straight back up. The opener's own
  // click is what closes it, so it is left to.
  const onRootBlur = (e: ReactFocusEvent<HTMLDivElement>) => {
    const next = e.relatedTarget;
    if (next && rootRef.current?.contains(next)) return;
    setFocused(null);
    if (next && next !== openerRef.current) onClose();
  };

  const onRowKeyDown = (e: ReactKeyboardEvent<HTMLButtonElement>) => {
    if (e.key === "ArrowDown" && list.length > 0) {
      e.preventDefault();
      focusPerson(cursor);
    }
  };

  const onPersonKeyDown = (e: ReactKeyboardEvent<HTMLButtonElement>, i: number, key: PersonaKey) => {
    const n = list.length;
    if (n === 0) return;
    if (e.key === "ArrowDown" || e.key === "ArrowRight") {
      e.preventDefault();
      focusPerson((i + 1) % n);
    } else if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
      e.preventDefault();
      if (i === 0) rowRefs.current.get(key)?.focus();
      else focusPerson(i - 1);
    } else if (e.key === "Home") {
      e.preventDefault();
      focusPerson(0);
    } else if (e.key === "End") {
      e.preventDefault();
      focusPerson(n - 1);
    }
  };

  /* ---- render ----------------------------------------------------------- */

  if (personas.length === 0) {
    return (
      <div className="ppick">
        <p className="ppick__empty">Loading who you can view as.</p>
      </div>
    );
  }

  const noun = open ? NOUN[open] : null;
  const announce =
    open && noun
      ? `${PICK_HEAD[open]}. ${list.length} ${list.length === 1 ? noun.one : noun.many} to choose from.`
      : "";

  // Handed over as a custom property rather than a min-height so the
  // stylesheet keeps the final say: it takes the larger of this and its own
  // token-built floor.
  const floorStyle = floor !== null ? ({ "--ppick-floor": `${floor}px` } as CSSProperties) : undefined;

  return (
    <div
      className="ppick"
      ref={rootRef}
      tabIndex={-1}
      role="group"
      aria-label="Choose who you are viewing as"
      style={floorStyle}
      onKeyDown={onRootKeyDown}
      onBlur={onRootBlur}
      onMouseLeave={onLeavePicker}
    >
      <p className="ppick__head" id={`${baseId}-head`}>
        View as
      </p>

      <ul className="ppick__list">
        {personas.map((p) => {
          const key = p.key;
          const isOpen = open === key;
          const isShy = open !== null && !isOpen;
          const isCurrent = key === activePersona;
          const panelId = `${baseId}-${key}`;
          const people = identities[key] ?? [];

          return (
            <li
              key={key}
              className={[
                "ppick__item",
                isOpen ? "ppick__item--open" : "",
                isShy ? "ppick__item--shy" : "",
                isCurrent ? "ppick__item--current" : "",
              ]
                .filter(Boolean)
                .join(" ")}
              data-persona={key}
              onMouseEnter={(e) => onEnterItem(key, e)}
              onFocus={() => setFocused(key)}
            >
              <button
                type="button"
                className="ppick__row"
                ref={(el) => {
                  rowRefs.current.set(key, el);
                }}
                aria-expanded={isOpen}
                aria-controls={panelId}
                aria-current={isCurrent ? "true" : undefined}
                onKeyDown={onRowKeyDown}
                onClick={() => choose(key)}
              >
                {/* The accent sits on the tile, not the mark: the mark inherits
                    it through currentColor, and so does the tile's own wash in
                    persona.css. One server-supplied colour, two surfaces, and
                    the stylesheet never has to know which persona this is. */}
                <span className="ppick__mark" style={{ color: p.accent }}>
                  <PersonaMark persona={key} size={34} color="currentColor" />
                </span>
                <span className="ppick__text">
                  <span className="ppick__name">{p.label}</span>
                  <span className="ppick__role">{p.role}</span>
                  <span className="ppick__jtbd">{p.jtbd}</span>
                </span>
                {isCurrent && <span className="ppick__now">You are here</span>}
              </button>

              {/* Mounted only while this role is open, so the tab order always
                  matches what is on screen and seventy buttons are not sitting
                  in the document waiting to be found by a screen reader. */}
              {isOpen && (
                <div className="ppick__people" id={panelId} role="group" aria-label={PICK_HEAD[key]}>
                  <p className="ppick__people-head">
                    <span>{PICK_HEAD[key]}</span>
                    {people.length > 1 && (
                      <span className="ppick__people-count">{people.length} to choose from</span>
                    )}
                  </p>

                  {people.length === 0 ? (
                    <p className="ppick__empty">
                      Nobody is listed under this role yet.
                    </p>
                  ) : (
                    <ul className="ppick__scroll">
                      {people.map((o, i) => {
                        const on = isCurrent && o.id === activeIdentity;
                        const count = countLine(o);
                        const team = teamLine(o);
                        return (
                          <li key={o.id} className="ppick__person-wrap">
                            <button
                              type="button"
                              className={`ppick__person${on ? " ppick__person--on" : ""}`}
                              ref={(el) => {
                                peopleRefs.current[i] = el;
                              }}
                              tabIndex={i === cursor ? 0 : -1}
                              aria-current={on ? "true" : undefined}
                              onFocus={() => setCursor(i)}
                              onKeyDown={(e) => onPersonKeyDown(e, i, key)}
                              onClick={() => choose(key, o.id)}
                            >
                              <span className="ppick__person-name">{o.label}</span>
                              {count && <span className="ppick__person-meta">{count}</span>}
                              {team && <span className="ppick__person-team">{team}</span>}
                            </button>
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ul>

      {/* Hover and focus change what is on screen without moving focus, so the
          change is spoken rather than left to be discovered. */}
      <p className="ppick__sr" role="status" aria-live="polite">
        {announce}
      </p>
    </div>
  );
}
