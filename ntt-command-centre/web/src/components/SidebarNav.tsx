/**
 * The persona's own navigation.
 *
 * Grouped rather than a flat tab rail, because fourteen pages across three
 * profiles is a product, not a dashboard — and the grouping is itself
 * information: an AE's pages are their day, their deals, their accounts and
 * their record, in that order, which is the order they think in.
 *
 * Built from `meta.pages`, which the server derives from the persona. A profile
 * cannot see the NAME of a page belonging to someone else's job, so there is no
 * disabled state here and nothing to grey out.
 *
 * Each item carries its page's stated question as its title and its second
 * line, because the question is what tells you whether this is the page you
 * want — "Calibration" alone does not.
 */
import type { Lens, PersonaKey } from "../api/types";
import { Icon, SIZE, type IconName } from "./icons";

export interface NavItem {
  key: Lens;
  label: string;
  question: string;
}

/** Which group a page belongs to, per persona. */
const GROUPS: Record<PersonaKey, { heading: string; pages: Lens[] }[]> = {
  ae: [
    { heading: "Today", pages: ["my-day"] },
    { heading: "My book", pages: ["my-deals", "my-accounts"] },
    { heading: "Me", pages: ["my-record"] },
  ],
  manager: [
    { heading: "This week", pages: ["pod-pulse"] },
    { heading: "The people", pages: ["rep-benchmark", "calibration"] },
    { heading: "The process", pages: ["process", "pod-whitespace"] },
  ],
  executive: [
    { heading: "The read", pages: ["tldr"] },
    { heading: "The numbers", pages: ["performance", "structure"] },
    { heading: "The decisions", pages: ["risks", "growth", "actions"] },
  ],
};

/**
 * The mark beside each page, from the shell's one icon set. Pages that ask
 * the same kind of question share a mark — growth and whitespace are both
 * "where is there more", performance is "how are we doing against plan" —
 * so the rail reads as families of questions rather than fifteen pictures.
 */
const ICON: Partial<Record<Lens, IconName>> = {
  "my-day": "today",
  "my-deals": "deals",
  "my-accounts": "accounts",
  "my-record": "record",
  "pod-pulse": "team",
  "rep-benchmark": "compare",
  process: "process",
  calibration: "calibrate",
  "pod-whitespace": "grow",
  tldr: "brief",
  performance: "plan",
  structure: "business",
  risks: "risks",
  growth: "grow",
  actions: "decisions",
};

export function SidebarNav({
  persona,
  pages,
  current,
  onNavigate,
  collapsed = false,
}: {
  persona: PersonaKey;
  pages: NavItem[];
  current: Lens;
  onNavigate: (page: Lens) => void;
  collapsed?: boolean;
}) {
  const byKey = new Map(pages.map((p) => [p.key, p]));
  // Anything the server sent that this file has not been told where to put
  // still appears, in its own group, rather than silently vanishing.
  const placed = new Set(GROUPS[persona].flatMap((g) => g.pages));
  const groups = [
    ...GROUPS[persona]
      .map((g) => ({ heading: g.heading, items: g.pages.map((k) => byKey.get(k)).filter(Boolean) }))
      .filter((g) => g.items.length > 0),
    ...(pages.some((p) => !placed.has(p.key))
      ? [{ heading: "More", items: pages.filter((p) => !placed.has(p.key)) }]
      : []),
  ] as { heading: string; items: NavItem[] }[];

  return (
    <nav className={`side${collapsed ? " side--collapsed" : ""}`} aria-label="Pages">
      {groups.map((g) => (
        <div className="side__group" key={g.heading}>
          <p className="side__heading">{g.heading}</p>
          <ul className="side__list">
            {g.items.map((p) => {
              const on = p.key === current;
              return (
                <li key={p.key}>
                  <button
                    type="button"
                    className={`side__item${on ? " side__item--on" : ""}`}
                    aria-current={on ? "page" : undefined}
                    title={p.question}
                    onClick={() => onNavigate(p.key)}
                  >
                    <span className="side__glyph" aria-hidden="true">
                      <Icon name={ICON[p.key] ?? "chevron"} size={SIZE.nav} />
                    </span>
                    <span className="side__text">
                      <span className="side__label">{p.label}</span>
                      <span className="side__question">{p.question}</span>
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </nav>
  );
}
