/**
 * One icon system for the whole shell.
 *
 * Before this file the chrome drew its marks with whatever Unicode glyph was
 * nearest — geometric shapes in the KPI tiles, dingbats down the sidebar, a
 * hamburger and two half-moons in the header — and a glyph is a character,
 * so every one of them took the size, weight and baseline of the font around
 * it. Twelve marks at eleven optical sizes was what the customer saw as
 * "outdated" and "very small". Here every mark is drawn, on one of two grids,
 * and sized only by the four constants below, so a control in the header and
 * an item in the rail carry the same amount of ink.
 *
 * TWO KINDS OF MARK.
 *
 * `Icon` is the line set: a 24-unit grid, `currentColor`, a 1.75 stroke with
 * round caps and joins. It labels controls and pages — the sidebar, the header
 * buttons, the Ask surfaces — where the mark sits beside text and must weigh
 * the same as the text.
 *
 * `Solid` is the isometric set, drawn the way `PersonaMark` draws the three
 * persona marks: a small object lit from the top left, the up face at full
 * strength, the left face shaded, the right face deepest, every face
 * `currentColor` at an opacity so one drawing is right on both themes and
 * tints to whatever colour wraps it. One object per KPI icon key the server
 * sends, because a KPI tile is a thing you pick up and turn over — the tile
 * that holds it is raised, and the popover it opens shows the same object
 * larger. The marks are read apart by silhouette, not detail, because at
 * 26px there is no detail.
 *
 * THE SIZES. `SIZE` is the only place a pixel size for an icon is written.
 * `Icon` and `Solid` accept nothing else, so a mark at some other size does
 * not typecheck; that is the mechanism behind "all icons a standard size"
 * rather than a convention to remember.
 */

/* ------------------------------------------------------------------ sizes */

/**
 * `control` — an icon-only button (the header's rail and theme toggles).
 * `nav` — an icon beside a two-line label (the sidebar).
 * `tile` — the Solid inside a KPI tile's raised square.
 * `hero` — the Solid at the head of a KPI popover.
 */
export const SIZE = { control: 20, nav: 20, tile: 26, hero: 32 } as const;
export type IconSize = (typeof SIZE)[keyof typeof SIZE];

/* ------------------------------------------------------------- line icons */

export type IconName =
  // the sidebar, one per page family
  | "today"
  | "deals"
  | "accounts"
  | "record"
  | "team"
  | "compare"
  | "process"
  | "calibrate"
  | "grow"
  | "brief"
  | "plan"
  | "business"
  | "risks"
  | "decisions"
  | "chevron"
  // the header
  | "menu"
  | "sun"
  | "moon"
  // the Ask surfaces: question shapes, and the house mark for the model
  | "rank"
  | "value"
  | "place"
  | "time"
  | "cause"
  | "act"
  | "spark";

interface LineIcon {
  mode: "stroke" | "fill";
  d: string[];
}

/** A circle as a path, so every icon is paths and nothing else. */
const circle = (cx: number, cy: number, r: number) =>
  `M${cx - r} ${cy}a${r} ${r} 0 1 0 ${2 * r} 0 ${r} ${r} 0 1 0-${2 * r} 0`;

/** A dial with two hands: the same drawing serves "today" and "when". */
const DIAL = [circle(12, 12, 9), "M12 7v5.2l3.4 2"];

/** Two columns on one baseline: "how this compares with that". */
const COLUMNS = ["M8 19V9", "M16 19V4.5", "M3.5 19h17"];

export const ICONS: Record<IconName, LineIcon> = {
  today: { mode: "stroke", d: DIAL },
  // A briefcase: the book of business.
  deals: {
    mode: "stroke",
    d: [
      "M4 8.5h16a1.5 1.5 0 0 1 1.5 1.5v8.5a1.5 1.5 0 0 1-1.5 1.5H4A1.5 1.5 0 0 1 2.5 18.5V10A1.5 1.5 0 0 1 4 8.5z",
      "M8.5 8.5V6a1.5 1.5 0 0 1 1.5-1.5h4A1.5 1.5 0 0 1 15.5 6v2.5",
      "M2.5 13.5h19",
    ],
  },
  // A building with windows and a door: the customers.
  accounts: {
    mode: "stroke",
    d: [
      "M5.5 20.5V4.5a1 1 0 0 1 1-1h11a1 1 0 0 1 1 1v16",
      "M3 20.5h18",
      "M9 7.5h1.5M13.5 7.5H15M9 11h1.5M13.5 11H15M9 14.5h1.5M13.5 14.5H15",
      "M10.5 20.5v-3h3v3",
    ],
  },
  // A medal: my record.
  record: { mode: "stroke", d: [circle(12, 9, 5), "M9.2 13.3 7.5 21l4.5-2.4 4.5 2.4-1.7-7.7"] },
  // Two people: the pod.
  team: {
    mode: "stroke",
    d: [
      circle(9.2, 8.5, 3.2),
      "M2.5 20v-1.4a4.1 4.1 0 0 1 4.1-4.1h5.2a4.1 4.1 0 0 1 4.1 4.1V20",
      "M16 5.4a3.2 3.2 0 0 1 0 6.2",
      "M21.5 20v-1.4a4.1 4.1 0 0 0-3.1-4",
    ],
  },
  compare: { mode: "stroke", d: COLUMNS },
  // A node on a line with an arrow out of it: a stage in a flow.
  process: { mode: "stroke", d: ["M3 12h5.5", circle(12, 12, 3.5), "M15.5 12H21", "M18.6 9.6 21 12l-2.4 2.4"] },
  // Two sliders: tuning.
  calibrate: {
    mode: "stroke",
    d: ["M3 8h9.5", "M17 8h4", circle(14.75, 8, 2.25), "M3 16h4", "M11.5 16H21", circle(9.25, 16, 2.25)],
  },
  // A line climbing to an arrow: growth, whitespace.
  grow: { mode: "stroke", d: ["M3 17l5.5-5.5 4 4L21 7", "M15 7h6v6"] },
  // A page with a folded corner and two lines: the brief.
  brief: {
    mode: "stroke",
    d: [
      "M7 3h7.5L20 8.5V20a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z",
      "M14.5 3v5.5H20",
      "M9.5 13h7",
      "M9.5 17h7",
    ],
  },
  // A target: performance against plan.
  plan: { mode: "stroke", d: [circle(12, 12, 9), circle(12, 12, 4.5), circle(12, 12, 0.8)] },
  // A treemap: how the business is made up.
  business: { mode: "stroke", d: ["M3.5 3.5h17v17h-17z", "M12.5 3.5v17", "M12.5 12h8", "M3.5 14.5h9"] },
  // A warning triangle.
  risks: {
    mode: "stroke",
    d: ["M12 3.8 2.9 19.3a1 1 0 0 0 .86 1.5h16.48a1 1 0 0 0 .86-1.5L12 3.8z", "M12 9.5v4.5", "M12 17.2v.3"],
  },
  // A checklist: the decisions.
  decisions: {
    mode: "stroke",
    d: [
      "M3.5 6.2l1.8 1.8 3.4-3.6",
      "M11.5 6h9",
      "M3.5 12.7l1.8 1.8 3.4-3.6",
      "M11.5 12.5h9",
      "M3.5 19.2l1.8 1.8 3.4-3.6",
      "M11.5 19h9",
    ],
  },
  // The fallback for a page this file has not been told about.
  chevron: { mode: "stroke", d: ["M9.5 6l6 6-6 6"] },

  menu: { mode: "stroke", d: ["M4 7h16", "M4 12h16", "M4 17h16"] },
  sun: {
    mode: "stroke",
    d: [
      circle(12, 12, 4),
      "M12 2.5v2.2",
      "M12 19.3v2.2",
      "M2.5 12h2.2",
      "M19.3 12h2.2",
      "M5.3 5.3l1.55 1.55",
      "M17.15 17.15l1.55 1.55",
      "M5.3 18.7l1.55-1.55",
      "M17.15 6.85l1.55-1.55",
    ],
  },
  moon: { mode: "stroke", d: ["M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"] },

  // Three rising bars: which, who, the top of a list.
  rank: { mode: "stroke", d: ["M5.5 20v-6", "M12 20V9", "M18.5 20V4"] },
  // A ruled circle: how much, how many.
  value: { mode: "stroke", d: [circle(12, 12, 8.5), "M8 10.3h8", "M8 13.9h5"] },
  // A diamond around a point: where.
  place: { mode: "stroke", d: ["M12 3.2 20.8 12 12 20.8 3.2 12z", circle(12, 12, 1.4)] },
  time: { mode: "stroke", d: DIAL },
  // One thing bending into another: why, what drove it.
  cause: { mode: "stroke", d: ["M3.5 18.5C9.5 18.5 10.5 8 18 8", "M14.5 4.5 18 8l-3.5 3.5"] },
  // An arrow pointing on: what to do.
  act: { mode: "stroke", d: ["M3.5 12h15.5", "M13.5 6.5 19 12l-5.5 5.5"] },
  // The four-point spark: the house mark for anything the model writes.
  spark: { mode: "fill", d: ["M12 3.3 14.03 9.98 20.7 12 14.03 14.03 12 20.7 9.98 14.03 3.3 12z"] },
};

/**
 * The paths of one line icon, without the `<svg>` around them. The Ask
 * surfaces size their marks from their own stylesheets and so draw their own
 * element; this keeps them on the same paths and the same stroke.
 */
export function IconPaths({ name }: { name: IconName }) {
  const icon = ICONS[name];
  const fill = icon.mode === "fill";
  return (
    <>
      {icon.d.map((d) => (
        <path
          key={d}
          d={d}
          fill={fill ? "currentColor" : "none"}
          stroke={fill ? "none" : "currentColor"}
          strokeWidth={1.75}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      ))}
    </>
  );
}

export interface IconProps {
  name: IconName;
  /** One of `SIZE`. Nothing else typechecks, on purpose. */
  size: IconSize;
  className?: string;
}

/** A line icon. Decorative: the control it sits in carries the label. */
export function Icon({ name, size, className }: IconProps) {
  return (
    <svg
      className={className ? `icon ${className}` : "icon"}
      data-icon={name}
      viewBox="0 0 24 24"
      width={size}
      height={size}
      aria-hidden="true"
      focusable="false"
    >
      <IconPaths name={name} />
    </svg>
  );
}

/* ------------------------------------------------------------ solid marks */

/**
 * The server's KPI icon vocabulary. `calendar` and `down` are sent today
 * alongside the ten the type comment lists, so they are drawn rather than
 * falling back to the generic bars.
 */
export type SolidName =
  | "pipeline"
  | "risk"
  | "clock"
  | "won"
  | "target"
  | "margin"
  | "account"
  | "people"
  | "growth"
  | "metric"
  | "calendar"
  | "down";

/**
 * One face of a solid. `o` is how much of the light it catches. A face with
 * `w` is a stroke of that width rather than a fill (the tick on the won cube);
 * `evenodd` cuts a hole (the rim of a dial).
 */
interface Face {
  d: string;
  o: number;
  w?: number;
  evenodd?: boolean;
}

/** Top-left light source, the same three tones as PersonaMark. */
const LIT = 1;
const SHADE = 0.66;
const DEEP = 0.36;
/** A cut face: a surface that is neither up nor a side. */
const CUT = 0.5;

/** Numbers in path data, trimmed so the markup stays readable in devtools. */
const n = (v: number) => String(Math.round(v * 100) / 100);

/**
 * An isometric box on the same 2:1 rhombus PersonaMark uses: `cx` the centre
 * line, `ty` the top vertex, `hw` the half width, `d` the drop of the sides.
 * Left face shaded, right face deep, up face lit.
 */
function box(cx: number, ty: number, hw: number, d: number, lit = LIT): Face[] {
  const l = cx - hw;
  const r = cx + hw;
  const m = ty + hw / 2;
  const b = ty + hw;
  return [
    { d: `M${n(l)} ${n(m)} L${n(cx)} ${n(b)} L${n(cx)} ${n(b + d)} L${n(l)} ${n(m + d)} Z`, o: SHADE },
    { d: `M${n(r)} ${n(m)} L${n(cx)} ${n(b)} L${n(cx)} ${n(b + d)} L${n(r)} ${n(m + d)} Z`, o: DEEP },
    { d: `M${n(cx)} ${n(ty)} L${n(r)} ${n(m)} L${n(cx)} ${n(b)} L${n(l)} ${n(m)} Z`, o: lit },
  ];
}

const ellipse = (cx: number, cy: number, rx: number, ry: number) =>
  `M${n(cx - rx)} ${n(cy)} A${n(rx)} ${n(ry)} 0 1 0 ${n(cx + rx)} ${n(cy)} A${n(rx)} ${n(ry)} 0 1 0 ${n(cx - rx)} ${n(cy)} Z`;

/** The front half of a cylinder's side, split at the front so the left half
 *  is shaded and the right half deep, the way a box's sides are. */
function band(cx: number, cy: number, rx: number, ry: number, h: number): Face[] {
  const l = n(cx - rx);
  const r = n(cx + rx);
  const f = n(cy + ry);
  return [
    {
      d: `M${l} ${n(cy)} A${n(rx)} ${n(ry)} 0 0 0 ${n(cx)} ${f} L${n(cx)} ${n(cy + ry + h)} A${n(rx)} ${n(ry)} 0 0 1 ${l} ${n(cy + h)} Z`,
      o: SHADE,
    },
    {
      d: `M${n(cx)} ${f} A${n(rx)} ${n(ry)} 0 0 0 ${r} ${n(cy)} L${r} ${n(cy + h)} A${n(rx)} ${n(ry)} 0 0 1 ${n(cx)} ${n(cy + ry + h)} Z`,
      o: DEEP,
    },
  ];
}

/** A short cylinder: side band, then a lit top. */
function puck(cx: number, cy: number, rx: number, ry: number, h: number): Face[] {
  return [...band(cx, cy, rx, ry, h), { d: ellipse(cx, cy, rx, ry), o: LIT }];
}

/** A flat ring on a top face. */
function ring(cx: number, cy: number, rx: number, ry: number, inset: number, o: number): Face {
  return {
    d: `${ellipse(cx, cy, rx, ry)} ${ellipse(cx, cy, rx - inset, ry - inset / 2)}`,
    o,
    evenodd: true,
  };
}

/** A staircase of three touching blocks, `tops` given left to right. */
function stairs(tops: [number, number, number]): Face[] {
  const floor = 29;
  const hw = 4.5;
  return [7.5, 16.5, 25.5].flatMap((cx, i) => box(cx, tops[i], hw, floor - tops[i] - hw));
}

/**
 * The wedge cut from the margin puck, as ellipse parameters: the slice from
 * three o'clock round to `ANGLE` on the front-right, where a cut is visible.
 */
const ANGLE = (70 * Math.PI) / 180;
const PX = 12 * Math.cos(ANGLE);
const PY = 6 * Math.sin(ANGLE);

export const SOLIDS: Record<SolidName, Face[]> = {
  // Three slabs stacked: the pipeline is layers of deals, one on another.
  pipeline: [...box(16, 15, 11, 3.5), ...box(16, 9, 11, 3.5), ...box(16, 3, 11, 3.5)],

  // A warning sign with depth: the front face shaded so the mark on it can be
  // lit, the extruded edge behind it deepest.
  risk: [
    { d: "M16 4 L19 2.5 L31 24.5 L28 26 Z", o: DEEP },
    { d: "M16 4 L28 26 L4 26 Z", o: SHADE },
    { d: "M14.8 11 L17.2 11 L17.2 18.6 L14.8 18.6 Z", o: LIT },
    { d: "M14.8 20.8 L17.2 20.8 L17.2 23.2 L14.8 23.2 Z", o: LIT },
  ],

  // A dial sunk into a puck: lit rim, shaded face, lit hands.
  clock: [
    ...band(16, 13, 12, 6, 7),
    { d: ellipse(16, 13, 9, 4.5), o: SHADE },
    ring(16, 13, 12, 6, 3, LIT),
    { d: "M15.1 13.4 L16.9 13.4 L16.9 9.8 L15.1 9.8 Z", o: LIT },
    { d: "M15.7 12.2 L21.6 14 L21.2 15.5 L15.4 13.7 Z", o: LIT },
    { d: ellipse(16, 13, 1.3, 0.8), o: LIT },
  ],

  // A cube with a tick cut into its deepest face, where it reads brightest.
  won: [...box(16, 5, 10, 10), { d: "M18 19.5 L20.5 20.75 L24.5 14.25", o: LIT, w: 2.2 }],

  // Concentric rings on a disc: the bullseye.
  target: [
    ...band(16, 14, 12, 6, 6),
    { d: ellipse(16, 14, 12, 6), o: SHADE },
    ring(16, 14, 12, 6, 2.5, LIT),
    ring(16, 14, 7, 3.5, 2.5, LIT),
    { d: ellipse(16, 14, 2, 1), o: LIT },
  ],

  // A puck with a wedge cut out and set beside it: the share that is margin.
  margin: [
    // the body, minus the slice from three o'clock to the cut
    { d: `M16 19 A12 6 0 0 1 4 13 L4 20 A12 6 0 0 0 16 26 Z`, o: SHADE },
    { d: `M16 13 L28 13 L28 20 L16 20 Z`, o: CUT },
    { d: `M${n(16 + PX)} ${n(13 + PY)} A12 6 0 0 1 16 19 L16 26 A12 6 0 0 0 ${n(16 + PX)} ${n(20 + PY)} Z`, o: DEEP },
    { d: `M16 13 L28 13 A12 6 0 1 0 ${n(16 + PX)} ${n(13 + PY)} Z`, o: LIT },
    // the slice, moved out to the front right
    { d: `M19 15.5 L${n(19 + PX)} ${n(15.5 + PY)} L${n(19 + PX)} ${n(20.5 + PY)} L19 20.5 Z`, o: SHADE },
    { d: `M31 15.5 A12 6 0 0 1 ${n(19 + PX)} ${n(15.5 + PY)} L${n(19 + PX)} ${n(20.5 + PY)} A12 6 0 0 0 31 20.5 Z`, o: DEEP },
    { d: `M19 15.5 L31 15.5 A12 6 0 0 1 ${n(19 + PX)} ${n(15.5 + PY)} Z`, o: LIT },
  ],

  // A stepped block: a building.
  account: [...box(16, 8, 9, 11), ...box(16, 5.5, 5, 5)],

  // Two rounded columns of unequal height: people, standing together.
  people: [...puck(10, 11, 5, 2.5, 15), ...puck(22, 7.5, 5, 2.5, 18.5)],

  // A staircase rising to the right.
  growth: stairs([19, 13, 7]),

  // Three thin bars on one floor: a measure, nothing more specific.
  metric: [...box(8, 15, 3, 9), ...box(16, 9, 3, 15), ...box(24, 18, 3, 6)],

  // A page with a spine along its top edge and one day marked on it.
  calendar: [
    ...box(16, 10, 11, 4),
    { d: "M5 13.5 L7 14.5 L7 16.5 L5 15.5 Z", o: SHADE },
    { d: "M18 9 L7 14.5 L7 16.5 L18 11 Z", o: DEEP },
    { d: "M5 13.5 L16 8 L18 9 L7 14.5 Z", o: LIT },
    ...box(19, 15.5, 2.5, 2),
  ],

  // A staircase falling to the right.
  down: stairs([7, 13, 19]),
};

/** The server's icon key, resolved to a mark; anything unknown is the bars. */
export function solidFor(key: string | undefined): SolidName {
  return key && key in SOLIDS ? (key as SolidName) : "metric";
}

export interface SolidProps {
  name: SolidName;
  size: IconSize;
  className?: string;
}

/** An isometric mark. Decorative: the tile's label says what it measures. */
export function Solid({ name, size, className }: SolidProps) {
  return (
    <svg
      className={className ? `solid ${className}` : "solid"}
      data-solid={name}
      viewBox="0 0 32 32"
      width={size}
      height={size}
      aria-hidden="true"
      focusable="false"
    >
      {SOLIDS[name].map((f) =>
        f.w ? (
          <path
            key={f.d}
            d={f.d}
            fill="none"
            stroke="currentColor"
            strokeWidth={f.w}
            strokeLinecap="round"
            strokeLinejoin="round"
            opacity={f.o}
          />
        ) : (
          <path key={f.d} d={f.d} fill="currentColor" fillRule={f.evenodd ? "evenodd" : undefined} opacity={f.o} />
        ),
      )}
    </svg>
  );
}
