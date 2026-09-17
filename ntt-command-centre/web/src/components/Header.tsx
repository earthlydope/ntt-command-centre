/**
 * The shell header.
 *
 * It holds exactly three things: the mark, the theme, and who you are.
 * Everything else that used to live here said something the page below
 * already says — or, in the case of the Ask pill and the findings bell, was
 * a second door to a room that already has one: Ask AI Expert lives on the
 * floating button in the bottom-right corner of every page, and what needs a
 * decision is the action rail, which is above the fold on every page.
 *
 * What was removed and why — the scope line ("Executive · North America · As of
 * 15 Sep 2026") repeated the persona pill on its right, the page's own subtitle
 * below it, and the footer's date stamp. Three statements of one fact in one
 * viewport is not thoroughness, it is noise, and it pushed the actual content
 * below the fold.
 *
 * **The logo is the real mark.** `ntt-logo.png` is the official lockup;
 * `ntt-logo-reversed.png` is the same file with the wordmark knocked out to
 * white for the dark shell. The symbol is never recoloured or distorted — the
 * theme picks a different FILE, because a black wordmark on a navy header is
 * invisible and a white one on paper is worse.
 *
 * The persona control is the visible face of row-level security. The client put
 * authentication at the API boundary, so a switch that changes the server's
 * predicate is his diagram made visible rather than a shortcut around SSO.
 */
import { useEffect, useRef, useState } from "react";
import type { MetaPayload, PersonaKey } from "../api/types";
import { useApp } from "../state/AppStateProvider";
import { useTheme } from "../theme/ThemeProvider";
import { Icon, SIZE } from "./icons";
import { PersonaMark, PersonaPicker } from "./PersonaPicker";

export function Header({
  meta,
  onToggleRail,
}: {
  meta: MetaPayload | null;
  onToggleRail?: () => void;
}) {
  const { state, setPersona, setIdentity } = useApp();
  const { theme, toggle } = useTheme();
  const [menu, setMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!menu) return;
    const onDown = (e: MouseEvent) => {
      if (
        !menuRef.current?.contains(e.target as Node) &&
        !triggerRef.current?.contains(e.target as Node)
      )
        setMenu(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setMenu(false);
        triggerRef.current?.focus();
      }
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [menu]);

  const active = meta?.persona.active;
  const personas = meta?.persona.personas ?? [];
  // The URL may carry no identity, in which case the server has chosen the
  // default and reported it back on `active`; that is the one to mark.
  const currentIdentity = state.identity || active?.identity || "";

  // A role on its own lands on that role's default person, which the server
  // chooses. A person under a different role is one dispatch, because the
  // reducer accepts both together and a two-step switch would fetch the
  // default book first and throw it away. The same role, different person, is
  // the only case that keeps the page.
  const pick = (persona: PersonaKey, identity?: string) => {
    if (persona !== state.persona) setPersona(persona, identity);
    else if (identity && identity !== currentIdentity) setIdentity(identity);
  };

  return (
    <header className="hdr">
      {onToggleRail && (
        <button
          type="button"
          className="ghost hdr-icon hdr-rail"
          onClick={onToggleRail}
          aria-label="Show or hide the menu"
        >
          <Icon name="menu" size={SIZE.control} />
        </button>
      )}

      <img
        className="hdr-logo"
        src={theme === "dark" ? "/ntt-logo-reversed.png" : "/ntt-logo.png"}
        alt="NTT DATA"
        width={92}
        height={33}
      />
      <span className="hdr-name">Deal Intelligence</span>

      <div className="hdr-right">
        {/* The mark is the theme you would switch TO, matching the label:
            a moon on the light shell, a sun on the dark one. */}
        <button
          type="button"
          className="ghost hdr-icon"
          onClick={toggle}
          aria-label={theme === "light" ? "Use dark theme" : "Use light theme"}
        >
          <Icon name={theme === "light" ? "moon" : "sun"} size={SIZE.control} />
        </button>

        <div className="who-wrap">
          <button
            type="button"
            ref={triggerRef}
            className="who"
            aria-haspopup="dialog"
            aria-expanded={menu}
            onClick={() => setMenu((v) => !v)}
          >
            {/* The same isometric mark the picker uses, so the thing in the
                header is recognisably the thing you chose. The tile keeps its
                solid persona fill; persona.css knocks the mark out against it. */}
            <span className="who-avatar" aria-hidden="true">
              <PersonaMark persona={state.persona} size={26} />
            </span>
            <span className="who-text">
              <span className="who-name">{active?.identityLabel ?? "Loading"}</span>
              <span className="who-role">{active?.label ?? ""}</span>
            </span>
            <span className="who-caret" aria-hidden="true">▾</span>
          </button>

          {/* One panel answers both questions — which role, and which person
              within it — by folding the other roles away when one is hovered
              or focused. The picker owns the keyboard model and Escape; this
              wrapper owns only where the panel sits and the outside click. */}
          {menu && (
            <div className="who-menu" ref={menuRef} role="dialog" aria-label="View as">
              <PersonaPicker
                personas={personas}
                identities={meta?.persona.identities ?? { ae: [], manager: [], executive: [] }}
                activePersona={state.persona}
                activeIdentity={currentIdentity}
                onPick={pick}
                onClose={() => setMenu(false)}
              />
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
