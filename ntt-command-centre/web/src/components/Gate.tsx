/**
 * The password on the door.
 *
 * main.tsx renders this AROUND the app. With no token the gate is shown
 * INSTEAD of the app — not over it — so nothing behind it mounts, nothing
 * fetches, and there is no page underneath to leak a figure through a scrim.
 * When the client sees a 401 anywhere (api/client.ts dispatches
 * `ntt:locked`) the app is unmounted the same way; on the next successful
 * login it mounts afresh and fetches with the new token. That is the whole
 * lifecycle: there is no retry queue, because nothing is ever waiting.
 *
 * What this is not. It is a shared demo password, not identity — it decides
 * whether you are in the demo, and the persona still decides what you see
 * (row-level security is attached at the API boundary, api/main.py). The
 * password rotates through NTT_ACCESS_PASSWORD on the server; the token is
 * bearer-only and lives in localStorage under "ntt.access".
 *
 * The mark follows the same rule as the header: the theme picks a different
 * FILE (the reversed lockup on dark), the symbol is never recoloured.
 */
import { useCallback, useEffect, useId, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { ApiError, LOCKED_EVENT, auth, readAccess } from "../api/client";
import { useTheme } from "../theme/ThemeProvider";

export function Gate({ children }: { children: ReactNode }) {
  // The stored token's expiry is checked here as well as on the server, so
  // an expired laptop sees the gate immediately rather than a page skeleton
  // that collapses into it on the first 401.
  const [open, setOpen] = useState<boolean>(() => readAccess() !== null);

  useEffect(() => {
    const onLocked = () => setOpen(false);
    window.addEventListener(LOCKED_EVENT, onLocked);
    return () => window.removeEventListener(LOCKED_EVENT, onLocked);
  }, []);

  const enter = useCallback(() => setOpen(true), []);

  if (open) return <>{children}</>;
  return <GateForm onEnter={enter} />;
}

function GateForm({ onEnter }: { onEnter: () => void }) {
  const { theme } = useTheme();
  const [password, setPassword] = useState("");
  const [reveal, setReveal] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // The shake is a class that animationend removes, so a second wrong guess
  // can play it again. Typing also clears it: the field stops looking wrong
  // the moment the user starts fixing it.
  const [shake, setShake] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const errorId = useId();

  const submit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (busy) return;
    if (!password) {
      inputRef.current?.focus();
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await auth.login(password);
      onEnter();
    } catch (err) {
      const wrong = err instanceof ApiError && err.status === 401;
      setError(wrong ? "That password is not right" : "The server could not be reached");
      setShake(true);
      // Keep what was typed, selected, so the next keystroke replaces it and
      // a near-miss can still be read with Show on.
      inputRef.current?.focus();
      inputRef.current?.select();
      setBusy(false);
    }
  };

  return (
    <main className="gate">
      <form
        className="gate__card"
        onSubmit={submit}
        aria-labelledby="gate-title"
        aria-busy={busy || undefined}
      >
        <div className="gate__brand">
          <img
            className="gate__logo"
            src={theme === "dark" ? "/ntt-logo-reversed.png" : "/ntt-logo.png"}
            alt="NTT DATA"
            width={92}
            height={33}
          />
          <h1 className="gate__name" id="gate-title">Deal Intelligence</h1>
        </div>

        <p className="gate__lede">Enter the password to open the platform.</p>

        <label className="gate__label" htmlFor="gate-password">Password</label>
        <div
          className={`gate__field${shake ? " gate__field--wrong" : ""}`}
          onAnimationEnd={() => setShake(false)}
        >
          <input
            ref={inputRef}
            id="gate-password"
            className="gate__input"
            type={reveal ? "text" : "password"}
            name="password"
            autoComplete="current-password"
            autoCapitalize="off"
            autoCorrect="off"
            spellCheck={false}
            autoFocus
            required
            value={password}
            onChange={(e) => {
              setPassword(e.target.value);
              setShake(false);
              if (error) setError(null);
            }}
            aria-invalid={error ? true : undefined}
            aria-describedby={error ? errorId : undefined}
            // readOnly rather than disabled while the server is checking: a
            // disabled input cannot take focus, and focus has to come back
            // here when the answer is "wrong".
            readOnly={busy}
          />
          <button
            type="button"
            className="gate__reveal"
            onClick={() => setReveal((v) => !v)}
            aria-pressed={reveal}
            aria-label={reveal ? "Hide password" : "Show password"}
          >
            {reveal ? "Hide" : "Show"}
          </button>
        </div>

        {/* role=alert announces the message when it appears; the slot keeps
            its height either way so the card does not jump. */}
        <p className="gate__error" id={errorId} role="alert">
          {error ?? ""}
        </p>

        <button type="submit" className="gate__enter" disabled={busy}>
          {busy ? "Checking…" : "Enter"}
        </button>

        <p className="gate__note">
          One password for the demo. Who you are, and what you can see, is
          still chosen inside.
        </p>
      </form>
    </main>
  );
}
