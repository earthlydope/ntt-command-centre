/// <reference types="vite/client" />

/**
 * `import.meta.env` is Vite's, not TypeScript's. Without this reference the
 * API base URL read in api/client.ts does not type-check, and the build would
 * fall back to a same-origin path silently.
 */
interface ImportMetaEnv {
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
