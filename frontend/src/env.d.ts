/**
 * Typed environment variables.
 *
 * The Vite 8 template registers `vite/client` types in tsconfig but no longer
 * ships a `vite-env.d.ts`, so declare our own variables here to keep
 * `import.meta.env` type-safe under `strict`.
 */
interface ImportMetaEnv {
  /** Base URL of the FastAPI backend, e.g. http://localhost:8000 */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
