/** Ensure NGL.js is loaded from CDN. Returns a promise that resolves with the global NGL object. */
declare global {
  interface Window {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    NGL: any;
  }
}

let _loadPromise: Promise<unknown> | null = null;

export function loadNGL(): Promise<unknown> {
  if (typeof window === "undefined") return Promise.reject(new Error("SSR"));
  if (window.NGL) return Promise.resolve(window.NGL);
  if (_loadPromise) return _loadPromise;
  _loadPromise = new Promise((resolve, reject) => {
    const existing = document.getElementById("ngl-cdn") as HTMLScriptElement | null;
    if (existing) {
      if (existing.dataset.loaded === "true") return resolve(window.NGL);
      existing.addEventListener("load", () => { existing.dataset.loaded = "true"; resolve(window.NGL); }, { once: true });
      existing.addEventListener("error", () => reject(new Error("NGL load failed")), { once: true });
      return;
    }
    const s = document.createElement("script");
    s.id = "ngl-cdn";
    s.src = "https://cdn.jsdelivr.net/npm/ngl/dist/ngl.js";
    s.async = true;
    s.onload = () => { s.dataset.loaded = "true"; resolve(window.NGL); };
    s.onerror = () => reject(new Error("NGL load failed"));
    document.head.appendChild(s);
  });
  return _loadPromise;
}

/** Suppress NGL's noisy deprecation warnings + stage log. */
let _patched = false;
export function suppressNglLogs(): void {
  if (_patched) return;
  _patched = true;
  const origWarn = console.warn;
  console.warn = (...args: unknown[]) => {
    if (typeof args[0] === "string" && args[0].includes("useLegacyLights")) return;
    origWarn.apply(console, args);
  };
  const origLog = console.log;
  console.log = (...args: unknown[]) => {
    if (typeof args[0] === "string" && args[0].startsWith("STAGE LOG")) return;
    origLog.apply(console, args);
  };
}
