/** REST client for the viz backend. */
export interface EpochInfo {
  epoch: number;
  dir: string;
  n_systems: number;
  mtime: number;
}

export interface SystemMeta {
  global_idx: number;
  sid: number;
  ads_id: number;
  n_atoms: number;
  n_steps: number;
  E_pred: number;
  E_gt: number;
  improvement: number;
  fmax_final: number;
  converged: boolean;
  status: string;        // "ok" | "uma_unconverged" | anomaly reason
  success: boolean;
}

export interface SystemsIndex {
  epoch_dir?: string;
  n_systems: number;
  systems: SystemMeta[];
}

export interface PerStepData {
  n_steps: number;
  energy: number[];
  fmax: number[];
}

async function json<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → ${r.status} ${r.statusText}`);
  return (await r.json()) as T;
}

// API paths are RELATIVE (no leading "/") so the app works when served under a
// reverse-proxy subpath (e.g. code-server's /proxy/3000/). A leading "/" would
// resolve against the host root and bypass the Next.js /api rewrite.
const API = "api";

export const api = {
  health: () => json<{ ok: boolean; viz_root: string; viz_root_exists: boolean }>(`${API}/health`),
  epochs: () => json<{ epochs: EpochInfo[]; count: number }>(`${API}/epochs`),
  systems: (epoch: number) => json<SystemsIndex>(`${API}/epochs/${epoch}/systems`),
  meta: (epoch: number, sys: number) => json<SystemMeta>(`${API}/epochs/${epoch}/systems/${sys}/meta`),
  data: (epoch: number, sys: number) => json<PerStepData>(`${API}/epochs/${epoch}/systems/${sys}/data`),
  // For "traj" the backend returns a multi-model PDB regardless of source
  // format (preferred .pdb on disk, falls back to converting .xyz).
  structureUrl: (epoch: number, sys: number, kind: "x0" | "x1_flow" | "x1_relaxed" | "traj") =>
    `${API}/epochs/${epoch}/systems/${sys}/structure/${kind}`,
};

export async function fetchText(url: string): Promise<string> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.text();
}
