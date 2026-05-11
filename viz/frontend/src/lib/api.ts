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

export const api = {
  health: () => json<{ ok: boolean; viz_root: string; viz_root_exists: boolean }>("/api/health"),
  epochs: () => json<{ epochs: EpochInfo[]; count: number }>("/api/epochs"),
  systems: (epoch: number) => json<SystemsIndex>(`/api/epochs/${epoch}/systems`),
  meta: (epoch: number, sys: number) => json<SystemMeta>(`/api/epochs/${epoch}/systems/${sys}/meta`),
  data: (epoch: number, sys: number) => json<PerStepData>(`/api/epochs/${epoch}/systems/${sys}/data`),
  // For "traj" the backend returns a multi-model PDB regardless of source
  // format (preferred .pdb on disk, falls back to converting .xyz).
  structureUrl: (epoch: number, sys: number, kind: "x0" | "x1_flow" | "x1_relaxed" | "traj") =>
    `/api/epochs/${epoch}/systems/${sys}/structure/${kind}`,
};

export async function fetchText(url: string): Promise<string> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.text();
}
