"use client";

import { useEffect, useMemo, useState } from "react";
import { Play, Pause, SkipBack, SkipForward } from "lucide-react";
import { api, EpochInfo, SystemMeta, PerStepData } from "@/lib/api";
import MoleculeViewer from "@/components/MoleculeViewer";
import EnergyPlot from "@/components/EnergyPlot";

function StatusBadge({ status, converged, success }: { status: string; converged: boolean; success: boolean }) {
  const color =
    success ? "bg-emerald-600/40 text-emerald-200 border-emerald-600" :
    status === "ok" ? "bg-blue-600/40 text-blue-200 border-blue-600" :
    status === "uma_unconverged" ? "bg-amber-600/40 text-amber-200 border-amber-600" :
    "bg-red-600/40 text-red-200 border-red-600";
  return (
    <span className={`text-[10px] px-1.5 py-0.5 rounded border ${color} font-mono`}>
      {success ? "SUCCESS" : status}
    </span>
  );
}

export default function Page() {
  const [epochs, setEpochs] = useState<EpochInfo[]>([]);
  const [selEpoch, setSelEpoch] = useState<number | null>(null);
  const [systems, setSystems] = useState<SystemMeta[]>([]);
  const [selSys, setSelSys] = useState<number | null>(null);
  const [perStep, setPerStep] = useState<PerStepData | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.epochs().then(({ epochs }) => {
      setEpochs(epochs);
      if (epochs.length && selEpoch === null) setSelEpoch(epochs[0].epoch);
    }).catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (selEpoch === null) return;
    api.systems(selEpoch).then(d => {
      setSystems(d.systems);
      if (d.systems.length && selSys === null) setSelSys(d.systems[0].global_idx);
    }).catch(e => setErr(String(e)));
  }, [selEpoch]);

  useEffect(() => {
    if (selEpoch === null || selSys === null) { setPerStep(null); return; }
    setPerStep(null); setFrame(0);
    api.data(selEpoch, selSys).then(setPerStep).catch(e => setErr(String(e)));
  }, [selEpoch, selSys]);

  // Autoplay
  useEffect(() => {
    if (!playing || !perStep) return;
    const id = setInterval(() => {
      setFrame(f => (f + 1 >= perStep.n_steps ? 0 : f + 1));
    }, 80);
    return () => clearInterval(id);
  }, [playing, perStep]);

  const selSysMeta = useMemo(
    () => systems.find(s => s.global_idx === selSys) ?? null,
    [systems, selSys],
  );

  const structureUrl = (kind: "x0" | "x1_flow" | "x1_relaxed" | "traj") =>
    selEpoch !== null && selSys !== null ? api.structureUrl(selEpoch, selSys, kind) : "";

  return (
    <main className="min-h-screen p-4">
      <header className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold text-gray-100">AdsorbGen Replay Viz</h1>
        <div className="text-xs text-gray-400 font-mono">
          {epochs.length} epoch(s) · {systems.length} system(s) in current
        </div>
      </header>

      {err && (
        <div className="mb-3 p-2 text-sm bg-red-900/30 border border-red-700 rounded text-red-200">
          {err}
        </div>
      )}

      <div className="grid grid-cols-12 gap-4">
        {/* Left sidebar: selectors + system list */}
        <aside className="col-span-3 space-y-3">
          <div className="bg-gray-900/50 border border-gray-700 rounded-lg p-3">
            <label className="block text-xs text-gray-400 mb-1">Epoch</label>
            <select
              value={selEpoch ?? ""}
              onChange={e => setSelEpoch(Number(e.target.value))}
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm"
            >
              {epochs.map(ep => (
                <option key={ep.epoch} value={ep.epoch}>
                  ep {ep.epoch} ({ep.n_systems} sys)
                </option>
              ))}
            </select>
          </div>

          <div className="bg-gray-900/50 border border-gray-700 rounded-lg">
            <div className="px-3 py-2 text-xs text-gray-400 border-b border-gray-700">
              Systems ({systems.length})
            </div>
            <div className="max-h-[70vh] overflow-y-auto">
              {systems.map(s => (
                <button
                  key={s.global_idx}
                  onClick={() => setSelSys(s.global_idx)}
                  className={`w-full text-left px-3 py-2 text-xs border-b border-gray-800 hover:bg-gray-800/60 ${
                    selSys === s.global_idx ? "bg-gray-800" : ""
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-gray-200">sys_{String(s.global_idx).padStart(3,"0")}</span>
                    <StatusBadge status={s.status} converged={s.converged} success={s.success} />
                  </div>
                  <div className="mt-1 text-[10px] text-gray-400 font-mono">
                    sid={s.sid} · {s.n_atoms}a · fmax={s.fmax_final?.toFixed(3)}
                  </div>
                </button>
              ))}
            </div>
          </div>
        </aside>

        {/* Right main: structure grid + energy plot */}
        <section className="col-span-9 space-y-3">
          {selSysMeta && (
            <div className="grid grid-cols-4 gap-3 text-xs font-mono bg-gray-900/50 border border-gray-700 rounded-lg p-3">
              <div><span className="text-gray-400">sid </span>{selSysMeta.sid}</div>
              <div><span className="text-gray-400">ads_id </span>{selSysMeta.ads_id}</div>
              <div><span className="text-gray-400">E_gt </span>{selSysMeta.E_gt?.toFixed(4)} eV</div>
              <div><span className="text-gray-400">E_pred </span>{selSysMeta.E_pred?.toFixed(4)} eV</div>
              <div><span className="text-gray-400">Δ </span>{selSysMeta.improvement?.toFixed(4)} eV</div>
              <div><span className="text-gray-400">fmax </span>{selSysMeta.fmax_final?.toFixed(4)}</div>
              <div><span className="text-gray-400">n_steps </span>{selSysMeta.n_steps}</div>
              <div><span className="text-gray-400">status </span>{selSysMeta.status}</div>
            </div>
          )}

          <div className="grid grid-cols-3 gap-3">
            <div>
              <div className="text-[11px] text-gray-400 mb-1">x_0 (prior placement)</div>
              {selEpoch !== null && selSys !== null && (
                <MoleculeViewer url={structureUrl("x0")} ext="pdb" label="x_0" height="340px" />
              )}
            </div>
            <div>
              <div className="text-[11px] text-gray-400 mb-1">x_1_flow (model prediction)</div>
              {selEpoch !== null && selSys !== null && (
                <MoleculeViewer url={structureUrl("x1_flow")} ext="pdb" label="x_1 flow" height="340px" />
              )}
            </div>
            <div>
              <div className="text-[11px] text-gray-400 mb-1">
                x_1_relaxed (UMA end){selSysMeta?.converged ? "" : " — unconverged"}
              </div>
              {selEpoch !== null && selSys !== null && selSysMeta?.converged && (
                <MoleculeViewer url={structureUrl("x1_relaxed")} ext="pdb" label="x_1 relaxed" height="340px" />
              )}
              {selSysMeta && !selSysMeta.converged && (
                <div className="h-[340px] border border-amber-700/40 bg-amber-900/10 rounded-lg flex items-center justify-center text-amber-300 text-xs text-center p-4">
                  Not converged; no final snapshot saved.
                </div>
              )}
            </div>
          </div>

          {/* Trajectory + energy plot */}
          {selEpoch !== null && selSys !== null && perStep && (
            <div className="bg-gray-900/50 border border-gray-700 rounded-lg p-3 space-y-2">
              <div className="flex items-center justify-between">
                <div className="text-sm text-gray-300">Relaxation trajectory</div>
                <div className="flex items-center gap-2">
                  <button onClick={() => setFrame(0)} className="p-1.5 rounded bg-gray-800 hover:bg-gray-700">
                    <SkipBack size={14} />
                  </button>
                  <button onClick={() => setPlaying(p => !p)} className="p-1.5 rounded bg-gray-800 hover:bg-gray-700">
                    {playing ? <Pause size={14} /> : <Play size={14} />}
                  </button>
                  <button onClick={() => setFrame(perStep.n_steps - 1)} className="p-1.5 rounded bg-gray-800 hover:bg-gray-700">
                    <SkipForward size={14} />
                  </button>
                  <span className="text-[11px] text-gray-400 font-mono w-20">
                    {frame} / {perStep.n_steps - 1}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-12 gap-3">
                <div className="col-span-5">
                  <MoleculeViewer url={structureUrl("traj")} ext="pdb" label={`traj · step ${frame}`} frame={frame} height="340px" />
                  <input
                    type="range"
                    min={0}
                    max={perStep.n_steps - 1}
                    value={frame}
                    onChange={e => setFrame(Number(e.target.value))}
                    className="w-full mt-2 accent-blue-500"
                  />
                </div>
                <div className="col-span-7">
                  <EnergyPlot
                    energy={perStep.energy}
                    fmax={perStep.fmax}
                    frame={frame}
                    onFrameChange={setFrame}
                  />
                </div>
              </div>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
