"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2, AlertCircle, Crosshair } from "lucide-react";
import { loadNGL, suppressNglLogs } from "@/lib/ngl";

interface Props {
  /** URL of a static structure (.pdb) or multi-frame trajectory (.xyz). */
  url: string;
  ext: "pdb" | "xyz";
  label?: string;
  /** For xyz trajectories: current frame index (0-based). Ignored for pdb. */
  frame?: number;
  /** Inline (embedded) or larger card. */
  height?: string;
}

export default function MoleculeViewer({ url, ext, label, frame = 0, height = "360px" }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const stageRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const componentRef = useRef<any>(null);
  // For multi-model PDB trajectories: number of MODEL records detected.
  // We scrub frames via component.setSelection(`/N`) which displays only
  // the atoms belonging to MODEL N. Without this, NGL renders ALL models
  // superimposed, which looks like a clump.
  const nModelsRef = useRef<number>(0);

  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [atomCount, setAtomCount] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    let ro: ResizeObserver | null = null;
    setReady(false); setError(null); setAtomCount(null);
    nModelsRef.current = 0;

    (async () => {
      try {
        await loadNGL();
        if (cancelled || !containerRef.current || !window.NGL) return;
        suppressNglLogs();

        if (stageRef.current) {
          try { stageRef.current.dispose(); } catch { /* ignore */ }
        }
        containerRef.current.innerHTML = "";
        const stage = new window.NGL.Stage(containerRef.current, { backgroundColor: "#0b1220" });
        stageRef.current = stage;
        ro = new ResizeObserver(() => stage.handleResize());
        ro.observe(containerRef.current);

        const component = await stage.loadFile(url, { ext, defaultRepresentation: false });
        if (cancelled) return;
        componentRef.current = component;

        // Detect multi-model PDB (== trajectory frames). NGL stores this on
        // ``structure.modelStore.count``. If > 1 we restrict the initial view
        // to MODEL 0 only; the frame cursor scrubs by changing the selection.
        let nModels = 0;
        try { nModels = component.structure?.modelStore?.count ?? 0; } catch { /* ignore */ }
        nModelsRef.current = nModels;
        const initSel = nModels > 1 ? `/${frame}` : "all";
        try { component.setDefaultAssembly?.(""); } catch { /* ignore */ }

        // ASE GUI-style: chunky spacefill atoms + thin licorice bonds + cell box.
        // ``radiusScale: 0.5`` matches ase.visualize.view(viewer='ngl') feel.
        // ``sele`` restricts to current frame for multi-model PDB.
        component.addRepresentation("spacefill", { colorScheme: "element", radiusScale: 0.5, sele: initSel });
        component.addRepresentation("licorice", { colorScheme: "element", radiusScale: 0.25, sele: initSel });
        component.addRepresentation("unitcell", { opacity: 0.4 });

        component.autoView(600);
        try { setAtomCount(component.structure.atomCount); } catch { /* ignore */ }
        setReady(true);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();

    return () => {
      cancelled = true;
      ro?.disconnect();
      if (stageRef.current) {
        try { stageRef.current.dispose(); } catch { /* ignore */ }
        stageRef.current = null;
      }
      componentRef.current = null;
      nModelsRef.current = 0;
    };
  }, [url, ext]);

  // Scrub frame for multi-model PDB trajectory: update each representation's
  // ``sele`` to ``/N`` where N is the current model index. NGL re-renders
  // showing only atoms in that MODEL.
  useEffect(() => {
    const comp = componentRef.current;
    if (!comp || nModelsRef.current <= 1) return;
    const idx = Math.min(Math.max(frame, 0), nModelsRef.current - 1);
    const sel = `/${idx}`;
    try {
      comp.eachRepresentation((repr: { name: string; setSelection: (s: string) => void }) => {
        if (repr.name === "spacefill" || repr.name === "licorice") {
          repr.setSelection(sel);
        }
      });
      stageRef.current?.viewer?.requestRender?.();
    } catch { /* ignore */ }
  }, [frame]);

  const handleReset = () => {
    try { componentRef.current?.autoView(600); } catch { /* ignore */ }
  };

  return (
    <div className="relative border border-gray-700 rounded-lg overflow-hidden bg-[#0b1220]" style={{ height }}>
      {label && (
        <div className="absolute top-1.5 left-2 z-10 px-2 py-0.5 rounded bg-black/60 text-[11px] text-gray-200 font-mono">
          {label}
        </div>
      )}
      {atomCount !== null && (
        <div className="absolute top-1.5 right-2 z-10 px-2 py-0.5 rounded bg-black/60 text-[11px] text-gray-300 font-mono">
          {atomCount} atoms
        </div>
      )}
      {!ready && !error && (
        <div className="absolute inset-0 flex items-center justify-center text-gray-400 z-10">
          <Loader2 size={16} className="animate-spin mr-2" />
          <span className="text-xs">Loading…</span>
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-red-400 gap-2 z-10 p-4">
          <AlertCircle size={18} />
          <span className="text-[11px] text-center break-all">{error}</span>
        </div>
      )}
      <div ref={containerRef} className="w-full h-full" />
      {ready && (
        <button
          onClick={handleReset}
          className="absolute bottom-2 right-2 z-10 flex items-center gap-1 px-2 py-1 bg-black/60 hover:bg-black/80 rounded text-[10px] text-gray-200"
        >
          <Crosshair size={10} />
          Reset
        </button>
      )}
    </div>
  );
}
