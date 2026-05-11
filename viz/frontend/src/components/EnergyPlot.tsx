"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";
import type { ComponentType } from "react";

// Plotly must be client-side only; silence the prop-type loss from dynamic().
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false }) as unknown as ComponentType<any>;

interface Props {
  energy: number[];
  fmax: number[];
  frame: number;                 // currently selected step
  onFrameChange?: (f: number) => void;
}

export default function EnergyPlot({ energy, fmax, frame, onFrameChange }: Props) {
  const x = useMemo(() => Array.from({ length: energy.length }, (_, i) => i), [energy.length]);
  const currentE = energy[Math.min(frame, energy.length - 1)];
  const currentF = fmax[Math.min(frame, fmax.length - 1)];

  return (
    <div className="w-full">
      {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
      <Plot
        data={[
          { x, y: energy, type: "scatter", mode: "lines", name: "E (eV)",
            yaxis: "y", line: { color: "#60a5fa", width: 2 } },
          { x, y: fmax, type: "scatter", mode: "lines", name: "fmax (eV/Å)",
            yaxis: "y2", line: { color: "#f97316", width: 2 } },
          { x: [frame], y: [currentE], type: "scatter", mode: "markers", name: "cursor",
            yaxis: "y", marker: { color: "#fff", size: 9, symbol: "diamond" },
            showlegend: false },
        ] as unknown[]}
        layout={{
          height: 260,
          margin: { l: 50, r: 50, t: 10, b: 40 },
          plot_bgcolor: "#0b1220",
          paper_bgcolor: "#0b1220",
          font: { color: "#e5e7eb", size: 11 },
          showlegend: true,
          legend: { x: 0.02, y: 0.98, bgcolor: "rgba(0,0,0,0.3)" },
          xaxis: { title: { text: "FIRE step" }, gridcolor: "#1f2937" },
          yaxis: { title: { text: "Energy (eV)" }, gridcolor: "#1f2937" },
          yaxis2: {
            title: { text: "fmax (eV/Å)" },
            overlaying: "y", side: "right",
            type: "log", gridcolor: "transparent",
          },
          annotations: [
            { xref: "paper", yref: "paper", x: 0.5, y: -0.22,
              text: `step ${frame} · E=${currentE?.toFixed(4)} eV · fmax=${currentF?.toFixed(4)} eV/Å`,
              showarrow: false, font: { color: "#94a3b8", size: 11 } },
          ],
        } as unknown as object}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
        onClick={(e: { points?: Array<{ x: number }> }) => {
          const p = e?.points?.[0];
          if (p && typeof p.x === "number") {
            onFrameChange?.(Math.round(p.x));
          }
        }}
      />
    </div>
  );
}
