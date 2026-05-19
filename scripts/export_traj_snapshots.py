#!/usr/bin/env python
"""Export per-system success trajectories as snapshot files for external
rendering (e.g. OVITO).

For each unique success sid in a viz epoch dir, concatenates the flow inference
trajectory (``flow_traj.xyz``: x0 -> x1_flow) and the UMA relaxation trajectory
(``traj.xyz``: x1_flow -> x1_relaxed) and writes, per system:
  <out>/sid<SID>/snap_XXXX.cif  - one CIF per snapshot, numbered (flow then relax)
  <out>/sid<SID>/combined.xyz   - the same frames as one multi-frame extxyz
                                  (OVITO opens this single file as an animation)
Each frame carries ``phase`` (FLOW/RELAX) and ``step`` in its info dict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read, write


def _trim_at_convergence(relax, fmax, thresh=0.05, pad=5):
    """Trim the relaxation at fmax convergence.

    Batched FIRE keeps stepping a chunk until its slowest member converges (or
    the step cap), so an early-converged system's tail is just post-convergence
    FIRE jitter. Keep up to the last step whose fmax still exceeds ``thresh``
    (the convergence criterion), plus a few settled frames.
    """
    fmax = np.asarray(fmax, dtype=float)
    if fmax.shape[0] != len(relax) or len(relax) < 3:
        return relax  # length mismatch / too short — keep as-is
    above = np.where(fmax > thresh)[0]
    if len(above) == 0:
        return relax[:min(len(relax), pad + 1)]
    end = min(len(relax), int(above[-1]) + 1 + pad)
    return relax[:end]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--viz-ep-dir", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    ep = Path(args.viz_ep_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    idx = json.loads((ep / "_index.json").read_text())
    seen: set = set()
    for s in idx["systems"]:
        sid = int(s["sid"])
        if sid in seen or not s.get("success"):
            continue
        d = ep / s["sys_dir_name"]
        if not ((d / "flow_traj.xyz").exists() and (d / "traj.xyz").exists()):
            continue
        seen.add(sid)

        flow = read(str(d / "flow_traj.xyz"), index=":")
        relax_full = read(str(d / "traj.xyz"), index=":")
        fmax = np.load(str(d / "data.npz"))["fmax"]
        relax = _trim_at_convergence(relax_full, fmax)
        frames = []
        for i, a in enumerate(flow):
            a.info["phase"] = "FLOW"
            a.info["phase_id"] = 0          # numeric tag for OVITO expressions
            a.info["step"] = i
            frames.append(a)
        for i, a in enumerate(relax):
            a.info["phase"] = "RELAX"
            a.info["phase_id"] = 1
            a.info["step"] = i
            frames.append(a)

        sysdir = out / f"sid{sid}"
        sysdir.mkdir(parents=True, exist_ok=True)
        for a in frames:
            # filename carries the phase (flow/relax) + that phase's step index
            name = f"{a.info['phase'].lower()}_{int(a.info['step']):04d}.cif"
            write(str(sysdir / name), a, format="cif")
        write(str(sysdir / "combined.xyz"), frames, format="extxyz")
        print(f"sid {sid}: {len(flow)} flow + {len(relax)} relax "
              f"(frozen tail {len(relax_full) - len(relax)} frames trimmed) "
              f"= {len(frames)} snapshots -> {sysdir}", flush=True)


if __name__ == "__main__":
    main()
