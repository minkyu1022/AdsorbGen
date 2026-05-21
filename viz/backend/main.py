"""Replay viz backend (FastAPI).

Serves structures + trajectories + per-step data captured by
``adsorbgen.replay.viz`` during training replay eval.

Run:
    uvicorn viz.backend.main:app --host 0.0.0.0 --port 8000 --reload

Environment:
    REPLAY_VIZ_ROOT   default /home/minkyu/Cat-bench/runs/full_run_w_replay/replay_viz

Data layout expected at VIZ_ROOT:
    ep{N}/
      sys_{XXX}/
        {x0.pdb, x1_flow.pdb, x1_relaxed.pdb, traj.xyz,
         data.npz, meta.json}
      _index.json
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

DEFAULT_VIZ_ROOT = Path(
    os.environ.get(
        "REPLAY_VIZ_ROOT",
        "/home/minkyu/Cat-bench/runs/full_run_w_replay/replay_viz",
    )
)

app = FastAPI(title="AdsorbGen Replay Viz", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- helpers ----------

_EPOCH_RE = re.compile(r"^ep(\d+)$")
# Match any ``sys_*`` folder; supports legacy ``sys_001``, sharded
# ``sys_s0_001``, and ad-hoc ``sys_success_002`` from viz-redo runs.
_SYS_RE = re.compile(r"^sys_.+$")
_STRUCTURE_KINDS = {"x0", "x1_flow", "x1_relaxed"}


def _viz_root() -> Path:
    root = DEFAULT_VIZ_ROOT
    if not root.exists():
        raise HTTPException(status_code=503, detail=f"VIZ_ROOT not found: {root}")
    return root


def _epoch_dir(epoch: int) -> Path:
    p = _viz_root() / f"ep{epoch}"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"epoch {epoch} not found")
    return p


def _sys_dir(epoch: int, sys_idx: int) -> Path:
    """Resolve sys_idx → sys_*/ folder for a given epoch.

    Folder naming evolved across the project:
      * legacy single-process replay: ``sys_{idx:03d}`` (e.g. ``sys_007``)
      * 4-GPU shard merge:            ``sys_s{shard}_{idx:03d}``
      * viz-redo (sub-targeting):     ``sys_{name_prefix}_{idx:03d}``
    Resolution strategy:
      1. Read ``_index.json`` and find the entry whose ``global_idx == sys_idx``;
         use its ``sys_dir_name`` to locate the folder.
      2. Fallback to the legacy literal name ``sys_{idx:03d}``.
    """
    d = _epoch_dir(epoch)
    idx_path = d / "_index.json"
    if idx_path.exists():
        try:
            payload = json.loads(idx_path.read_text())
            for entry in payload.get("systems", []):
                if int(entry.get("global_idx", -1)) == int(sys_idx):
                    name = entry.get("sys_dir_name")
                    if name:
                        p = d / name
                        if p.exists():
                            return p
        except Exception:
            pass
    # Legacy literal fallback (single-process replay before merge)
    p = d / f"sys_{sys_idx:03d}"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"sys_{sys_idx} not found in ep{epoch}")
    return p


# ---------- endpoints ----------

@app.get("/api/health")
def health() -> Dict[str, Any]:
    root = DEFAULT_VIZ_ROOT
    return {
        "ok": True,
        "viz_root": str(root),
        "viz_root_exists": root.exists(),
    }


@app.get("/api/epochs")
def list_epochs() -> Dict[str, Any]:
    """Enumerate ``ep{N}`` subdirs. Newest first."""
    root = _viz_root()
    epochs: List[Dict[str, Any]] = []
    for child in root.iterdir():
        m = _EPOCH_RE.match(child.name)
        if not m or not child.is_dir():
            continue
        n = int(m.group(1))
        idx_path = child / "_index.json"
        n_systems = 0
        if idx_path.exists():
            try:
                idx = json.loads(idx_path.read_text())
                n_systems = int(idx.get("n_systems", 0))
            except Exception:
                pass
        epochs.append({
            "epoch": n,
            "dir": child.name,
            "n_systems": n_systems,
            "mtime": child.stat().st_mtime,
        })
    epochs.sort(key=lambda e: e["epoch"], reverse=True)
    return {"epochs": epochs, "count": len(epochs)}


@app.get("/api/epochs/{epoch}/systems")
def list_systems(epoch: int) -> Dict[str, Any]:
    """Return per-system metadata for an epoch (from ``_index.json`` if present,
    else scan subdirs)."""
    d = _epoch_dir(epoch)
    idx_path = d / "_index.json"
    if idx_path.exists():
        try:
            payload = json.loads(idx_path.read_text())
            return payload
        except Exception:
            pass
    # Fallback: scan sys_*/meta.json
    systems = []
    for child in sorted(d.iterdir()):
        m = _SYS_RE.match(child.name)
        if not m:
            continue
        meta_path = child / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                systems.append(meta)
            except Exception:
                pass
    return {"epoch_dir": d.name, "n_systems": len(systems), "systems": systems}


@app.get("/api/epochs/{epoch}/systems/{sys_idx}/meta")
def get_meta(epoch: int, sys_idx: int) -> Dict[str, Any]:
    d = _sys_dir(epoch, sys_idx)
    p = d / "meta.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="meta.json not found")
    try:
        return json.loads(p.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read meta.json: {e}")


@app.get("/api/epochs/{epoch}/systems/{sys_idx}/structure/{kind}")
def get_structure(epoch: int, sys_idx: int, kind: str):
    """Return one of ``x0.pdb``, ``x1_flow.pdb``, ``x1_relaxed.pdb``, ``traj.xyz``.

    ``kind`` ∈ {x0, x1_flow, x1_relaxed, traj}.
    """
    d = _sys_dir(epoch, sys_idx)
    if kind == "traj":
        # Prefer multi-model PDB (NGL native); fall back to converting xyz on the fly.
        p_pdb = d / "traj.pdb"
        if p_pdb.exists():
            return FileResponse(str(p_pdb), media_type="chemical/x-pdb", filename="traj.pdb")
        p_xyz = d / "traj.xyz"
        if not p_xyz.exists():
            raise HTTPException(status_code=404, detail="trajectory file not found")
        # Convert .xyz → multi-model PDB on the fly (legacy data path).
        try:
            from ase.io import read as ase_read, write as ase_write
            from io import StringIO
            frames = ase_read(str(p_xyz), index=":")
            buf = StringIO()
            ase_write(buf, frames, format="proteindatabank")
            text = buf.getvalue()
            return PlainTextResponse(content=text, media_type="chemical/x-pdb")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"xyz→pdb conversion failed: {e}")
    if kind in _STRUCTURE_KINDS:
        p = d / f"{kind}.pdb"
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"{kind}.pdb not found")
        return FileResponse(str(p), media_type="chemical/x-pdb", filename=f"{kind}.pdb")
    raise HTTPException(status_code=400, detail=f"unknown kind {kind!r}; expected one of {_STRUCTURE_KINDS | {'traj'}}")


@app.get("/api/epochs/{epoch}/systems/{sys_idx}/data")
def get_per_step_data(epoch: int, sys_idx: int) -> Dict[str, Any]:
    """Return per-FIRE-step ``energy`` and ``fmax`` as JSON arrays.

    Positions are also available in data.npz but not returned here (sent via
    traj.xyz). Keeps JSON payload small.
    """
    d = _sys_dir(epoch, sys_idx)
    p = d / "data.npz"
    if not p.exists():
        raise HTTPException(status_code=404, detail="data.npz not found")
    try:
        npz = np.load(str(p))
        return {
            "n_steps": int(npz["energy"].shape[0]),
            "energy": npz["energy"].astype(float).tolist(),
            "fmax": npz["fmax"].astype(float).tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read data.npz: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
