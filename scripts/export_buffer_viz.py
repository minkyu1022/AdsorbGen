#!/usr/bin/env python
"""Export ReplayStream buffer entries to a viz epoch as final-structure-only.

Unlike ``replay_viz_for_sids.py`` (which re-runs flow+UMA and is therefore
stochastic / not guaranteed to reproduce the saved sample), this reads the
*exact* relaxed structures already stored in the ReplayStream chunk pkls and
renders one ``sys_*/`` folder per buffer entry. No GPU, no re-run.

Only ``x1_relaxed.pdb`` + ``meta.json`` are written — x0 / x1_flow / trajectory
are not stored in a ReplayEntry, so they are intentionally absent.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if (_REPO / "adsorbgen").is_dir():
    sys.path.insert(0, str(_REPO))

from adsorbgen.replay_viz import save_structure_pdb  # noqa: E402


def _get(entry, key):
    return entry[key] if isinstance(entry, dict) else getattr(entry, key)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stream-dir", required=True,
                   help="ReplayStream root containing shard_*/chunk_*.pkl")
    p.add_argument("--viz-root", required=True,
                   help="replay_viz dir; ep{TAG}/ is created/overwritten")
    p.add_argument("--epoch-tag", type=int, default=2)
    args = p.parse_args()

    stream = Path(args.stream_dir)
    chunks = sorted(stream.glob("shard_*/chunk_*.pkl"))
    entries = []
    for c in chunks:
        with open(c, "rb") as f:
            entries.extend(pickle.load(f))
    print(f"[export] {len(entries)} buffer entries from {len(chunks)} chunk file(s)")

    ep_dir = Path(args.viz_root) / f"ep{args.epoch_tag}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    systems = []
    for g, e in enumerate(entries):
        sd = ep_dir / f"sys_buf_{g:04d}"
        sd.mkdir(exist_ok=True)
        numbers = _get(e, "atomic_numbers")
        save_structure_pdb(
            numbers, _get(e, "pos_relaxed"), _get(e, "cell"), _get(e, "tags"),
            sd / "x1_relaxed.pdb",
        )
        meta = {
            "global_idx": g,
            "sid": int(_get(e, "sid")),
            "ads_id": int(_get(e, "ads_id")),
            "n_atoms": int(len(numbers)),
            "n_steps": 0,
            "E_pred": float(_get(e, "E_sys_pred")),
            "E_gt": float(_get(e, "E_sys_gt")),
            "improvement": float(_get(e, "improvement")),
            "fmax_final": 0.0,
            "converged": True,
            "status": "ok",
            "success": True,
            "sys_dir_name": sd.name,
            "buffer_export": True,   # exact saved sample, not a re-run
        }
        (sd / "meta.json").write_text(json.dumps(meta, indent=2))
        systems.append(meta)

    index = {"epoch_dir": ep_dir.name, "n_systems": len(systems), "systems": systems}
    (ep_dir / "_index.json").write_text(json.dumps(index, indent=2))
    print(f"[export] wrote {len(systems)} systems → {ep_dir}")


if __name__ == "__main__":
    main()
