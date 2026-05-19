#!/usr/bin/env python
"""Prepare task-consistent gas/slab energy caches for replay.

Writes:
  E_gas_only.pkl  : sid -> gas reference energy from oc20 atomic potentials
  E_slab_only.pkl : sid -> pristine slab energy from is2res pristine cache

Both are keyed by sid because downstream replay/adsorption-energy checks are
sid-centric. The records include the source key/adsorbate string for auditing.
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path


_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def _composition_from_adsorbate(label: str) -> dict[str, int]:
    """Parse OC20 adsorbate labels like '*CH2' or '*N*NH' into atom counts."""
    clean = label.replace("*", "")
    counts: dict[str, int] = {}
    for elem, num in _TOKEN_RE.findall(clean):
        counts[elem] = counts.get(elem, 0) + (int(num) if num else 1)
    if not counts:
        raise ValueError(f"could not parse adsorbate label: {label!r}")
    return counts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gt-index", default="/home/irteam/data/replay/gt_index_by_sid.pkl")
    p.add_argument("--gas-refs", default="/home/irteam/data/pkls/gas_phase_refs_oc20.pkl")
    p.add_argument("--pristine-slabs", default="/home/irteam/results/pristine_slabs/is2res.pkl")
    p.add_argument("--pristine-index", default="/home/irteam/results/pristine_slabs/is2res.sid_index.pkl")
    p.add_argument("--out-dir", default="/home/irteam/data/replay")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.gt_index, "rb") as f:
        gt_index = pickle.load(f)
    with open(args.gas_refs, "rb") as f:
        gas_refs = pickle.load(f)
    with open(args.pristine_slabs, "rb") as f:
        slabs = pickle.load(f)
    with open(args.pristine_index, "rb") as f:
        sid_to_slab_key = pickle.load(f)

    atomic_potentials = gas_refs["atomic_potentials"]

    gas_by_sid = {}
    slab_by_sid = {}
    missing_slab = 0
    missing_gas = 0

    for sid, info in gt_index.items():
        if not isinstance(info, dict) or info.get("system_key") is None:
            continue

        system_key = tuple(info["system_key"])
        adsorbate = str(system_key[-1])
        try:
            comp = _composition_from_adsorbate(adsorbate)
            e_gas = sum(float(atomic_potentials[el]) * n for el, n in comp.items())
        except Exception:
            missing_gas += 1
        else:
            gas_by_sid[int(sid)] = {
                "e_total": float(e_gas),
                "adsorbate": adsorbate,
                "composition": comp,
                "model": gas_refs.get("model"),
                "task": gas_refs.get("task"),
            }

        slab_key = sid_to_slab_key.get(int(sid))
        slab_rec = slabs.get(slab_key) if slab_key is not None else None
        if slab_rec is None or slab_rec.get("e_total") is None:
            missing_slab += 1
            continue
        slab_by_sid[int(sid)] = {
            "e_total": float(slab_rec["e_total"]),
            "slab_key": slab_key,
            "converged": bool(slab_rec.get("converged", False)),
            "fmax": float(slab_rec.get("forces_max", float("nan"))),
            "n_steps": int(slab_rec.get("n_steps", 0) or 0),
            "n_atoms": int(slab_rec.get("n_atoms", 0) or 0),
        }

    gas_path = out_dir / "E_gas_only.pkl"
    slab_path = out_dir / "E_slab_only.pkl"
    with open(gas_path, "wb") as f:
        pickle.dump(gas_by_sid, f)
    with open(slab_path, "wb") as f:
        pickle.dump(slab_by_sid, f)

    print(f"Wrote {len(gas_by_sid)} gas records -> {gas_path}")
    print(f"Wrote {len(slab_by_sid)} slab records -> {slab_path}")
    print(f"Missing gas={missing_gas}, missing slab={missing_slab}")


if __name__ == "__main__":
    main()
