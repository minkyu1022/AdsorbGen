"""Extract a compact ``system_id -> E_slab`` cache from the phase2 slab results.

The upstream ``results/phase2/slab_results.pkl`` is ~11 GB because it carries
full ASE ``Atoms`` objects with fairchem calculator state. For evaluation we
only need the scalar ``E_slab`` per OC20-Dense system, keyed by
``system_id``. This script does a one-time pass to build a tiny JSON cache.

The slab-results key is ``(bulk_mpid, miller_index, shift, top)``. OC20-Dense
exposes those four fields per ``system_id`` via ``oc20dense_mapping.pkl``, so
we join on the tuple.

Run this with an environment that can import fairchem (slab_results.pkl
references ``fairchem.*`` classes inside Atoms objects). The resulting JSON
is portable.

Usage:
    /home/minkyu/micromamba/envs/catbench/bin/python \
        -m adsorbgen.scripts.build_slab_energy_cache \
        --slab-pkl results/phase2/slab_results.pkl \
        --mapping-pkl data/oc20dense/oc20dense_mapping.pkl \
        --out data/oc20dense/slab_energy_cache.json
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _canon_miller(mi) -> tuple:
    return tuple(int(x) for x in mi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slab-pkl", required=True)
    p.add_argument("--mapping-pkl", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shift-digits", type=int, default=3, help="round shift to N digits for matching")
    args = p.parse_args()

    with open(args.mapping_pkl, "rb") as f:
        mapping = pickle.load(f)

    sys_to_key: dict[str, tuple] = {}
    for entry in mapping.values():
        sid = entry["system_id"]
        if sid in sys_to_key:
            continue
        mpid = entry["mpid"]
        miller = _canon_miller(entry["miller_idx"])
        shift = round(float(entry["shift"]), args.shift_digits)
        top = bool(entry["top"])
        sys_to_key[sid] = (mpid, miller, shift, top)
    print(f"[mapping] {len(sys_to_key)} unique system_ids", flush=True)

    print(f"[slab] loading {args.slab_pkl} (this takes a while)...", flush=True)
    with open(args.slab_pkl, "rb") as f:
        slab = pickle.load(f)
    print(f"[slab] {len(slab)} slab entries loaded", flush=True)

    slab_lookup: dict[tuple, dict] = {}
    for key, val in slab.items():
        mpid, miller, shift, top = key
        miller = _canon_miller(miller)
        shift_r = round(float(shift), args.shift_digits)
        slab_lookup[(mpid, miller, shift_r, bool(top))] = val

    cache = {}
    n_match = 0
    n_miss = 0
    for sid, key in sys_to_key.items():
        rec = slab_lookup.get(key)
        if rec is None:
            n_miss += 1
            continue
        e_slab = rec.get("E_slab")
        if e_slab is None or (isinstance(e_slab, float) and math.isnan(e_slab)):
            n_miss += 1
            continue
        cache[sid] = {
            "E_slab": float(e_slab),
            "n_atoms": int(rec.get("n_atoms", 0)),
            "converged": bool(rec.get("converged", False)),
            "bulk_mpid": str(key[0]),
            "miller_index": list(key[1]),
            "shift": float(key[2]),
            "top": bool(key[3]),
        }
        n_match += 1
    print(f"[cache] matched={n_match} missing={n_miss}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"[cache] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
