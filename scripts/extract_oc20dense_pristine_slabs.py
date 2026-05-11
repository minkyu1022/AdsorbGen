#!/usr/bin/env python
"""Extract OC20-Dense clean-surface references from the trajectory tarball.

The OC20-Dense tar contains one clean-surface trajectory per dense system:

    trajs/{system_key}/{system_key}_surface.traj

This script reads the final frame of each surface trajectory and writes a small
pristine slab DB compatible with ``adsorbgen.eval.load_pristine_context``:

    results/pristine_slabs/oc20dense.pkl
    results/pristine_slabs/oc20dense.system_index.pkl

The DB is keyed by slab keys derived from ``oc20dense_mapping.pkl`` when
available, while the system index maps each Dense ``system_key`` to that slab
key. If mapping is unavailable, both keys fall back to the system_key string.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np


SURFACE_RE = re.compile(r"^trajs/([^/]+)/\1_surface\.traj$")


def _load_system_to_slab_key(mapping_pkl: Path | None) -> dict[str, object]:
    if mapping_pkl is None or not mapping_pkl.exists():
        return {}
    with open(mapping_pkl, "rb") as f:
        mapping = pickle.load(f)
    out: dict[str, object] = {}
    for rec in mapping.values():
        sys_key = str(rec["system_id"])
        if sys_key in out:
            continue
        out[sys_key] = (
            rec.get("mpid"),
            tuple(int(x) for x in rec.get("miller_idx", ())),
            float(rec.get("shift")),
            bool(rec.get("top")),
        )
    return out


def _read_final_atoms_from_member(tf: tarfile.TarFile, member):
    from ase.io.trajectory import Trajectory

    f = tf.extractfile(member)
    if f is None:
        raise RuntimeError(f"failed to extract {member.name}")
    data = f.read()
    tmp = tempfile.NamedTemporaryFile(suffix=".traj", delete=False)
    try:
        tmp.write(data)
        tmp.flush()
        tmp.close()
        traj = Trajectory(tmp.name, "r")
        if len(traj) == 0:
            raise RuntimeError(f"empty trajectory: {member.name}")
        return traj[-1]
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path,
                   default=Path("data/oc20dense/oc20_dense_trajectories.tar.gz"))
    p.add_argument("--mapping-pkl", type=Path,
                   default=Path("data/oc20dense/oc20dense_mapping.pkl"))
    p.add_argument("--out", type=Path,
                   default=Path("results/pristine_slabs/oc20dense.pkl"))
    p.add_argument("--system-index-out", type=Path, default=None)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--max-surfaces", type=int, default=None,
                   help="debug cap; omit for full extraction")
    args = p.parse_args()

    if args.system_index_out is None:
        args.system_index_out = args.out.with_suffix(".system_index.pkl")

    assert args.src.exists(), f"missing tarball: {args.src}"
    system_to_slab = _load_system_to_slab_key(args.mapping_pkl)
    print(f"[mapping] systems with slab keys: {len(system_to_slab)}", flush=True)

    db: dict[object, dict] = {}
    system_index: dict[str, object] = {}
    n_seen = n_surface = n_errors = 0
    t0 = time.time()

    with tarfile.open(args.src, mode="r|gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith("_surface.traj"):
                continue
            n_seen += 1
            m = SURFACE_RE.match(member.name)
            if m is None:
                continue
            system_key = m.group(1)
            slab_key = system_to_slab.get(system_key, system_key)
            try:
                atoms = _read_final_atoms_from_member(tf, member)
            except Exception as exc:
                n_errors += 1
                print(f"[warn] failed {member.name}: {type(exc).__name__}: {exc}", flush=True)
                continue

            if slab_key not in db:
                db[slab_key] = {
                    "system_key": system_key,
                    "slab_key": slab_key,
                    "pos": np.asarray(atoms.get_positions(), dtype=np.float64),
                    "atomic_numbers": np.asarray(atoms.get_atomic_numbers(), dtype=np.int64),
                    "cell": np.asarray(atoms.get_cell(), dtype=np.float64),
                    "pbc": np.asarray(atoms.get_pbc(), dtype=bool),
                }
            system_index[system_key] = slab_key
            n_surface += 1
            if args.max_surfaces is not None and n_surface >= args.max_surfaces:
                break

            if n_surface % max(args.log_every, 1) == 0:
                dt = time.time() - t0
                print(
                    f"[extract] surfaces={n_surface} unique_slabs={len(db)} "
                    f"errors={n_errors} elapsed={dt:.1f}s",
                    flush=True,
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(args.system_index_out, "wb") as f:
        pickle.dump(system_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("[done]", flush=True)
    print(f"  surfaces read: {n_surface}", flush=True)
    print(f"  unique slabs:  {len(db)}", flush=True)
    print(f"  errors:        {n_errors}", flush=True)
    print(f"  db:            {args.out}", flush=True)
    print(f"  system index:  {args.system_index_out}", flush=True)


if __name__ == "__main__":
    main()
