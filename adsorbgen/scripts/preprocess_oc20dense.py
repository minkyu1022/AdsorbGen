"""One-time preprocessing of OC20-Dense trajectories into a clean displacement LMDB.

Source tar layout (confirmed):
    trajs/{system_id}/{system_id}_{config_id}.traj
    trajs/{system_id}/{system_id}_surface.traj      <- skipped
Each .traj is an ASE Trajectory with multiple frames. We take traj[0] as
x_ref and traj[-1] as x_relax, and read the relaxed energy from traj[-1].

Per-system relative ΔE = E_config - min_{configs in same system} E_config, so
each system's best config has ΔE = 0. This is used as CFG conditioning at
finetune time; at inference we set ΔE = 0 to bias toward global minima.

Streaming: we read the tarball sequentially (`mode="r|gz"`) so we never hold
the whole archive in memory. Each .traj is written to a temporary file so
ASE's Trajectory reader can lazily seek to the first and last frames only.

Usage:
    PYTHONPATH=AdsorbGen python -m adsorbgen.scripts.preprocess_oc20dense \
        --src data/oc20dense/oc20_dense_trajectories.tar.gz \
        --tags-pkl data/oc20dense/oc20dense_tags.pkl \
        --dst-train data/processed/oc20dense_train.lmdb \
        --dst-val   data/processed/oc20dense_val.lmdb \
        --val-frac 0.2 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import sys
import tarfile
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Tuple

import lmdb
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _to_np(v, dtype):
    return np.asarray(v, dtype=dtype)


def _entry_from_traj(
    system_key: str,
    config_key: str,
    atoms_init,
    atoms_final,
    e_init: float,
    e_final: float,
    tags_override: Optional[np.ndarray] = None,
    ads_id: int = -1,
) -> dict:
    """Build a preprocessed entry dict from ASE Atoms objects.

    `tags_override` is required because the OC20-Dense Trajectory files do not
    always carry the canonical (bulk/surface/adsorbate) tags — we read those
    from `oc20dense_tags.pkl`.
    """
    pos_init = _to_np(atoms_init.get_positions(), np.float32)
    pos_relaxed = _to_np(atoms_final.get_positions(), np.float32)
    cell = _to_np(np.array(atoms_init.get_cell()), np.float32)
    atomic_numbers = _to_np(atoms_init.get_atomic_numbers(), np.int64)

    if tags_override is not None:
        tags = _to_np(tags_override, np.int64)
    else:
        tags = _to_np(atoms_init.get_tags(), np.int64)

    fixed = np.zeros(len(atoms_init), dtype=np.int64)
    for c in getattr(atoms_init, "constraints", []) or []:
        idx = getattr(c, "index", None)
        if idx is not None:
            fixed[np.asarray(idx, dtype=int)] = 1

    movable = ((tags == 1) | (tags == 2)) & (fixed == 0)
    preprocess_shift = np.zeros((3,), dtype=np.float32)
    if movable.any():
        center = pos_init[movable].mean(axis=0, keepdims=True).astype(np.float32)
        preprocess_shift = (-center).reshape(3).astype(np.float32)
        pos_init = pos_init - center
        pos_relaxed = pos_relaxed - center

    return {
        "pos": pos_init.astype(np.float32),
        "pos_relaxed": pos_relaxed.astype(np.float32),
        "cell": cell.astype(np.float32),
        "tags": tags.astype(np.int64),
        "fixed": fixed.astype(np.int64),
        "atomic_numbers": atomic_numbers.astype(np.int64),
        "sid": -1,
        "ads_id": int(ads_id),
        "y_init": float(e_init),
        "y_relaxed": float(e_final),
        "delta_e": 0.0,  # filled later (per-system relative)
        "system_key": system_key,
        "config_key": config_key,
        "anomaly": 0,  # OC20-Dense is curated; no per-sample OC20 anomaly label
        "preprocess_shift": preprocess_shift,
        "preprocess_shift_mode": "pos_movable",
    }


_TRAJ_RE = re.compile(r"^trajs/([^/]+)/([^/]+)\.traj$")


def _iter_trajectories(
    src_tar: Path,
    tags_map: Optional[dict] = None,
    log_every: int = 500,
) -> Iterable[Tuple[str, str, "object", "object", float, float]]:
    """Stream (system_id, config_id, atoms_init, atoms_final, e_init, e_final) tuples.

    Implementation notes:
        - Tarball layout: ``trajs/{sid}/{sid}_{config}.traj`` (gzipped).
        - Skip ``*_surface.traj`` (clean-slab references, not adsorbate configs).
        - Stream the tarball sequentially with ``mode="r|gz"`` so memory stays
          bounded to one trajectory at a time.
        - Extract each member to a NamedTemporaryFile and use
          ``ase.io.trajectory.Trajectory`` for lazy indexing (we only touch
          frames 0 and -1, avoiding decoding intermediate frames).
    """
    from ase.io.trajectory import Trajectory  # local import: ASE is heavyweight

    t0 = time.time()
    n_seen = 0
    n_yielded = 0
    n_skipped_surface = 0
    n_errors = 0
    with tarfile.open(str(src_tar), mode="r|gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".traj"):
                continue
            n_seen += 1
            m = _TRAJ_RE.match(member.name)
            if m is None:
                continue
            sid, stem = m.group(1), m.group(2)
            prefix = f"{sid}_"
            if not stem.startswith(prefix):
                continue
            config_id = stem[len(prefix):]
            if config_id == "surface":
                n_skipped_surface += 1
                continue

            f = tf.extractfile(member)
            if f is None:
                continue
            data = f.read()
            tmp = tempfile.NamedTemporaryFile(suffix=".traj", delete=False)
            try:
                tmp.write(data)
                tmp.flush()
                tmp.close()
                try:
                    traj = Trajectory(tmp.name, "r")
                    if len(traj) < 2:
                        continue
                    atoms_init = traj[0]
                    atoms_final = traj[-1]
                    try:
                        e_init = float(atoms_init.get_potential_energy())
                    except Exception:
                        e_init = 0.0
                    e_final = float(atoms_final.get_potential_energy())
                except Exception as exc:
                    n_errors += 1
                    if n_errors <= 5:
                        print(f"[warn] failed to read {member.name}: {exc}", flush=True)
                    continue
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass

            yield sid, config_id, atoms_init, atoms_final, e_init, e_final
            n_yielded += 1

            if n_yielded % log_every == 0:
                dt = time.time() - t0
                rate = n_yielded / max(dt, 1e-6)
                print(
                    f"[iter] seen={n_seen} yielded={n_yielded} skipped_surface={n_skipped_surface} "
                    f"errors={n_errors} rate={rate:.1f}/s elapsed={dt:.0f}s",
                    flush=True,
                )

    print(
        f"[iter] done: seen={n_seen} yielded={n_yielded} skipped_surface={n_skipped_surface} "
        f"errors={n_errors} elapsed={time.time() - t0:.0f}s",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="OC20-Dense tar.gz path")
    p.add_argument("--tags-pkl", type=str, default=None, help="oc20dense_tags.pkl (overrides traj tags)")
    p.add_argument("--mapping-pkl", type=str, default=None,
                   help="oc20dense_mapping.pkl; required to stamp ads_id per entry")
    p.add_argument("--adsorbates-pkl", type=str, default=None,
                   help="adsorbates.pkl; required to map SMILES -> ads_id")
    p.add_argument("--dst-train", required=True)
    p.add_argument("--dst-val", required=True)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--map-size-gb", type=int, default=16)
    p.add_argument("--max-systems", type=int, default=None, help="debug: stop after N systems")
    args = p.parse_args()

    src_tar = Path(args.src)
    assert src_tar.exists(), f"missing {src_tar}"

    tags_map = None
    if args.tags_pkl:
        with open(args.tags_pkl, "rb") as f:
            tags_map = pickle.load(f)
        print(f"[tags] loaded {len(tags_map)} entries from {args.tags_pkl}", flush=True)

    smiles_to_ads_id: dict[str, int] = {}
    inv_mapping: dict[Tuple[str, str], dict] = {}
    if args.adsorbates_pkl and args.mapping_pkl:
        with open(args.adsorbates_pkl, "rb") as f:
            ads_pkl = pickle.load(f)
        for k, v in ads_pkl.items():
            smiles_to_ads_id[v[1]] = int(k)
        print(f"[ads] loaded {len(smiles_to_ads_id)} canonical adsorbates", flush=True)
        with open(args.mapping_pkl, "rb") as f:
            dense_map = pickle.load(f)
        for v in dense_map.values():
            inv_mapping[(v["system_id"], v["config_id"])] = v
        print(f"[map] loaded {len(inv_mapping)} (system,config) entries", flush=True)
    else:
        print("[warn] --mapping-pkl/--adsorbates-pkl not given; ads_id will be -1", flush=True)

    n_ads_unmatched = 0
    by_system: dict[str, list[dict]] = defaultdict(list)
    for sid, cid, ai, af, ei, ef in _iter_trajectories(src_tar, tags_map=tags_map):
        override = tags_map.get(sid) if tags_map is not None else None
        ads_id = -1
        meta = inv_mapping.get((sid, cid))
        if meta is not None:
            smiles = meta.get("adsorbate")
            ads_id = smiles_to_ads_id.get(smiles, -1)
            if ads_id < 0:
                n_ads_unmatched += 1
        try:
            entry = _entry_from_traj(sid, cid, ai, af, ei, ef,
                                     tags_override=override, ads_id=ads_id)
        except Exception as exc:
            print(f"[warn] entry build failed for {sid}/{cid}: {exc}", flush=True)
            continue
        if args.max_systems is not None and sid not in by_system and len(by_system) >= args.max_systems:
            break  # tarball is grouped by sid, so we're past the cutoff
        by_system[sid].append(entry)
    print(f"loaded {sum(len(v) for v in by_system.values())} configs across {len(by_system)} systems", flush=True)
    if smiles_to_ads_id:
        print(f"[ads] unmatched SMILES: {n_ads_unmatched}", flush=True)

    # Per-system relative ΔE.
    for sys_key, configs in by_system.items():
        e_min = min(c["y_relaxed"] for c in configs)
        for c in configs:
            c["delta_e"] = float(c["y_relaxed"] - e_min)

    rng = random.Random(args.seed)
    system_keys = sorted(by_system.keys())
    rng.shuffle(system_keys)
    n_val = max(1, int(round(len(system_keys) * args.val_frac)))
    val_keys = set(system_keys[:n_val])
    train_keys = set(system_keys[n_val:])
    print(f"split: train_systems={len(train_keys)} val_systems={len(val_keys)}", flush=True)

    def _write(entries: list[dict], dst_path: str):
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        env = lmdb.open(dst_path, subdir=False, map_size=args.map_size_gb * (1 << 30))
        with env.begin(write=True) as txn:
            for i, e in enumerate(entries):
                txn.put(str(i).encode("ascii"), pickle.dumps(e, protocol=pickle.HIGHEST_PROTOCOL))
            txn.put(b"length", pickle.dumps(len(entries)))
            mask = np.zeros(len(entries), dtype=np.int8)
            txn.put(b"anomaly_mask", pickle.dumps(mask, protocol=pickle.HIGHEST_PROTOCOL))
        env.sync()
        env.close()

    train_entries = [c for k in train_keys for c in by_system[k]]
    val_entries = [c for k in val_keys for c in by_system[k]]
    rng.shuffle(train_entries)
    rng.shuffle(val_entries)
    _write(train_entries, args.dst_train)
    _write(val_entries, args.dst_val)
    print(f"wrote {len(train_entries)} train / {len(val_entries)} val", flush=True)

    side = {
        "src": str(src_tar),
        "val_frac": args.val_frac,
        "seed": args.seed,
        "n_systems": len(by_system),
        "n_train_systems": len(train_keys),
        "n_val_systems": len(val_keys),
        "train_systems": sorted(train_keys),
        "val_systems": sorted(val_keys),
    }
    with open(Path(args.dst_train).with_suffix(".split.json"), "w") as f:
        json.dump(side, f, indent=2)


if __name__ == "__main__":
    main()
