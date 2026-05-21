"""Apply the official AdsorbGen unwrap/center geometry stage to an LMDB.

This is the reproducible version of the earlier ad-hoc unwrap conversion.  It
operates on already-preprocessed AdsorbGen LMDBs whose values are pickled dicts
with ``pos``, ``pos_relaxed``, ``cell``, ``tags``, ``fixed``, and
``atomic_numbers``.

Example:
    PYTHONPATH=AdsorbGen python -m adsorbgen.scripts.unwrap_preprocess \
        --src data/processed/is2res_train.lmdb \
        --dst data/processed/is2res_train_unwrap_centered.lmdb \
        --adsorbates-pkl data/pkls/adsorbates.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adsorbgen.data.pbc_unwrap import (  # noqa: E402
    PBC_XY,
    PBC_XYZ,
    load_adsorbate_reference_db,
    preprocess_entry_geometry,
    summarize_geometry_stats,
)


def _read_length(env: lmdb.Environment) -> int:
    with env.begin() as txn:
        raw = txn.get(b"length")
        if raw is not None:
            return int(pickle.loads(raw))
        return int(txn.stat()["entries"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Input preprocessed LMDB")
    p.add_argument("--dst", required=True, help="Output LMDB")
    p.add_argument("--adsorbates-pkl", default="data/pkls/adsorbates.pkl")
    p.add_argument("--center-mode", default="relaxed_all",
                   choices=["none", "pos_movable", "relaxed_movable", "pos_all",
                            "relaxed_all", "pos_ads", "relaxed_ads"])
    p.add_argument("--pbc-axes", choices=["xy", "xyz"], default="xy")
    p.add_argument("--max-samples", type=int, default=0, help="0 = all")
    p.add_argument("--map-size-gb", type=int, default=64)
    p.add_argument("--no-unwrap", action="store_true",
                   help="Only recenter, preserving existing periodic images.")
    args = p.parse_args()

    src = lmdb.open(args.src, subdir=False, readonly=True, lock=False, readahead=False)
    n_total = _read_length(src)
    n = n_total if args.max_samples <= 0 else min(n_total, args.max_samples)
    pbc = PBC_XYZ if args.pbc_axes == "xyz" else PBC_XY

    ref_db = None
    unwrap_adsorbate = not args.no_unwrap
    if unwrap_adsorbate:
        print(f"loading adsorbate reference graphs {args.adsorbates_pkl} ...", flush=True)
        ref_db = load_adsorbate_reference_db(args.adsorbates_pkl)
        print(f"adsorbate graphs: {len(ref_db)} entries", flush=True)

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    dst = lmdb.open(args.dst, subdir=False, map_size=args.map_size_gb * (1 << 30))

    stats_rows = []
    skipped = 0
    copied_meta_keys = (b"anomaly_mask",)
    with src.begin() as rtxn, dst.begin(write=True) as wtxn:
        for i in range(n):
            raw = rtxn.get(str(i).encode("ascii"))
            if raw is None:
                skipped += 1
                continue
            try:
                entry = pickle.loads(raw)
                entry, stats = preprocess_entry_geometry(
                    entry,
                    unwrap_adsorbate=unwrap_adsorbate,
                    ref_db=ref_db,
                    center_mode=args.center_mode,
                    pbc=pbc,
                )
            except Exception as exc:
                skipped += 1
                if skipped <= 5:
                    print(f"[warn] skipped {i}: {exc}", flush=True)
                continue
            out_i = len(stats_rows)
            wtxn.put(str(out_i).encode("ascii"), pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL))
            stats_rows.append(stats)
            if (out_i + 1) % 10000 == 0:
                print(f"  written {out_i + 1}/{n}", flush=True)

        wtxn.put(b"length", pickle.dumps(len(stats_rows)))
        for key in copied_meta_keys:
            value = rtxn.get(key)
            if value is not None and len(stats_rows) == n_total:
                wtxn.put(key, value)
            elif key == b"anomaly_mask":
                # If max-samples or skipped rows changed the length, keep the
                # schema valid but conservative.
                wtxn.put(key, pickle.dumps(np.zeros(len(stats_rows), dtype=np.int8)))

    src.close()
    dst.sync()
    dst.close()

    report = {
        "src": args.src,
        "dst": args.dst,
        "n_total": n_total,
        "requested": n,
        "written": len(stats_rows),
        "skipped": skipped,
        "unwrap_adsorbate": unwrap_adsorbate,
        "center_mode": args.center_mode,
        "pbc_axes": args.pbc_axes,
        **summarize_geometry_stats(stats_rows),
    }
    report_path = Path(args.dst).with_suffix(".report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"geometry report -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
