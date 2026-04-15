"""One-time preprocessing of OC20 IS2RES LMDB into a clean displacement LMDB.

The raw OC20 LMDB contains pickled legacy `torch_geometric.data.Data` objects
which break under modern torch_geometric. We read them once via `__dict__`,
extract the fields we need, center the movable atoms' centroid at the origin,
and write a clean numpy-only LMDB consumable by
`adsorbgen.dataset.PreprocessedDisplacementDataset` with `unconditional=True`.

Usage:
    PYTHONPATH=AdsorbGen python -m adsorbgen.scripts.preprocess_is2res \
        --src data/oc20/train/data.lmdb \
        --dst data/processed/is2res_train.lmdb \
        --max-samples 0
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _to_np(v, dtype):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    return np.asarray(v, dtype=dtype)


def _extract(data_obj, metadata: dict | None = None) -> dict:
    d = data_obj.__dict__
    pos = _to_np(d["pos"], np.float32)
    pos_relaxed = _to_np(d["pos_relaxed"], np.float32)
    cell = _to_np(d["cell"], np.float32)
    if cell.ndim == 3:
        cell = cell[0]
    tags = _to_np(d["tags"], np.int64)
    fixed = _to_np(d["fixed"], np.int64)
    atomic_numbers = _to_np(d["atomic_numbers"], np.int64)
    sid = int(d.get("sid", -1))
    y_init = float(d.get("y_init", 0.0) or 0.0)
    y_relaxed = float(d.get("y_relaxed", 0.0) or 0.0)

    movable = ((tags == 1) | (tags == 2)) & (fixed == 0)
    if movable.any():
        center = pos[movable].mean(axis=0, keepdims=True).astype(np.float32)
        pos = pos - center
        pos_relaxed = pos_relaxed - center

    anomaly = 0
    ads_id = -1
    if metadata is not None and sid >= 0:
        meta = metadata.get(f"random{sid}", {})
        anomaly = int(meta.get("anomaly", 0))
        ads_id = int(meta.get("ads_id", -1))

    return {
        "pos": pos.astype(np.float32),
        "pos_relaxed": pos_relaxed.astype(np.float32),
        "cell": cell.astype(np.float32),
        "tags": tags.astype(np.int64),
        "fixed": fixed.astype(np.int64),
        "atomic_numbers": atomic_numbers.astype(np.int64),
        "sid": sid,
        "ads_id": ads_id,
        "y_init": y_init,
        "y_relaxed": y_relaxed,
        "anomaly": anomaly,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Raw OC20 IS2RES LMDB")
    p.add_argument("--dst", required=True, help="Output preprocessed LMDB")
    p.add_argument("--metadata", type=str, default=None,
                   help="oc20_metadata.pkl; if given, stamp anomaly label (0-4) per entry and write anomaly_mask key")
    p.add_argument("--max-samples", type=int, default=0, help="0 = all")
    p.add_argument("--map-size-gb", type=int, default=64)
    args = p.parse_args()

    metadata = None
    if args.metadata:
        print(f"loading metadata {args.metadata} ...", flush=True)
        with open(args.metadata, "rb") as f:
            metadata = pickle.load(f)
        print(f"metadata: {len(metadata)} entries", flush=True)

    src = lmdb.open(args.src, subdir=False, readonly=True, lock=False)
    with src.begin() as txn:
        n_total = txn.stat()["entries"]
        raw = txn.get(b"length")
        if raw is not None:
            try:
                n_total = int(pickle.loads(raw))
            except Exception:
                pass

    n = n_total if args.max_samples <= 0 else min(n_total, args.max_samples)
    print(f"src={args.src}  total={n_total}  processing={n}", flush=True)

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    dst = lmdb.open(args.dst, subdir=False, map_size=args.map_size_gb * (1 << 30))

    written = 0
    skipped = 0
    anomaly_list: list[int] = []
    with src.begin() as rtxn, dst.begin(write=True) as wtxn:
        for i in range(n):
            raw = rtxn.get(str(i).encode("ascii"))
            if raw is None:
                skipped += 1
                continue
            try:
                obj = pickle.loads(raw)
                entry = _extract(obj, metadata=metadata)
            except Exception as e:
                skipped += 1
                if skipped < 5:
                    print(f"  skipped {i}: {e}", flush=True)
                continue
            wtxn.put(str(written).encode("ascii"), pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL))
            anomaly_list.append(int(entry["anomaly"]))
            written += 1
            if (written + 1) % 1000 == 0:
                print(f"  written {written}/{n}", flush=True)
        wtxn.put(b"length", pickle.dumps(written))
        mask = np.asarray(anomaly_list, dtype=np.int8)
        wtxn.put(b"anomaly_mask", pickle.dumps(mask, protocol=pickle.HIGHEST_PROTOCOL))

    src.close()
    dst.sync()
    dst.close()
    n_anom = int((mask != 0).sum()) if len(anomaly_list) else 0
    print(f"done. written={written} skipped={skipped} anomaly_flagged={n_anom} ({n_anom / max(written, 1) * 100:.1f}%) -> {args.dst}", flush=True)


if __name__ == "__main__":
    main()
