"""Inspect the first N members of the OC20-Dense trajectories tarball.

Safe to run on a partially-downloaded .tar.gz — we stream members sequentially
and stop after collecting a few samples. Prints member names, sizes, and for
.traj files, uses ase.io.trajectory.Trajectory lazy access to read traj[0] and
traj[-1] from each.

Usage:
    PYTHONPATH=AdsorbGen python -m adsorbgen.scripts.inspect_oc20dense_tar \
        --src data/oc20dense/oc20_dense_trajectories.tar.gz --n 6
"""

from __future__ import annotations

import argparse
import tarfile
import tempfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--n", type=int, default=6)
    args = p.parse_args()

    names_seen = 0
    trajs_seen = 0
    with tarfile.open(args.src, mode="r|gz") as tf:
        for member in tf:
            if names_seen < args.n * 3:
                print(f"{member.size:>10}  {member.name}", flush=True)
                names_seen += 1
            if member.isfile() and member.name.endswith(".traj"):
                if trajs_seen < args.n:
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    data = f.read()
                    with tempfile.NamedTemporaryFile(suffix=".traj", delete=True) as tmp:
                        tmp.write(data)
                        tmp.flush()
                        from ase.io.trajectory import Trajectory
                        traj = Trajectory(tmp.name, "r")
                        n_frames = len(traj)
                        first = traj[0]
                        last = traj[-1]
                        try:
                            e_last = last.get_potential_energy()
                        except Exception as exc:  # pragma: no cover - diagnostic
                            e_last = f"<err:{exc}>"
                        print(
                            f"  -> {member.name}: frames={n_frames} n_atoms={len(first)} "
                            f"e_last={e_last} has_constraints={bool(first.constraints)}",
                            flush=True,
                        )
                    trajs_seen += 1
            if trajs_seen >= args.n:
                break

    print(f"done. trajs_seen={trajs_seen}")


if __name__ == "__main__":
    main()
