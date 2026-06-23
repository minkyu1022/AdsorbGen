#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def symlink_force(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def copytree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    ignore = shutil.ignore_patterns(
        "__pycache__",
        "*.pyc",
        ".git",
        "wandb",
        "runs",
        "data",
        "data-vol1",
    )
    shutil.copytree(src, dst, ignore=ignore)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_tar(stage: Path, tar_path: Path) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    if tar_path.exists():
        tar_path.unlink()
    if shutil.which("zstd"):
        cmd = ["tar", "-I", "zstd -T0 -3", "-cf", str(tar_path), "-C", str(stage), "."]
    else:
        if not str(tar_path).endswith(".tar.gz"):
            tar_path = tar_path.with_suffix(".tar.gz")
        cmd = ["tar", "-czf", str(tar_path), "-C", str(stage), "."]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-dir", required=True)
    ap.add_argument("--package-root", required=True)
    ap.add_argument("--adsorbates-pkl", default="/home1/irteam/data-vol1/minkyu/data/pkls/adsorbates.pkl")
    ap.add_argument("--repo", default="/home1/irteam/AdsorbGen")
    ap.add_argument("--fast-dynamics", default="/home1/irteam/fast_dynamics")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-tar", action="store_true")
    args = ap.parse_args()

    full_dir = Path(args.full_dir).resolve()
    flow_jobs = full_dir / "flow_jobs"
    package_root = Path(args.package_root).resolve()
    stage = package_root / "remote_half_package"
    local_jobs = full_dir / "local_half_flow_jobs"
    remote_jobs = stage / "flow_jobs_remote"
    code_root = stage / "code"
    assets = stage / "assets"

    jobs = sorted(flow_jobs.glob("jobs_*.pkl"))
    if not jobs:
        raise FileNotFoundError(f"no jobs_*.pkl in {flow_jobs}")

    if package_root.exists() and args.force:
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True, exist_ok=True)
    if stage.exists():
        shutil.rmtree(stage)
    if local_jobs.exists() or local_jobs.is_symlink():
        if local_jobs.is_symlink():
            local_jobs.unlink()
        elif args.force:
            shutil.rmtree(local_jobs)
        else:
            raise FileExistsError(local_jobs)

    remote_jobs.mkdir(parents=True, exist_ok=True)
    local_jobs.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    code_root.mkdir(parents=True, exist_ok=True)

    local = []
    remote = []
    for i, p in enumerate(jobs):
        target = local if i % 2 == 0 else remote
        target.append(p)

    for p in local:
        symlink_force(p, local_jobs / p.name)
        meta = p.with_suffix(".json")
        if meta.exists():
            symlink_force(meta, local_jobs / meta.name)

    for p in remote:
        link_or_copy(p, remote_jobs / p.name)
        meta = p.with_suffix(".json")
        if meta.exists():
            link_or_copy(meta, remote_jobs / meta.name)

    repo = Path(args.repo).resolve()
    copytree_clean(repo / "geoopt", code_root / "AdsorbGen" / "geoopt")
    copytree_clean(repo / "adsorbgen", code_root / "AdsorbGen" / "adsorbgen")
    fd = Path(args.fast_dynamics).resolve()
    if (fd / "fast_dynamics").exists():
        copytree_clean(fd / "fast_dynamics", code_root / "fast_dynamics" / "fast_dynamics")
    shutil.copy2(args.adsorbates_pkl, assets / "adsorbates.pkl")

    remote_out_rel = "run_remote_half"
    local_out = full_dir / "local_half_relax_run"
    common_relax = (
        "--gpus ${GPUS} "
        "--uma-model uma-s-1p2 --uma-task oc20 "
        "--fmax 0.05 --max-steps 300 --max-atoms 32768 --maxstep 0.2 "
        "--lbfgs-memory 100 --lbfgs-damping 1.0 --lbfgs-alpha 70.0 "
        "--lbfgs-streaming --lbfgs-check-interval 10 --save-result-pkl"
    )

    write_text(
        stage / "run_remote_relax.sh",
        f"""#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PY="${{PYTHON_BIN:-python}}"
GPUS="${{GPUS:-0 1 2 3 4 5 6 7}}"
export PYTHONPATH="${{ROOT}}/code/AdsorbGen:${{ROOT}}/code/AdsorbGen/geoopt:${{ROOT}}/code/fast_dynamics:${{PYTHONPATH:-}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export ADSORBATES_PKL="${{ROOT}}/assets/adsorbates.pkl"
REMOTE_OUT="${{ROOT}}/{remote_out_rel}"
mkdir -p "${{REMOTE_OUT}}" "${{REMOTE_OUT}}/logs"
ADSORBGEN_ALLOW_RELAX=1 "${{PY}}" "${{ROOT}}/code/AdsorbGen/geoopt/two_stage_full_replay.py" relax \\
  --repo "${{ROOT}}/code/AdsorbGen" \\
  --adsorbates-pkl "${{ROOT}}/assets/adsorbates.pkl" \\
  --out-dir "${{REMOTE_OUT}}" \\
  --jobs-dir "${{ROOT}}/flow_jobs_remote" \\
  {common_relax} 2>&1 | tee "${{REMOTE_OUT}}/logs/remote_relax.log"
""",
    )
    os.chmod(stage / "run_remote_relax.sh", 0o755)

    write_text(
        package_root / "run_local_relax.sh",
        f"""#!/usr/bin/env bash
set -euo pipefail
PY="${{PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}}"
GPUS="${{GPUS:-0 1 2 3 4 5 6 7}}"
export PYTHONPATH="/home1/irteam/AdsorbGen:/home1/irteam/AdsorbGen/geoopt:/home1/irteam/fast_dynamics:${{PYTHONPATH:-}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
mkdir -p "{local_out}" "{local_out}/logs"
ADSORBGEN_ALLOW_RELAX=1 "${{PY}}" "/home1/irteam/AdsorbGen/geoopt/two_stage_full_replay.py" relax \\
  --repo "/home1/irteam/AdsorbGen" \\
  --adsorbates-pkl "{args.adsorbates_pkl}" \\
  --out-dir "{local_out}" \\
  --jobs-dir "{local_jobs}" \\
  {common_relax} 2>&1 | tee "{local_out}/logs/local_relax.log"
""",
    )
    os.chmod(package_root / "run_local_relax.sh", 0o755)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "full_dir": str(full_dir),
        "split": "sorted jobs_*.pkl, even indices local, odd indices remote",
        "total_job_shards": len(jobs),
        "local_job_shards": len(local),
        "remote_job_shards": len(remote),
        "local_jobs_dir": str(local_jobs),
        "local_out_dir": str(local_out),
        "remote_package_stage": str(stage),
        "remote_jobs_dir_inside_package": "flow_jobs_remote",
        "remote_out_dir_inside_package": remote_out_rel,
        "paper_style_passk_note": "For scoring later: n=all candidates, c=valid and final_E_sys - global_min <= 0.1 eV.",
    }
    write_text(stage / "MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True))
    write_text(package_root / "MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True))
    write_text(
        stage / "README_REMOTE_RELAX.md",
        """# Remote Half Relaxation Package

This package contains one half of the generated full-replay flow jobs.
Do not run flow inference again. Run only UMA batched LBFGS relaxation.

## Requirements

- Python environment with fairchem / UMA, nvalchemi, nvalchemiops, torch, ASE, lmdb.
- GPU access. Default command uses GPUs `0 1 2 3 4 5 6 7`; override with `GPUS`.
- UMA model used by this experiment: `uma-s-1p2`, task `oc20`.
- LBFGS setting: ASE-default-style `fmax=0.05`, `maxstep=0.2`, `memory=100`, `max_steps=300`.

## Run

```bash
cd <unpacked-package>
PYTHON_BIN=/path/to/python GPUS="0 1 2 3 4 5 6 7" ./run_remote_relax.sh
```

## Return

After completion, send back:

- `run_remote_half/relax_results/`
- `run_remote_half/relax_aggregate.json`
- `run_remote_half/relax_summary.json`
- `run_remote_half/logs/`

The result rows contain `global_i`, so local and remote halves can be merged
without relying on shard IDs.
""",
    )

    tar_path = package_root.with_suffix(".tar.zst")
    if not args.no_tar:
        make_tar(stage, tar_path)
        manifest["tar_path"] = str(tar_path)
        manifest["tar_size_bytes"] = tar_path.stat().st_size
        write_text(stage / "MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True))
        write_text(package_root / "MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True))
        print(json.dumps({"package_root": str(package_root), "tar_path": str(tar_path), **manifest}, sort_keys=True))
    else:
        print(json.dumps({"package_root": str(package_root), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
