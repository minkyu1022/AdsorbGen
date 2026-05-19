#!/usr/bin/env python
"""Render per-system success trajectory videos with POV-Ray 3D rendering.

For each unique sid in a viz epoch dir, picks one success ``sys_*/`` folder and
renders an MP4 that plays the flow inference trajectory (``flow_traj.xyz``:
x0 -> x1_flow) followed by the UMA relaxation trajectory (``traj.xyz``:
x1_flow -> x1_relaxed) as one continuous animation.

Each frame is ray-traced with POV-Ray (shaded spheres + ball-and-stick bonds),
unlike a flat 2D matplotlib projection.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ase.io import read
from ase.io.pov import write_pov, get_bondpairs
from PIL import Image, ImageDraw, ImageFont

ROT = "-72x,18y,0z"
CANVAS = 760


def _pick_success_folder_per_sid(ep_dir: Path) -> dict:
    idx = json.loads((ep_dir / "_index.json").read_text())
    chosen: dict = {}
    for s in idx["systems"]:
        sid = int(s["sid"])
        if sid in chosen:
            continue
        d = ep_dir / s["sys_dir_name"]
        if (s.get("success") and (d / "flow_traj.xyz").exists()
                and (d / "traj.xyz").exists()):
            chosen[sid] = (d, s)
    return chosen


def _render_frame(atoms, work: Path, caption: str) -> Image.Image:
    pov = write_pov(
        str(work / "f.pov"), atoms,
        rotation=ROT, radii=0.62, show_unit_cell=2,
        povray_settings=dict(
            canvas_width=CANVAS,
            bondatoms=get_bondpairs(atoms, radius=1.1),
            background="White",
        ),
    )
    pov.render()  # -> work/f.png
    im = Image.open(work / "f.png").convert("RGB")
    bar_h = 30
    out = Image.new("RGB", (im.width, im.height + bar_h), "black")
    out.paste(im, (0, 0))
    d = ImageDraw.Draw(out)
    d.text((8, im.height + 7), caption, fill="white")
    return out


def _render(sid, folder, meta, out_path, fps):
    flow = read(str(folder / "flow_traj.xyz"), index=":")
    relax = read(str(folder / "traj.xyz"), index=":")
    frames = [("FLOW", i, a) for i, a in enumerate(flow)] \
        + [("RELAX", i, a) for i, a in enumerate(relax)]
    nfl = len(flow)
    head = (f"sid {sid} ads_id {meta['ads_id']}  "
            f"E_gt {meta['E_gt']:.1f} -> E_pred {meta['E_pred']:.1f} eV "
            f"(d {meta['improvement']:.1f})")

    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        for k, (phase, step, atoms) in enumerate(frames):
            cap = f"{head}   [{phase}] step {step}" + (f"/{nfl-1}" if phase == "FLOW" else "")
            img = _render_frame(atoms, work, cap)
            if img.width % 2 or img.height % 2:  # libx264 needs even dims
                img = img.crop((0, 0, img.width - img.width % 2,
                                img.height - img.height % 2))
            img.save(work / f"frame_{k:05d}.png")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", str(work / "frame_%05d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
            check=True,
        )
    return len(flow), len(relax)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--viz-ep-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--only-sid", type=int, default=0, help="render just this sid (0=all)")
    args = p.parse_args()

    ep_dir = Path(args.viz_ep_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = _pick_success_folder_per_sid(ep_dir)
    if args.only_sid:
        chosen = {k: v for k, v in chosen.items() if k == args.only_sid}
    print(f"[render] systems: {sorted(chosen)}", flush=True)
    for sid, (folder, meta) in sorted(chosen.items()):
        out = out_dir / f"sid{sid}_traj.mp4"
        nf, nr = _render(sid, folder, meta, out, fps=args.fps)
        print(f"[render] sid {sid}: flow {nf}f + relax {nr}f -> {out}", flush=True)


if __name__ == "__main__":
    main()
