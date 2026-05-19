"""Streamlit web UI for inspecting flow inference trajectories.

Run:
    streamlit run scripts/trajectory_viewer_app.py --server.port 8501 --server.address 0.0.0.0

Open in browser:  http://<host>:8501  (use an ssh tunnel if remote)
"""
from __future__ import annotations
import io
import json
import pickle
from pathlib import Path

import lmdb
import numpy as np
import py3Dmol
import streamlit as st
from ase import Atoms
from ase.io import read, write
import streamlit.components.v1 as components

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
RUN_DIRS = {
    "CatFlow center+rel (no aux loss)": "/home/irteam/runs/dissoc_traj/catflow_center_rel",
    "abs coord + ads_pair_l1=1.0":      "/home/irteam/runs/dissoc_traj/ads_pair_dist_loss",
}
ADSORBATES_PKL = "/home/irteam/data/pkls/adsorbates.pkl"
OC20DENSE_LMDB = "/home/irteam/data/processed/oc20dense.lmdb"

st.set_page_config(page_title="Flow Trajectory Viewer", layout="wide")

# -----------------------------------------------------------------------------
# Resource loaders (cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ads_db():
    with open(ADSORBATES_PKL, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def open_lmdb_env():
    return lmdb.open(OC20DENSE_LMDB, subdir=False, readonly=True, lock=False, readahead=False)

@st.cache_data
def load_entries(run_dir: str):
    p = Path(run_dir) / "trajectories" / "_index.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())

@st.cache_data
def load_frames(traj_path: str):
    return read(traj_path, index=":")

@st.cache_data
def load_anomaly_summary(run_dir: str):
    p = Path(run_dir) / "anomaly_summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())

def get_ads_info(record_index: int):
    env = open_lmdb_env()
    with env.begin() as txn:
        entry = pickle.loads(txn.get(str(int(record_index)).encode("ascii")))
    ads_id = int(entry.get("ads_id", -1))
    db = load_ads_db()
    atoms, smiles, binding_idx, label = db[ads_id]
    return {"ads_id": ads_id, "smiles": smiles, "binding_idx": binding_idx,
            "canonical_atoms": atoms, "lmdb_entry": entry}

# -----------------------------------------------------------------------------
# 3D rendering
# -----------------------------------------------------------------------------
def atoms_to_xyz(atoms: Atoms) -> str:
    buf = io.StringIO()
    write(buf, atoms, format="xyz")
    return buf.getvalue()

def render_3dmol(atoms: Atoms, height: int = 360, highlight_tags: bool = True,
                 show_cell: bool = True) -> str:
    """Return raw 3Dmol.js HTML for the given ASE Atoms (one frame)."""
    view = py3Dmol.view(width="100%", height=height)
    view.addModel(atoms_to_xyz(atoms), "xyz")
    view.setStyle({}, {"sphere": {"radius": 0.45}, "stick": {"radius": 0.18}})

    if highlight_tags and atoms.has("tags"):
        tags = np.asarray(atoms.get_tags())
        for axis_tag, color in [(2, "red"), (1, "lightgrey"), (0, "darkgrey")]:
            idx = np.where(tags == axis_tag)[0].tolist()
            if not idx:
                continue
            if axis_tag == 2:
                view.setStyle({"serial": [i + 1 for i in idx]},
                              {"sphere": {"radius": 0.55, "color": color},
                               "stick":  {"radius": 0.22, "color": color}})
            else:
                view.setStyle({"serial": [i + 1 for i in idx]},
                              {"sphere": {"radius": 0.35, "color": color},
                               "stick":  {"radius": 0.14, "color": color}})

    if show_cell and atoms.cell is not None and np.linalg.det(atoms.cell.array) > 1e-6:
        cell = atoms.cell.array
        corners = []
        for ix in (0, 1):
            for iy in (0, 1):
                for iz in (0, 1):
                    corners.append(ix * cell[0] + iy * cell[1] + iz * cell[2])
        edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
        for (a, b) in edges:
            p, q = corners[a], corners[b]
            view.addLine({"start": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
                          "end":   {"x": float(q[0]), "y": float(q[1]), "z": float(q[2])},
                          "color": "yellow"})

    view.zoomTo()
    return view._make_html()

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("Flow Inference Trajectory Viewer")

with st.sidebar:
    st.header("Selection")
    run_label = st.selectbox("Run", list(RUN_DIRS.keys()))
    run_dir = RUN_DIRS[run_label]
    entries = load_entries(run_dir)
    summary = load_anomaly_summary(run_dir)

    if summary:
        st.caption(
            f"Overall (n={summary.get('total','?')}): "
            f"valid={summary.get('valid_strict_rate', 0):.1%}, "
            f"dissoc={summary.get('dissoc_rate', 0):.1%}, "
            f"overlap={summary.get('overlap_rate', 0):.1%}"
        )

    if not entries:
        st.warning("No _index.json entries found in this run.")
        st.stop()

    def fmt(i):
        e = entries[i]
        flags = []
        if e.get("has_dissoc"): flags.append("DISSOC")
        if e.get("has_overlap"): flags.append("OVERLAP")
        if e.get("has_desorbed"): flags.append("DESORB")
        if e.get("has_intercalated"): flags.append("INTER")
        if e.get("has_surf_changed"): flags.append("SURF")
        tag = "|".join(flags) if flags else "VALID"
        return f"#{e['sample_index']:>2}  [{tag}]  min_pair={e.get('min_pair_distance_A', float('nan')):.2f} Å"

    sample_idx = st.selectbox("Sample", list(range(len(entries))),
                              format_func=fmt, index=0)
    entry = entries[sample_idx]

    # Frame slider lives in sidebar so users can scrub without scrolling.
    frames = load_frames(entry["traj"])
    n_frames = len(frames)
    frame_idx = st.slider("Euler step", 0, n_frames - 1, n_frames - 1,
                          help="0 = prior (x_0), last = final prediction (x_1).")

frame = frames[frame_idx]

# -----------------------------------------------------------------------------
# Top metrics row
# -----------------------------------------------------------------------------
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("valid_strict", "✓" if entry.get("valid_strict") else "✗")
m2.metric("dissoc",       "anomaly" if entry.get("has_dissoc") else "ok")
m3.metric("overlap",      "anomaly" if entry.get("has_overlap") else "ok")
m4.metric("intercalated", "anomaly" if entry.get("has_intercalated") else "ok")
m5.metric("min pair Å",   f"{entry.get('min_pair_distance_A', float('nan')):.2f}")
m6.metric("frame",        f"{frame_idx}/{n_frames - 1}")

st.caption(f"record_index = **{entry['record_index']}** | sid = {entry.get('sid')} | system_key = {entry.get('system_key')}")

# -----------------------------------------------------------------------------
# Side-by-side 3D
# -----------------------------------------------------------------------------
info = None
try:
    info = get_ads_info(entry["record_index"])
except Exception as exc:
    st.warning(f"Could not resolve canonical adsorbate from LMDB/pkl: {exc}")

L, R = st.columns([2, 1])
with L:
    st.subheader(f"Model trajectory — full system (step {frame_idx})")
    components.html(render_3dmol(frame, height=420, show_cell=True), height=440)

with R:
    st.subheader("Predicted ads atoms only")
    tags = np.asarray(frame.get_tags())
    ads_idx = np.where(tags == 2)[0]
    if ads_idx.size:
        ads_sub = frame[ads_idx]
        ads_sub.set_pbc(False)
        ads_sub.set_cell(None)
        components.html(render_3dmol(ads_sub, height=200, show_cell=False), height=210)
    if info is not None:
        st.markdown(f"**ads_id**={info['ads_id']}  |  **smiles**=`{info['smiles']}`  |  binding_idx={info['binding_idx']}")
        st.subheader("Canonical (gas-phase)")
        components.html(render_3dmol(info["canonical_atoms"], height=200, show_cell=False), height=210)

# -----------------------------------------------------------------------------
# Pair distance over trajectory (if available from index entry)
# -----------------------------------------------------------------------------
with st.expander("Pair-distance evolution (ads-ads & ads-slab)"):
    n = len(frame)
    if "ads_idx_cache" not in st.session_state:
        st.session_state["ads_idx_cache"] = {}
    cache_key = (entry["traj"], "min_per_frame")
    series = st.session_state["ads_idx_cache"].get(cache_key)
    if series is None:
        ads_idx_arr = np.where(np.asarray(frames[0].get_tags()) == 2)[0]
        mins = []
        for f in frames:
            d = f.get_all_distances(mic=True)
            np.fill_diagonal(d, np.inf)
            if ads_idx_arr.size >= 2:
                ads_min = d[np.ix_(ads_idx_arr, ads_idx_arr)].min()
            else:
                ads_min = float("nan")
            other = np.delete(np.arange(len(f)), ads_idx_arr)
            if ads_idx_arr.size and other.size:
                as_min = d[np.ix_(ads_idx_arr, other)].min()
            else:
                as_min = float("nan")
            mins.append((ads_min, as_min))
        series = np.asarray(mins)
        st.session_state["ads_idx_cache"][cache_key] = series

    import pandas as pd
    df = pd.DataFrame({
        "ads-ads min (Å)":   series[:, 0],
        "ads-slab min (Å)":  series[:, 1],
    })
    st.line_chart(df, height=240)
    st.caption("Horizontal slider step above corresponds to row index here.")
