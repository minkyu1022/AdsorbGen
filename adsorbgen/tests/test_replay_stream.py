"""ReplayStream{Writer,Reader} unit tests.

Verifies:
  * single-shard roundtrip
  * chunk auto-flush at chunk_size
  * multi-shard parallel writes are independently readable
  * reader is incremental (second call returns only new chunks)
  * partially-written manifest line / chunk file does not corrupt reader state
  * reader picks up shards created after construction (refresh_shards)
"""
from __future__ import annotations
import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np

from adsorbgen.replay import (
    ReplayBuffer,
    ReplayEntry,
    ReplayStreamReader,
    ReplayStreamWriter,
)


def _make_entry(sid: int, n_atoms: int = 4, ads_id: int = 0) -> ReplayEntry:
    rng = np.random.default_rng(sid)
    return ReplayEntry(
        system_key=("sys", sid),
        sid=int(sid),
        ads_id=int(ads_id),
        pos_relaxed=rng.normal(size=(n_atoms, 3)).astype(np.float32),
        tags=np.full(n_atoms, 2, dtype=np.int64),
        atomic_numbers=np.full(n_atoms, 6, dtype=np.int64),
        fixed=np.zeros(n_atoms, dtype=np.int64),
        cell=np.eye(3, dtype=np.float32) * 10.0,
        E_sys_pred=-1.0 * sid,
        E_sys_gt=-0.5 * sid,
        improvement=0.5 * sid,
        epoch_added=int(sid // 10),
        source_placement="random_heuristic",
    )


def test_single_shard_roundtrip(tmp_path: Path):
    w = ReplayStreamWriter(tmp_path, shard_id=0, chunk_size=4)
    for s in range(3):
        w.append(_make_entry(s))
    assert w.flush() == 0          # 3 < chunk_size, manual flush returns chunk 0
    assert w.flush() is None       # buffer empty

    r = ReplayStreamReader(tmp_path)
    entries = r.load_new_chunks()
    assert [e.sid for e in entries] == [0, 1, 2]
    assert r.n_consumed() == {0: 1}

    # second read returns nothing
    assert r.load_new_chunks() == []


def test_auto_flush_at_chunk_size(tmp_path: Path):
    w = ReplayStreamWriter(tmp_path, shard_id=0, chunk_size=2)
    for s in range(5):
        w.append(_make_entry(s))
    # at sid=1 and sid=3 the buffer hit size 2 and auto-flushed → 2 chunks of 2
    # leftover 1 entry remains in buffer
    r = ReplayStreamReader(tmp_path)
    entries = r.load_new_chunks()
    assert [e.sid for e in entries] == [0, 1, 2, 3]

    w.flush()  # flush final partial chunk
    leftovers = r.load_new_chunks()
    assert [e.sid for e in leftovers] == [4]


def test_multi_shard_independent(tmp_path: Path):
    writers = [ReplayStreamWriter(tmp_path, shard_id=s, chunk_size=4) for s in range(3)]
    for sid in range(12):
        writers[sid % 3].append(_make_entry(sid))
    for w in writers:
        w.flush()

    r = ReplayStreamReader(tmp_path)
    entries = r.load_new_chunks()
    # all sids 0..11 should be present (order across shards not guaranteed,
    # but within a shard order preserved)
    assert sorted(e.sid for e in entries) == list(range(12))
    assert set(r.n_consumed().keys()) == {0, 1, 2}


def test_incremental_reads(tmp_path: Path):
    w = ReplayStreamWriter(tmp_path, shard_id=0, chunk_size=2)
    for s in range(4):
        w.append(_make_entry(s))      # auto flushes 2 chunks

    r = ReplayStreamReader(tmp_path)
    first = r.load_new_chunks()
    assert [e.sid for e in first] == [0, 1, 2, 3]

    # writer continues
    for s in range(4, 6):
        w.append(_make_entry(s))
    w.flush()

    second = r.load_new_chunks()
    assert [e.sid for e in second] == [4, 5]

    # nothing new
    assert r.load_new_chunks() == []


def test_partial_manifest_line_is_tolerated(tmp_path: Path):
    w = ReplayStreamWriter(tmp_path, shard_id=0, chunk_size=2)
    for s in range(2):
        w.append(_make_entry(s))      # auto flush at 2 → chunk 0

    # Simulate a writer crash mid-manifest-write: append a truncated json line
    manifest = tmp_path / "shard_0" / "manifest.jsonl"
    with open(manifest, "a") as f:
        f.write('{"chunk": 1, "n_entries":')  # truncated, no newline

    r = ReplayStreamReader(tmp_path)
    entries = r.load_new_chunks()
    assert [e.sid for e in entries] == [0, 1]
    # consumed should advance only past chunk 0; chunk 1 partial line not counted
    assert r.n_consumed() == {0: 1}

    # complete the line and add a chunk: reader picks it up on next call
    with open(manifest, "a") as f:
        f.write('\n')                          # finish the bad line as no-op
    for s in range(2, 4):
        w._buffer.append(_make_entry(s))       # bypass auto flush to control chunk
    w.flush()                                  # writes chunk_00001.pkl + manifest line
    again = r.load_new_chunks()
    assert [e.sid for e in again] == [2, 3]


def test_writer_separates_after_truncated_manifest_line(tmp_path: Path):
    w = ReplayStreamWriter(tmp_path, shard_id=0, chunk_size=2)
    for s in range(2):
        w.append(_make_entry(s))

    manifest = tmp_path / "shard_0" / "manifest.jsonl"
    with open(manifest, "a") as f:
        f.write('{"chunk": 1, "n_entries":')  # no newline, simulates crash

    for s in range(2, 4):
        w.append(_make_entry(s))

    r = ReplayStreamReader(tmp_path)
    entries = r.load_new_chunks()
    assert [e.sid for e in entries] == [0, 1, 2, 3]


def test_reader_picks_up_late_shard(tmp_path: Path):
    # reader created before any shard exists
    r = ReplayStreamReader(tmp_path)
    assert r.load_new_chunks() == []

    w = ReplayStreamWriter(tmp_path, shard_id=7, chunk_size=1)
    w.append(_make_entry(42))                  # auto flush, chunk 0

    entries = r.load_new_chunks()
    assert len(entries) == 1
    assert entries[0].sid == 42
    assert 7 in r.n_consumed()


def test_writer_to_buffer_consume_pipeline(tmp_path: Path):
    """Smoke: daemon-style writer → consumer-style reader → ReplayBuffer.

    Mirrors what AdsorbGenModule does for --external-replay-dir at training
    epoch end: read newly appended chunks and add() into the buffer.
    """
    w0 = ReplayStreamWriter(tmp_path, shard_id=0, chunk_size=2)
    w1 = ReplayStreamWriter(tmp_path, shard_id=1, chunk_size=2)
    for sid in range(6):
        (w0 if sid % 2 == 0 else w1).append(_make_entry(sid))
    w0.flush(); w1.flush()

    reader = ReplayStreamReader(tmp_path)
    buf = ReplayBuffer(mode="append", per_system_cap=10, global_cap=1000)
    n_accepted = 0
    for e in reader.load_new_chunks():
        if buf.add(e):
            n_accepted += 1
    assert n_accepted == 6
    assert len(buf) == 6
    assert buf.n_systems() == 6
