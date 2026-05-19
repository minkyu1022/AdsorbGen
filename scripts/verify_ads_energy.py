#!/usr/bin/env python
"""Validate the UMA adsorption-energy scheme against OC20 metadata targets.

For N sampled OC20 training systems, computes
    E_ads(UMA) = E_sys(UMA) - E_slab(UMA) - E_gas(UMA-ref)
all consistently with UMA-s-1p1 / task=oc20, and compares against the dataset's
own stored adsorption-energy target ``y_relaxed``.

  E_sys   : UMA oc20 single-point of the GT relaxed adslab (LMDB pos_relaxed)
  E_slab  : pristine clean-slab e_total from is2res.pkl (already UMA oc20)
  E_gas   : sum of per-element OC20 reference potentials (gas_phase_refs_oc20)
"""
from __future__ import annotations

import argparse
import pickle

import lmdb
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from fairchem.core import pretrained_mlip
from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

TRAIN = "/home/irteam/data/processed/is2res_train.lmdb"
GTIDX = "/home/irteam/data/replay/gt_index_by_sid.pkl"
SLABS = "/home/irteam/results/pristine_slabs/is2res.pkl"
GASREF = "/home/irteam/data-vol1/minkyu/data/pkls/gas_phase_refs_oc20.pkl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    gt = pickle.load(open(GTIDX, "rb"))
    slabs = pickle.load(open(SLABS, "rb"))
    gas = pickle.load(open(GASREF, "rb"))
    atom_pot = gas["atomic_potentials"]
    print(f"gas ref: model={gas['model']} task={gas['task']}  atomic_potentials={atom_pot}")

    pu = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
    calc = FAIRChemCalculator(pu, task_name="oc20")

    env = lmdb.open(TRAIN, subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        n_tot = int(pickle.loads(txn.get(b"length")))
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(n_tot)
        rows = []
        for i in order:
            if len(rows) >= args.n:
                break
            e = pickle.loads(txn.get(str(int(i)).encode()))
            sid = int(e["sid"])
            gi = gt.get(sid)
            if not (isinstance(gi, dict) and gi.get("system_key")):
                continue
            slab = slabs.get(tuple(gi["system_key"][:4]))
            if slab is None:
                continue
            z = np.asarray(e["atomic_numbers"]); tags = np.asarray(e["tags"])
            ads_z = z[tags == 2]
            if len(ads_z) == 0:
                continue
            syms = [chemical_symbols[int(x)] for x in ads_z]
            if any(s not in atom_pot for s in syms):
                continue
            rows.append((sid, e, slab, syms))
    env.close()

    print(f"\n{'sid':>9} {'adsorbate':>12} {'E_sys':>11} {'E_slab':>11} {'E_gas':>10}"
          f" {'E_ads(UMA)':>12} {'y_relaxed':>11} {'diff':>9}")
    diffs = []
    for sid, e, slab, syms in rows:
        z = np.asarray(e["atomic_numbers"])
        atoms = Atoms(numbers=z, positions=np.asarray(e["pos_relaxed"]),
                      cell=np.asarray(e["cell"]).reshape(3, 3), pbc=True)
        atoms.calc = calc
        e_sys = float(atoms.get_potential_energy())
        e_slab = float(slab["e_total"])
        e_gas = sum(atom_pot[s] for s in syms)
        e_ads = e_sys - e_slab - e_gas
        y = float(e["y_relaxed"])
        d = e_ads - y
        diffs.append(d)
        formula = "".join(sorted(syms))
        print(f"{sid:>9} {formula:>12} {e_sys:>11.3f} {e_slab:>11.3f} {e_gas:>10.3f}"
              f" {e_ads:>12.3f} {y:>11.3f} {d:>9.3f}")

    d = np.asarray(diffs)
    print(f"\nE_ads(UMA) vs y_relaxed:  MAE={np.abs(d).mean():.3f} eV  "
          f"mean diff={d.mean():+.3f}  std={d.std():.3f}  (n={len(d)})")


if __name__ == "__main__":
    main()
