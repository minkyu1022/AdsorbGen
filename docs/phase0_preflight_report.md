# Phase 0 Preflight Report — COMPLETE

## 0a. Phase3 anomaly flag verification — DONE

**Finding**: phase3_adsorption.py:117 (`_atoms_from_lmdb_data`) builds `atoms_init` from LMDB's `pos_relaxed`, not `pos`. So phase3 re-relaxes a DFT-relaxed structure with UMA — anomaly checks compare (DFT-relaxed) ↔ (DFT-relaxed + tiny UMA delta). 4 of 5 flags are uniformly False because the trajectory is near-identity.

**Implications**:
- phase3's `E_slab_ads` (UMA energy of DFT-relaxed) is still a valid GT reference.
- Check functions themselves are correctly coded.
- Replay pipeline reuses them to filter OUR model predictions (placement→model-output pair). They'll work correctly there.
- GT eligibility filter: `converged=True AND surface_changed=False` (92.4% pass; other flags not informative).

**Verdict**: no phase3 rerun needed.

## 0b. OC20-Dense merged test set — DONE

- Script: `scripts/merge_oc20dense_test.py`
- Output: `data/processed/oc20dense.lmdb`
- Entries: 65,073 data + 2 metadata (`length`, `anomaly_mask`)
- Keys 0..65072 sequentially
- Metadata: `anomaly_mask` concatenated (all zeros = all clean per upstream filter)

## 0c. GT replay index — DONE

- Script: `scripts/build_replay_gt_index.py`
- Outputs: `data/replay/gt_index_by_system.pkl`, `data/replay/gt_index_by_sid.pkl`
- 544,182 phase3 entries → 529,266 unique physical systems
- 489,453 systems eligible (92.5%), 39,813 excluded (no GT passes filter)
- Headroom p50=0, p90=0, max=11.93 eV. Most systems are singletons (1 config each).
- 2.2% of sids have >0.05 eV internal headroom (multi-config systems)

## 0d. Placement preflight — DONE

- Script: `scripts/preflight_placement.py`
- 500 random samples from is2res_train.lmdb
- **100% placement success** (0 failures)
- Cost: ~27ms/sample single-threaded. With 8 dataloader workers: ~3.4ms effective/sample during training. Acceptable.
- **Decision**: runtime try/except in PlacementPriorDataset (log + skip failures). No explicit preflight clean-index needed.
