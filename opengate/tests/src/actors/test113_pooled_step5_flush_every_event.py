#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fifth step: pooled with flush_every_event and a digitizer reading the output.

A digitizer only sees what is written to its input during its own event. Pooled
across events, it would get photons of other events and never see the batch
flushed at the end of the run, so the actor must refuse such a reader unless
flush_every_event is set. With it, the reader must get every photon, and every
photon must still belong to its own hit.

The reader draws random numbers, so this run cannot be compared with step 1.
"""

import shutil

import numpy as np

import opengate as gate
import opengate.tests.utility as tu

from test112_generative_bundle_helpers import (
    TORCH_AVAILABLE,
    make_vector_torchscript_bundle,
)
from test113_pooled_helpers import (
    PATHS_FOLDER,
    SEED,
    N_MODULES,
    TARGET_BATCH,
    read_tree,
)
from og_actor109_pet_sim_setup import create_simulation


def create_with_reader(paths, bundle_dir, flush):
    sim, _, _ = create_simulation(
        paths, generator=str(bundle_dir), n_modules=N_MODULES
    )
    sim.random_seed = SEED
    sim.actor_manager.get_actor("Hits").output_filename = "hits_flush.root"
    og = sim.actor_manager.get_actor("SyntheticPhotons")
    og.output_filename = "pooled_flush.root"
    og.flush_every_event = flush
    # keeps every row, so it must end up with exactly the photons it was given
    reader = sim.add_actor("DigitizerEfficiencyActor", "Reader")
    reader.attached_to = og.attached_to
    reader.authorize_repeated_volumes = True
    reader.input_digi_collection = og.name
    reader.efficiency = 1.0
    reader.output_filename = "reader.root"
    return sim, og


if __name__ == "__main__":
    paths = tu.get_default_test_paths(__file__, output_folder=PATHS_FOLDER)
    paths.output.mkdir(parents=True, exist_ok=True)

    if not TORCH_AVAILABLE:
        print("torch is not installed, skipping test113 (nothing to check).")
        tu.test_ok(True)
        raise SystemExit(0)

    bundle_dir = paths.output / "bundle_vector_flush"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    make_vector_torchscript_bundle(bundle_dir, target_batch=TARGET_BATCH)

    is_ok = True

    # without flush_every_event the reader is refused before Geant4 starts
    _, og = create_with_reader(paths, bundle_dir, flush=False)
    try:
        og.resolve_and_validate_config()
        print("FAIL: a reader of the pooled output was accepted without flush_every_event")
        is_ok = False
    except Exception as e:
        if "flush_every_event" in str(e):
            print(f"OK: a reader without flush_every_event raised a clear error: {e}")
        else:
            print(f"FAIL: exception did not mention flush_every_event: {e}")
            is_ok = False

    sim, _ = create_with_reader(paths, bundle_dir, flush=True)
    sim.run(start_new_process=False)

    photons = read_tree(paths.output / "pooled_flush.root")
    reader = read_tree(paths.output / "reader.root")
    hits = read_tree(paths.output / "hits_flush.root")

    # the reader got every photon, the last ones of the run included
    n_photons = len(photons["SourceHitIndex"])
    n_read = len(reader["SourceHitIndex"])
    if n_photons == 0 or n_read != n_photons:
        print(f"FAIL: the reader got {n_read} of {n_photons} photons")
        is_ok = False
    else:
        print(f"OK: the reader got all {n_photons} photons")
        for name in sorted(set(photons) & set(reader)):
            if not np.array_equal(photons[name], reader[name]):
                print(f"FAIL: {name} differs between the output and the reader")
                is_ok = False

    # one thread, so SourceHitIndex counts the hits with energy from 0; the
    # test model delays every photon by 1 ns after its own hit
    rows = np.flatnonzero(hits["TotalEnergyDeposit"] > 0)
    index = photons["SourceHitIndex"].astype(np.int64)
    if n_photons and index.max() >= len(rows):
        print("FAIL: a SourceHitIndex points past the last hit")
        is_ok = False
    elif n_photons:
        expected = hits["GlobalTime"][rows[index]] + 1.0 * gate.g4_units.ns
        if not np.array_equal(photons["GlobalTime"], expected):
            print("FAIL: photons do not carry the time of their own hit")
            is_ok = False
        else:
            print("OK: every photon carries the time of its own hit plus 1 ns")

    tu.test_ok(is_ok)
