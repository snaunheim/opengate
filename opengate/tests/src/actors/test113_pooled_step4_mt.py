#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fourth of four: runs the pooled bundle with several threads and checks that
nothing is lost.

In MT each worker has its own pool, filled and flushed independently, and
the leftover of a partly filled pool is written by that worker's own
EndOfRunAction. What this checks is that every hit still ends up with
exactly the photons it asked for, and that the photons of one hit all carry
that hit's own values.

Event order and the Poisson draws differ between runs with different thread
counts, so the two runs cannot be compared row by row. The invariants below
hold regardless.
"""

import shutil

import numpy as np

import opengate.tests.utility as tu

from test112_generative_bundle_helpers import (
    TORCH_AVAILABLE,
    make_vector_torchscript_bundle,
)
from test113_pooled_helpers import PATHS_FOLDER, SEED, N_MODULES, TARGET_BATCH
from og_actor109_pet_sim_setup import create_simulation

N_THREADS = 4


def run_mt(paths, bundle_dir, output_name, hits_name, seed, n_threads):
    sim, hits_path, output_path = create_simulation(
        paths, generator=str(bundle_dir), n_modules=N_MODULES
    )
    sim.random_seed = seed
    sim.number_of_threads = n_threads
    sim.actor_manager.get_actor("Hits").output_filename = hits_name
    og = sim.actor_manager.get_actor("SyntheticPhotons")
    og.output_filename = output_name
    sim.run(start_new_process=False)
    return output_path.parent / output_name, hits_path.parent / hits_name


def group_bounds(arrays):
    """
    Sorts the photons by the hit they came from and returns that order plus
    the start of each group. Each thread numbers its hits from its own
    offset, so SourceHitIndex identifies a hit across threads as well.
    """
    hit = arrays["SourceHitIndex"].astype(np.int64)
    order = np.argsort(hit, kind="stable")
    h = hit[order]
    starts = np.flatnonzero(np.concatenate(([True], h[1:] != h[:-1])))
    return order, starts


def check_groups(arrays, label):
    """Every photon of one hit has to repeat that hit's own values."""
    ok = True
    order, starts = group_bounds(arrays)
    ends = np.append(starts[1:], len(order))
    print(f"  {label}: {len(order)} photons in {len(starts)} hit groups")

    # a value the actor takes from the hit must not vary inside a group;
    # if pooling mixed two hits up, it would
    for name in ("GlobalTime", "PostPosition_X", "PostPosition_Y"):
        if name not in arrays:
            continue
        values = arrays[name][order]
        bad = [s for s, e in zip(starts, ends) if not np.all(values[s:e] == values[s])]
        if bad:
            print(f"  FAIL: {name} varies inside {len(bad)} hit groups in {label}")
            ok = False
        else:
            print(f"  OK: {name} is constant within every hit group")
    return ok, len(starts)


if __name__ == "__main__":
    paths = tu.get_default_test_paths(__file__, output_folder=PATHS_FOLDER)
    paths.output.mkdir(parents=True, exist_ok=True)

    if not TORCH_AVAILABLE:
        print("torch is not installed, skipping test113 (nothing to check).")
        tu.test_ok(True)
        raise SystemExit(0)

    bundle_dir = paths.output / "bundle_vector_mt"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    make_vector_torchscript_bundle(bundle_dir, target_batch=TARGET_BATCH)

    out, hits_out = run_mt(
        paths, bundle_dir, "pooled_mt.root", "hits_mt.root", SEED, N_THREADS
    )

    import uproot

    with uproot.open(out) as f:
        arrays = f[list(f.keys())[0]].arrays(library="np")
    with uproot.open(hits_out) as f:
        hits = f["Hits"].arrays(library="np")

    is_ok = True
    n_photons = len(arrays["SourceHitIndex"])
    if n_photons == 0:
        print(f"FAIL: the {N_THREADS}-thread run produced no photons")
        is_ok = False
        tu.test_ok(is_ok)
        raise SystemExit(0)

    print(f"OK: the {N_THREADS}-thread run produced {n_photons} photons")
    ok, n_groups = check_groups(arrays, f"{N_THREADS} threads")
    is_ok &= ok

    # every hit that deposited energy had its photons sampled and must have
    # reached the output; a pool that was not flushed would lose the last ones
    n_hits_with_edep = int((hits["TotalEnergyDeposit"] > 0).sum())
    if n_groups != n_hits_with_edep:
        print(
            f"FAIL: {n_hits_with_edep} hits deposited energy but only "
            f"{n_groups} of them produced photons. A pool was not flushed."
        )
        is_ok = False
    else:
        print(f"OK: all {n_hits_with_edep} hits with energy produced photons")

    tu.test_ok(is_ok)
