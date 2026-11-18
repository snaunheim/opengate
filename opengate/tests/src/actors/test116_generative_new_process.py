#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A run in its own process pickles the actors to send them back, and with them
the bundle. The bundle leaves its model out, so this works for a directory
and a zip bundle, and a copy of the bundle can load the model again.
"""

import pickle
import shutil
import zipfile

import numpy as np

import opengate as gate
import opengate.tests.utility as tu
from opengate.contrib.optical.generative_bundle import GenerativeModelBundle

from test112_generative_bundle_helpers import (
    TORCH_AVAILABLE,
    make_vector_torchscript_bundle,
)
from test113_pooled_helpers import read_tree
from og_actor109_pet_sim_setup import create_simulation


def run_in_new_process(paths, bundle, name):
    sim, _, _ = create_simulation(paths, generator=str(bundle), n_modules=4)
    sim.random_seed = 12345
    sim.run_timing_intervals = [[0, 0.05 * gate.g4_units.s]]
    sim.actor_manager.get_actor("Hits").output_filename = f"hits_{name}.root"
    og = sim.actor_manager.get_actor("SyntheticPhotons")
    og.output_filename = f"photons_{name}.root"
    sim.run(start_new_process=True)
    return read_tree(paths.output / f"photons_{name}.root")


if __name__ == "__main__":
    paths = tu.get_default_test_paths(
        __file__, output_folder="test116_generative_new_process"
    )
    paths.output.mkdir(parents=True, exist_ok=True)

    if not TORCH_AVAILABLE:
        print("torch is not installed, skipping test116 (nothing to check).")
        tu.test_ok(True)
        raise SystemExit(0)

    bundle_dir = paths.output / "bundle"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    make_vector_torchscript_bundle(bundle_dir, target_batch=4096)
    bundle_zip = paths.output / "bundle.zip"
    with zipfile.ZipFile(bundle_zip, "w") as z:
        for f in bundle_dir.iterdir():
            z.write(f, f.name)

    is_ok = True

    # same seed and same model, so both runs give the same photons
    photons_dir = run_in_new_process(paths, bundle_dir, "dir")
    photons_zip = run_in_new_process(paths, bundle_zip, "zip")
    n = len(photons_dir["SourceHitIndex"])
    ok = n > 0 and all(
        np.array_equal(photons_dir[k], photons_zip[k]) for k in photons_dir
    )
    print(f"{'OK' if ok else 'FAIL'}: both runs came back, {n} photons each")
    is_ok &= ok

    # after the runs: a forked run cannot use CUDA once this process did
    x = np.linspace(-1.0, 1.0, 100)
    for bundle in (bundle_dir, bundle_zip):
        original = GenerativeModelBundle(bundle)
        copy = pickle.loads(pickle.dumps(original))
        ok = copy._model is None
        copy.load_model()
        before = original.generate_pooled(x, x, x, x)
        after = copy.generate_pooled(x, x, x, x)
        ok &= all(np.array_equal(b, a) for b, a in zip(before, after))
        print(f"{'OK' if ok else 'FAIL'}: pickled {bundle.name}, model loaded again")
        is_ok &= ok

    tu.test_ok(is_ok)
