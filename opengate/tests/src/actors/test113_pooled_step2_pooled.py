#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Second of three: runs the same PET ring with a bundle whose model takes one
conditioning entry per photon, so the actor pools hits until the batch is
full before calling it.

Same geometry, same seed and same output schema as step 1.
"""

import shutil

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
    run_simulation,
)

if __name__ == "__main__":
    paths = tu.get_default_test_paths(__file__, output_folder=PATHS_FOLDER)
    paths.output.mkdir(parents=True, exist_ok=True)

    if not TORCH_AVAILABLE:
        print("torch is not installed, skipping test113 (nothing to check).")
        tu.test_ok(True)
        raise SystemExit(0)

    bundle_dir = paths.output / "bundle_vector"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    make_vector_torchscript_bundle(bundle_dir, target_batch=TARGET_BATCH)

    out = run_simulation(
        paths, bundle_dir, "pooled.root", seed=SEED, n_modules=N_MODULES
    )
    print(f"OK: pooled run wrote {out}")
    tu.test_ok(out.exists())
