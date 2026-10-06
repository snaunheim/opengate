#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
First of three: runs the PET ring with a bundle whose model takes one
position per call, so the actor calls it once per hit.

test113_pooled_step3_compare.py compares this output with the pooled one.
A separate script per run, since a SimulationEngine can only run once per
process and a loaded TorchScript model cannot be pickled into a subprocess.
"""

import shutil

import opengate.tests.utility as tu

from test112_generative_bundle_helpers import (
    TORCH_AVAILABLE,
    make_torchscript_bundle,
)
from test113_pooled_helpers import PATHS_FOLDER, SEED, N_MODULES, run_simulation

if __name__ == "__main__":
    paths = tu.get_default_test_paths(__file__, output_folder=PATHS_FOLDER)
    paths.output.mkdir(parents=True, exist_ok=True)

    if not TORCH_AVAILABLE:
        print("torch is not installed, skipping test113 (nothing to check).")
        tu.test_ok(True)
        raise SystemExit(0)

    bundle_dir = paths.output / "bundle_scalar"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    make_torchscript_bundle(bundle_dir)

    out = run_simulation(
        paths, bundle_dir, "per_hit.root", seed=SEED, n_modules=N_MODULES
    )
    print(f"OK: per-hit run wrote {out}")
    tu.test_ok(out.exists())
