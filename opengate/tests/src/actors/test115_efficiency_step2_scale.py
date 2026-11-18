#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Second step: a bundle without an efficiency map, but efficiency_scale set.
The scale is then the efficiency everywhere.
"""

import shutil

import numpy as np

import opengate as gate
import opengate.tests.utility as tu

from test112_generative_bundle_helpers import (
    TORCH_AVAILABLE,
    make_vector_torchscript_bundle,
)
from test115_efficiency_helpers import (
    PATHS_FOLDER,
    create,
    read_counts,
    check_total,
)

SCALE = 0.25

if __name__ == "__main__":
    paths = tu.get_default_test_paths(__file__, output_folder=PATHS_FOLDER)
    paths.output.mkdir(parents=True, exist_ok=True)

    if not TORCH_AVAILABLE:
        print("torch is not installed, skipping test115 (nothing to check).")
        tu.test_ok(True)
        raise SystemExit(0)

    bundle_dir = paths.output / "bundle_plain"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    make_vector_torchscript_bundle(bundle_dir, target_batch=4096)

    sim, og = create(
        paths, bundle_dir, "scale", 4, 0.05 * gate.g4_units.s, scale=SCALE
    )
    sim.run(start_new_process=False)
    counts, _, edep = read_counts(paths, "scale")
    mu = edep * og._scintillation_yield
    is_ok = check_total(
        f"no map, efficiency_scale={SCALE}",
        counts,
        np.full(len(mu), SCALE),
        mu,
        og._resolution_scale,
    )

    tu.test_ok(is_ok)
