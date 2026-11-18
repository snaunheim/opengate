#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
First step: a bundle with an efficiency map.

Each released photon reaches the sensor with the probability the map gives at
its hit, so the photons per hit are binomial around efficiency * released.
The test map falls linearly with depth and halves on one side of the module
(nearest in transverse), and its axes are in a different order than GATE's
module frame, so a wrong axis, sign or interpolation shows in the counts.
"""

import shutil

import numpy as np

import opengate as gate
import opengate.tests.utility as tu

from test112_generative_bundle_helpers import TORCH_AVAILABLE
from test115_efficiency_helpers import (
    PATHS_FOLDER,
    make_map_bundle,
    expected_efficiency,
    create,
    read_counts,
    expected_variance,
    check_total,
)

if __name__ == "__main__":
    paths = tu.get_default_test_paths(__file__, output_folder=PATHS_FOLDER)
    paths.output.mkdir(parents=True, exist_ok=True)

    if not TORCH_AVAILABLE:
        print("torch is not installed, skipping test115 (nothing to check).")
        tu.test_ok(True)
        raise SystemExit(0)

    bundle_dir = paths.output / "bundle_map"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    make_map_bundle(bundle_dir)

    is_ok = True
    sec = gate.g4_units.s

    # a scale that lifts the map above 1 is refused before the run
    _, og = create(paths, bundle_dir, "refused", 4, 0.05 * sec, scale=20)
    try:
        og.resolve_and_validate_config()
        print("FAIL: efficiency_scale=20 with a map up to 0.08 was accepted")
        is_ok = False
    except Exception as e:
        if "efficiency_scale" in str(e):
            print(f"OK: efficiency_scale=20 was refused: {e}")
        else:
            print(f"FAIL: unexpected error: {e}")
            is_ok = False

    sim, og = create(paths, bundle_dir, "map", 20, 0.2 * sec, scale=1.0)
    sim.run(start_new_process=False)
    counts, local, edep = read_counts(paths, "map")
    mu = edep * og._scintillation_yield
    rscale = og._resolution_scale
    eff = expected_efficiency(local)

    is_ok &= check_total("all hits", counts, eff, mu, rscale)
    for label, m in [
        ("transverse < 0", local[:, 1] < 0),
        ("transverse > 0", local[:, 1] > 0),
        ("front half", local[:, 0] < 0),
        ("back half", local[:, 0] > 0),
    ]:
        is_ok &= check_total(label, counts[m], eff[m], mu[m], rscale)

    # per hit: the binomial reduction gives unit spread, a fixed factor would not.
    # large hits only, so the released count is hardly ever cut at zero
    big = mu > 200
    n_big = big.sum()
    var = expected_variance(eff, mu, rscale)
    pull = (counts - eff * mu)[big] / np.sqrt(var[big])
    tol = 4 / np.sqrt(2 * n_big)
    ok = abs(pull.mean()) < 4 / np.sqrt(n_big) and abs(pull.std() - 1) < tol
    print(
        f"{'OK' if ok else 'FAIL'}: {n_big} hits, pull mean "
        f"{pull.mean():+.3f}, std {pull.std():.3f} (tolerance {tol:.3f})"
    )
    is_ok &= ok

    # the check above only means something if a fixed factor would fail it
    factor_std = np.sqrt(np.mean((eff**2 * rscale**2 * mu / var)[big]))
    ok = abs(factor_std - 1) > 2 * tol
    print(
        f"{'OK' if ok else 'FAIL'}: a fixed factor would give a pull std of "
        f"{factor_std:.3f}"
    )
    is_ok &= ok

    tu.test_ok(is_ok)
