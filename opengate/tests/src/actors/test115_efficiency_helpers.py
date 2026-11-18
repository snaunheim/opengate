#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import numpy as np

from test112_generative_bundle_helpers import make_vector_torchscript_bundle
from test113_pooled_helpers import read_tree
from og_actor109_pet_sim_setup import create_simulation

PATHS_FOLDER = "test115_generative_efficiency"
SEED = 98765

# model frame of the test bundle: transverse, axial, depth (0 at the sensor
# face, i.e. GATE's local depth + 5 mm), a different order than GATE's.
# Small efficiencies as in real arrays, where the binomial spread dominates.
DEPTH_AXIS = [0.0, 10.0]
DEPTH_EFF = [0.08, 0.02]
TRANSVERSE_AXIS = [-0.75, 0.75]
TRANSVERSE_FACTOR = [1.0, 0.5]


def make_map_bundle(bundle_dir):
    """the vector test bundle, plus an efficiency map in its own frame"""
    make_vector_torchscript_bundle(bundle_dir, target_batch=4096)
    with open(bundle_dir / "manifest.json") as f:
        manifest = json.load(f)
    manifest["coordinates"] = {
        "axes_order": ["transverse", "axial", "depth"],
        "offset": [0.0, 0.0, -5.0],
        "unit": "mm",
    }
    manifest["efficiency"] = {
        "file": "efficiency.npz",
        "interpolation": ["nearest", "nearest", "linear"],
    }
    with open(bundle_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
    values = np.array(TRANSVERSE_FACTOR)[:, None, None] * np.array(DEPTH_EFF)
    np.savez(
        bundle_dir / "efficiency.npz",
        transverse=TRANSVERSE_AXIS,
        axial=[0.0],
        depth=DEPTH_AXIS,
        efficiency=values,
    )


def expected_efficiency(local):
    """the test map at GATE module-local positions (x=depth, y=transverse)"""
    depth = np.clip(local[:, 0] + 5.0, DEPTH_AXIS[0], DEPTH_AXIS[1])
    eff = np.interp(depth, DEPTH_AXIS, DEPTH_EFF)
    return eff * np.where(local[:, 1] < 0, *TRANSVERSE_FACTOR)


def create(paths, bundle_dir, name, n_modules, duration, scale):
    sim, _, _ = create_simulation(
        paths, generator=str(bundle_dir), n_modules=n_modules
    )
    sim.random_seed = SEED
    sim.run_timing_intervals = [[0, duration]]
    sim.actor_manager.get_actor("Hits").output_filename = f"hits_{name}.root"
    og = sim.actor_manager.get_actor("SyntheticPhotons")
    og.output_filename = f"photons_{name}.root"
    og.efficiency_scale = scale
    return sim, og


def read_counts(paths, name):
    """photons per hit, local position and edep of each hit with energy"""
    hits = read_tree(paths.output / f"hits_{name}.root")
    photons = read_tree(paths.output / f"photons_{name}.root")
    # one thread, so SourceHitIndex counts the hits with energy from 0
    rows = np.flatnonzero(hits["TotalEnergyDeposit"] > 0)
    counts = np.bincount(
        photons["SourceHitIndex"].astype(np.int64), minlength=len(rows)
    )
    local = np.stack(
        [hits[f"PostPositionLocalModule_{c}"][rows] for c in "XYZ"], axis=1
    )
    return counts, local, hits["TotalEnergyDeposit"][rows]


def expected_variance(eff, mu, resolution_scale):
    # released N has variance R^2 mu, the reduction adds mu eps (1 - eps)
    return eff**2 * resolution_scale**2 * mu + mu * eff * (1 - eff)


def check_total(label, counts, eff, mu, resolution_scale):
    total, expected = counts.sum(), (eff * mu).sum()
    n_sigma = (total - expected) / np.sqrt(
        expected_variance(eff, mu, resolution_scale).sum()
    )
    ok = abs(n_sigma) < 4
    print(
        f"{'OK' if ok else 'FAIL'}: {label}: {total} photons, expected "
        f"{expected:.0f} ({n_sigma:+.1f} sigma)"
    )
    return ok
