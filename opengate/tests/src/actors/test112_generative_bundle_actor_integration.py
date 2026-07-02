#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil

import opengate.tests.utility as tu

from test110_generative_bundle_helpers import TORCH_AVAILABLE, make_torchscript_bundle
from test108_optical_generative_actor_simulation import create_simulation

if __name__ == "__main__":
    paths = tu.get_default_test_paths(
        __file__, output_folder="test112_generative_bundle_actor_integration"
    )
    paths.output.mkdir(parents=True, exist_ok=True)

    if not TORCH_AVAILABLE:
        print("torch is not installed, skipping test112 (nothing to check).")
        tu.test_ok(True)
        raise SystemExit(0)

    is_ok = True

    # --- actor integration: generator as a bundle path (string) ---------
    bundle_dir_actor = paths.output / "bundle_actor"
    if bundle_dir_actor.exists():
        shutil.rmtree(bundle_dir_actor)
    bundle_dir_actor.mkdir(parents=True)
    make_torchscript_bundle(bundle_dir_actor)

    sim, hits_path, output_path = create_simulation(
        paths, generator=str(bundle_dir_actor)
    )
    sim.run(start_new_process=False)
    import uproot

    with uproot.open(output_path) as f:
        tree = f[list(f.keys())[0]]
        n_rows = len(tree["X"].array())
    if n_rows == 0:
        print("FAIL: actor with a bundle-path generator produced 0 output rows")
        is_ok = False
    else:
        print(f"OK: actor with a bundle-path generator produced {n_rows} rows")

    # --- actor integration: contradicting local_axes_order fails loudly -
    # (checked directly against the actor's bundle-resolution step, since a
    # GATE SimulationEngine can only run once per process)
    sim2, _, _ = create_simulation(paths, generator=str(bundle_dir_actor))
    og2 = sim2.actor_manager.get_actor("SyntheticPhotons")
    og2.local_axes_order = [1, 2, 0]  # contradicts the bundle's identity manifest
    try:
        og2._resolve_bundle_generator()
        print("FAIL: expected an exception for a contradicting local_axes_order")
        is_ok = False
    except Exception as e:
        if "local_axes_order" in str(e):
            print(f"OK: contradicting local_axes_order raised a clear error: {e}")
        else:
            print(f"FAIL: exception did not mention local_axes_order: {e}")
            is_ok = False

    tu.test_ok(is_ok)
