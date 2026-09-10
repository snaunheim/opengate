#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil

import opengate.tests.utility as tu

from test112_generative_bundle_helpers import (
    TORCH_AVAILABLE,
    make_torchscript_bundle,
)
from og_actor108_sim_setup import create_simulation

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

    # --- actor integration: a bundle, end to end ------------------------
    # Exercises constant 'value' entries, declared units and a model column
    # count that differs from the number of declared outputs.
    bundle_dir_v2 = paths.output / "bundle_actor"
    if bundle_dir_v2.exists():
        shutil.rmtree(bundle_dir_v2)
    bundle_dir_v2.mkdir(parents=True)
    make_torchscript_bundle(bundle_dir_v2)

    sim, hits_path, output_path = create_simulation(paths, generator=str(bundle_dir_v2))
    sim.run(start_new_process=False)
    import numpy as np
    import uproot

    with uproot.open(output_path) as f:
        tree = f[list(f.keys())[0]]
        arrays = tree.arrays(library="np")
    n_rows = len(arrays["PostPositionLocalModule_X"])
    if n_rows == 0:
        print("FAIL: actor with a bundle-path generator produced 0 output rows")
        is_ok = False
    else:
        print(f"OK: actor with a bundle-path generator produced {n_rows} rows")

    # the declared constants reach the output unchanged
    for column, expected in (
        ("PostPositionLocalModule_X", -5.0),
        ("Direction_Y", 0.0),
        ("Direction_Z", 0.0),
    ):
        if not np.allclose(arrays[column], expected):
            print(f"FAIL: {column} is not the declared constant {expected}")
            is_ok = False
        else:
            print(f"OK: {column} carries its declared constant {expected}")

    # 3 eV declared in eV must arrive as MeV, GATE's internal unit
    if not np.allclose(arrays["KineticEnergy"], 3.0e-6):
        print(
            f"FAIL: KineticEnergy is {arrays['KineticEnergy'][:3]}, expected "
            f"3e-06 MeV (3 eV as declared)"
        )
        is_ok = False
    else:
        print("OK: the declared eV unit was converted to MeV")

    # --- the derived world-frame position is the module-local one, back ---
    # In this setup the module sits at the world origin with no rotation, so
    # the inverse transform is the identity and the two frames must agree
    # exactly. That pins down the transform without depending on a particular
    # placement; test109's rotated ring exercises a non-trivial one.
    for axis in ("X", "Y", "Z"):
        world = arrays.get(f"PostPosition_{axis}")
        local = arrays.get(f"PostPositionLocalModule_{axis}")
        if world is None:
            continue
        if not np.allclose(world, local, atol=1e-9):
            print(
                f"FAIL: PostPosition_{axis} does not match "
                f"PostPositionLocalModule_{axis} for an unrotated module at the "
                f"origin (max diff {np.abs(world - local).max():g})"
            )
            is_ok = False
        else:
            print(f"OK: derived PostPosition_{axis} matches the module frame")

    # --- a generator object with no 'outputs' fails loudly ---------------
    # A bundle takes its schema from the manifest; a plain object has none, so
    # the actor cannot guess what the columns it returns mean.
    sim1, _, _ = create_simulation(paths)
    og1 = sim1.actor_manager.get_actor("SyntheticPhotons")
    og1.outputs = None
    try:
        og1.resolve_and_validate_config()
        print("FAIL: expected an exception for a generator without 'outputs'")
        is_ok = False
    except Exception as e:
        if "outputs" in str(e):
            print(f"OK: a generator without 'outputs' raised a clear error: {e}")
        else:
            print(f"FAIL: exception did not mention outputs: {e}")
            is_ok = False

    # --- actor integration: contradicting local_axes_order fails loudly -
    # (checked against the actor's configuration phase, since a GATE
    # SimulationEngine can only run once per process; this is also the phase
    # in which a real run would reject the contradiction, before Geant4 starts)
    sim2, _, _ = create_simulation(paths, generator=str(bundle_dir_v2))
    og2 = sim2.actor_manager.get_actor("SyntheticPhotons")
    # contradicts the bundle's [0, 1, 2]; also not the default, which would be
    # indistinguishable from not having set it at all
    og2.local_axes_order = [2, 1, 0]
    try:
        og2.resolve_and_validate_config()
        print("FAIL: expected an exception for a contradicting local_axes_order")
        is_ok = False
    except Exception as e:
        if "local_axes_order" in str(e):
            print(f"OK: contradicting local_axes_order raised a clear error: {e}")
        else:
            print(f"FAIL: exception did not mention local_axes_order: {e}")
            is_ok = False

    tu.test_ok(is_ok)
