#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import shutil

import opengate.tests.utility as tu
from opengate.contrib.optical.generative_bundle import GenerativeModelBundle

from test110_generative_bundle_helpers import (
    TORCH_AVAILABLE,
    make_torchscript_bundle,
    check_generate_batch_output,
)

if __name__ == "__main__":
    paths = tu.get_default_test_paths(__file__, output_folder="test110")
    paths.output.mkdir(parents=True, exist_ok=True)

    if not TORCH_AVAILABLE:
        print("torch is not installed, skipping test110 (nothing to check).")
        tu.test_ok(True)
        raise SystemExit(0)

    is_ok = True

    # --- a valid bundle loads and generate_batch works -----------------
    bundle_dir = paths.output / "bundle_valid"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    make_torchscript_bundle(bundle_dir)

    bundle = GenerativeModelBundle(bundle_dir)
    x, y, z, time, n_photons = 1.0, 2.0, 3.0, 100.0, 5
    X, Y, dX, dY, dZ, Ekine, Time = bundle.generate_batch(x, y, z, time, n_photons)
    is_ok &= check_generate_batch_output(X, Y, dX, dY, dZ, Ekine, Time, n_photons, x, y)

    # default coordinates convention: identity axes order, no offset
    expected_axes = [0, 1, 2]
    if bundle.expected_local_axes_order() != expected_axes:
        print(
            f"FAIL: expected_local_axes_order() = "
            f"{bundle.expected_local_axes_order()}, expected {expected_axes}"
        )
        is_ok = False
    else:
        print("OK: expected_local_axes_order() matches identity manifest convention")

    # --- a bundle with a permuted axes_order --------------------------
    bundle_dir_permuted = paths.output / "bundle_permuted"
    if bundle_dir_permuted.exists():
        shutil.rmtree(bundle_dir_permuted)
    bundle_dir_permuted.mkdir(parents=True)
    make_torchscript_bundle(
        bundle_dir_permuted, axes_order=["transverse", "axial", "depth"]
    )
    bundle_permuted = GenerativeModelBundle(bundle_dir_permuted)
    expected_permuted = [1, 2, 0]
    if bundle_permuted.expected_local_axes_order() != expected_permuted:
        print(
            f"FAIL: expected_local_axes_order() = "
            f"{bundle_permuted.expected_local_axes_order()}, "
            f"expected {expected_permuted}"
        )
        is_ok = False
    else:
        print("OK: expected_local_axes_order() matches permuted manifest convention")

    # --- sensor_face origin implies a depth offset ----------------------
    bundle_dir_sensor = paths.output / "bundle_sensor_face"
    if bundle_dir_sensor.exists():
        shutil.rmtree(bundle_dir_sensor)
    bundle_dir_sensor.mkdir(parents=True)
    make_torchscript_bundle(bundle_dir_sensor, origin="sensor_face")
    bundle_sensor = GenerativeModelBundle(bundle_dir_sensor)
    expected_offset = [1.5, 0.0, 0.0]  # crystal_size_mm[0] / 2, depth is axis 0
    if bundle_sensor.expected_local_position_offset() != expected_offset:
        print(
            f"FAIL: expected_local_position_offset() = "
            f"{bundle_sensor.expected_local_position_offset()}, "
            f"expected {expected_offset}"
        )
        is_ok = False
    else:
        print("OK: expected_local_position_offset() matches sensor_face convention")

    # --- missing manifest.json fails loudly -----------------------------
    bundle_dir_missing = paths.output / "bundle_missing_manifest"
    if bundle_dir_missing.exists():
        shutil.rmtree(bundle_dir_missing)
    bundle_dir_missing.mkdir(parents=True)
    try:
        GenerativeModelBundle(bundle_dir_missing)
        print("FAIL: expected an exception for a bundle with no manifest.json")
        is_ok = False
    except Exception as e:
        if "manifest.json" in str(e):
            print(f"OK: missing manifest.json raised a clear error: {e}")
        else:
            print(f"FAIL: exception did not mention manifest.json: {e}")
            is_ok = False

    # --- unknown format fails loudly ------------------------------------
    bundle_dir_bad_format = paths.output / "bundle_bad_format"
    if bundle_dir_bad_format.exists():
        shutil.rmtree(bundle_dir_bad_format)
    bundle_dir_bad_format.mkdir(parents=True)
    manifest = make_torchscript_bundle(bundle_dir_bad_format)
    manifest["format"] = "unsupported_format"
    with open(bundle_dir_bad_format / "manifest.json", "w") as f:
        json.dump(manifest, f)
    try:
        GenerativeModelBundle(bundle_dir_bad_format)
        print("FAIL: expected an exception for an unknown format")
        is_ok = False
    except Exception as e:
        if "format" in str(e):
            print(f"OK: unknown format raised a clear error: {e}")
        else:
            print(f"FAIL: exception did not mention format: {e}")
            is_ok = False

    tu.test_ok(is_ok)
