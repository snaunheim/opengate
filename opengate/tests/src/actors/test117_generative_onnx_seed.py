#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
An ONNX model that draws noise in its graph gives the same photons for the
same GATE seed and other photons for another seed. Two actors and two runs
per simulation, as both share onnxruntime's generator. A run with two
threads must still work, but is not compared.

The model is written by hand, since torch.onnx.export needs the onnx package.
"""

import importlib.util
import json
import shutil
import struct

import numpy as np

import opengate as gate
import opengate.tests.utility as tu
from opengate.contrib.optical.generative_bundle import GenerativeModelBundle

from test113_pooled_helpers import read_tree
from og_actor109_pet_sim_setup import create_simulation

ORT_AVAILABLE = importlib.util.find_spec("onnxruntime") is not None
ACTORS = ("SyntheticPhotons", "SyntheticPhotons2")


# -- a minimal protobuf writer, enough for an ONNX model -------------------


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | 0x80 if n else b)
        if not n:
            return bytes(out)


def _int(field, n):
    return _varint(field << 3) + _varint(n)


def _bytes(field, b):
    b = b.encode() if isinstance(b, str) else b
    return _varint(field << 3 | 2) + _varint(len(b)) + b


def _node(op, inputs, outputs):
    b = b"".join(_bytes(1, i) for i in inputs)
    b += b"".join(_bytes(2, o) for o in outputs)
    return _bytes(1, b + _bytes(4, op))


def _float_vector(field, name):
    # a float tensor of shape [n]
    shape = _bytes(1, _bytes(2, "n"))
    tensor_type = _int(1, 1) + _bytes(2, shape)
    return _bytes(field, _bytes(1, name) + _bytes(2, _bytes(1, tensor_type)))


def noise_model():
    """(x, y, z, time) -> (y + noise, z + noise, |noise|), RandomNormalLike without a seed"""
    graph = b"".join(
        [
            _node("RandomNormalLike", ["x"], ["n1"]),
            _node("RandomNormalLike", ["x"], ["n2"]),
            _node("RandomNormalLike", ["x"], ["n3"]),
            _node("Add", ["y", "n1"], ["pos_t"]),
            _node("Add", ["z", "n2"], ["pos_a"]),
            _node("Abs", ["n3"], ["delay"]),
            _bytes(2, "noise"),
            *[_float_vector(11, k) for k in ("x", "y", "z", "time")],
            *[_float_vector(12, k) for k in ("pos_t", "pos_a", "delay")],
        ]
    )
    opset = _bytes(1, "") + _int(2, 17)
    return _int(1, 8) + _bytes(8, opset) + _bytes(7, graph)


def make_onnx_noise_bundle(bundle_dir):
    (bundle_dir / "model.onnx").write_bytes(noise_model())
    manifest = {
        "schema_version": 3,
        "format": "onnx",
        "model_file": "model.onnx",
        "outputs": [
            {"attribute": "PostPositionLocalModule", "component": "y", "unit": "mm"},
            {"attribute": "PostPositionLocalModule", "component": "z", "unit": "mm"},
            {"attribute": "GlobalTime", "unit": "ns", "semantics": "relative_to_hit"},
            {
                "attribute": "PostPositionLocalModule",
                "component": "x",
                "unit": "mm",
                "value": -5.0,
            },
            {"attribute": "Direction", "component": "x", "unit": "1", "value": 1.0},
            {"attribute": "Direction", "component": "y", "unit": "1", "value": 0.0},
            {"attribute": "Direction", "component": "z", "unit": "1", "value": 0.0},
            {"attribute": "KineticEnergy", "unit": "eV", "value": 3.0},
            {"attribute": "PostPosition", "derived": "world_from_module"},
        ],
        "coordinates": {
            "axes_order": ["depth", "transverse", "axial"],
            "offset": [0.0, 0.0, 0.0],
            "unit": "mm",
        },
        "training": {"crystal_size_mm": [10.0, 3.0, 3.0], "material": "BGO"},
        "batch": {"input_mode": "vector", "target_batch": 4096},
    }
    with open(bundle_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)


# -- the runs ----------------------------------------------------------------


def run(paths, bundle_dir, name, seed, n_threads=1, opened=False):
    sec = gate.g4_units.s
    sim, _, _ = create_simulation(paths, generator=str(bundle_dir), n_modules=4)
    sim.random_seed = seed
    sim.number_of_threads = n_threads
    sim.run_timing_intervals = [[0, 0.025 * sec], [0.025 * sec, 0.05 * sec]]
    sim.actor_manager.get_actor("Hits").output_filename = f"hits_{name}.root"
    og = sim.actor_manager.get_actor("SyntheticPhotons")
    og2 = sim.add_actor("DigitizerOpticalGenerativeActor", "SyntheticPhotons2")
    og2.attached_to = og.attached_to
    og2.authorize_repeated_volumes = True
    og2.input_digi_collection = og.input_digi_collection
    for actor in (og, og2):
        actor.output_filename = f"{actor.name}_{name}.root"
        # built now, so the session exists before the seed is known
        if opened:
            actor.generator = GenerativeModelBundle(bundle_dir)
        else:
            actor.generator = str(bundle_dir)
    # the last run uses this process, so CUDA is only touched after the forks
    sim.run(start_new_process=not opened)
    return {a: read_tree(paths.output / f"{a}_{name}.root") for a in ACTORS}


def bitwise_equal(a, b):
    return a.keys() == b.keys() and all(
        a[k].dtype == b[k].dtype and a[k].tobytes() == b[k].tobytes() for k in a
    )


if __name__ == "__main__":
    paths = tu.get_default_test_paths(
        __file__, output_folder="test117_generative_onnx_seed"
    )
    paths.output.mkdir(parents=True, exist_ok=True)

    if not ORT_AVAILABLE:
        print("onnxruntime is not installed, skipping test117 (nothing to check).")
        tu.test_ok(True)
        raise SystemExit(0)

    bundle_dir = paths.output / "bundle"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    make_onnx_noise_bundle(bundle_dir)

    a1 = run(paths, bundle_dir, "a1", seed=123)
    a2 = run(paths, bundle_dir, "a2", seed=123)
    b = run(paths, bundle_dir, "b", seed=124)
    mt = run(paths, bundle_dir, "mt", seed=123, n_threads=2)
    opened = run(paths, bundle_dir, "opened", seed=123, opened=True)

    is_ok = True
    for actor in ACTORS:
        photons = a1[actor]
        n = len(photons["SourceHitIndex"])
        n_hits = len(np.unique(photons["SourceHitIndex"]))
        # photons of one hit get different positions only through the noise
        n_y = len(np.unique(photons["PostPositionLocalModule_Y"]))
        ok = n > 0 and n_y > n_hits
        print(f"{'OK' if ok else 'FAIL'}: {actor}, {n} photons of {n_hits} hits, {n_y} distinct y")
        is_ok &= ok

        ok = bitwise_equal(photons, a2[actor])
        print(f"{'OK' if ok else 'FAIL'}: {actor}, same seed gives the same photons")
        is_ok &= ok

        ok = not np.array_equal(
            photons["PostPositionLocalModule_Y"], b[actor]["PostPositionLocalModule_Y"]
        )
        print(f"{'OK' if ok else 'FAIL'}: {actor}, another seed gives other photons")
        is_ok &= ok

        ok = len(mt[actor]["SourceHitIndex"]) > 0
        print(f"{'OK' if ok else 'FAIL'}: {actor}, two threads give photons")
        is_ok &= ok

        ok = bitwise_equal(photons, opened[actor])
        print(f"{'OK' if ok else 'FAIL'}: {actor}, opened bundle gives the same photons")
        is_ok &= ok

    tu.test_ok(is_ok)
