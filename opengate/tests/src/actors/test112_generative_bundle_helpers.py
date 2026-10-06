#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.util
import json

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def make_torchscript_bundle(bundle_dir):
    """
    Builds a TorchScript bundle in bundle_dir. The model
    returns five columns - two position components, energy and time, plus a
    constant depth declared in the manifest rather than produced by the model -
    so the test exercises 'value' entries and a column count that differs from
    the number of declared outputs.

    Only called if TORCH_AVAILABLE is True.
    """
    import torch
    import torch.nn as nn
    from typing import Tuple

    class TinyGenerator(nn.Module):
        def forward(
            self, x: float, y: float, z: float, time: float, n_photons: int
        ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            pos_t = torch.full((n_photons,), x)
            pos_a = torch.full((n_photons,), y)
            # a direction straight along +depth, as two of its components are
            # constants in the manifest
            d_depth = torch.ones(n_photons)
            ekine = torch.full((n_photons,), 3.0)  # eV, declared as such
            delay = torch.ones(n_photons)  # 1 ns after the hit
            return pos_t, pos_a, d_depth, ekine, delay

    model = torch.jit.script(TinyGenerator())
    model_file = "model.pt"
    model.save(str(bundle_dir / model_file))

    manifest = _manifest(model_file)
    # this model takes one position per call, so the actor calls it once per hit
    manifest["batch"] = {"max_batch": None, "input_mode": "scalar"}
    with open(bundle_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    return manifest


def _manifest(model_file):
    """The manifest both bundles share, except for the 'batch' section."""
    outputs = [
        {"attribute": "PostPositionLocalModule", "component": "y", "unit": "mm"},
        {"attribute": "PostPositionLocalModule", "component": "z", "unit": "mm"},
        {"attribute": "Direction", "component": "x", "unit": "1"},
        {"attribute": "KineticEnergy", "unit": "eV"},
        {
            "attribute": "GlobalTime",
            "unit": "ns",
            "semantics": "relative_to_hit",
        },
        # not produced by the model: photons leave at the sensor face,
        # 5 mm from the center of a 10 mm deep module
        {
            "attribute": "PostPositionLocalModule",
            "component": "x",
            "unit": "mm",
            "value": -5.0,
        },
        {"attribute": "Direction", "component": "y", "unit": "1", "value": 0.0},
        {"attribute": "Direction", "component": "z", "unit": "1", "value": 0.0},
        # the same position again, taken back into the world frame by the
        # actor rather than produced by the model
        {"attribute": "PostPosition", "derived": "world_from_module"},
    ]

    return {
        "schema_version": 3,
        "format": "torchscript",
        "model_file": model_file,
        "outputs": outputs,
        "coordinates": {
            "axes_order": ["depth", "transverse", "axial"],
            "offset": [-5.0, 0.0, 0.0],
            "unit": "mm",
        },
        "training": {"crystal_size_mm": [10.0, 3.0, 3.0], "material": "BGO"},
    }


def make_vector_torchscript_bundle(bundle_dir, target_batch=4096):
    """
    Same model and same output schema as make_torchscript_bundle, but with
    per-photon conditioning, so the actor pools hits into one call. Both
    bundles produce the same photons for the same hits, which is what
    test112 compares.

    Only called if TORCH_AVAILABLE is True.
    """
    import torch
    import torch.nn as nn
    from typing import Tuple

    class TinyVectorGenerator(nn.Module):
        def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            z: torch.Tensor,
            time: torch.Tensor,
        ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            # the conditioning already has one entry per photon, so the
            # columns are just passed through instead of being broadcast
            n = x.shape[0]
            pos_t = x
            pos_a = y
            d_depth = torch.ones(n)
            ekine = torch.full((n,), 3.0)  # eV, declared as such
            delay = torch.ones(n)  # 1 ns after the hit
            return pos_t, pos_a, d_depth, ekine, delay

    model = torch.jit.script(TinyVectorGenerator())
    model_file = "model.pt"
    model.save(str(bundle_dir / model_file))

    # same schema as the scalar bundle, so the two runs are comparable
    manifest = _manifest(model_file)
    manifest["batch"] = {
        "max_batch": None,
        "input_mode": "vector",
        "target_batch": target_batch,
    }
    with open(bundle_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    return manifest
