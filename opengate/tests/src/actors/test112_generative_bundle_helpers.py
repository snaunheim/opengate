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

    manifest = {
        "schema_version": 2,
        "format": "torchscript",
        "model_file": model_file,
        "outputs": outputs,
        "coordinates": {
            "axes_order": ["depth", "transverse", "axial"],
            "offset": [-5.0, 0.0, 0.0],
            "unit": "mm",
        },
        "training": {"crystal_size_mm": [10.0, 3.0, 3.0], "material": "BGO"},
        "batch": {"max_batch": None},
    }
    with open(bundle_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    return manifest
