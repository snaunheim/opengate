#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.util
import json

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def make_torchscript_bundle(bundle_dir, axes_order=None, origin="crystal_center"):
    """
    Builds a tiny TorchScript model bundle (manifest.json + model file) in
    bundle_dir, conforming to docs/generative_bundle_contract.md. The model
    takes (x, y, z, time) per photon and deterministically returns
    (X, Y, dX, dY, dZ, Ekine, Time) so tests can check values exactly.

    Only called if TORCH_AVAILABLE is True.
    """
    import torch
    import torch.nn as nn

    if axes_order is None:
        axes_order = ["depth", "transverse", "axial"]

    class TinyGenerator(nn.Module):
        def forward(self, xyzt):
            x, y, z, t = xyzt[:, 0], xyzt[:, 1], xyzt[:, 2], xyzt[:, 3]
            n = xyzt.shape[0]
            X = x
            Y = y
            dX = torch.zeros(n)
            dY = torch.zeros(n)
            dZ = torch.ones(n)
            Ekine = torch.full((n,), 3.0)
            Time = torch.ones(n)  # 1 ns delay, relative to hit time
            return torch.stack([X, Y, dX, dY, dZ, Ekine, Time], dim=1)

    model = torch.jit.script(TinyGenerator())
    model_file = "model.pt"
    model.save(str(bundle_dir / model_file))

    manifest = {
        "schema_version": 1,
        "format": "torchscript",
        "model_file": model_file,
        "coordinates": {
            "axes_order": axes_order,
            "origin": origin,
            "units": "mm",
        },
        "training": {
            "crystal_size_mm": [3.0, 3.0, 20.0],
            "material": "BGO",
        },
        "batch": {"dynamic": True, "max_batch": None},
    }
    with open(bundle_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    return manifest


def check_generate_batch_output(X, Y, dX, dY, dZ, Ekine, Time, n_photons, x, y):
    ok = True
    for name, arr in (
        ("X", X),
        ("Y", Y),
        ("dX", dX),
        ("dY", dY),
        ("dZ", dZ),
        ("Ekine", Ekine),
        ("Time", Time),
    ):
        arr = np.asarray(arr)
        if arr.shape != (n_photons,):
            print(f"FAIL: {name} has shape {arr.shape}, expected ({n_photons},)")
            ok = False

    if not np.allclose(X, x):
        print(f"FAIL: X={X} does not match input x={x}")
        ok = False
    if not np.allclose(Y, y):
        print(f"FAIL: Y={Y} does not match input y={y}")
        ok = False
    if np.any(np.asarray(Time) <= 0):
        print("FAIL: Time contains non-positive values")
        ok = False

    if ok:
        print(f"OK: generate_batch returned {n_photons} well-formed photon records")
    return ok
