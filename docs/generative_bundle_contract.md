# Generative model bundle contract

This document defines the interface that a generative optical-photon model must implement to be used as the `generator` of a `DigitizerOpticalGenerativeActor` via a bundle path, and the on-disk layout `GenerativeModelBundle` (`opengate/contrib/optical/generative_bundle.py`) expects when loading one.

Status: `schema_version: 1`.

## What the actor gives you and what it expects back

`DigitizerOpticalGenerativeActor` does not ask a model to invent optical photons out of nothing. For every energy-deposition hit in the input digi collection, the actor itself first draws a Poisson-distributed photon count from the hit's deposited energy and the crystal's `SCINTILLATIONYIELD` (read from `OpticalProperties.xml`). Only then does it call the model once, with that count, to ask what those photons look like on arrival. So the model is responsible for the *shape* of the light (position, direction, energy, timing), not the *yield*.

Concretely, the actor calls:

```python
X, Y, dX, dY, dZ, Ekine, Time = generate_batch(x, y, z, time, n_photons)
```

once per hit, batching all `n_photons` of that hit into a single call.

### Inputs

| name | type | units | meaning |
|---|---|---|---|
| `x, y, z` | float | mm | energy-deposition position, in the crystal-local frame, axis order and origin as declared in `coordinates` (see below) |
| `time` | float | ns | global time of the energy deposition |
| `n_photons` | int | n/a | number of photons to generate for this hit, already sampled by the actor; always ≥ 1 |

A model that was never trained on timing information is allowed to ignore `time` internally, but the function signature must still accept it, the actor always passes five arguments.

### Outputs

Seven one-dimensional arrays, each of length `n_photons`:

| name | units | meaning |
|---|---|---|
| `X, Y` | mm | photon position at the point the model represents (typically the sensor face), in the same frame as the input `x, y, z` |
| `dX, dY, dZ` | n/a | direction vector at that point; should be unit length |
| `Ekine` | MeV | photon kinetic energy |
| `Time` | ns | **relative** delay of the photon after the hit's `time`, i.e. `Time >= 0` |

The actor adds the hit's own global time to `Time` afterwards. A model must never add the input `time` into its own output.

`X, Y, dX, dY, dZ, Ekine, Time` must all be plain 1-D sequences of length `n_photons` that `np.asarray(..., dtype=np.float64)` can consume without error (e.g. NumPy arrays, PyTorch tensors, Python lists).

## Bundle layout

A bundle is either a directory or a zip file containing a manifest and the serialized model it describes:

```
my_bundle/
  manifest.json
  model.pt          # or model.onnx, or a directory of AOTInductor artifacts
```

`GenerativeModelBundle` opens whichever of the two you point it at
(`GenerativeModelBundle("path/to/my_bundle")` or `GenerativeModelBundle("path/to/my_bundle.zip")`) and reads `manifest.json` from the root in both cases. Nothing else in the bundle is inspected by name. `model_file` in the manifest is what tells the loader which file to open.

### manifest.json

```json
{
  "schema_version": 1,
  "format": "torchscript",
  "model_file": "model.pt",
  "coordinates": {
    "axes_order": ["depth", "transverse", "axial"],
    "origin": "crystal_center",
    "units": "mm"
  },
  "training": {
    "crystal_size_mm": [3.0, 3.0, 20.0],
    "material": "BGO"
  },
  "batch": {
    "dynamic": true,
    "max_batch": null
  }
}
```

**`schema_version`**: must be `1`. Any other value is rejected at load time.

**`format`**: one of `"torchscript"`, `"onnx"`, `"aotinductor"`. Selects which loader is used; see below.

**`model_file`**: path to the model file, relative to the bundle root (or to the directory, for `aotinductor`, which cannot be loaded from inside a zip).

**`coordinates`**: how to interpret and produce the position axes:

- `axes_order`: a permutation of `["depth", "transverse", "axial"]`. GATE's own crystal-local frame (`PostPositionLocal` / `PostPositionLocalModule`) is `(x=depth, y=transverse, z=axial)`. If your model was trained with a different ordering, declare it (`["transverse", "axial", "depth"]`) and the actor will permute GATE's coordinates into that order before calling your model, and permute the model's `X, Y` output back. You do not need to set `local_axes_order` on the actor yourself when using a bundle; it is derived from this field.
- `origin`: where `(0, 0, 0)` sits along the depth axis. `"crystal_center"` means depth runs from `-d/2` to `+d/2`, matching GATE's default local frame directly. `"sensor_face"` means depth runs from `0` at the sensor to `d` at the entrance face; the actor computes and applies the corresponding offset for you, using `training.crystal_size_mm`. Anything else is accepted as a free-text note but is treated as no offset.
- `units`: currently always `"mm"`. Present for forward compatibility, not read by the loader today.

**`training`**: descriptive metadata about what the model was trained on. `crystal_size_mm` (a `[depth, transverse, axial]`-ordered triple, matching `axes_order`) is used to compute the `sensor_face` offset above; `material` is not consumed by the loader but should be kept accurate, since it is the first thing to check when a model's output looks physically wrong for the crystal it is being applied to.

**`batch`**: describes the model's batching behavior. `dynamic: true` means the model accepts an arbitrary `n_photons` per call, which is the normal case since hit multiplicities vary. `max_batch` can record an upper bound the model was validated up to, if one exists; leave it `null` if there isn't one. Neither field changes actor behavior today; they exist so a model's batching limits are documented next to the model itself rather than only in whoever trained it's memory.

### Choosing a format

- **torchscript**: a model saved with `torch.jit.script(...).save(...)` or `torch.jit.trace(...).save(...)`. Loaded with `torch.jit.load` and run under `torch.no_grad()`. Requires `torch` to be installed; the import is
  deferred to load time, so bundles using other formats do not need it.
- **onnx**: a model exported to ONNX. Loaded with `onnxruntime.InferenceSession`, preferring the CUDA execution provider and falling back to the CPU (see [Running on the GPU](#running-on-the-gpu)). Requires `onnxruntime`. Unlike the other two formats, inputs are passed by name, not by position; see below.
- **aotinductor**: a model compiled with `torch._export.aot_compile`. Loaded with `torch._export.aot_load`. Requires a directory bundle: the compiled `.so` cannot be read out of a zip, so an `aotinductor` bundle passed as a zip path is rejected at load time.

### A minimal torchscript example

The smallest model that satisfies the contract, one that just echoes its inputs back, useful as a starting point or as a fixture in tests:

```python
import torch
import torch.nn as nn
from typing import Tuple

class TinyGenerator(nn.Module):
    def forward(
        self, x: float, y: float, z: float, time: float, n_photons: int
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        X = torch.full((n_photons,), x)
        Y = torch.full((n_photons,), y)
        dX = torch.zeros(n_photons)
        dY = torch.zeros(n_photons)
        dZ = torch.ones(n_photons)
        Ekine = torch.full((n_photons,), 3.0)      # MeV
        Time = torch.ones(n_photons)               # 1 ns delay
        return X, Y, dX, dY, dZ, Ekine, Time

model = torch.jit.script(TinyGenerator())
model.save("my_bundle/model.pt")
```

paired with the `manifest.json` shown above (with `"model_file": "model.pt"` and `"format": "torchscript"`).

Note the signature: five separate scalar arguments (`x, y, z, time, n_photons`), not a single batched tensor. `GenerativeModelBundle` calls torchscript and aotinductor models this way:
`model(float(x), float(y), float(z), float(time), int(n_photons))`.
The model itself is responsible for broadcasting into `n_photons`-length output, as in the example above.

### Using an ONNX model

Switching an existing bundle from torchscript to ONNX is mostly a matter of changing two fields in `manifest.json`:

```json
{
  "format": "onnx",
  "model_file": "model.onnx"
}
```

with `coordinates`, `training`, and `batch` left as they are. `onnxruntime` is only imported when a bundle declares `format: "onnx"`, so a pure-ONNX bundle does not require `torch` to be installed at all.

The one part that is not a drop-in swap is the calling convention. Where torchscript and aotinductor models are called positionally, `GenerativeModelBundle` calls an ONNX model with a `feeds` dict (`generative_bundle.py`, `_generate_batch_onnx`):

```python
feeds = {
    "x": np.array(x, dtype=np.float32),
    "y": np.array(y, dtype=np.float32),
    "z": np.array(z, dtype=np.float32),
    "time": np.array(time, dtype=np.float32),
    "n_photons_carrier": np.zeros(n_photons, dtype=np.int64),
}
session.run(None, feeds)
```

so your exported graph's input names must be exactly `x`, `y`, `z`, `time`, and `n_photons_carrier`; `onnxruntime` binds by name, and there is no positional fallback. `n_photons_carrier` does not carry any real data: it is a zero array of length `n_photons`, present purely so the batch size is available inside the graph (for example through a `Shape` node), since a bare Python `int` cannot be passed to an ONNX graph the way it can to a scripted `forward()`.

Your graph also needs a dynamic batch dimension on its outputs so it can produce `n_photons`-length arrays for whatever `n_photons` a given hit needs, since hit multiplicities vary across a run. If you export from PyTorch, that means listing the batch dimension in `dynamic_axes`:

```python
torch.onnx.export(
    model,
    (x, y, z, time, n_photons_carrier),
    "model.onnx",
    input_names=["x", "y", "z", "time", "n_photons_carrier"],
    output_names=["X", "Y", "dX", "dY", "dZ", "Ekine", "Time"],
    dynamic_axes={name: {0: "n_photons"} for name in
                  ["n_photons_carrier", "X", "Y", "dX", "dY", "dZ", "Ekine", "Time"]},
)
```

A model exported without `dynamic_axes` will only run for the exact batch size it was traced with, which `batch.dynamic: true` in the manifest is meant to document but does not itself enforce.

## Running on the GPU

Inference is the expensive part of a run that uses a generative bundle, so it is worth checking that it actually runs where you think it does. Both backends prefer the GPU and fall back to the CPU rather than failing, so a misconfigured environment costs speed, not correctness.

`GenerativeModelBundle` warns when it ends up on the CPU, naming the reason, so you do not have to go looking. If you see no such warning, the model is on the GPU.

**torchscript / aotinductor** use the GPU when `torch.cuda.is_available()`; if not, install a CUDA build of torch.

**onnx** needs `onnxruntime-gpu` (the plain `onnxruntime` package has no CUDA provider at all) *and* needs onnxruntime to find its CUDA and cuDNN shared libraries at load time. The second part is the one that usually bites: unlike torch, which locates the libraries it ships with, onnxruntime searches only the loader path. So an environment where `torch.cuda.is_available()` is `True` can still run ONNX on the CPU, and the only sign is an onnxruntime log line about a library it could not open (e.g. `libcublasLt.so.13`).

If torch is installed with CUDA support, those libraries are usually already in site-packages and only need to be on the path:

```bash
SP=$(python -c 'import site;print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$SP/nvidia/cu13/lib:$SP/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

Adjust `cu13` to the CUDA major version your `onnxruntime-gpu` was built against; the [onnxruntime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) table lists which versions pair with which. To make it permanent for a conda environment, put those two lines in `$CONDA_PREFIX/etc/conda/activate.d/`.

## Using a bundle

Once you have a bundle directory or zip on disk, point the actor at it directly. There is no need to construct `GenerativeModelBundle` yourself unless you want to inspect or call it standalone:

```python
og = sim.add_actor("DigitizerOpticalGenerativeActor", "SyntheticPhotons")
og.attached_to = crystal.name
og.input_digi_collection = hits_actor.name
og.generator = "path/to/my_bundle"   # or "path/to/my_bundle.zip"
```

`local_axes_order` and `local_position_offset` are then derived from the bundle's manifest automatically. Setting either explicitly is only needed if you are passing a plain Python object as `generator` instead of a bundle path; if you do set them on an actor that also uses a bundle, they must match what the manifest implies, or simulation setup fails with an error naming the mismatch.

## Validation

Loading a bundle checks, in order: that `manifest.json` exists (directory or zip); that `schema_version` is `1`; that `format` is one of the three supported values; that `model_file` is present in the manifest and the file it names exists in the bundle; that `coordinates`, `training`, and `batch` sections are all present; and that `coordinates.axes_order` is a valid permutation of `depth`, `transverse`, `axial`. Any failure raises with a message naming the bundle path and the specific problem, rather than failing later inside a forward pass.

### When validation happens

When a bundle is used through `DigitizerOpticalGenerativeActor`, these checks run in the actor's `resolve_and_validate_config` phase, together with the reconciliation of `local_axes_order` / `local_position_offset` against the manifest and the lookup of the crystal's `SCINTILLATIONYIELD`. That phase runs before any Geant4 object exists, so a misconfigured bundle is rejected before the simulation engine starts — and, for a split simulation, before the run is packaged into jobs.

The model file itself is deliberately *not* deserialized in that phase: the actor constructs the bundle with `defer_model_load=True` and calls `load_model()` in `StartSimulationAction`, so importing torch/onnxruntime and opening a GPU session happens only when a simulation actually runs. Constructing a `GenerativeModelBundle` directly, without that flag, still loads the model immediately, as before.
