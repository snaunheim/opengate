# Generative model bundle contract

This document provides detailed information about the usage of the `DigitizerOpticalGenerativeActor` in combination with a generative model.

Status: `schema_version: 2`.

## General Information about the Actor
The `DigitizerOpticalGenerativeActor` aims to replace time-consuming optical photon tracking with a faster alternative based on generative AI models. Instead of tracking the particle through the sensitive volume until they reach a sensor surface (e.g., a SiPM), the model allows direct generation of individual optical photons being detected at the sensor. To do so, the model is conditioned on the spatial location an energy-deposition has occured inside the sensitive volume. In order to collect the relevant information about hits recorded in the senstive volume the `DigitizerHitsCollectionActor`can be used before the `DigitizerOpticalGenerativeActor`.

### Reference Frame
To correctly reproduce light-sharing characteristics and similar effects, a generative model typically replaces the optical photon transport of the whole detector, which might be an array of multiple scintillation crystals. The volume which encloses the whole detector is in the following text called module. Spatial postions that are exchanged between the model and the actor are given in the **module-local** frame, with the origin being in the iso-center of the module. For an array with N*N scintillation crystals and N being an odd-number, the module's iso-center therefore coincides with the iso-center of the central crystal. Even if the model represents a monolithic detector, the user needs to define the module volume which might be matching with the crystal volume. The reason is that the actor requires `PostPositionLocalModule`on its input collection in order to extract the condition forwarded to the model.

### Inputs

| name | type | units | meaning |
|---|---|---|---|
| `x, y, z` | float | as declared in `coordinates.unit` | energy-deposition position, in the array-local frame, axis order and offset as declared in `coordinates` (see below) |
| `time` | float | ns | global time of the energy deposition |
| `n_photons` | int | n/a | number of photons being emitted for this hit, already sampled by the actor; always ≥ 1 |

A model that was never trained on timing information is allowed to ignore `time` internally, but the function signature must still accept it, the actor always passes five arguments.

### Outputs

The model returns multiple one-dimensional arrays, each 1-D sequence should be in a form such that `np.asarray(..., dtype=np.float64)` can consume without error. An array represents a specific quantity (e.g., detected X location), and one row represents one optical photon. The output that GATE produces and how it interacts with the model is defined in the manifest's `outputs` list. The number of `outputs` entries (that do not declare a constant `value`) must equal to number of columns return by the model.

The output digi collection carries the declared attributes under their GATE names (so a `PostPositionLocalModule` entry appears as the branches `PostPositionLocalModule_X/_Y/_Z`), plus `SourceHitIndex`, the index of the input row that produced each photon.

## Bundle Layout

A bundle is either a directory or a zip file containing a manifest and the serialized model it describes.
```
my_bundle/
  manifest.json
  model.onnx          # or model.onnx, or a directory of AOTInductor artifacts
```

`GenerativeModelBundle` opens the directory or zip file and reads `manifest.json` from the root. The entry `model_file` in the manifest declares the model file that should be used.

### manifest.json
This is an example.
```json
{
  "schema_version": 2,
  "format": "onnx",
  "model_file": "model.onnx",

  "outputs": [
    {"attribute": "PostPositionLocalModule", "component": "x", "unit": "mm", "offset": 0.0},
    {"attribute": "PostPositionLocalModule", "component": "y", "unit": "mm", "offset": 0.0},
    {"attribute": "Direction",               "component": "x", "unit": "1"},
    {"attribute": "Direction",               "component": "y", "unit": "1"},
    {"attribute": "Direction",               "component": "z", "unit": "1"},
    {"attribute": "KineticEnergy", "unit": "eV"},
    {"attribute": "GlobalTime",    "unit": "ns", "semantics": "relative_to_hit"},

    {"attribute": "PostPositionLocalModule", "component": "z", "unit": "mm", "value": -5.0}
  ],

  "coordinates": {
    "axes_order": ["depth", "transverse", "axial"],
    "offset": [0.0, 0.0, -5.0],
    "unit": "mm"
  },
  "training": {"crystal_size_mm": [3.0, 3.0, 10.0], "material": "BGO", "module_configuration":"3x3"},

  "batch": {"max_batch": null}
}
```

**`schema_version`**: The current version is `2`.

**`format`**: Needs to be either `"torchscript"`, `"onnx"`, or `"aotinductor"`. It selects which loader is used. Attention: Only ONNX shipped models have been tested extensively so far.

**`model_file`**: Path to the model file, relative to the bundle root.

**`outputs`**: Defines which quantites the model/actor produces. There might be additional quantites that are not directly generated by the model but derived from other outputs of the model. In general the model output might be a subset of the outputs defined here.

**`coordinates`**: Defines the interpretation of position axis.
- `axes_order`: Declares the permutation of `["depth", "transverse", "axial"]`. GATE's own module-local frame (`PostPositionLocalModule`) is `(x=depth, y=transverse, z=axial)`. If your model was trained with a different ordering, declare it (`["transverse", "axial", "depth"]`) and the actor will permute GATE's coordinates into that order before calling your model. `axes_order` describes only the input side, it has no effect on `outputs`.
- `offset`: Origin offset in the model's own axis order, each meaning *the value added to that coordinate so that (0,0,0) lies at the module center*. A model whose depth axis starts at the sensor surface of a 10 mm deep module declares `-5.0` on its depth axis. 
- `unit`: Declares the unit the coordinates axes, as a name from `opengate.g4_units`.

**`training`**: Descriptive metadata of the model that is being used. This is propably the first thing to cross check if the model's output doesn't match with prior physics expectation.

**`batch`**:
- `max_batch`: Defines the maximum number of synthetic optical photons being created in a single call. This is especially usefull if your GPU memory is limited and you want to prevent out-of-memory errors. However, it will increase the time that is needed to run the simulation since each call carries a significant amount of overhead. Setting the value to `null` allows the model to do the inference without a limit.

### The Outputs List
`outputs` declares the columns being used. **The order of the list is the order in which the model returns its columns**, so no second mapping table is needed. Names and types are checked against `GateDigiAttributeManager` itself, not against a list maintained in OpenGATE, so the schema cannot go stale when GATE gains new attributes.

| field | required | meaning |
|---|---|---|
| `attribute` | yes | name of a GATE digi attribute |
| `unit` | yes | unit in which the model delivers this quantity |
| `component` | only for 3-vectors | `"x"`, `"y"` or `"z"` |
| `offset` | no | added to the value, in the same unit as the quantity; see `coordinates.offset` for the sign convention |
| `value` | no | a fixed value instead of a model column, see below |
| `semantics` | no | currently only needed for time: `"absolute"` (default) or `"relative_to_hit"` |
| `derived` | no | currently only needed for world coordinates: the actor computes this one itself, see below |

#### Units

Each attribute needs to have a unit. The unit names are those of `opengate.g4_units`, scaling it performed by the actor so that the model can stay in the units it was trained on. Attributes without a unit should get the unit `1`. `offset` and `value` are given in the same unit as the quantity itself.

#### Constant Attributes

A model might not generate all the attributes needed, e.g., it generates only the x- and y-component (since the z-component is not of interest). To complete the 3-vector without having the model to generate constant values, there is the option to declare such an attribute with the `value` keyword.
Attributes marked with `value` are not a model column and are not considered for the list order corresponding to the model's output order. Offsets are not applied to those attributes.

#### Semantics

This field currently only applies to the time attribute and declares whether the generated timestamps are return as sum of the initial hit timestamp and model's generated values (`relative_to_hit`), or if the the return is only the model's generated values (`absolute`).

#### Derived outputs (`derived`)

Some outputs are needed, which are not directly generated by the model. Those attributes are marked with the `derived`keyword and take no `unit`, `component`, `value` or `offset`. The current version has two derivations defined.

##### `"world_from_module"`

A model works in the module-local frame, so it cannot produce a world-frame position. Instead the actor can provide this information, by inverting the very transform GATE used to make `PostPositionLocalModule` in the first place.

```json
{"attribute": "PostPosition", "derived": "world_from_module"}
```

Two requirements are needed:

- `PostPositionLocalModule` must itself be declared and complete in the same manifest. There is nothing to transform otherwise.
- The **input** digi collection must carry `PreStepUniqueVolumeID`, which is what stores the navigation history the transform is read from:

  ```python
  hc.attributes = [..., "PreStepUniqueVolumeID"]
  ```

  The actor requires it only when a manifest actually declares a derived output, so manifests that do not are unaffected.


##### `"from_hit"`

The optical output carries no unique identity of its own (e.g., what specific module was hit etc.) Every module writes into the same `PostPositionLocalModule` range, so without help the only way back to the module a photon came from is a join on the hits tree via `SourceHitIndex`. A `"from_hit"` entry copies an attribute of the source hit onto each of its photons:

```json
{"attribute": "PreStepUniqueVolumeID", "derived": "from_hit"}
{"attribute": "EventID", "derived": "from_hit"}
```



The attribute must be present on the **input** digi collection:

```python
hc.attributes = [..., "PreStepUniqueVolumeID", "EventID"]
```