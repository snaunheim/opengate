import json
import zipfile
from pathlib import Path

import numpy as np

from opengate.exception import fatal, warning
from opengate.utility import g4_units

SUPPORTED_SCHEMA_VERSION = 2
SUPPORTED_FORMATS = ("torchscript", "onnx", "aotinductor")

# GATE's module-local frame (PostPositionLocalModule) is
# (x=depth/radial, y=transverse, z=axial); see og_actor109_pet_sim_setup.py.
GATE_LOCAL_AXES = ("depth", "transverse", "axial")

# Not a official g4_units name. Should be used for dimensionless quantities and
# is used just as a scaling vector
DIMENSIONLESS_UNIT = "1"

VECTOR_COMPONENTS = ("x", "y", "z")

# The world coordinates are not an output of the model, so they are derived
# from an exisiting output.
MODULE_LOCAL_ATTRIBUTE = "PostPositionLocalModule"
DERIVED_WORLD_FROM_MODULE = "world_from_module"

DERIVED_FROM_HIT = "from_hit"
FROM_HIT_ATTRIBUTES = ("PreStepUniqueVolumeID", "EventID")

SUPPORTED_DERIVATIONS = (DERIVED_WORLD_FROM_MODULE, DERIVED_FROM_HIT)


def unit_factor(unit, what, bundle_path):
    """
    The factor that converts a value given in 'unit' into GATE's internal
    units. '1' means dimensionless; every other name must be one g4_units
    knows
    """
    if unit is None:
        fatal(
            f"Generative model bundle '{bundle_path}': {what} does not declare "
            f"a 'unit'."
        )
    if unit == DIMENSIONLESS_UNIT:
        return 1.0
    if unit not in g4_units:
        fatal(
            f"Generative model bundle '{bundle_path}': {what} declares "
            f"unit={unit!r}, which is not a unit GATE knows. Use a name from "
            f"opengate.g4_units (e.g. 'mm', 'ns', 'eV', 'MeV'), or "
            f"'{DIMENSIONLESS_UNIT}' for a dimensionless quantity."
        )
    return float(g4_units[unit])


class GenerativeModelBundle:
    """
    Loads a generative-model bundle (directory or zip) conforming to
    docs/generative_bundle_contract.md and exposes generate_batch(), so it
    can be used directly as the 'generator' of a DigitizerOpticalGenerativeActor
    or of opengate.contrib.optical.compose_event.

    All model loading (reading manifest.json, importing torch/onnxruntime,
    deserializing the model file) happens once, here, in __init__ - or, with
    defer_model_load=True, the manifest is validated in __init__ and the model
    is deserialized later by load_model().
    """

    def __init__(self, bundle_path, defer_model_load=False):
        """
        With defer_model_load=True, the manifest is read and validated but the
        model itself is not deserialized, so no torch/onnxruntime import and no
        GPU session happens yet. Call load_model() before generate_batch().
        So we can validate a bundle's configuration early (e.g. in an
        actor's resolve_and_validate_config) and have potential loading cost only when
        the simulation actually starts.
        """
        self.bundle_path = Path(bundle_path)
        self._zip = None
        self.manifest = self._load_manifest()
        self._validate_manifest()
        self.format = self.manifest["format"]
        self.coordinates = self.manifest["coordinates"]
        self.outputs = validate_outputs(self.manifest["outputs"], self.bundle_path)
        self._model = None if defer_model_load else self._load_model()

    def load_model(self):
        """Deserialize the model if it was not loaded in __init__. Idempotent."""
        if self._model is None:
            self._model = self._load_model()

    def close(self):
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    # -- manifest -----------------------------------------------------

    def _load_manifest(self):
        if self.bundle_path.is_dir():
            manifest_path = self.bundle_path / "manifest.json"
            if not manifest_path.exists():
                fatal(
                    f"Generative model bundle '{self.bundle_path}' is missing "
                    f"manifest.json."
                )
            with open(manifest_path) as f:
                return json.load(f)
        elif zipfile.is_zipfile(self.bundle_path):
            self._zip = zipfile.ZipFile(self.bundle_path)
            if "manifest.json" not in self._zip.namelist():
                fatal(
                    f"Generative model bundle '{self.bundle_path}' is missing "
                    f"manifest.json."
                )
            with self._zip.open("manifest.json") as f:
                return json.load(f)
        else:
            fatal(
                f"Generative model bundle path '{self.bundle_path}' is neither "
                f"a directory nor a zip file."
            )

    def _validate_manifest(self):
        schema_version = self.manifest.get("schema_version")
        if schema_version != SUPPORTED_SCHEMA_VERSION:
            fatal(
                f"Generative model bundle '{self.bundle_path}' has "
                f"schema_version={schema_version!r}, but only "
                f"schema_version={SUPPORTED_SCHEMA_VERSION} is supported."
            )
        fmt = self.manifest.get("format")
        if fmt not in SUPPORTED_FORMATS:
            fatal(
                f"Generative model bundle '{self.bundle_path}' declares "
                f"format={fmt!r}, which is not one of {SUPPORTED_FORMATS}."
            )
        if "model_file" not in self.manifest:
            fatal(
                f"Generative model bundle '{self.bundle_path}' manifest is "
                f"missing required field 'model_file'."
            )
        for required_section in ("coordinates", "training", "batch", "outputs"):
            if required_section not in self.manifest:
                fatal(
                    f"Generative model bundle '{self.bundle_path}' manifest is "
                    f"missing required section '{required_section}'."
                )
        axes_order = self.manifest["coordinates"].get("axes_order")
        if sorted(axes_order or []) != ["axial", "depth", "transverse"]:
            fatal(
                f"Generative model bundle '{self.bundle_path}' manifest has "
                f"invalid coordinates.axes_order={axes_order!r}; expected a "
                f"permutation of ['depth', 'transverse', 'axial']."
            )
        unit_factor(
            self.manifest["coordinates"].get("unit"), "coordinates", self.bundle_path
        )
        offset = self.manifest["coordinates"].get("offset", [0.0, 0.0, 0.0])
        if len(offset) != 3:
            fatal(
                f"Generative model bundle '{self.bundle_path}' manifest has "
                f"coordinates.offset={offset!r}; expected three numbers, one "
                f"per axis, in the model's own axis order."
            )
        max_batch = self.manifest["batch"].get("max_batch")
        if max_batch is not None and (
            not isinstance(max_batch, int)
            or isinstance(max_batch, bool)
            or max_batch < 1
        ):
            fatal(
                f"Generative model bundle '{self.bundle_path}' manifest has "
                f"batch.max_batch={max_batch!r}; expected a positive whole "
                f"number, or null when the model has no batch size limit."
            )

    @property
    def max_batch(self):
        """
        The largest n_photons this model may be called with, or None when it
        has no limit. A hit that produces more photons than this is split
        across several calls, see generate_batch.
        """
        return self.manifest["batch"].get("max_batch")

    # -- model loading --------------------------------------------------

    def _read_model_bytes(self):
        model_file = self.manifest["model_file"]
        if self._zip is not None:
            return self._zip.read(model_file)
        model_path = self.bundle_path / model_file
        if not model_path.exists():
            fatal(
                f"Generative model bundle '{self.bundle_path}' manifest "
                f"declares model_file='{model_file}', but that file does not "
                f"exist."
            )
        return model_path.read_bytes()

    def _load_model(self):
        fmt = self.format
        if fmt == "torchscript":
            return self._load_torchscript_model()
        if fmt == "onnx":
            return self._load_onnx_model()
        if fmt == "aotinductor":
            return self._load_aotinductor_model()
        fatal(f"Unknown generative model bundle format '{fmt}'.")

    def _load_torchscript_model(self):
        try:
            import torch
        except ImportError:
            fatal(
                f"Generative model bundle '{self.bundle_path}' requires torch "
                f"(format='torchscript'), but torch is not installed."
            )
        import io

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            warning(
                f"Generative model bundle '{self.bundle_path}' is running on the "
                f"CPU: torch.cuda.is_available() is False."
            )
        buffer = io.BytesIO(self._read_model_bytes())
        model = torch.jit.load(buffer, map_location=device)
        model.eval()
        return model

    def _load_onnx_model(self):
        try:
            import onnxruntime
        except ImportError:
            fatal(
                f"Generative model bundle '{self.bundle_path}' requires "
                f"onnxruntime (format='onnx'), but onnxruntime is not "
                f"installed."
            )
        requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = onnxruntime.InferenceSession(
            self._read_model_bytes(), providers=requested
        )

        # If onnxruntime falls back to the CPU:
        if "CUDAExecutionProvider" not in session.get_providers():
            available = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" in available:
                hint = (
                    "onnxruntime ships the CUDA provider but could not load it, "
                    "which usually means its CUDA/cuDNN libraries are not on the "
                    "loader path. If torch is installed with CUDA support, they "
                    "are typically already in site-packages, e.g.:\n"
                    "  export LD_LIBRARY_PATH="
                    "$(python -c 'import site;print(site.getsitepackages()[0])')"
                    "/nvidia/cu13/lib:$LD_LIBRARY_PATH\n"
                    "Check the onnxruntime log lines above for the exact "
                    "library it failed to open."
                )
            else:
                hint = (
                    "this onnxruntime build has no CUDA provider."
                )
            warning(
                f"Generative model bundle '{self.bundle_path}' is running on the "
                f"CPU: {hint}"
            )
        return session

    def _load_aotinductor_model(self):
        try:
            import torch
        except ImportError:
            fatal(
                f"Generative model bundle '{self.bundle_path}' requires torch "
                f"(format='aotinductor'), but torch is not installed."
            )
        if self._zip is not None:
            fatal(
                f"Generative model bundle '{self.bundle_path}': format "
                f"'aotinductor' requires a directory bundle (the compiled "
                f".so cannot be loaded from within a zip)."
            )
        model_path = self.bundle_path / self.manifest["model_file"]
        if not model_path.exists():
            fatal(
                f"Generative model bundle '{self.bundle_path}' manifest "
                f"declares model_file='{self.manifest['model_file']}', but "
                f"that file does not exist."
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch._export.aot_load(str(model_path), device=device)

    # -- coordinate convention ------------------------------------------

    def expected_local_axes_order(self):
        """
        Permutation of GATE's local axes (depth, transverse, axial) that the bundle expects, as indices in (x, y, z) of
        PostPositionLocalModule, derived from manifest coordinates.axes_order.
        """
        axes_order = self.coordinates["axes_order"]
        return [GATE_LOCAL_AXES.index(name) for name in axes_order]

    def expected_local_position_offset(self):
        """
        The (dx, dy, dz) offset, in the model's own axis order, to apply to the
        interaction position before handing it to the model.

        coordinates.offset declares WHAT MUST BE ADDED to a model coordinate to
        put (0,0,0) at the module iso-center.
        """
        factor = unit_factor(
            self.coordinates.get("unit"), "coordinates", self.bundle_path
        )
        offset = self.coordinates.get("offset", [0.0, 0.0, 0.0])
        return [-v * factor for v in offset]

    # -- output schema ----------------------------------------------------

    def resolved_outputs(self):
        """The validated output schema, see validate_outputs()."""
        return self.outputs

    # -- inference --------------------------------------------------------

    def generate_batch(self, x, y, z, time, n_photons):
        """
        Returns one 1-D array per declared model column, in manifest order.
        All of them share one length, which may be smaller than n_photons if
        the model drops photons that never reach the sensor.
        """
        if self._model is None:
            fatal(
                f"Generative model bundle '{self.bundle_path}' was created with "
                f"defer_model_load=True and its model has not been loaded yet. "
                f"Call load_model() before generate_batch()."
            )
        if self.format == "torchscript" or self.format == "aotinductor":
            return self._generate_batch_torch(x, y, z, time, n_photons)
        if self.format == "onnx":
            return self._generate_batch_onnx(x, y, z, time, n_photons)
        fatal(f"Unknown generative model bundle format '{self.format}'.")

    def _generate_batch_torch(self, x, y, z, time, n_photons):
        import torch

        with torch.no_grad():
            output = self._model(
                float(x), float(y), float(z), float(time), int(n_photons)
            )
        return tuple(np.asarray(t.cpu().numpy(), dtype=np.float64) for t in output)

    def _generate_batch_onnx(self, x, y, z, time, n_photons):
        feeds = {
            "x": np.array(x, dtype=np.float32),
            "y": np.array(y, dtype=np.float32),
            "z": np.array(z, dtype=np.float32),
            "time": np.array(time, dtype=np.float32),
            "n_photons_carrier": np.zeros(n_photons, dtype=np.int64),
        }
        outputs = self._model.run(None, feeds)
        return tuple(np.asarray(o, dtype=np.float64) for o in outputs)


def validate_outputs(outputs, bundle_path):
    """
    Validate 'outputs' list against GATE's own digi attributes.
    """
    import opengate_core as g4

    if not isinstance(outputs, (list, tuple)) or len(outputs) == 0:
        fatal(
            f"Generative model bundle '{bundle_path}': 'outputs' must be a "
            f"non-empty list of declared output attributes."
        )

    manager = g4.GateDigiAttributeManager.GetInstance()
    available = set(manager.GetAvailableDigiAttributeNames())

    resolved = []
    seen_components = {}  # attribute name -> set of components already declared
    column_index = 0

    for i, entry in enumerate(outputs):
        where = f"outputs[{i}]"
        attribute = entry.get("attribute")
        if attribute is None:
            fatal(
                f"Generative model bundle '{bundle_path}': {where} is missing "
                f"the required field 'attribute'."
            )
        if attribute not in available:
            fatal(
                f"Generative model bundle '{bundle_path}': {where} declares "
                f"attribute={attribute!r}, which is not a GATE digi attribute. "
                f"See GateDigiAttributeManager for the available names."
            )

        att_type = manager.GetDigiAttributeByName(attribute).GetDigiAttributeType()
        derived = entry.get("derived")
        # A from_hit value is copied from the input hit, not produced by the
        # model, so it is not restricted to the model's floating point types.
        allowed_types = (
            ("D", "3", "U", "I") if derived == DERIVED_FROM_HIT else ("D", "3")
        )
        if att_type not in allowed_types:
            fatal(
                f"Generative model bundle '{bundle_path}': {where} declares "
                f"attribute={attribute!r}, whose type is '{att_type}'. A "
                f"generative model produces floating point numbers, so only "
                f"attributes of type 'D' (double) and '3' (3-vector) can be "
                f"filled from a model output."
            )

        # A derived output is computed by the actor from another declared
        # attribute, therefore no model column, no component and no unit.
        if derived is not None:
            if derived not in SUPPORTED_DERIVATIONS:
                fatal(
                    f"Generative model bundle '{bundle_path}': {where} declares "
                    f"derived={derived!r}, which is not a derivation this "
                    f"version knows. Supported: {list(SUPPORTED_DERIVATIONS)}."
                )
            for forbidden in ("value", "offset", "component", "unit"):
                if forbidden in entry:
                    fatal(
                        f"Generative model bundle '{bundle_path}': {where} "
                        f"declares both 'derived' and {forbidden!r}. A derived "
                        f"output is computed by the actor from another "
                        f"attribute, so it takes no {forbidden}."
                    )
            if derived == DERIVED_WORLD_FROM_MODULE and att_type != "3":
                fatal(
                    f"Generative model bundle '{bundle_path}': {where} declares "
                    f"derived={derived!r} on attribute {attribute!r}, whose "
                    f"type is '{att_type}'. A derived position must be a "
                    f"3-vector attribute."
                )
            if derived == DERIVED_FROM_HIT and attribute not in FROM_HIT_ATTRIBUTES:
                fatal(
                    f"Generative model bundle '{bundle_path}': {where} declares "
                    f"derived={DERIVED_FROM_HIT!r} on attribute {attribute!r}. "
                    f"Only {list(FROM_HIT_ATTRIBUTES)} can be carried over from "
                    f"the hit that produced the photon."
                )
            if attribute in seen_components:
                fatal(
                    f"Generative model bundle '{bundle_path}': {where} declares "
                    f"attribute {attribute!r} a second time. Each output can be "
                    f"filled only once."
                )
            # world_from_module fills all three components at once; a from_hit
            # attribute is a scalar and must not claim components it has not.
            seen_components[attribute] = (
                set(VECTOR_COMPONENTS) if att_type == "3" else {""}
            )
            resolved.append(
                {
                    "attribute": attribute,
                    "component": None,
                    "relative_to_hit": False,
                    "derived": derived,
                    "column_index": -1,
                    "constant": 0.0,
                    "factor": 1.0,
                    "offset": 0.0,
                }
            )
            continue

        component = entry.get("component")
        if att_type == "3":
            if component not in VECTOR_COMPONENTS:
                fatal(
                    f"Generative model bundle '{bundle_path}': {where} declares "
                    f"the 3-vector attribute {attribute!r} but "
                    f"component={component!r}; expected one of "
                    f"{list(VECTOR_COMPONENTS)}."
                )
        elif component is not None:
            fatal(
                f"Generative model bundle '{bundle_path}': {where} declares "
                f"component={component!r} for {attribute!r}, which is a scalar "
                f"attribute, not a 3-vector."
            )

        declared = seen_components.setdefault(attribute, set())
        key = component if att_type == "3" else ""
        if key in declared:
            label = (
                f"component {component!r} of {attribute!r}"
                if att_type == "3"
                else f"attribute {attribute!r}"
            )
            fatal(
                f"Generative model bundle '{bundle_path}': {where} declares "
                f"{label} a second time. Each output can be filled only once."
            )
        declared.add(key)

        has_value = "value" in entry
        has_offset = "offset" in entry
        if has_value and has_offset:
            fatal(
                f"Generative model bundle '{bundle_path}': {where} declares "
                f"both 'value' and 'offset'. A constant is already given in "
                f"the GATE frame, so no offset applies to it."
            )

        semantics = entry.get("semantics", "absolute")
        if semantics not in ("absolute", "relative_to_hit"):
            fatal(
                f"Generative model bundle '{bundle_path}': {where} declares "
                f"semantics={semantics!r}; expected 'absolute' or "
                f"'relative_to_hit'."
            )
        if semantics == "relative_to_hit" and "Time" not in attribute:
            fatal(
                f"Generative model bundle '{bundle_path}': {where} declares "
                f"semantics='relative_to_hit' on attribute {attribute!r}, "
                f"which is not a time. Only a time can be a delay relative to "
                f"the hit."
            )

        factor = unit_factor(entry.get("unit"), where, bundle_path)

        resolved_entry = {
            "attribute": attribute,
            "component": component if att_type == "3" else None,
            "relative_to_hit": semantics == "relative_to_hit",
            "derived": None,
        }
        if has_value:
            # a constant is given directly in the GATE frame, no offset applies
            resolved_entry["column_index"] = -1
            resolved_entry["constant"] = float(entry["value"]) * factor
            resolved_entry["factor"] = 1.0
            resolved_entry["offset"] = 0.0
        else:
            resolved_entry["column_index"] = column_index
            resolved_entry["constant"] = 0.0
            resolved_entry["factor"] = factor
            resolved_entry["offset"] = float(entry.get("offset", 0.0)) * factor
            column_index += 1
        resolved.append(resolved_entry)

    if column_index == 0:
        fatal(
            f"Generative model bundle '{bundle_path}': every entry in 'outputs' "
            f"declares a constant 'value', so no model column is read at all. "
            f"A bundle whose outputs are all constants does not need a model."
        )

    # all three components of a 3-vector must be known because of the writing routine
    for attribute, declared in seen_components.items():
        att_type = manager.GetDigiAttributeByName(attribute).GetDigiAttributeType()
        if att_type == "3" and declared != set(VECTOR_COMPONENTS):
            missing = sorted(set(VECTOR_COMPONENTS) - declared)
            fatal(
                f"Generative model bundle '{bundle_path}': the 3-vector "
                f"attribute {attribute!r} is missing the component(s) "
                f"{missing}. All three must be defined either by a model column "
                f"or by a constant 'value'."
            )

    # a world-frame position is computed from the module-local one, so that
    # attribute has to be in the schema and complete (the loop above already
    # guarantees completeness for whatever is present)
    if any(e["derived"] == DERIVED_WORLD_FROM_MODULE for e in resolved):
        if MODULE_LOCAL_ATTRIBUTE not in seen_components:
            fatal(
                f"Generative model bundle '{bundle_path}': an output declares "
                f"derived={DERIVED_WORLD_FROM_MODULE!r}, which is computed from "
                f"{MODULE_LOCAL_ATTRIBUTE!r}, but that attribute is not part of "
                f"the outputs. Declare its three components as well."
            )

    return resolved
