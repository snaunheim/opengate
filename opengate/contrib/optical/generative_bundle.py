import json
import zipfile
from pathlib import Path

import numpy as np

from opengate.exception import fatal

SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_FORMATS = ("torchscript", "onnx", "aotinductor")

# GATE's crystal-local frame (PostPositionLocal / PostPositionLocalModule) is
# (x=depth/radial, y=transverse, z=axial); see og_actor109_pet_sim_setup.py.
GATE_LOCAL_AXES = ("depth", "transverse", "axial")


class GenerativeModelBundle:
    """
    Loads a generative-model bundle (directory or zip) conforming to
    docs/generative_bundle_contract.md and exposes generate_batch(), so it
    can be used directly as the 'generator' of a DigitizerOpticalGenerativeActor
    or of opengate.contrib.optical.compose_event.

    All model loading (reading manifest.json, importing torch/onnxruntime,
    deserializing the model file) happens once, here, in __init__. Nothing
    heavy is done in generate_batch beyond the forward pass itself.
    """

    def __init__(self, bundle_path):
        self.bundle_path = Path(bundle_path)
        self._zip = None
        self.manifest = self._load_manifest()
        self._validate_manifest()
        self.format = self.manifest["format"]
        self.coordinates = self.manifest["coordinates"]
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
        for required_section in ("coordinates", "training", "batch"):
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
                f"(format='torchscript'), but torch is not installed. Try: "
                f"pip install torch"
            )
        import io

        buffer = io.BytesIO(self._read_model_bytes())
        model = torch.jit.load(buffer, map_location="cpu")
        model.eval()
        return model

    def _load_onnx_model(self):
        try:
            import onnxruntime
        except ImportError:
            fatal(
                f"Generative model bundle '{self.bundle_path}' requires "
                f"onnxruntime (format='onnx'), but onnxruntime is not "
                f"installed. Try: pip install onnxruntime"
            )
        return onnxruntime.InferenceSession(
            self._read_model_bytes(), providers=["CPUExecutionProvider"]
        )

    def _load_aotinductor_model(self):
        try:
            import torch
        except ImportError:
            fatal(
                f"Generative model bundle '{self.bundle_path}' requires torch "
                f"(format='aotinductor'), but torch is not installed. Try: "
                f"pip install torch"
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
        return torch._export.aot_load(str(model_path), device="cpu")

    # -- coordinate convention ------------------------------------------

    def expected_local_axes_order(self):
        """
        The permutation of GATE's local axes (depth, transverse, axial) that
        this bundle expects, as indices into (x, y, z) of
        PostPositionLocal(Module), derived from manifest coordinates.axes_order.
        """
        axes_order = self.coordinates["axes_order"]
        return [GATE_LOCAL_AXES.index(name) for name in axes_order]

    def expected_local_position_offset(self):
        """
        The (dx, dy, dz) offset, in the model's own axis order, implied by
        coordinates.origin: bundles trained with origin='sensor_face' expect
        the depth axis shifted from GATE's centered frame ([-d/2, +d/2]) to a
        sensor-origin frame ([0, d]), where d is training.crystal_size_mm
        along the depth axis. 'crystal_center' (or any other origin) implies
        no offset, since GATE's local frame is already crystal-centered.
        """
        origin = self.coordinates.get("origin")
        offset = [0.0, 0.0, 0.0]
        if origin == "sensor_face":
            crystal_size = self.manifest["training"]["crystal_size_mm"]
            axes_order = self.coordinates["axes_order"]
            depth_position = axes_order.index("depth")
            offset[depth_position] = crystal_size[depth_position] / 2.0
        return offset

    # -- inference --------------------------------------------------------

    def generate_batch(self, x, y, z, time, n_photons):
        """
        Implements the frozen generator contract: see
        docs/generative_bundle_contract.md. Only the forward pass happens
        here; loading already happened in __init__.
        """
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
