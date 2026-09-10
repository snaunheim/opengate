#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# The schema FixedNMockGenerator's seven columns fill, in the order it returns
# them. A generator object has no manifest, so the actor's 'outputs' has to say
# this instead. The two transverse position columns are the model's own; the
# depth is a constant, since the mock emits everything at the module center.
MOCK_GENERATOR_OUTPUTS = [
    {"attribute": "PostPositionLocalModule", "component": "y", "unit": "mm"},
    {"attribute": "PostPositionLocalModule", "component": "z", "unit": "mm"},
    {"attribute": "Direction", "component": "x", "unit": "1"},
    {"attribute": "Direction", "component": "y", "unit": "1"},
    {"attribute": "Direction", "component": "z", "unit": "1"},
    {"attribute": "KineticEnergy", "unit": "MeV"},
    {"attribute": "GlobalTime", "unit": "ns", "semantics": "relative_to_hit"},
    {
        "attribute": "PostPositionLocalModule",
        "component": "x",
        "unit": "mm",
        "value": 0.0,
    },
]


class FixedNMockGenerator:
    """
    Deterministic mock generator for testing DigitizerOpticalGenerativeActor.

    Called once per hit with the batch size n_photons determined by the actor
    via Poisson sampling from edep * scintillation_yield. Returns n_photons
    records as arrays — matching the interface a real GPU model would use.

    Its seven columns are declared by MOCK_GENERATOR_OUTPUTS above.

    generate_batch(x, y, z, time, n_photons)
        -> (X, Y, dX, dY, dZ, Ekine, Time)  — each a numpy array of length n_photons
    """

    def __init__(self):
        self.call_count = 0

    def generate_batch(self, x, y, z, time, n_photons):
        self.call_count += 1
        X = np.full(n_photons, x)
        Y = np.full(n_photons, y)
        dX = np.zeros(n_photons)
        dY = np.zeros(n_photons)
        dZ = np.ones(n_photons)
        Ekine = np.full(n_photons, 3.0e-6)  # MeV, ~3 eV optical photon in BGO
        # a delay after the hit, not an absolute time: the actor adds the hit's
        # own GlobalTime to it (semantics 'relative_to_hit')
        Time = np.full(n_photons, 1.0)
        return X, Y, dX, dY, dZ, Ekine, Time


def check_output(output_root_path, hits_root_path):
    """
    Read the actor output and verify:
      1. Output has more rows than input hits (Poisson yield > 0 on average)
      2. The GlobalTime column is linear (not log-domain): all values > 0
      3. SourceHitIndex never decreases within a contiguous block
         (photons from the same hit share the same index value)
    Returns True if all checks pass, False otherwise.

    The columns carry the names of the GATE digi attributes the schema
    declares, so the time column is GlobalTime, not Time.
    """
    import uproot

    ok = True

    with uproot.open(output_root_path) as f:
        tree_name = list(f.keys())[0]
        tree = f[tree_name]
        time = tree["GlobalTime"].array(library="np")
        src_idx = tree["SourceHitIndex"].array(library="np")
        n_output = len(time)

    with uproot.open(hits_root_path) as f:
        tree_name = list(f.keys())[0]
        tree = f[tree_name]
        edep = tree["TotalEnergyDeposit"].array(library="np")
        n_hits = len(edep)

    if n_output == 0:
        print(f"FAIL: output has 0 rows")
        ok = False
    elif n_output < n_hits:
        print(
            f"FAIL: output rows ({n_output}) < input hits ({n_hits}); "
            f"expected more photons than hits on average"
        )
        ok = False
    else:
        mean_per_hit = n_output / n_hits
        print(
            f"OK: {n_output} photon rows from {n_hits} hits "
            f"(mean {mean_per_hit:.1f} photons/hit)"
        )

    if np.any(time <= 0):
        print(
            "FAIL: GlobalTime column contains non-positive values "
            "(still in log domain?)"
        )
        ok = False
    else:
        print(f"OK: all GlobalTime values are positive (min={time.min():.3e})")

    # photons from the same hit are contiguous and share the same SourceHitIndex
    if n_output > 1:
        transitions = np.where(np.diff(src_idx) != 0)[0]
        # check that within each contiguous block all values are identical
        block_starts = np.concatenate([[0], transitions + 1])
        block_ends = np.concatenate([transitions + 1, [n_output]])
        all_consistent = all(
            np.all(src_idx[s:e] == src_idx[s]) for s, e in zip(block_starts, block_ends)
        )
        if not all_consistent:
            print("FAIL: SourceHitIndex not consistent within photon groups")
            ok = False
        else:
            print("OK: SourceHitIndex is consistent within each photon group")

    return ok
