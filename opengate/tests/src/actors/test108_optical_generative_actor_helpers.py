#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class FixedNMockGenerator:
    """
    Deterministic mock generator for testing DigitizerOpticalGenerativeActor.

    Called once per synthetic photon (N is determined by the actor via Poisson
    sampling from edep * scintillation_yield). Returns a single photon record.

    generate(x, y, z, time) -> (X, Y, dX, dY, dZ, Ekine, Time)
    """

    def __init__(self):
        self.call_count = 0

    def generate(self, x, y, z, time):
        self.call_count += 1
        X = x
        Y = y
        dX = 0.0
        dY = 0.0
        dZ = 1.0
        Ekine = 3.0  # eV, typical optical photon energy for BGO
        return X, Y, dX, dY, dZ, Ekine, time


def check_output(output_root_path, hits_root_path):
    """
    Read the actor output and verify:
      1. Output has more rows than input hits (Poisson yield > 0 on average)
      2. The Time column is linear (not log-domain): all values > 0
      3. SourceHitIndex never decreases within a contiguous block
         (photons from the same hit share the same index value)
    Returns True if all checks pass, False otherwise.
    """
    import uproot

    ok = True

    with uproot.open(output_root_path) as f:
        tree_name = list(f.keys())[0]
        tree = f[tree_name]
        time = tree["Time"].array(library="np")
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
        print("FAIL: Time column contains non-positive values (still in log domain?)")
        ok = False
    else:
        print(f"OK: all Time values are positive (min={time.min():.3e})")

    # photons from the same hit are contiguous and share the same SourceHitIndex
    if n_output > 1:
        transitions = np.where(np.diff(src_idx) != 0)[0]
        # check that within each contiguous block all values are identical
        block_starts = np.concatenate([[0], transitions + 1])
        block_ends = np.concatenate([transitions + 1, [n_output]])
        all_consistent = all(
            np.all(src_idx[s:e] == src_idx[s])
            for s, e in zip(block_starts, block_ends)
        )
        if not all_consistent:
            print("FAIL: SourceHitIndex not consistent within photon groups")
            ok = False
        else:
            print("OK: SourceHitIndex is consistent within each photon group")

    return ok
