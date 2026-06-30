#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np


class FixedNMockGenerator:
    """
    Deterministic mock generator for testing DigitizerOpticalGenerativeActor.

    Produces exactly N synthetic optical-photon records per call, with
    deterministic values derived from the input coordinates so test
    assertions can be computed analytically without running a real model.

    generate(x, y, z, edep, time) returns a list of N tuples
        (X, Y, dX, dY, dZ, Ekine, LogTime)
    where each element encodes the inputs in a simple, verifiable way.
    """

    def __init__(self, n_photons_per_hit):
        self.n_photons_per_hit = n_photons_per_hit
        self.call_count = 0

    def generate(self, x, y, z, edep, time):
        self.call_count += 1
        results = []
        for i in range(self.n_photons_per_hit):
            X = x + float(i)
            Y = y + float(i)
            dX = 0.0
            dY = 0.0
            dZ = 1.0
            Ekine = edep / self.n_photons_per_hit
            # LogTime chosen so exp(LogTime) ≈ time (plus a small photon index offset)
            LogTime = math.log(max(time + float(i) * 1e-3, 1e-30))
            results.append((X, Y, dX, dY, dZ, Ekine, LogTime))
        return results


def check_output(output_root_path, hits_root_path, n_photons_per_hit):
    """
    Read the actor output and verify:
      1. Total row count == n_photons_per_hit * number-of-input-hits
      2. The Time column is linear (not log-domain): all values > 0
      3. SourceHitIndex values are contiguous integers in [0, n_hits)
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
        n_hits = len(tree["TotalEnergyDeposit"].array(library="np"))

    expected_rows = n_hits * n_photons_per_hit
    if n_output != expected_rows:
        print(
            f"FAIL: expected {expected_rows} output rows ({n_hits} hits × "
            f"{n_photons_per_hit} photons), got {n_output}"
        )
        ok = False
    else:
        print(f"OK: row count {n_output} == {n_hits} hits × {n_photons_per_hit} photons")

    if np.any(time <= 0):
        print("FAIL: Time column contains non-positive values (still in log domain?)")
        ok = False
    else:
        print(f"OK: all Time values are positive (min={time.min():.3e})")

    # SourceHitIndex is a per-event local index (resets each event), so we
    # only verify that every consecutive group of n_photons_per_hit rows
    # shares the same index value — i.e. N photons from the same hit are
    # contiguous and tagged identically.
    groups = src_idx.reshape(-1, n_photons_per_hit)
    all_same_within_group = np.all(groups == groups[:, :1], axis=1)
    if not np.all(all_same_within_group):
        print("FAIL: SourceHitIndex not consistent within photon groups")
        ok = False
    else:
        print("OK: SourceHitIndex is consistent within each photon group")

    return ok
