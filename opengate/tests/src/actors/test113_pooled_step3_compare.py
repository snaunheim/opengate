#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Third of three: compares the two outputs of steps 1 and 2 branch by branch.

Pooling batches photons from several hits, and several events, into one model
call. The values that are constant per hit rather than per photon are the
ones this can get wrong: the hit time a 'relative_to_hit' GlobalTime is
added to, the module transform PostPosition is derived with, the copied
PreStepUniqueVolumeID and EventID, and SourceHitIndex. So the two runs have
to agree exactly, not approximately.
"""

import numpy as np

import opengate.tests.utility as tu

from test112_generative_bundle_helpers import TORCH_AVAILABLE
from test113_pooled_helpers import PATHS_FOLDER, read_tree

if __name__ == "__main__":
    paths = tu.get_default_test_paths(__file__, output_folder=PATHS_FOLDER)

    if not TORCH_AVAILABLE:
        print("torch is not installed, skipping test113 (nothing to check).")
        tu.test_ok(True)
        raise SystemExit(0)

    per_hit_path = paths.output / "per_hit.root"
    pooled_path = paths.output / "pooled.root"
    for p in (per_hit_path, pooled_path):
        if not p.exists():
            print(f"FAIL: {p} is missing, run steps 1 and 2 first")
            tu.test_ok(False)
            raise SystemExit(0)

    per_hit = read_tree(per_hit_path)
    pooled = read_tree(pooled_path)

    is_ok = True

    n_per_hit = len(next(iter(per_hit.values())))
    n_pooled = len(next(iter(pooled.values())))
    if n_per_hit == 0:
        print("FAIL: the per-hit run produced 0 rows, nothing to compare")
        is_ok = False
    elif n_per_hit != n_pooled:
        print(f"FAIL: {n_per_hit} rows per hit but {n_pooled} rows pooled")
        is_ok = False
    else:
        print(f"OK: both runs produced {n_per_hit} photon rows")

    if sorted(per_hit) != sorted(pooled):
        print(f"FAIL: different branches: {sorted(per_hit)} vs {sorted(pooled)}")
        is_ok = False

    if is_ok:
        for name in sorted(per_hit):
            a, b = per_hit[name], pooled[name]
            if len(a) != len(b):
                print(f"FAIL: {name} has {len(a)} vs {len(b)} entries")
                is_ok = False
            elif not np.array_equal(a, b):
                print(f"FAIL: {name} differs between the two runs")
                is_ok = False
            else:
                print(f"OK: {name} is identical")

    tu.test_ok(is_ok)
