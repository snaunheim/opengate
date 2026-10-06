#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import uproot

from og_actor109_pet_sim_setup import create_simulation

PATHS_FOLDER = "test113_generative_bundle_pooled"

# both runs need the same primaries, otherwise there is nothing to compare
SEED = 123456
N_MODULES = 4

# small enough that the pool overflows several times in this run, so the
# comparison covers both full batches and the remainder flushed at the end
TARGET_BATCH = 4096


def run_simulation(paths, bundle_dir, output_name, seed, n_modules):
    sim, _, output_path = create_simulation(
        paths, generator=str(bundle_dir), n_modules=n_modules
    )
    sim.random_seed = seed
    og = sim.actor_manager.get_actor("SyntheticPhotons")
    og.output_filename = output_name
    sim.run(start_new_process=False)
    return output_path.parent / output_name


def read_tree(path):
    with uproot.open(path) as f:
        return f[list(f.keys())[0]].arrays(library="np")
