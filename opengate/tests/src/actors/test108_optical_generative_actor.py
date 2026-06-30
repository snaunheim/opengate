#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate.tests.utility as tu
from test108_optical_generative_actor_simulation import create_simulation
from test108_optical_generative_actor_helpers import check_output

if __name__ == "__main__":
    paths = tu.get_default_test_paths(__file__, output_folder="test108")

    print(f"Output dir: {paths.output}")
    paths.output.mkdir(parents=True, exist_ok=True)

    sim, hits_path, output_path, n_photons = create_simulation(paths)
    sim.run(start_new_process=False)

    import os
    print(f"Hits file exists: {os.path.exists(hits_path)}")
    print(f"Optical file exists: {os.path.exists(output_path)}")

    is_ok = check_output(output_path, hits_path, n_photons)
    tu.test_ok(is_ok)
