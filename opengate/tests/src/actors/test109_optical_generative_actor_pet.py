#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate.tests.utility as tu
from test109_optical_generative_actor_pet_simulation import create_simulation

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from test108_optical_generative_actor_helpers import check_output

if __name__ == "__main__":
    paths = tu.get_default_test_paths(__file__, output_folder="test109")

    print(f"Output dir: {paths.output}")
    paths.output.mkdir(parents=True, exist_ok=True)

    sim, hits_path, output_path = create_simulation(paths)
    sim.run(start_new_process=False)

    print(f"Hits file exists:    {os.path.exists(hits_path)}")
    print(f"Optical file exists: {os.path.exists(output_path)}")

    is_ok = check_output(output_path, hits_path)
    tu.test_ok(is_ok)
