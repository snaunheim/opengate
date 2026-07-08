#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
from test108_optical_generative_actor_helpers import FixedNMockGenerator


def create_simulation(paths, generator=None):
    mm = gate.g4_units.mm
    MeV = gate.g4_units.MeV
    Bq = gate.g4_units.Bq
    sec = gate.g4_units.s
    m = gate.g4_units.m

    sim = gate.Simulation()
    sim.visu = False
    sim.random_seed = 42
    sim.output_dir = paths.output

    sim.volume_manager.add_material_database(
        gate.utility.get_contrib_path() / "GateMaterials.db"
    )

    sim.world.size = [1 * m, 1 * m, 1 * m]
    sim.world.material = "G4_AIR"

    crystal = sim.add_volume("Box", "crystal")
    crystal.size = [3 * mm, 3 * mm, 20 * mm]
    crystal.material = "BGO"

    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"

    source = sim.add_source("GenericSource", "gamma")
    source.particle = "gamma"
    source.energy.mono = 0.511 * MeV
    source.activity = 1000 * Bq
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, -1]
    source.position.translation = [0, 0, 25 * mm]

    hits_filename = paths.output / "test108_hits.root"
    output_filename = paths.output / "test108_optical.root"

    hc = sim.add_actor("DigitizerHitsCollectionActor", "Hits")
    hc.attached_to = crystal.name
    hc.output_filename = hits_filename
    hc.attributes = [
        "PostPositionLocal",
        "TotalEnergyDeposit",
        "GlobalTime",
    ]

    if generator is None:
        generator = FixedNMockGenerator()

    og = sim.add_actor("DigitizerOpticalGenerativeActor", "SyntheticPhotons")
    og.attached_to = crystal.name
    og.input_digi_collection = hc.name
    og.generator = generator
    og.output_filename = output_filename

    sim.run_timing_intervals = [[0, 0.1 * sec]]

    return sim, hits_filename, output_filename
