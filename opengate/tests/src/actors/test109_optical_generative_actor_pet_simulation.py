#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
import opengate.geometry.utility as geo_util
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from test108_optical_generative_actor_helpers import FixedNMockGenerator



def create_simulation(paths, generator=None):
    mm = gate.g4_units.mm
    MeV = gate.g4_units.MeV
    Bq = gate.g4_units.Bq
    sec = gate.g4_units.s
    deg = gate.g4_units.deg

    sim = gate.Simulation()
    sim.visu = False
    sim.random_seed = 42
    sim.output_dir = paths.output

    sim.volume_manager.add_material_database(
        gate.utility.get_contrib_path() / "GateMaterials.db"
    )

    # ---------------------------------------------------------------------------
    # Geometry: simple cylindrical PET ring
    #
    # Each detector module contains a 3×3 array of BGO crystals (3×3×10 mm each).
    # The module face is 9×9 mm; crystals are 10 mm deep (radial direction).
    # Ring inner radius = 100 mm  →  circumference ≈ 628 mm
    # Module arc width ≈ 10 mm (crystal depth in radial dir) → 20 modules fit comfortably.
    # ---------------------------------------------------------------------------

    crystal_xy = 3 * mm      # transverse crystal size
    crystal_z = 10 * mm      # axial crystal size (also depth in radial direction)
    n_crystals_xy = 3        # 3×3 array per module
    gap = 0.2 * mm           # inter-crystal gap
    pitch = crystal_xy + gap

    module_xy = n_crystals_xy * crystal_xy + (n_crystals_xy - 1) * gap  # 9.4 mm
    module_z = module_xy                                                   # square face
    module_depth = crystal_z                                               # 10 mm radial depth

    ring_inner_r = 100 * mm
    ring_outer_r = ring_inner_r + module_depth                            # 110 mm
    ring_mid_r = ring_inner_r + module_depth / 2                          # 105 mm
    n_modules = 20

    # World
    world_r = ring_outer_r + 50 * mm
    sim.world.size = [2 * world_r, 2 * world_r, 4 * crystal_z]
    sim.world.material = "G4_AIR"

    # Ring air container (purely for logical grouping — optional but clean)
    ring = sim.add_volume("Tubs", "ring")
    ring.rmax = ring_outer_r + 1 * mm
    ring.rmin = ring_inner_r - 1 * mm
    ring.dz = module_z / 2 + 1 * mm
    ring.material = "G4_AIR"

    # Detector module (air box — holds the crystal array)
    module = sim.add_volume("Box", "module")
    module.mother = ring.name
    module.size = [module_depth, module_xy, module_z]   # x = radial, y/z = transverse/axial
    module.material = "G4_AIR"

    translations_ring, rotations_ring = geo_util.get_circular_repetition(
        n_modules,
        [ring_mid_r, 0.0, 0.0],
        start_angle_deg=0,
        axis=[0, 0, 1],
    )
    module.translation = translations_ring
    module.rotation = rotations_ring

    # BGO crystal (3×3×10 mm) — the sensitive volume
    crystal = sim.add_volume("Box", "crystal")
    crystal.mother = module.name
    crystal.size = [crystal_z, crystal_xy, crystal_xy]  # radial × transverse × axial
    crystal.material = "BGO"

    crystal.translation = geo_util.get_grid_repetition(
        [1, n_crystals_xy, n_crystals_xy],
        [0, pitch, pitch],
    )

    # Physics: standard EM — no optical photon tracking
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"

    # Back-to-back 511 keV annihilation photon source at the FOV center
    source = sim.add_source("GenericSource", "annihilation")
    source.particle = "back_to_back"
    source.activity = 5000 * Bq
    source.position.type = "point"
    source.position.translation = [0, 0, 0]
    source.direction.theta = [90 * deg, 90 * deg]   # transaxial plane only
    source.direction.phi = [0, 360 * deg]

    # Hits collection on the BGO crystals
    hits_filename = paths.output / "test109_hits.root"
    output_filename = paths.output / "test109_optical.root"

    hc = sim.add_actor("DigitizerHitsCollectionActor", "Hits")
    hc.attached_to = crystal.name
    hc.authorize_repeated_volumes = True
    hc.output_filename = hits_filename
    hc.attributes = [
        "PostPositionLocal",
        "TotalEnergyDeposit",
        "GlobalTime",
    ]

    # Optical generative actor — replaces expensive optical photon tracking
    if generator is None:
        generator = FixedNMockGenerator()

    og = sim.add_actor("DigitizerOpticalGenerativeActor", "SyntheticPhotons")
    og.attached_to = crystal.name
    og.authorize_repeated_volumes = True
    og.input_digi_collection = hc.name
    og.generator = generator
    og.output_filename = output_filename

    sim.run_timing_intervals = [[0, 0.05 * sec]]

    return sim, hits_filename, output_filename
