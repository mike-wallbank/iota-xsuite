#
#
#

import os, sys
import pickle

import numpy as np
from scipy import constants

import xobjects as xo
import xtrack as xt
import xpart as xp
import xfields as xf

from cpymad.madx import Madx

if __name__ == "__main__":

    # Options
    pic = True
    num_particles = int(1e7)
    num_turns = int(1e3)

    # Beam configuration
    ref_energy = 2.5e6 # 2.5 MeV
    emit_x = 4.3e-6
    emit_y = 3.0e-6
    delta_sigma = 2.1e-3

    ref_momentum = ref_energy * np.sqrt(1 + 2*xp.PROTON_MASS_EV/ref_energy)
    ref_total_energy_gev = 1e-9 * (ref_energy + xp.PROTON_MASS_EV)
    gamma = (ref_energy+xp.PROTON_MASS_EV) / xp.PROTON_MASS_EV
    beta = np.sqrt(1 - pow(gamma, -2))
    beta_gamma = ref_momentum / xp.PROTON_MASS_EV
    nemit_x = beta_gamma * emit_x
    nemit_y = beta_gamma * emit_y

    # Import lattice
    madx = Madx(stdout=False)
    madx.call(file="SpaceChargeLattice_IPAC2024.seq")
    madx.command.beam(particle="PROTON", energy=ref_total_energy_gev)
    madx.use(period="iota")
    line = xt.Line.from_madx_sequence(madx.sequence["iota"], allow_thick=True)
    for element in line.elements:
        if isinstance(element, xt.Cavity):
            # Protons are below transition
            element.lag = 0.

    circumference = line.get_length()
    orbit_time = circumference / (beta * constants.c)

    aperture_diameter = 50.e-3
    aperture = xt.LimitEllipse(a=aperture_diameter/2., b=aperture_diameter/2.)
    line.insert_element(name='aperture', element=aperture, at=0)

    # Bunch intensity
    beam_current = 1.e-3 #1mA
    harmonic_number = 4
    bunch_current = beam_current / harmonic_number
    bunch_intensity = int(bunch_current * orbit_time / constants.e)
    print(f"Bunch intensity {bunch_intensity}")

    # Define reference particle
    line.particle_ref = xp.Particles(p0c=ref_momentum,
                                     q0=1,
                                     mass0=xp.PROTON_MASS_EV)

    # Define the context
    context = xo.ContextCupy()

    # Get sigma_s
    line.build_tracker(_context=context, compile=False)
    buffer = line._buffer
    twiss = line.twiss()
    line.discard_tracker()
    beta_s = twiss['bets0']
    sigma_s = beta_s * delta_sigma

    # Slice lattice into thin slices
    line.slice_thick_elements(
        slicing_strategies=[
            xt.slicing.Strategy(slicing=xt.slicing.Teapot(100)),
            xt.slicing.Strategy(slicing=xt.slicing.Uniform(100), element_type=xt.Bend),
            xt.slicing.Strategy(slicing=xt.slicing.Teapot(100), element_type=xt.Quadrupole),
            xt.slicing.Strategy(slicing=None, element_type=xt.Solenoid)
        ])

    # Create particle bunch
    line.build_tracker(_context=context, compile=False)
    macroparticles = \
        xp.generate_matched_gaussian_bunch(line=line,
                                           num_particles=num_particles,
                                           total_intensity_particles=bunch_intensity,
                                           nemitt_x=nemit_x,
                                           nemitt_y=nemit_y,
                                           sigma_z=sigma_s,
                                           _context=context)

    # Add test particles
    bunch_test_mm = np.linspace(-6.e-3, 6.e-3, 100) # 100 particles between -6 and 6 mm
    bunch_test_particles_x = line.build_particles(x=bunch_test_mm,
                                                  weight=0)
    bunch_test_particles_y = line.build_particles(y=bunch_test_mm,
                                                  weight=0)
    all_particles = xp.Particles.merge([macroparticles,
                                        bunch_test_particles_x,
                                        bunch_test_particles_y],
                                       _context=context)

    # Set up monitors
    line.discard_tracker()
    bunch_test_monitor = xt.ParticlesMonitor(start_at_turn=0,
                                             stop_at_turn=num_turns,
                                             particle_id_range=[num_particles,
                                                                num_particles+200],
                                             _buffer=buffer)
    line.insert_element(name='bunch_test_monitor',
                        element=bunch_test_monitor, at=0)

    # Space charge
    lprofile = xf.LongitudinalProfileQGaussian(number_of_particles=bunch_intensity,
                                               sigma_z=sigma_s,
                                               z0=0.,
                                               q_parameter=1.,
                                               _context=context)
    sc_elements = \
        xf.install_spacecharge_frozen(line=line,
                                      longitudinal_profile=lprofile,
                                      nemitt_x=nemit_x,
                                      nemitt_y=nemit_y,
                                      sigma_z=sigma_s,
                                      num_spacecharge_interactions=200)
    if pic:
        pic_collection, all_pics = \
            xf.replace_spacecharge_with_PIC(line=line,
                                            n_sigmas_range_pic_x=5.0,
                                            n_sigmas_range_pic_y=5.0,
                                            nx_grid=64,
                                            ny_grid=64,
                                            nz_grid=64,
                                            n_lims_x=5,
                                            n_lims_y=5,
                                            z_range=(-2.5*sigma_s,
                                                     2.5*sigma_s),
                                            solver="FFTSolver2p5DAveraged")

    # Re-build tracker with the new elements in place
    line.build_tracker(_context=context)

    # Ensure correct context
    assert all_particles._context == context
    assert len(set([ee._buffer for ee in line.elements])) == 1 # All on same context
    assert line._context == context

    # Track
    line.track(all_particles,
               num_turns=num_turns,
               with_progress=True)

    # Save output
    output_data = {}
    output_data["bunch_test"] = bunch_test_monitor.to_dict()
    with open("iota_proton_sc.pickle", 'wb') as outFile:
        pickle.dump(output_data, outFile)

