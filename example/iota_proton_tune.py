#
#
#

import numpy as np
import PyNAFF as naff

def frac_tune(coords: np.ndarray):
    tune_ana = naff.naff(coords)
    fractional_tune = tune_ana[0][1]
    return fractional_tune

import pickle
from xtrack import ParticlesMonitor

with open("iota_proton_sc-frozen.pickle", 'rb') as inFile:
    data_frozen = pickle.load(inFile)
with open("iota_proton_sc-pic.pickle", 'rb') as inFile:
    data_pic = pickle.load(inFile)
bunch_test_frozen = ParticlesMonitor.from_dict(data_frozen["bunch_test"])
bunch_test_pic = ParticlesMonitor.from_dict(data_pic["bunch_test"])

def analyze_tunes(bunch_test: ParticlesMonitor):

    x_offsets = []
    y_offsets = []
    x_tunes = []
    y_tunes = []

    for particle in range(100):

        x_offsets.append(bunch_test.x[particle, 0])
        y_offsets.append(bunch_test.y[particle+100, 0])

        try:
            x_tune = frac_tune(bunch_test.x[particle])
        except:
            x_tune = 0.
        try:
            y_tune = frac_tune(bunch_test.y[particle+100])
        except:
            y_tune = 0.

        x_tunes.append(x_tune)
        y_tunes.append(y_tune)

    return x_offsets, x_tunes, y_offsets, y_tunes

x_offsets_frozen, x_tunes_frozen, y_offsets_frozen, y_tunes_frozen = analyze_tunes(bunch_test_frozen)
x_offsets_pic, x_tunes_pic, y_offsets_pic, y_tunes_pic = analyze_tunes(bunch_test_pic)

import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.plot(x_offsets_frozen, x_tunes_frozen, label='Frozen')
plt.plot(x_offsets_pic, x_tunes_pic, label='PIC')
plt.xlabel("x offset [m]")
plt.ylabel(r"$\nu_x$")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(y_offsets_frozen, y_tunes_frozen, label='Frozen')
plt.plot(y_offsets_pic, y_tunes_pic, label='PIC')
plt.xlabel("y offset [m]")
plt.ylabel(r"$\nu_y$")
plt.legend()
plt.tight_layout()
plt.savefig("tune_shift_pic.png")
