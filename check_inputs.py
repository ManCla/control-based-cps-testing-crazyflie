import sys # for command line arguments
from os.path import exists      # to check if test has already been performed
import matplotlib.pyplot as plt # for plotting
from CrazyflieSimulationPython.cfSimulator.zTest import zTest
from CrazyflieSimulationPython.cfSimulator import ZAnalysis as fdh
import numpy as np

from ControlBasedTesting.faCharacterization import faCharacterization

'''
Script that uses a stored characterization to check a given input on it.
Mainly for avoiding having to re-run phases 1, 2, and 3 every time.
'''

try_plot_test = False # use if you want to try plotting also the actual execution of the test

directory = "cfdata_nlmax015/"
dt = 0.001
non_linear_threshold = 0.15

# unpack command line inputs
faCharact = faCharacterization.open(sys.argv[1])
file_name = sys.argv[2].split('/')[1]

# unpack test name
print("Analysing test "+file_name)
use_case  = file_name.split('-')
shape   = use_case[0]
a_gain  = float(use_case[1])
t_scale = float(use_case[2])

if try_plot_test and exists(directory+file_name) :
    test_results = fdh()
    test_results.open(directory+file_name)
    test_results.show_all()

ref = zTest(shape,a_gain,t_scale)

# faCharact.plot_amp_lower_bound(non_linear_threshold)
faCharact.plot_input_mapping_on_characterization(ref, dt, non_linear_threshold)

risk_out_bounds = faCharact.check_input_on_characterization(ref, dt)
print("The risk of this input triggering non-linear behaviour is: {}".format(risk_out_bounds))

plt.show()
