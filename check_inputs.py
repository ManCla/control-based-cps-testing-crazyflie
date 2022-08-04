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

directory = "cfdata_nlmax015/"
dt = 0.001
non_linear_threshold = 0.15
faCharact = faCharacterization.open(sys.argv[1])

shape = 'ud1'
a_gain = 4.6717
t_scale = 0.4573714
file_name = shape+'-'+str(a_gain)+'-'+str(t_scale)
if exists(directory+file_name) :
    test_results = fdh()
    test_results.open(directory+file_name)
    test_results.show_all()

ref = zTest(shape,a_gain,t_scale)

faCharact.plot_amp_lower_bound(non_linear_threshold)
faCharact.plot_input_mapping_on_characterization(ref, dt, non_linear_threshold)

plt.show()
