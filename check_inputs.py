import sys # for command line arguments
import matplotlib.pyplot as plt # for plotting
from CrazyflieSimulationPython.cfSimulator.zTest import zTest
import numpy as np

from ControlBasedTesting.faCharacterization import faCharacterization

'''
Script that uses a stored characterization to check a given input on it.
Mainly for avoiding having to re-run phases 1, 2, and 3 every time.
'''

dt = 0.001
faCharact = faCharacterization.open(sys.argv[1])

ref = zTest('ud1',10.5, 0.2)

faCharact.plot_input_mapping_on_characterization(ref, dt, 0.3)

plt.show()
