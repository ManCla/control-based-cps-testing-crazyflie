import sys # for command line arguments
import matplotlib.pyplot as plt # for plotting
import numpy as np

from ControlBasedTesting.faCharacterization import faCharacterization

'''
Script that uses a stored characterization to check a given input on it.
Mainly for avoiding having to re-run phases 1, 2, and 3 every time.
'''

ref = np.linspace(0,1,101)

faCharact = faCharacterization.open(sys.argv[1])

faCharact.check_input(ref, 0.01, 0.3)

plt.show()
