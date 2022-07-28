import sys # for command line arguments
import matplotlib.pyplot as plt # for plotting

from ControlBasedTesting.faCharacterization import faCharacterization

'''
Script that uses a stored characterization to check a given input on it.
Mainly for avoiding having to re-run phases 1, 2, and 3 every time.
'''

faCharact = faCharacterization.open(sys.argv[1])
faCharact.check_input([1.1],[1.1])

plt.show()
