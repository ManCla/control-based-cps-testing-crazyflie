import sys # for command line arguments
from os.path import exists      # to check if test has already been performed
import matplotlib.pyplot as plt # for plotting
import numpy as np

from ControlBasedTesting.faCharacterization import faCharacterization

'''
Script that opens a stored characterization given
from command line and plots it
'''

nl_max = 0.15

faCharact = faCharacterization.open(sys.argv[1])

# plot obtained characterization
faCharact.plot_non_linearity_characterization(nl_max)
faCharact.plot_filtering_characterization(nl_max)
faCharact.plot_motors_saturation_characterization()
faCharact.plot_hit_ground_characterization()

plt.show()
