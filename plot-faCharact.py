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
shape = 'none'

faCharact = faCharacterization.open(sys.argv[1])

faCharact.check_freq_ordering()

for s in ['steps', 'ramp', 'trapezoidal', 'triangular'] :
    faCharact.evaluate_shape_bandwidth(s,nl_max,show_plot=False)

# plot obtained characterization
faCharact.plot_non_linearity_characterization(nl_max,shape=shape)
faCharact.plot_filtering_characterization(nl_max,shape=shape)
faCharact.plot_motors_saturation_characterization(shape=shape)
faCharact.plot_hit_ground_characterization(shape=shape)

plt.show()
