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

shapes = ['steps', 'ramp', 'trapezoidal', 'triangular']

# faCharact_only_main = faCharacterization(nl_max)
# faCharact_all = faCharacterization(nl_max)
faCharact_only_main = faCharacterization.open('characterization-only-main')
faCharact_all = faCharacterization.open('characterization-all')

faCharact_only_main.plot_motors_saturation_characterization(shape='none',title='Fig. 12a: Crazyflie Actuator Saturation')

#plots separated by shapes
for s in shapes :
    faCharact_all.evaluate_shape_bandwidth(s,nl_max,show_plot=True, title='Fig. 15: degree of filtering')
    faCharact_only_main.plot_non_linearity_characterization(nl_max,shape=s, title='Fig. 14: degree of non-linearity')    

plt.show()
