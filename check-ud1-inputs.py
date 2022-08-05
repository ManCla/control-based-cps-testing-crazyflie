import sys # for command line arguments
import os # to get list of files in data directory
from os.path import exists      # to check if test has already been performed
import matplotlib.pyplot as plt # for plotting
from CrazyflieSimulationPython.cfSimulator.zTest import zTest
from CrazyflieSimulationPython.cfSimulator import ZAnalysis as fdh
import numpy as np

from ControlBasedTesting.faCharacterization import faCharacterization

'''
Script that  check all ud1 inputs and enables the comparison of the
predicted non-linear degree and the actual one.

Uses a stored characterization (passed as command line argument)
to avoid having to re-run phases 1, 2, and 3 every time.
'''

directory = "cfdata_nlmax015/"
dt = 0.001
non_linear_threshold = 0.15

# unpack command line inputs
faCharact = faCharacterization.open(sys.argv[1])

dir_content = os.listdir(directory)
# filter out directories and look only at files
dir_content = list(filter(lambda x:os.path.isfile(directory+'/'+x), dir_content))

nl_prediction = np.array([])
nl_actual = np.array([])

for file in dir_content :

    use_case  = file.split('-')
    shape   = use_case[0]
    if not(shape=='ud1') :
        continue
    a_gain  = float(use_case[1])
    t_scale = float(use_case[2])
    
    # open file and get 
    test_results = fdh()
    test_results.open(directory+file, silent=True)
    real_nldg = test_results.get_z_non_linear_degree() # TODO: this does not fft the settling/warm-up!!!
    nl_actual = np.append(nl_actual, np.array([real_nldg]))
    # print("Test {} has an actual nldg of: {}".format(file,real_nldg))

    ref = zTest(shape,a_gain,t_scale)

    risk_out_bounds = faCharact.check_input_on_characterization(ref, dt)
    nl_prediction = np.append(nl_prediction, np.array([risk_out_bounds]))
    # print("The risk of this input triggering non-linear behaviour is: {}".format(risk_out_bounds))

fig, axs = plt.subplots(1, 1)
axs.title.set_text("Closeness to performance bound VS non linear degree")
axs.set_xlabel("Non-linear degree from test execution")
axs.set_ylabel("Risk assessment from our characterizationn")
axs.grid()
lim = max(max(nl_actual),max(nl_prediction))+0.2
axs.set_xlim([0, lim])
axs.set_ylim([0, lim])
axs.scatter(nl_actual, nl_prediction, s=2)
plt.show()
