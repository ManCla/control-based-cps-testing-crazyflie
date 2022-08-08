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
shape_for_validation = 'trapezoidal'

# unpack command line inputs
faCharact = faCharacterization.open(sys.argv[1])

dir_content = os.listdir(directory)
# filter out directories and look only at files
dir_content = list(filter(lambda x:os.path.isfile(directory+'/'+x), dir_content))

classifier_result = np.array([])
regressor_result = np.array([])
nl_actual = np.array([])

total_count = 0
false_negative = 0
false_positive = 0

# iteration over test files
for file in dir_content :

    # get test details
    use_case  = file.split('-')
    shape   = use_case[0]
    if not(shape==shape_for_validation) :
        continue
    a_gain  = float(use_case[1])
    t_scale = float(use_case[2])
    
    # open file and get ground truth
    test_results = fdh()
    test_results.open(directory+file, silent=True)
    real_nldg = test_results.get_z_non_linear_degree() # TODO: this does not fft the settling/warm-up!!!
    nl_actual = np.append(nl_actual, np.array([real_nldg]))

    # create reference generator object and evaluate regressor and classifier
    ref = zTest(shape,a_gain,t_scale)
    risk_out_bounds_probability = faCharact.check_input_on_rfc(ref, dt)[0][1]
    classifier_result = np.append(classifier_result, np.array([risk_out_bounds_probability]))

    nl_prediction = faCharact.check_input_on_rfr(ref, dt)[0]
    regressor_result = np.append(regressor_result, np.array([nl_prediction]))

    # some statistics and print out wrong predictions
    total_count = total_count+1
    if real_nldg<0.15 : # non linear bh appears
        if risk_out_bounds_probability>0.5 :
            false_positive = false_positive+1
    else :
        if risk_out_bounds_probability<0.5 :
            false_negative = false_negative+1
            # print("false negative: "+file)
    if risk_out_bounds_probability<0.01 :
        print("test with zero risk? : "+file)

print("--> total count: "+str(total_count))
print("--> false positives: "+str(false_positive))
print("--> false negatives: "+str(false_negative))

##### PLOTTING #####

fig, axs = plt.subplots(2, 1)

### PLOT CLASSIFIER RESUTLS
axs[0].title.set_text("CLASSIFIER RESULTS")
axs[0].set_xlabel("Non-linear degree from test execution")
axs[0].set_ylabel("Probability of non-lin-deg>0.15 according to classifier")
axs[0].grid()
axs[0].set_xlim([0, max(nl_actual)+0.2])
axs[0].set_ylim([0, 1.1])
axs[0].plot([faCharact.rfc_non_linear_threshold]*2,[0,2], linestyle='dashed', c='black')
axs[0].scatter(nl_actual, classifier_result, s=2)

### PLOT REGRESSOR RESUTLS
axs[1].title.set_text("REGRESSOR RESULTS")
axs[1].set_xlabel("Non-linear degree from test execution")
axs[1].set_ylabel("Prediction of non linear degree from random forest")
axs[1].grid()
axs[1].set_xlim([0, max(nl_actual)+0.2])
axs[1].set_ylim([0, max(regressor_result)+0.2])
axs[1].plot([0,3],[faCharact.rfc_non_linear_threshold]*2, linestyle='dashed', c='black')
axs[1].plot([faCharact.rfc_non_linear_threshold]*2,[0,2], linestyle='dashed', c='black')
axs[1].scatter(nl_actual, regressor_result, s=2)

plt.show()
