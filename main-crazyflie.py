import matplotlib.pyplot as plt # for plotting
import random as rnd            # for setting the seed of the random generator
from os.path import exists      # to check if test has already been performed
import numpy as np

# utilities of testing approach
from ControlBasedTesting.binary_search_sinus_freq import binary_search_sinus_freq
from ControlBasedTesting.NLthUpperbound import NLthUpperbound
from ControlBasedTesting.shapeTestSet import shapeTestSet

# for Crazyflie Testing
from CrazyflieSimulationPython.cfSimulator import cfSimulation, ZAnalysis, zTest

data_directory = 'cfdata/'

### TODO: set seed for random test generation ###

#####################################################################
### PHASE 1: sinusoidal-based non-linearity threshold exploration ###
#####################################################################
# INPUTS: f_min, f_max, delta_amp, max_amp
# OUTPUT: non-linear threshold

f_min      = 0.1  # [Hz]
f_max      = 2    # [Hz]
delta_amp  = 0.05 # [m]
delta_freq = 0.01 # [Hz] maximum resolution on frequency
max_amp    = 6    # [m]  assumed that max_amp>delta_amp
nl_max     = 0.3  # value of non-linear degree above which a test
                  # is considered too non-linear and not interesting 

# crate object to store upperbound of nonlinear th based on sinus tests
sinusoidal_upper_bound = NLthUpperbound(delta_amp, delta_freq, f_min, f_max)

### Find th for f_min ###
lower, upper = binary_search_sinus_freq(cfSimulation, zTest, ZAnalysis, \
                                        f_min, delta_amp, max_amp, nl_max, data_directory)
sinusoidal_upper_bound.add_sample(f_min, lower, upper) # add sample to threshold

### Find th for f_max ###
lower, upper = binary_search_sinus_freq(cfSimulation, zTest, ZAnalysis, \
                                        f_max, delta_amp, max_amp, nl_max, data_directory)
sinusoidal_upper_bound.add_sample(f_max, lower, upper) # add sample to threshold

freq = sinusoidal_upper_bound.sample()
while freq : # iteration over frequency axis
    # this function call implements the search along the amplitude axis
    lower, upper = binary_search_sinus_freq(cfSimulation, zTest, ZAnalysis, \
                                            freq, delta_amp, max_amp, nl_max, data_directory)
    # print("At freq: {} upper: {} lower: {}".format(freq,upper,lower))
    sinusoidal_upper_bound.add_sample(freq, lower, upper) # add sample to threshold
    freq = sinusoidal_upper_bound.sample()
print("Phase 1 done: I have sampled {} frequencies".format(sinusoidal_upper_bound.nlth.size))

####################################
### PHASE 2: test set generation ###
####################################
# INPUTS: shapes, f_min, f_max, nl_th, num_tests per shape
# OUTPUT: a vector of length num_tests (d,A) pairs for each shape

# uniform sampling of frequencies
num_freqs = int((f_max-f_min)/sinusoidal_upper_bound.get_freq_resolution())
freqs_under_test = np.linspace(f_min,f_max,num=num_freqs)
print("desired frequency resolution seems to be {}. Will sample {} freqs for each shape".format\
     (sinusoidal_upper_bound.get_freq_resolution(),num_freqs))

test_set = []

rnd.seed(1) # seed for repeatibility

# generate testset
for i, s in enumerate(zTest.shapes) :
    if not(s=='sinus') : # we are not interested in sinus test cases at this step
        print('generating test set for shape: '+s)
        # TODO: dt should not be hardcoded
        test_set.append(shapeTestSet(zTest,s,sinusoidal_upper_bound,0.001))
        test_set[i].generate_test_set(freqs_under_test)
        # test_set[i].plot_test_set()
    else :
        # just for consistency of vector but we don't really
        # want to generate more sinus test cases
        test_set.append(shapeTestSet(zTest,s,sinusoidal_upper_bound,0.001))

# plt.show()

########################################################
### PHASE 3: Tests Execution & check of MRs on tests ###
########################################################
# INPUTS: test set
# OUTPUT: frequency-amplitude points and associated behaviour

for i, s in enumerate(zTest.shapes) :      ## iterate over shapes
    if not(s=='sinus') :     # we do not want to run sinus test cases at this step
        for test in test_set[i].test_cases : ## iterate over test cases
            print("Running test: Shape: {} Amp_scale: {} Time_scale: {}".format(s,test['a_gain'],test['t_scale']))
            test_input = zTest(s,test['a_gain'],test['t_scale'])
            file_path = data_directory+s+'-'+str(test['a_gain'])+'-'+str(test['t_scale'])
            if not(exists(file_path)) :
                sut    = cfSimulation()      # initialize simulation object
                result = sut.run(test_input)       # test execution
                result.save(name=file_path)
            else :
                print("-> test already executed: "+file_path)

# the results of the tests can be plotted on the frequency-amplitude
# plane with the plot-ztests-fft.py script

# check MRs (plot sanity checks):
# 1. filtering frequencies have to be higher than good tracking frequencies,
# 2. non-linear degree should only increase for higher amplitudes, and
# 3. we expect a monotonic decrease in the accepted amplitudes when 
#    the frequency increases.

# iterate over shapes
for i,s in enumerate(zTest.shapes) :
    if not(s=='sinus') :     # sinus tests are handled differently since the test set
                             # was built for the threshold upper bounding rather than
                             # the sampling of the space
        print(" --- Sanity Check of "+s+" tests --- ")
        # iterate over pairs of tests (this is computationally heavy)
        for ii,test in enumerate(test_set[i].test_cases) : ## iterate over test cases
            # check if MRs apply
            test_file = s+'-'+str(test['a_gain'])+'-'+str(test['t_scale'])
            if not(exists(data_directory+test_file)) : # check if we find test file
                print("WARNING -- main-crazyflie : couldn't find test: "+test_file)
            for test2 in test_set[i].test_cases[(ii+1):] :
                test2_file = s+'-'+str(test2['a_gain'])+'-'+str(test2['t_scale'])
                if not(exists(data_directory+test2_file)) : # check if we find test file
                    print("WARNING -- main-crazyflie : couldn't find test: "+test2_file)

#############################################################
### PHASE 4: Aggregate Results and Build Characterization ###
#############################################################
# INPUTS: frequency-amplitude behaviour points
# OUTPUT: frequency-amplitude characterization

# aggregate results

# check MRs
