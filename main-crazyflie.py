import random as rnd            # for setting the seed of the random generator
import os                       # for handling paths
import numpy as np

# utilities of testing approach
from ControlBasedTesting.binary_search_sinus_freq import binary_search_sinus_freq
from ControlBasedTesting.NLthUpperbound import NLthUpperbound
from ControlBasedTesting.shapeTestSet import shapeTestSet
from ControlBasedTesting.faCharacterization import faCharacterization

# for Crazyflie Testing
from CrazyflieSimulationPython.cfSimulator import cfSimulation, ZAnalysis, zTest

data_directory = 'cfdata_nlmax015/'
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

rnd.seed(1) # seed for repeatability

#######################
### APPROACH INPUTS ###
#######################
f_min      = 0.1  # [Hz]
f_max      = 2    # [Hz]
delta_amp  = 0.05 # [m]  desired amplitude resolution
delta_freq = 0.05 # [Hz] minimum gap on frequency axis sampling
max_amp    = 6    # [m]  assumed that max_amp>delta_amp
nl_max     = 0.15 # value of non-linear degree above which a test
                  # is considered too non-linear and not interesting 
# shapes to be used for the test case generation
test_shapes = ['steps', 'ramp', 'trapezoidal', 'triangular']

####################################################################
### STEP 1: sinusoidal-based non-linearity threshold exploration ###
###################################################################
# INPUTS: f_min, f_max, delta_amp, max_amp
# OUTPUT: non-linear best-case bound of threshold
print("Starting Phase 1: upper bounding of non-linear threshold with sinusoidal inputs")

phase_1_tests_counter = 0

# crate object to store upperbound of nonlinear th based on sinus tests
sinusoidal_upper_bound = NLthUpperbound(delta_amp, delta_freq, f_min, f_max)

### Find th for f_min ###
print("Phase 1: amplitude binary search along minimum frequency: {}".format(f_min))
lower, upper, c= binary_search_sinus_freq(cfSimulation, zTest, ZAnalysis, \
                                          f_min, delta_amp, max_amp, nl_max, data_directory)
phase_1_tests_counter = phase_1_tests_counter+c
sinusoidal_upper_bound.add_sample(f_min, lower, upper) # add sample to threshold

### Find th for f_max ###
print("Phase 1: amplitude binary search along maximum frequency: {}".format(f_max))
lower, upper, c= binary_search_sinus_freq(cfSimulation, zTest, ZAnalysis, \
                                          f_max, delta_amp, max_amp, nl_max, data_directory)
phase_1_tests_counter = phase_1_tests_counter+c
sinusoidal_upper_bound.add_sample(f_max, lower, upper) # add sample to threshold

freq = sinusoidal_upper_bound.sample()
while freq : # iteration over frequency axis
    print("Phase 1: amplitude binary search along frequency: {}".format(freq))
    # this function call implements the search along the amplitude axis
    lower, upper, c= binary_search_sinus_freq(cfSimulation, zTest, ZAnalysis, \
                                              freq, delta_amp, max_amp, nl_max, data_directory)
    phase_1_tests_counter = phase_1_tests_counter+c
    # print("At freq: {} upper: {} lower: {}".format(freq,upper,lower))
    sinusoidal_upper_bound.add_sample(freq, lower, upper) # add sample to threshold
    freq = sinusoidal_upper_bound.sample()

print("Phase 1 done: I have sampled {} frequencies and executed {} tests".format\
    (sinusoidal_upper_bound.nlth.size,phase_1_tests_counter))

###################################
### STEP 2: test set generation ###
###################################
# INPUTS: shapes, f_min, f_max, nl_th, delta_amp
# OUTPUT: a vector of length num_tests of (d,A) pairs for each shape
print('Starting Phase 2: test set generation')

# uniform sampling of frequencies
num_freqs = int((f_max-f_min)/sinusoidal_upper_bound.get_freq_resolution())
freqs_under_test = np.linspace(f_min,f_max,num=num_freqs)
print("desired frequency resolution seems to be {}. Will sample {} freqs for each shape".format\
     (sinusoidal_upper_bound.get_freq_resolution(),num_freqs))

test_set = [] # initialize vector of shape test sets

# generate testset
for s in test_shapes :
    print('Phase 2: generating test set for shape: '+s)
    # TODO: dt should not be hardcoded
    test_set.append(shapeTestSet(zTest,s,sinusoidal_upper_bound,0.001))
    test_set[-1].generate_test_set(freqs_under_test)
    # test_set[-1].plot_test_set()

print("Phase 2 done: I have generated {} test cases for each of the {} shapes".\
      format(len(test_set[0].test_cases),len(test_set)))

###############################
### STEP 3: Tests Execution ###
###############################
# INPUTS: test set
# OUTPUT: frequency-amplitude points and associated behaviour
print('Starting Phase 3: tests execution')

# create characterization object to store the sampled points
faCharact = faCharacterization(sinusoidal_upper_bound)

for i, s in enumerate(zTest.shapes) :      ## iterate over shapes
    if s in test_shapes :
        print("Phase 3: running tests for shape {}".format(s))
        for test in test_set[i].test_cases : ## iterate over test cases
            print("Running test: Shape: {} Amp_scale: {} Time_scale: {}".format(s,test['a_gain'],test['t_scale']))
            test_input = zTest(s,test['a_gain'],test['t_scale'])
            file_path = data_directory+s+'-'+str(test['a_gain'])+'-'+str(test['t_scale'])
            if not(os.path.exists(file_path)) :
                sut    = cfSimulation()      # initialize simulation object
                result = sut.run(test_input, inital_drone_state=test_input.get_initial_state())       # test execution
                result.save(name=file_path)
            else :
                print("-> test already executed: "+file_path)
                result = ZAnalysis()
            result.open(file_path, silent=True)
            faCharact.add_test(result.get_z_fft_freq_peaks(),\
                               result.get_z_ref_fft_peaks(),\
                               result.get_z_filter_degree(),\
                               result.get_z_non_linear_degree(),\
                               result.get_motors_saturated(),\
                               result.get_hit_ground(),\
                               s, test['t_scale'], test['a_gain'])

# the results of the tests can be plotted on the frequency-amplitude
# plane with the plot-ztests-fft.py script

# Use this to store a characterization object
faCharact.save()
