import matplotlib.pyplot as plt # for plotting
import random as rnd            # for setting the seed of the random generator
from os.path import exists      # to check if test has already been performed
import numpy as np

# utilities of testing approach
from ControlBasedTesting.binary_search_sinus_freq import binary_search_sinus_freq
from ControlBasedTesting.NLthUpperbound import NLthUpperbound
from ControlBasedTesting.shapeTestSet import shapeTestSet
from ControlBasedTesting.faCharacterization import faCharacterization
from ControlBasedTesting.sanity_checks import check_filtering, check_non_lin_amp, check_non_lin_freq

# for Crazyflie Testing
from CrazyflieSimulationPython.cfSimulator import cfSimulation, ZAnalysis, zTest

data_directory = 'cfdata_nlmax015/'

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


#####################################################################
### PHASE 1: sinusoidal-based non-linearity threshold exploration ###
#####################################################################
# INPUTS: f_min, f_max, delta_amp, max_amp
# OUTPUT: non-linear best-case bound of threshold
print("Starting Phase 1: upper bounding of non-linear threshold with sinusoidal inputs")

# crate object to store upperbound of nonlinear th based on sinus tests
sinusoidal_upper_bound = NLthUpperbound(delta_amp, delta_freq, f_min, f_max)

### Find th for f_min ###
print("Phase 1: amplitude binary search along minimum frequency: {}".format(f_min))
lower, upper = binary_search_sinus_freq(cfSimulation, zTest, ZAnalysis, \
                                        f_min, delta_amp, max_amp, nl_max, data_directory)
sinusoidal_upper_bound.add_sample(f_min, lower, upper) # add sample to threshold

### Find th for f_max ###
print("Phase 1: amplitude binary search along maximum frequency: {}".format(f_max))
lower, upper = binary_search_sinus_freq(cfSimulation, zTest, ZAnalysis, \
                                        f_max, delta_amp, max_amp, nl_max, data_directory)
sinusoidal_upper_bound.add_sample(f_max, lower, upper) # add sample to threshold

freq = sinusoidal_upper_bound.sample()
while freq : # iteration over frequency axis
    print("Phase 1: amplitude binary search along frequency: {}".format(freq))
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

# plt.show()
print("Phase 2 done: I have generated {} test cases for each of the {} shapes".\
      format(len(test_set[0].test_cases),len(test_set)))

################################
### PHASE 3: Tests Execution ###
################################
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
            if not(exists(file_path)) :
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
# faCharact.save()

# plot obtained characterization
faCharact.plot_non_linearity_characterization(nl_max)
faCharact.plot_filtering_characterization(nl_max)
plt.show()

##################################################
### PHASE 4: Check Properties on Tests results ###
##################################################
# INPUTS: frequency-amplitude behaviour points
# OUTPUT: pass or fail of properties
print('Starting Phase 4: properties verification')

# iterate over shapes
for i,s in enumerate(test_shapes) :
    print(" --- Sanity Check of "+s+" tests --- ")
    # iterate over pairs of tests (this is computationally heavy)
    for ii,test1 in enumerate(test_set[i].test_cases) : ## iterate over test cases
        # check if MRs apply
        test1_file = s+'-'+str(test1['a_gain'])+'-'+str(test1['t_scale'])
        if not(exists(data_directory+test1_file)) : # check if we find test file
            print("WARNING -- main-crazyflie : couldn't find test: "+test1_file)
        else :
            t1_data = ZAnalysis()
            # t1_data.open(data_directory+test1_file, silent=True)
            for test2 in test_set[i].test_cases[(ii+1):] :
                test2_file = s+'-'+str(test2['a_gain'])+'-'+str(test2['t_scale'])
                if not(exists(data_directory+test2_file)) : # check if we find test file
                    print("WARNING -- main-crazyflie : couldn't find test: "+test2_file)
                else :
                    t2_data = ZAnalysis()
                    # t2_data.open(data_directory+test2_file, silent=True)
                    check_filtering(test1['t_scale'],test2['t_scale'],t1_data, t2_data)
                    check_non_lin_amp(test1['a_gain'],test2['a_gain'],t1_data, t2_data)
                    check_non_lin_freq(test1['t_scale'],test2['t_scale'],t1_data, t2_data)
