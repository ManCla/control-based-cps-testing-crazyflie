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
faCharacterization_directory = 'rfr_characterizations'

### TODO: set seed for random test generation ###

#####################################################################
### PHASE 1: sinusoidal-based non-linearity threshold exploration ###
#####################################################################
# INPUTS: f_min, f_max, delta_amp, max_amp
# OUTPUT: non-linear threshold

f_min      = 0.1  # [Hz]
f_max      = 2    # [Hz]
delta_amp  = 0.05 # [m]
delta_freq = 0.05 # [Hz] minimum gap on frequency axis sampling
max_amp    = 6    # [m]  assumed that max_amp>delta_amp
                  # NOTE: this is an absolute maximum of the input, not of the (f,A) point.
nl_max     = 0.15 # value of non-linear degree above which a test
                  # is considered too non-linear and not interesting 

exclude_high_freq_tests = True  # when true, tests that involve only freq peaks above a given
                                # estimate of the bandwidth are excluded from the characterization

# number of frequency-amplitude components used for each test from the random forest regression
num_fa_components_rfr = 40
# number of classifiers used for random forest regression
num_trees_rfr = 2000

# crate object to store upperbound of nonlinear th based on sinus tests
sinusoidal_upper_bound = NLthUpperbound(delta_amp, delta_freq, f_min, f_max)

print("Starting Phase 1: upper bounding of non-linear threshold with sinusoidal inputs")

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
# INPUTS: shapes, f_min, f_max, nl_th, num_tests per shape
# OUTPUT: a vector of length num_tests (d,A) pairs for each shape

# uniform sampling of frequencies
freq_res = 0.05 # sinusoidal_upper_bound.get_freq_resolution()
num_freqs = int((f_max-f_min)/freq_res)
freqs_under_test = np.linspace(f_min,f_max,num=num_freqs)
print(" -- desired frequency resolution seems to be {}. Will sample {} freqs for each shape".format\
     (freq_res, num_freqs))

test_set = []

# generate testset
for i, s in enumerate(zTest.shapes) :
    if not(s=='sinus') : # we are not interested in sinus test cases at this step
        print('Phase 2: generating test set for shape: '+s)
        # TODO: dt should not be hardcoded
        test_set.append(shapeTestSet(zTest,s,sinusoidal_upper_bound,0.001))
        seed = i*100000+1 # used to be able to increase tests gradually while keeping randomness
        test_set[i].generate_test_set(freqs_under_test,seed)
        # test_set[i].plot_test_set()
    else :
        # just for consistency of vector but we don't really
        # want to generate more sinus test cases
        test_set.append(shapeTestSet(zTest,s,sinusoidal_upper_bound,0.001))

# plt.show()
print("Phase 2 done: I have generated {} test cases for each of the {} shapes".\
      format(len(test_set[0].test_cases),len(test_set)))

########################################################
### PHASE 3: Tests Execution & check of MRs on tests ###
########################################################
# INPUTS: test set
# OUTPUT: frequency-amplitude points and associated behaviour

# create characterization object to store the sampled points
faCharact = faCharacterization(freq_res, delta_amp, num_fa_components_rfr, num_trees_rfr, nlth=sinusoidal_upper_bound,)

for i, s in enumerate(zTest.shapes) :      ## iterate over shapes
    if not(s=='sinus') :     # we do not want to run sinus test cases at this step
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
            if not(s=='impulse' or s=='ud1') : # TMP - fix discussion on shapes for characterization generation
                faCharact.add_test(result.get_z_fft_freq_peaks(),\
                                   result.get_z_ref_fft_peaks(),\
                                   result.get_z_filter_degree(),\
                                   result.get_z_non_linear_degree(),\
                                   s, test['t_scale'], test['a_gain'])
                # include only tests with at least one relevant freq
                # TODO: condition should be lin bh and all freqs filtered
                all_filtered = result.get_z_non_linear_degree()<nl_max and min(result.get_z_filter_degree()[1:])>0.5
                if exclude_high_freq_tests and all_filtered :
                    print("I'm excluding from rf dataset: "+file_path)
                else :
                    top_freqs, top_amps = result.maxima_ref_fa_components(num_fa_components_rfr)
                    faCharact.add_test_random_forest_dataset(top_freqs, top_amps, result.get_z_non_linear_degree())

print("Phase 3: building random forest regressor")
faCharact.create_forest_regressor()
print("Phase 3: building random forest classifier")
faCharact.create_forest_classifier(nl_max)
print("Phase 3 done: finished building random forest regressor and classifier")

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

#############################################################
### PHASE 4: Aggregate Results and Build Characterization ###
#############################################################
# INPUTS: frequency-amplitude behaviour points
# OUTPUT: frequency-amplitude characterization

# Store the new characterization object if it doesn't exist already
charact_file_name = "{}faComponents_{}trees".format(num_fa_components_rfr,num_trees_rfr)
if exclude_high_freq_tests :
    charact_file_name = charact_file_name+"_noHighFreq"
charact_file_path = faCharacterization_directory+'/'+charact_file_name
if not(exists(charact_file_path)) :
    faCharact.save(charact_file_path)

# plot obtained characterization
faCharact.plot_non_linearity_characterization(nl_max)
faCharact.plot_filtering_characterization(nl_max)
plt.show()

# aggregate results

# check MRs
