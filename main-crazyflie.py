import matplotlib.pyplot as plt # for plotting
import random as rnd            # for setting the seed of the random generator
from os.path import exists      # to check if test has already been performed

# utilities of testing approach
from ControlBasedTesting.binary_search_sinus_freq import binary_search_sinus_freq
from ControlBasedTesting.NLthUpperbound import NLthUpperbound
from ControlBasedTesting.shapeTestSet import shapeTestSet

# for Crazyflie Testing
from CrazyflieSimulationPython.cfSimulator import cfSimulation, ZAnalysis, zTest

data_directory = 'cfdata/'
rnd.seed(1)

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
while freq :
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

num_tests = 2 # number of randomly generated tests per shape

test_set = []

# generate testset
i=0
for s in zTest.shapes :
    if not(s=='sinus') : # we are not interested in sinus test cases at this step
        print('generating test set for shape: '+s)
        test_set.append(shapeTestSet(zTest,s,sinusoidal_upper_bound,0.001))
        test_set[i].generate_test_set(num_tests)
        # test_set[i].plot_test_set()
        i=i+1

# plt.show()

########################################################
### PHASE 3: Tests Execution & check of MRs on tests ###
########################################################
# INPUTS: test set
# OUTPUT: frequency-amplitude points and associated behaviour

i=0
for s in zTest.shapes :      ## iterate over shapes
    if not(s=='sinus') :     # we do not want to run sinus test cases at this step
        for test in test_set[i].test_cases : ## iterate over
            #
            print("Running test: Shape: {} Amp_scale: {} Time_scale: {}".format(s,test['a_gain'],test['t_scale']))
            test_input = zTest(s,test['a_gain'],test['t_scale'])
            file_path = data_directory+s+'-'+str(test['a_gain'])+'-'+str(test['t_scale'])
            if not(exists(file_path)) :
                sut    = cfSimulation()      # initialize simulation object
                result = sut.run(test_input)       # test execution
                result.save(name=file_path)
            else :
                print("test already executed: "+file_path)
        i=i+1

# check MRs

#############################################################
### PHASE 4: Aggregate Results and Build Characterization ###
#############################################################
# INPUTS: frequency-amplitude behaviour points
# OUTPUT: frequency-amplitude characterization

# aggregate results

# check MRs
