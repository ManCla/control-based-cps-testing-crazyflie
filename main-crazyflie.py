
# utilities of testing approach
from ControlBasedTesting.binary_search_sinus_freq import binary_search_sinus_freq
from ControlBasedTesting.NLthUpperbound import NLthUpperbound

# for Crazyflie Testing
from CrazyflieSimulationPython.cfSimulator import cfSimulation, ZAnalysis, zTest

data_directory = 'cfdata/'

#####################################################################
### PHASE 1: sinusoidal-based non-linearity threshold exploration ###
#####################################################################
# INPUTS: f_min, f_max, delta_amp, max_amp
# OUTPUT: non-linear threshold

f_min     = 0.6     # [Hz]
f_max     = 4.5     # [Hz]
delta_amp = 0.25    # [m]
max_amp   = 6       # [m]  assumed that max_amp>delta_amp
nl_max    = 0.3     # value of non-linear degree above which a test
                    # is considered too non-linear and not interesting 

# crate object to store upperbound of nonlinear th based on sinus tests
sinusoidal_upper_bound = NLthUpperbound(delta_amp, f_min, f_max)

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
    sinusoidal_upper_bound.add_sample(freq, lower, upper) # add sample to threshold
    freq = sinusoidal_upper_bound.sample()

sinusoidal_upper_bound.plot()

####################################
### PHASE 2: test set generation ###
####################################
# INPUTS: shapes, f_min, f_max, nl_th
# OUTPUT: a vector of (d,A) pairs for each shape

# list of shapes
# Shape type: ...

# generate testset

########################################################
### PHASE 3: Tests Execution & check of MRs on tests ###
########################################################
# INPUTS: test set
# OUTPUT: frequency-amplitude points and associated behaviour

# run tests

# check MRs

#############################################################
### PHASE 4: Aggregate Results and Build Characterization ###
#############################################################
# INPUTS: frequency-amplitude behaviour points
# OUTPUT: frequency-amplitude characterization

# aggregate results

# check MRs
