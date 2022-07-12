from os.path import exists # to check if test has already been performed

# for Crazyflie Testing
from CrazyflieSimulationPython.cfSimulator import cfSimulation, ZAnalysis, zTest

data_directory = 'cfdata/'

#####################################################################
### PHASE 1: sinusoidal-based non-linearity threshold exploration ###
#####################################################################
# INPUTS: f_min, f_max, delta_amp, max_amp
# OUTPUT: non-linear threshold

f_min     = 0.01    # [Hz]
f_max     = 15      # [Hz]
delta_amp = 0.1     # [m]
max_amp   = 6       # [m]  assumed that max_amp>delta_amp
nl_max    = 0.3     # value of non-linear degree above which a test
                    # is considered too non-linear and not interesting 

# crate object to store upperbound of nonlinear th based on sinus tests
# sinusoidal_upper_bound = NLthUpperbound(delta_amp, f_min, f_max)

### Find th for f_min ###
nl_deg = 1       # initialize non-linear degree
amp    = max_amp # initialize amplitude search
lower  = 0       # search lower bound
upper  = max_amp # search upper bound

while abs(upper-lower)>delta_amp : # while not close enough to non-lin. th.
    test      = zTest('sinus', amp, 2)
    file_path = data_directory+'sinus'+'-'+str(amp)+'-'+str(test.time_coef)
    if not(exists(file_path)) :
        sut    = cfSimulation()      # initialize simulation object
        result = sut.run(test)       # test execution
        result.save(name=file_path)
    else :
        print("test already executed: "+file_path)
    test_data = ZAnalysis()
    test_data.open(file_path)
    nl_deg    = test_data.get_z_non_linear_degree() # get degree of non-linearity

    # binary search
    if nl_deg > nl_max : # non-lin behaviour with large input
        upper = amp         # above this everything should be non-linear
    elif amp == max_amp : # lin behaviour with large input
        lower = amp         # search is over, for this frequency we couldn't
        upper = amp         # make the system behave non-linear
    else : # linear behaviour with small input
        lower = amp         # below this everything should be linear
    amp = (upper+lower)/2   # binary search
    print("upper: "+str(upper)+" lower: "+str(lower))

# sinusoidal_upper_bound.add_sample(f_min, lower, upper) # add sample to threshold

# find th for f_max


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
