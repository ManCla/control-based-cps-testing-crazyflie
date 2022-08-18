from os.path import exists      # to check if test has already been performed

# for test execution
from CrazyflieSimulationPython.cfSimulator import cfSimulation, zTest
from CrazyflieSimulationPython.cfSimulator import ZAnalysis as fdh

# directory to store tests results
preliminary_tests_directory = 'input-repetitions-evaluation-tests/'

max_num_periods = 10 # maximum number of input repetitions considered

# dirty trick to run longer test otherwise hard-coded in the zTest class
zTest.num_periods = max_num_periods

### input parameters ###
#  we use f_max and max_amp
#  the reason is that in the higher frequency tests
#  the dnl is more difficult to evaluate for faster tests
#  (as the freq of the input moves further away from the 
#  ones at which the system will respond)
reference = "sinus"   # shape function 
amplitude_scale = 6  # amplitude coefficient
time_scale = 2        # time scaling coefficient

file_name = preliminary_tests_directory+reference+'-'+str(amplitude_scale)+'-'+str(time_scale)

# initialize simulation objects
sim = cfSimulation()
ref = zTest(reference,amplitude_scale,time_scale)

print("Running long test to evaluate the number of periods needed for dnl computation")
print("Test is : (shape:{}, amp:{}, time:{})".\
      format(reference, amplitude_scale, time_scale))
print("I will simulate {} seconds of flight".format(ref.duration))

if not(exists(file_name)):
    # actual test execution
    test_results = sim.run(ref, inital_drone_state=ref.get_initial_state())
    # store simulation results
    test_results.save(name=file_name)
else :
    print("-->Test has already been executed")

# to show time-domain plots of test
# test_results = fdh()
# test_results.open(file_name,silent=True)
# test_results.show_all()

for i in range(1,max_num_periods+1):
    test_data = fdh(num_periods_anlyse=i)
    test_data.open(file_name,silent=True)
    print(" analysis with {} periods: dnl={}".format(i,test_data.get_z_non_linear_degree()))
