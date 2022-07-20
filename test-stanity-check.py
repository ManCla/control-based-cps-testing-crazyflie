# to open test results
from CrazyflieSimulationPython.cfSimulator import ZAnalysis as fdh
# sanity check functions under test
from ControlBasedTesting.sanity_checks import check_filtering, check_non_lin_amp, check_non_lin_freq

data_directory = 'cfdata/'

test_file_name = []

# test inputs
test_file_name.append('triangular-0.19107476-1.95') # [1]
test_file_name.append('triangular-0.21019-1.86')    # [2]
test_file_name.append('triangular-0.23134285-1.99') # [3]
test_file_name.append('triangular-0.23955497-1.88') # [4]
test_file_name.append('triangular-0.2738388-1.88')  # [5]
test_file_name.append('triangular-0.46535274-1.81') # [6]
test_file_name.append('triangular-0.59653693-1.24') # [7]
test_file_name.append('triangular-0.63561916-1.25') # [8]
test_file_name.append('triangular-0.6394237-1.7')   # [9]
test_file_name.append('triangular-0.726753-1.64')   # [10]
test_file_name.append('triangular-0.73198205-1.42') # [11]
test_file_name.append('triangular-0.9388064-1.54')  # [12]
test_file_name.append('triangular-0.94483775-0.94') # [13]
test_file_name.append('triangular-0.96640795-1.14') # [14]
test_file_name.append('triangular-0.9771422-1.06')  # [15]
test_file_name.append('triangular-0.99612164-1.07') # [16]
test_file_name.append('triangular-1.0789068-1.5')   # [17]
test_file_name.append('triangular-1.0918465-1.19')  # [18]
test_file_name.append('triangular-1.1527857-1.31')  # [19]
test_file_name.append('triangular-1.1713417-1.61')  # [20]
test_file_name.append('triangular-1.1799632-1.02')  # [21]
test_file_name.append('triangular-1.2232162-0.67')  # [22]
test_file_name.append('triangular-1.252594-1.51')   # [23]
test_file_name.append('triangular-1.3000097-1.23')  # [24]
test_file_name.append('triangular-1.371891-1.03')   # [25]
test_file_name.append('triangular-1.3864089-0.61')  # [26]
test_file_name.append('triangular-1.4875317-0.83')  # [27]
test_file_name.append('triangular-1.5449678-1.21')  # [28]
test_file_name.append('triangular-1.6131307-0.9')   # [29]
test_file_name.append('triangular-1.7912475-0.78')  # [30]
test_file_name.append('triangular-2.2804132-0.38')  # [31]
test_file_name.append('triangular-2.8564863-0.31')  # [32]
test_file_name.append('triangular-3.1092021-0.33')  # [33]
test_file_name.append('triangular-4.060921-0.18')   # [34]
test_file_name.append('triangular-4.2084823-0.25')  # [35]
test_file_name.append('triangular-4.5423594-0.31')  # [36]
test_file_name.append('triangular-4.834569-0.16')   # [37]
test_file_name.append('triangular-5.3735037-0.15')  # [38]
test_file_name.append('triangular-5.5586143-0.11')  # [39]
test_file_name.append('triangular-5.8492455-0.17')  # [40]

test_number1 = 1
test_number2 = 2
test_results1 = fdh()
test_results1.open(data_directory+test_file_name[test_number1])
test_results2 = fdh()
test_results2.open(data_directory+test_file_name[test_number2])

check_filtering(test_results1, test_results2)

