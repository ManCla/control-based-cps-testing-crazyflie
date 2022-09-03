from os.path import exists # to check if test has already been performed

'''
Binary search for threshold of non-linear behaviour along a given frequency
'''
def binary_search_sinus_freq(Simulator, TestCase, TestData, \
                             freq, delta_amp, max_amp, nl_max, data_directory,\
                             silent=True):
    nl_deg = 1               # initialize non-linear degree
    lower  = delta_amp       # search lower bound: we use delta_amp to avoid
                             # issues num noise/resolution and small inputs
    upper  = max_amp         # search upper bound
    amp    = (upper+lower)/2 # initialize amplitude search

    tests_counter = 0

    while abs(upper-lower)>delta_amp : # while not close enough to non-lin. th.
        test      = TestCase('sinus', amp, freq)
        file_path = data_directory+'sinus'+'-'+str(amp)+'-'+str(test.time_coef)
        if not(exists(file_path)) :
            sut    = Simulator()      # initialize simulation object
            print("executing: "+file_path)
            result = sut.run(test, inital_drone_state=test.get_initial_state())       # test execution
            result.save(name=file_path)
        elif not(silent) :
            print("test already executed: "+file_path)
        test_data = TestData()
        test_data.open(file_path, silent=True)
        nl_deg    = test_data.get_z_non_linear_degree() # get degree of non-linearity
        tests_counter = tests_counter+1 # increase counter of tests

        # binary search
        if nl_deg > nl_max :  # non-linear behaviour with large input
            upper = amp          # above this everything should be non-linear
        elif amp == max_amp : # linear behaviour with large input
            lower = amp          # search is over, for this frequency we couldn't
            upper = amp          # make the system behave non-linear
        else :                # linear behaviour with small input
            lower = amp          # below this everything should be linear
        amp = (upper+lower)/2    # binary search
        if amp<delta_amp :
            print("amp<delta_amp this should not happen")

    # this final normalization makes the threshold usable across shapes
    # NOTE: test_data.get_z_ref_fft_peaks()[1] is always the max in sinus tests
    fft_amp_ratio = amp/test_data.get_z_ref_fft_peaks()[1]
    return lower/fft_amp_ratio, upper/fft_amp_ratio, tests_counter
