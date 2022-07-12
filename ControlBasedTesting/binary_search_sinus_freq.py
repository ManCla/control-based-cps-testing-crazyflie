from os.path import exists # to check if test has already been performed

'''
Binary search for threshold of non-linear behaviour along a given frequency
'''
def binary_search_sinus_freq(Simulator, TestCase, TestData, \
                             freq, delta_amp, max_amp, nl_max, data_directory):
    nl_deg = 1       # initialize non-linear degree
    amp    = max_amp # initialize amplitude search
    lower  = 0       # search lower bound
    upper  = max_amp # search upper bound

    while abs(upper-lower)>delta_amp : # while not close enough to non-lin. th.
        test      = TestCase('sinus', amp, freq)
        file_path = data_directory+'sinus'+'-'+str(amp)+'-'+str(test.time_coef)
        if not(exists(file_path)) :
            sut    = Simulator()      # initialize simulation object
            result = sut.run(test)       # test execution
            result.save(name=file_path)
        else :
            print("test already executed: "+file_path)
        test_data = TestData()
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

    return lower, upper
