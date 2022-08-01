'''
This file contains the functions that implement the sanity checks that
we perform on the tests. The properties that we check for are the following:

1. filtering frequencies have to be higher than good tracking frequencies,
2. non-linear degree should only increase for higher amplitudes, and
3. non-linear degree should only increase for higher frequencies.

All of the three functions take as input two test results objects (called
test1 and test2).
These objects are expected to expose the following interface:
 - TODO: write interface of test results classes
'''


'''
Sanity check for filtering action when frequency increases
'''
def check_filtering(t_scale1, t_scale2, test1, test2):
    # if t_scale2>t_scale1 : # enforce that t_scale1>t_scale2
    #     tmp = t_scale1
    #     t_scale1 = t_scale2
    #     t_scale2 = tmp
    #     tmp = test1
    #     test1 = test2
    #     test2 = tmp
    # # for f in test2.get_z_fft_freq_peaks() : # iterate over main points of input of slower test
    # if not(len(test2.get_z_fft_freq_peaks())==len(test1.get_z_fft_freq_peaks())) :
    #     print("pair of tests with different number of main frequencies")
    # return
    pass

'''
'''
def check_non_lin_amp(a_gain1, a_gain2, test1, test2):
    pass

'''
'''
def check_non_lin_freq(t_scale1, t_scale2, test1, test2):
    pass
