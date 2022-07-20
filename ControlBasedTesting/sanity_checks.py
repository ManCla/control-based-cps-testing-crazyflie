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
Sanity check for filtering
'''
def check_filtering(test1, test2):
    pass

'''
'''
def check_non_lin_amp(test1, test2):
    pass

'''
'''
def check_non_lin_freq(test1, test2):
    pass
