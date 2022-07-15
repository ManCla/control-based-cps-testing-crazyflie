'''
Class that takes a shape and generates the associated test set for
 a given estimation of the non-linearity threshold.
'''

class shapeTestSet(object):
    
    def __init__(self, shape, nlThreshold):
        self.shape = shape
        self.nlThreshold = nlThreshold

    '''
    Generate the actual test set by sampling uniformly in the rectangular
    range between the points [(f_min,a_max), (f_max,a_min)]
    '''
    def generate_test_set():
        pass

    '''
    Find lower-right point in input space
    '''
    def _find_f_min_a_max(self):
        # self.f_min
        # self.a_max
        pass

    '''
    Find upper-left point in input space
    '''
    def _find_f_max_a_min(self):
        # self.f_max
        # self.a_min
        pass

    '''
    Given a time scaling coefficient and an amplitude coefficient
    returns the corresponding vector of frequency amplitude 
    coordinates
    '''
    def get_test_coordinates(self, a_gain, t_scale):
        pass

    '''
    Given a time scaling coefficient and an amplitude coefficient
    plot the corresponding coordinates in the frequency-amplitude
    plane.
    '''
    def plot_test(self, a_gain, t_scale)
        pass
