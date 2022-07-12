'''
Class to store and manage the upper-bound estimation
of the non-linearity threshold obtained with sinusoidal
inputs.
'''

class NLthUpperbound():
    
    def __init__(self, delta_amp, w_min, w_max):
        self.delta_amp  = delta_amp # max gap accepted between amp th samples
        self.w_min = w_min        # frequency range to explore
        self.w_max = w_max

    def add_sample(self, frequency, amplitude_min, amplitude_max):
        # add sample to threshold
        pass

    def sample(self):
        # return frequency to test because of too large gap between
        # adjacent samples
        pass
