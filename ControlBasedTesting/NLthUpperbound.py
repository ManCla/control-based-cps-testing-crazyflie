import numpy as np

'''
Class to store and manage the upper-bound estimation
of the non-linearity threshold obtained with sinusoidal
inputs.
'''

th_sample_type = np.dtype([('freq', '<f4'), ('A_min', '<f4'), ('A_max', '<f4')])

class NLthUpperbound():
    
    def __init__(self, delta_amp, f_min, f_max):
        self.delta_amp  = delta_amp # max gap accepted between amp th samples
        self.f_min = f_min          # frequency range to explore
        self.f_max = f_max

        # init vector for the [freq, A_min, A_max] triplets
        self.nlth = np.array([], dtype=th_sample_type)

    def add_sample(self, frequency, amplitude_min, amplitude_max):
        # check if frequency is in range. Rise just warning otherwise
        if frequency<self.f_min or frequency>self.f_max :
            print("WARNING--NLthUpperbound: you sampled outside out of the freq bounds")
        # 
        for i, th in enumerate(self.nlth) :
            if frequency<th['freq'] :
                self.nlth = np.insert(self.nlth, i, [(frequency, amplitude_min, amplitude_max)])
                return
        # if we get here it means we have reached the end of the thresholds vector
        self.nlth = np.append(self.nlth,np.array((frequency, amplitude_min, amplitude_max),\
                                                 dtype=th_sample_type))

    def sample(self):
        # return frequency to test because of too large gap between
        # adjacent samples
        pass

if __name__ == "__main__":
    a = NLthUpperbound(1, 1, 5)
    a.add_sample(1.5,4,5)
    a.add_sample(2.5,4,5)
    a.add_sample(2,4,5)
    a.add_sample(4,4,5)
    print(a.nlth)
