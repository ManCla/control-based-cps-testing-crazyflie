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

    '''
    Add frequency sample to bound to non-linearity threshold
    A sample contains a lower and upper bound of the threshold
    '''
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

    '''
    Search samples in threshold for "jumps" in threshold that are larger than delta_amp
    If found return a frequency where the jump is that should be sampled
    '''
    def sample(self):
        # check for nlth to have at least two elements
        if self.nlth.size<2 :
            print("ERROR--NLthUpperbound: need at least two samples in nlth to evaluate gaps")
            return
        for i, th in enumerate(self.nlth) :
            if i == self.nlth.size-1 : 
                print("Phase 1 done: no jumps >delta_amp in threshold preliminary evaluation")
            else :
                a_avg_prev = (  self.nlth[i]['A_max'] +   self.nlth[i]['A_min']) /2
                a_avg_next = (self.nlth[i+1]['A_max'] + self.nlth[i+1]['A_min']) /2
                if abs(a_avg_next-a_avg_prev)>self.delta_amp :
                    return (self.nlth[i]['freq'] + self.nlth[i+1]['freq']) /2

if __name__ == "__main__":
    a = NLthUpperbound(0.5, 1, 5)
    print(a.sample())
    a.add_sample(1.5,4,5)
    a.add_sample(2.5,3,4)
    a.add_sample(2.25,3.5,4.5)
    a.add_sample(3.25,3.5,4.5)
    a.add_sample(4,4,5)
    print(a.sample())
