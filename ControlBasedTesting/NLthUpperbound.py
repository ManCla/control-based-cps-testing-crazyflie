import numpy as np
import matplotlib.pyplot as plt
from CrazyflieSimulationPython.cfSimulator import ZAnalysis as fdh # just used for some plotting parameters

'''
Class to store and manage the upper-bound estimation
of the non-linearity threshold obtained with sinusoidal
inputs.
'''

# Type of element in threshold vector
th_sample_type = np.dtype([('freq', '<f4'), ('A_min', '<f4'), ('A_max', '<f4')])

# Some plotting parameters
x_label = "Frequency [Hz]"
y_label = "Amplitude [m]"
x_min   = 0.005 # used as x coordinate for zero frequency component of tests in log scale

class NLthUpperbound():
    
    def __init__(self, delta_amp, delta_freq, f_min, f_max):
        self.delta_amp  = delta_amp  # max gap accepted between amp th samples
        self.delta_freq = delta_freq # min gap accepted between amp th samples
        self.f_min = f_min           # frequency range to explore
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
        ## TODO check if frequency has already been sampled
        if frequency in self.nlth['freq'] :
            print("WARNING--NLthUpperbound: trying to add the same frequency twice")
            return

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
    If found, returns a frequency where the jump is that should be sampled.
    If something is wrong or search is over returns False
    '''
    def sample(self):
        # check for nlth to have at least two elements
        if self.nlth.size<2 :
            print("ERROR--NLthUpperbound: need at least two samples in nlth to evaluate gaps")
            return False
        for i, th in enumerate(self.nlth) :
            if i == self.nlth.size-1 : 
                print("Phase 1 done: no jumps >delta_amp in threshold preliminary evaluation")
                return False
            else :
                a_avg_prev = (  self.nlth[i]['A_max'] +   self.nlth[i]['A_min']) /2
                a_avg_next = (self.nlth[i+1]['A_max'] + self.nlth[i+1]['A_min']) /2
                amp_gap  = abs(a_avg_next-a_avg_prev)>self.delta_amp
                freq_gap = abs(self.nlth[i+1]['freq']-self.nlth[i]['freq'])>self.delta_freq
                if amp_gap and freq_gap :
                    return (self.nlth[i]['freq'] + self.nlth[i+1]['freq']) /2

    '''
    Plot upper and lower bounds
    '''
    def plot(self) :
        # 
        non_lin_fig , non_lin_ax = plt.subplots(1, 1)
        non_lin_fig.tight_layout()
        non_lin_ax.grid(color=fdh.chosen_grid_color, linestyle=fdh.chosen_grid_linestyle, linewidth=fdh.chosen_grid_linewidth)
        plt.setp(non_lin_ax, xlabel=x_label, ylabel=y_label)
        non_lin_ax.set_xscale('log')
        non_lin_ax.set_yscale('log')
        # non_lin_ax.set_ylim(0,6.3)
        non_lin_ax.set_xlim(self.f_min,self.f_max)
        non_lin_ax.title.set_text("Non Linear Degree All Shapes Together")

        non_lin_ax.plot(self.nlth['freq'],self.nlth['A_min'])
        non_lin_ax.plot(self.nlth['freq'],self.nlth['A_max'])

        plt.show()

    '''
    return maximum amplitude accepted by upper bound
    '''
    def maximum_amp(self) :
        maximum_amp = 0
        for th in self.nlth :
            if th['A_max']>maximum_amp :
                maximum_amp = th['A_max']
        return maximum_amp

if __name__ == "__main__":
    a = NLthUpperbound(0.5, 1, 5)
    print(a.sample())
    a.add_sample(1.5,4,5)
    a.add_sample(2.5,3,4)
    a.add_sample(2.25,3.5,4.5)
    a.add_sample(3.25,3.5,4.5)
    a.add_sample(4,4,5)
    print(a.sample())
