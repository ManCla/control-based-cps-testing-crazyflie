import numpy as np # for structured arrays
import matplotlib.pyplot as plt # for plotting
import pickle as pk # for saving object
import time # for naming of data-files to save
import scipy.fft as fft

'''
Class that contains the frequency amplitude characterization
and exposes the relevant methods for using it.
The class stores the fa points with the behaviour observed
from all of the tests: for each point we have
 - fa coordinates
 - degree of filtering
 - degree of non-linearity
 - test that was used to sample the point 
   (shape, amplitude_gain and time_scaling)

Note that the class is about the individual points rather than the tests!
E.g. when plotting you don't need to iterate over both tests and points,
but you can just iterate over the points.
'''

date_format    = '%Y%m%d_%H%M%S'

spectrum_amp_threshold = 0.05 # percentage of max ref peak above which we look for more peaks in spectra
use_peaks_only = True  # use an absolute threshold to select only relevant components of input to check

###################
### LOCAL TYPES ###
###################

# NOTE: shape names are assumed to be strings no longer than 20 characters
faPoint_type = np.dtype([('freq', 'f4'),\
                         ('amp', 'f4'),\
                         ('deg_filtering', 'f4'),\
                         ('deg_non_lin', 'f4'),\
                         ('shape', np.unicode_, 20),\
                         ('t_scale', 'f4'),\
                         ('a_gain', 'f4')
                        ])

#######################
### LOCAL FUNCTIONS ###
#######################

# local function to get colour associated to non-linear degree.
# defines staining colours of non-linearity plot
def get_nldg_colour(nldg) : 
    nldg = min(1,nldg)     # get behaviour
    if nldg<0.5 :
        # gradient from blue to yellow
        non_lin_color = [2*nldg,2*nldg,2*(0.5-nldg)] # transform into rgb colour
    else :
        # gradient from yellow to red
        non_lin_color = [1,2*(0.5-(nldg-0.5)),0] # transform into rgb colour
    return non_lin_color

# local function to get colour associated to non-linear degree.
# defines staining colours of non-linearity plot
def get_filtering_colour(dof) :
    dof_sat = min(dof,1)
    filter_color = [0,dof_sat,1-dof_sat]
    return filter_color

# local function that computes the fa_mapping for an arbitrary input
# used both for plotting and analysis 
def fa_mapping_for_input(ref, dt) :
    # compute input fft
    num_samples = int(ref.duration//dt)+1
    z_fft_freq = fft.fftfreq(num_samples, d=dt)
    ref_time_series = [ref.refGen(x)[2] for x in np.linspace(0,ref.duration, num_samples)]
    z_ref_fft  = [abs(x) for x in fft.fft(ref_time_series, norm="forward", workers=-1, overwrite_x=True)]
    # spectrum is symmetric
    z_fft_freq = np.array(z_fft_freq)[:len(z_fft_freq)//2]
    z_ref_fft  = np.array(z_ref_fft)[:len(z_ref_fft)//2]

    if use_peaks_only :
        ref_peaks_indexes = [i for i in range(len(z_ref_fft)) if z_ref_fft[i]>spectrum_amp_threshold ]
        freqs = z_fft_freq[ref_peaks_indexes]
        amps  = z_ref_fft[ref_peaks_indexes]
    else :
        freqs = z_fft_freq
        amps  = z_ref_fft
    return freqs, amps


class faCharacterization():
    
    def __init__(self, nlth=0):
        self.faPoints =  np.array([], dtype=faPoint_type) # init main vector containing all the points
        self.nlth = nlth # non-linear threshold upper-bound, optional and used for plotting

    '''
    save object containing current characterization
    '''
    def save(self, name="no-name"):
        if name=="no-name" :
            # saves itself to file named as current date and time
            filename = time.strftime(date_format, time.localtime())
        else :
            # if provided, save with custom filename
            filename = name 
        filename = "characterization-"+filename
        with open(filename, "wb") as f:
            pk.dump(self, f, protocol=pk.HIGHEST_PROTOCOL)

    '''
    open file containing characterization of interest
    '''
    def open(data_location, silent=False):
        with open(data_location, 'rb') as f:
            data = pk.load(f)
        if not(silent):
            print('Read data from file: \033[4m' + data_location + '\033[0m')
        return data

    '''
    add a set of points coming from the same test
    '''
    def add_test(self, freqs, amps, deg_filtering, deg_non_lin, test_shape, test_tScale, test_aScale):
        # check that vectors contain info about consistent number of points
        if not(len(freqs)==len(amps) and len(amps)==len(deg_filtering)) :
            print("ERROR -- faCharacterization: inconsistent number of points info")
        # iterate over points
        for i in range(len(freqs)) :
            # add each of the points
            point = (freqs[i],\
                     amps[i],\
                     deg_filtering[i],\
                     deg_non_lin,\
                     test_shape,\
                     test_tScale,\
                     test_aScale,\
                    )
            self.faPoints = np.append(self.faPoints,np.array(point,dtype=faPoint_type))

    '''
    Function that evaluates the closed loop bandwidth (i.e.
    the frequency threshold above which input signals are not tracked)
    '''
    def evaluate_bandwidth(self) :
        # do not consider non-linear tests
        # find frequency threshold
        pass

    '''
    plot non-linearity characterization
    '''
    def plot_non_linearity_characterization(self) :
        
        fig, axs = plt.subplots(1, 1)
        if not(self.nlth==0) : # if sinusoidal based threshold is provided, use it to plot target area
            # plot frequency limits
            axs.plot([self.nlth.f_min,self.nlth.f_min],[0,self.nlth.get_maximum_amp()], linestyle='dashed', c='black')
            axs.plot([self.nlth.f_max,self.nlth.f_max],[0,self.nlth.get_maximum_amp()], linestyle='dashed', c='black')
            # plot linearity upper bounds as from pre-estimation
            axs.plot(self.nlth.nlth['freq'],self.nlth.nlth['A_min'])
            axs.plot(self.nlth.nlth['freq'],self.nlth.nlth['A_max'])
        # plot aesthetics
        axs.set_xscale('log')
        axs.set_yscale('log')
        axs.grid()

        nldg_colours = [get_nldg_colour(x) for x in self.faPoints['deg_non_lin']]
        axs.scatter(self.faPoints['freq'], self.faPoints['amp'], s=2, c=nldg_colours)

        return axs # used for adding more elements to the plot

    '''
    plot filtering characterization
    '''
    def plot_filtering_characterization(self, non_linear_threshold) :

        fig, axs = plt.subplots(1, 1)
        if not(self.nlth==0) : # if sinusoidal based threshold is provided, use it to plot target area
            # plot frequency limits
            axs.plot([self.nlth.f_min,self.nlth.f_min],[0,self.nlth.get_maximum_amp()], linestyle='dashed', c='black')
            axs.plot([self.nlth.f_max,self.nlth.f_max],[0,self.nlth.get_maximum_amp()], linestyle='dashed', c='black')
            # plot linearity upper bounds as from pre-estimation
            axs.plot(self.nlth.nlth['freq'],self.nlth.nlth['A_min'])
            axs.plot(self.nlth.nlth['freq'],self.nlth.nlth['A_max'])
        # plot aesthetics
        axs.set_xscale('log')
        axs.set_yscale('log')
        axs.grid()

        lin_points = np.array([x for x in self.faPoints if x['deg_non_lin']<non_linear_threshold], dtype=faPoint_type)
        dof_colours = [get_filtering_colour(x['deg_filtering']) for x in lin_points]
        axs.scatter(lin_points['freq'], lin_points['amp'], s=2, c=dof_colours)

        return axs # used for adding more elements to the plot

    '''
    plot the freq-amp mapping of a given input against the characterization.
    INPUT:
     - reference generator object
     - sampling time dt of input
     - (optional) the non-linear th. upper bound (used for plotting only)
    '''
    def plot_input_mapping_on_characterization(self, ref, dt, non_linear_threshold) :

        freqs, amps = fa_mapping_for_input(ref, dt) # compute fa mapping
        # actual plotting
        axs_nl = self.plot_non_linearity_characterization()
        axs_nl.scatter(freqs, amps, s=25, c='black', marker="P")
        axs_df = self.plot_filtering_characterization(non_linear_threshold)
        axs_df.scatter(freqs, amps, s=25, c='black', marker="P")

    '''
    check acceptance metric for an arbitrary input 
    '''
    def check_input_on_characterization() :
        pass


if __name__ == "__main__":
    charact = faCharacterization()

    charact.add_test([1],[1],[0],0,'my_shape',2,3)
    charact.add_test([1],[2],[0],1,'my_shape',2,3)
    charact.add_test([2],[1],[0],1,'my_shape',2,3)
    charact.add_test([2],[2],[0],1,'my_shape',2,3)
    charact.plot_characterization()
    plt.show()
