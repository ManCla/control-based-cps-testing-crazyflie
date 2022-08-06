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

use_lower_th_input_analysis = True  # use an absolute threshold to select only relevant components of input to check
spectrum_amp_threshold = 0.05 # absolute value above which we consider components in spectra
                              # consider setting to delta_amp or another characterization parameter

###################
### LOCAL TYPES ###
###################

# NOTE: shape names are assumed to be strings no longer than 20 characters
faPoint_type = np.dtype([('freq', 'f4'),\
                         ('amp', 'f4'),\
                         ('weight', 'f4'),\
                         ('deg_filtering', 'f4'),\
                         ('deg_non_lin', 'f4'),\
                         ('shape', np.unicode_, 20),\
                         ('t_scale', 'f4'),\
                         ('a_gain', 'f4')
                        ])

# Type of element in lower bound threshold vector
# used only in build_lower_bound() method.
# NOTE that said lower bound is not really used, if not for plotting
# consider removing
amp_lower_bound_point_type = np.dtype([('freq', '<f4'), ('amp', '<f4')])

#######################
### LOCAL FUNCTIONS ###
#######################

# local function to get colour associated to non-linear degree.
# defines staining colours of non-linearity plot
def get_nldg_colour(nldg, nl_th) :
    nldg = min(1,nldg)     # get behaviour
    if nldg<nl_th  :
        # gradient from green to red
        non_lin_color = [nldg/nl_th,(nl_th-nldg)/nl_th,0]
    else :
        non_lin_color = [1,0,0]
    return non_lin_color

# local function to get colour associated to non-linear degree.
# defines staining colours of non-linearity plot
def get_filtering_colour(dof) :
    dof_sat = min(dof,1)
    filter_color = [0,1-dof_sat,dof_sat]
    return filter_color

# local function that computes the fa_mapping for an arbitrary input
# used both for plotting and analysis. But not used for building the
# characterization, in that case the coordinates are given from outside.
def fa_mapping_for_input(ref, dt) :
    # compute input fft
    num_samples = int(ref.duration//dt)+1
    z_fft_freq = fft.fftfreq(num_samples, d=dt)
    ref_time_series = [ref.refGen(x)[2] for x in np.linspace(0,ref.duration, num_samples)]
    z_ref_fft  = [abs(x) for x in fft.fft(ref_time_series, norm="forward", workers=-1, overwrite_x=True)]
    # spectrum is symmetric
    z_fft_freq = np.array(z_fft_freq)[:len(z_fft_freq)//2]
    z_ref_fft  = np.array(z_ref_fft)[:len(z_ref_fft)//2]

    if use_lower_th_input_analysis : # TODO: we just remove the very small components. Think it through
        ref_peaks_indexes = [i for i in range(len(z_ref_fft)) if z_ref_fft[i]>spectrum_amp_threshold ]
        freqs = z_fft_freq[ref_peaks_indexes]
        amps  = z_ref_fft[ref_peaks_indexes]
    else :
        freqs = z_fft_freq
        amps  = z_ref_fft
    return freqs, amps


class faCharacterization():
    
    # TODO: nlth at this point should be required and not just optional
    def __init__(self, df, da, nlth=0):
        self.faPoints =  np.array([], dtype=faPoint_type) # init main vector containing all the points
        self.nlth = nlth # non-linear threshold upper-bound, optional and used for plotting

        # resolution on frequency and amplitude axes: used for analysis
        self.freq_res = df
        self.amp_res  = da

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
        signal_power = sum(amps)

        i  = 0 # index to cover the freqs vector
        ii = 0 # index used in the case that we insert two adjacent points
        # first insert the tests that are in frequency ranges already explored
        for j, f in enumerate(self.faPoints['freq']) :
            while i<len(freqs) and f>freqs[i] :
                weight = amps[i]/signal_power # points local weight: a test with many points has less
                                              # local information, weight using power percentage
                point = (freqs[i],\
                         amps[i],\
                         weight,\
                         deg_filtering[i],\
                         deg_non_lin,\
                         test_shape,\
                         test_tScale,\
                         test_aScale,\
                        )
                self.faPoints = np.insert(self.faPoints,j+ii,np.array(point,dtype=faPoint_type))
                i  = i+1
                ii = ii+1
        # append remaining elements of this test
        while i<len(freqs):
            weight = amps[i]/signal_power # points local weight: a test with many points has less
                                          # local information, weight using power percentage
            point = (freqs[i],\
                     amps[i],\
                     weight,\
                     deg_filtering[i],\
                     deg_non_lin,\
                     test_shape,\
                     test_tScale,\
                     test_aScale,\
                    )
            self.faPoints = np.append(self.faPoints,np.array(point,dtype=faPoint_type))
            i = i+1

    '''
    verification function that checks if add_test has been doing a good job at inserting 
    the fa points ordered by frequency
    '''
    def check_freq_ordering(self) :
        if any([self.faPoints['freq'][i]-self.faPoints['freq'][i-1]<0 for i in range(1,len(self.faPoints['freq']))]) :
            print("ERROR faCharacterization -- faPoints not ordered by frequency")
        exit()

    ########################
    ### ANALYSIS METHODS ###
    ########################

    '''
    Function that evaluates the closed loop bandwidth (i.e.
    the frequency threshold above which input signals are not tracked)
    '''
    def evaluate_bandwidth(self) :
        # do not consider non-linear tests
        # find frequency threshold
        pass

    '''
    function that computes a lower bound in [f_min, f_max] below which we
    will consider all points to be safe
    NOTE: this  is not used since we use the number of neighbours combined with
          the amplitude to evaluate such an occurrence. Kept for plotting.
    '''
    def build_lower_bound(self) :
        # iterate over frequencies and store minimum amplitude available for each of them
        amp_lower_bound = np.array([],dtype=amp_lower_bound_point_type)
        f = self.faPoints[0]['freq']
        a = 1000 # just needs to be a large number, would make sense to use A_max
        for pt in self.faPoints[1:] :
            if pt['freq']==f : # if we are still on the previous frequency
                if pt['amp']<a :
                    a = pt['amp']
            else : # if we fond a new frequency
                amp_lower_bound = np.append(amp_lower_bound, np.array((f,a),dtype=amp_lower_bound_point_type))
                f = pt['freq']
                a = pt['amp']
        self.amp_lower_bound = amp_lower_bound

    '''
    function that finds all neighbours of a given point
    '''
    def find_neighbours(self, freq, amp) :
        # TODO: test different values for neighbours bounds
        # filter frequencies
        neighbours = [pt for pt in self.faPoints if abs(pt['freq']-freq)<1*self.freq_res]
        # filter amplitudes
        neighbours = [pt for pt in neighbours if abs(pt['amp']-amp)<1*self.amp_res]
        return np.array(neighbours, dtype=faPoint_type)

    '''
    compute how much a given component at given frequency and amplitude is likely
    to cause non-linear behaviour on its own.
    4 possibilities:
     CASE (1) : out of frequency bounds
     CASE (2) : in freq bounds and amplitude>threshold
     CASE (3) : in freq bounds and amplitude<<<threshold
     CASE (4) : in freq bounds and amplitude around threshold
    '''
    def compute_nl_deg_for_fa_point(self, freq, amp) :
        ## CASE (1)
        if freq<self.nlth.f_min : # frequency too low
            return 0 # very unlikely that such a slow input will push the sys out of linearity
        if freq>self.nlth.f_max : # frequency too high
            # now, if amplitude is also very high this is suspicious, raise warning
            if amp>self.nlth.nlth[-1]['A_max'] :
                print("WARNING - checking input and it has large high frequency component, are you sure?")
            else :
                return 0 # this should be just filtered and not really affect the system much
        # now on we can assume that we are within the frequency bounds

        ## CASE (2)
        local_nl_threshold = self.nlth.get_th_at_freq(freq)
        if amp>local_nl_threshold :
            return 1 # we are above the upper bound of the threshold, danger zone
        # now we can assume we are in the frequency bounds and below the threshold

        # find neighbours of point:
        neighbours = self.find_neighbours(freq,amp)
        print ("point (f:{},a:{}) has {} neighbours".format(freq,amp,len(neighbours)))

        ## CASE (3)
        # if they are less than 2 and the amplitude is low (guaranteed by the
        # condition above) then we are in case 3 and consider the point safe
        # TODO: consider increasing this
        if len(neighbours)<2 :
            return 0

        ## CASE (4)
        # use neighbours to evaluate potential behaviour of this component
        # TMP: visualize the neighbours
        axs = self.plot_non_linearity_characterization(0.15)
        axs.scatter(neighbours['freq'],neighbours['amp'], c='b', s=2)
        # TODO: are you sure that average here is the right thing to do?
        #       weight over the weight of the points
        #       defined as the number of mates they have?
        return np.average(neighbours['deg_non_lin'], weights=neighbours['weight'])

    '''
    check acceptance metric for an arbitrary input
    '''
    def check_input_on_characterization(self, ref, dt) :
        freqs, amps = fa_mapping_for_input(ref, dt) # compute fa mapping

        # init variable for risk evaluation of non-linear behaviour appearance
        pt_nl_risk = np.zeros((len(freqs)-1))

        # TODO: might not be best to do this one point at a time. An alternative could be to
        #       filter all the points in the characterization to those that are neat any of the input
        for i,f in enumerate(freqs[1:]) : # skip zero freq - NOTE: counter is not the index then!!
            pt_nl_risk[i] = self.compute_nl_deg_for_fa_point(f,amps[i+1])

        # TODO: maybe want to account also for filtering degree in evaluation?
        # TODO: some weighting over the input under analysis?
        return (len(amps)-1)*np.average(pt_nl_risk, weights=amps[1:])

    ########################
    ### PLOTTING METHODS ###
    ########################

    '''
    plot non-linearity characterization
     - nl_max if the nldeg threshold above which a test is considered too much non-linear
       here it is used for defining the colour gradient when plotting
    '''
    def plot_non_linearity_characterization(self, non_linear_threshold) :
        
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

        nldg_colours = [get_nldg_colour(x, non_linear_threshold) for x in self.faPoints['deg_non_lin']]
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
    plot the lower bound of the amplitude below which we consider
    all inputs in the [f_min, f_max] range accepted.
    NOTE: this is not really needed because such points will be excluded
          already by the fact that they have no neighbours
    '''
    def plot_amp_lower_bound(self, non_linear_threshold) :

        axs = self.plot_non_linearity_characterization(non_linear_threshold)

        self.build_lower_bound() # TODO: only if it hasn't already been computed
        axs.plot(self.amp_lower_bound['freq'],self.amp_lower_bound['amp'], c=[0,1,0])

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
        axs_nl = self.plot_non_linearity_characterization(non_linear_threshold)
        axs_nl.scatter(freqs, amps, s=25, c='black', marker="P")
        axs_df = self.plot_filtering_characterization(non_linear_threshold)
        axs_df.scatter(freqs, amps, s=25, c='black', marker="P")

if __name__ == "__main__":
    charact = faCharacterization()

    charact.add_test([1],[1],[0],0,'my_shape',2,3)
    charact.add_test([3],[1],[0],1,'my_shape',2,3)
    charact.add_test([3],[1],[0],1,'my_shape',2,3)
    charact.add_test([4],[2],[0],1,'my_shape',2,3)
    charact.add_test([2],[2],[0],1,'my_shape',2,3)
    charact.add_test([2.5],[2],[0],1,'my_shape',2,3)
    charact.add_test([3],[1],[0],1,'my_shape',2,3)

    # test if frequency ordering is working
    freqs = charact.faPoints['freq']
    deltas = [freqs[i]-freqs[i-1] for i in range(1,len(freqs))]
    if any( [ d< -0.001 for d in deltas] ) :
        print("ERROR: faPoints vector is not ordered!")

    charact.plot_non_linearity_characterization(0.3)
    plt.show()
