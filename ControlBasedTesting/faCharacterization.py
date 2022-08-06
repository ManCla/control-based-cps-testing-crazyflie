import numpy as np # for structured arrays
import matplotlib.pyplot as plt # for plotting
import pickle as pk # for saving object
import time # for naming of data-files to save
import scipy.fft as fft
from sklearn.ensemble import RandomForestRegressor as RFR # used for random forest classification
from sklearn.ensemble import RandomForestClassifier as RFC # used for random forest classification

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
def fa_mapping_for_input(ref, dt, num_maxima) :
    # compute input fft
    num_samples = int(ref.duration//dt)+1
    z_fft_freq = fft.fftfreq(num_samples, d=dt)
    ref_time_series = [ref.refGen(x)[2] for x in np.linspace(0,ref.duration, num_samples)]
    z_ref_fft  = [abs(x) for x in fft.fft(ref_time_series, norm="forward", workers=-1, overwrite_x=True)]
    # spectrum is symmetric
    z_fft_freq = np.array(z_fft_freq)[:len(z_fft_freq)//2]
    z_ref_fft  = np.array(z_ref_fft)[:len(z_ref_fft)//2]

    #get indexes of maxima elements - exclude zero frequency
    ordered_amp_indexes = np.array(z_ref_fft[1:]).argsort()[-num_maxima:]+1
    freqs = np.array(z_fft_freq)[ordered_amp_indexes]
    amps  = np.array(z_ref_fft)[ordered_amp_indexes]
    return freqs, amps


class faCharacterization():
    
    # TODO: nlth at this point should be required and not just optional
    def __init__(self, df, da, num_comp_rfr, num_trees, nlth=0):
        self.faPoints =  np.array([], dtype=faPoint_type) # init main vector containing all the points
        self.fa_rfr =  np.array([]) # table of fa points used for random forest regression
        self.nlth = nlth # non-linear threshold upper-bound, optional and used for plotting

        # resolution on frequency and amplitude axes: used for analysis
        self.freq_res = df
        self.amp_res  = da
        # number of fa components used for random forest regression
        self.num_comp_rfr = num_comp_rfr

        # initialize random forest regressor and classifier
        self.rfr = RFR(num_trees)
        self.rfc = RFC(num_trees)

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
        filename = filename
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

    def add_test_random_forest_dataset(self, freqs, amps, deg_non_lin) :
        if not(len(freqs)==self.num_comp_rfr and len(amps)==self.num_comp_rfr) :
            print("ERROR -- faCharacterization- rand forest: incorrect number of points info")
        new_row = np.hstack((freqs,amps,deg_non_lin)) # new row to be added to table
        if self.fa_rfr.size==0 :
            self.fa_rfr = new_row
        else :
            self.fa_rfr = np.vstack((self.fa_rfr,new_row))

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
    build random forest
    '''
    def create_forest_regressor(self) :
        self.rfr = self.rfr.fit(self.fa_rfr[:,:-1],self.fa_rfr[:,-1])

    '''
    check acceptance metric for an arbitrary input
    '''
    def check_input_on_rfr(self, ref, dt) :
        freqs, amps = fa_mapping_for_input(ref, dt, self.num_comp_rfr) # compute fa mapping
        # this one should be just an execution of the obtained random tree forest
        return self.rfr.predict([np.hstack((freqs,amps))])

    '''
    build random forest
    '''
    def create_forest_classifier(self, non_linear_threshold) :
        self.rfc_non_linear_threshold = non_linear_threshold
        self.rfc = self.rfc.fit(self.fa_rfr[:,:-1],[x>non_linear_threshold for x in self.fa_rfr[:,-1]])

    '''
    check acceptance metric for an arbitrary input
    '''
    def check_input_on_rfc(self, ref, dt) :
        freqs, amps = fa_mapping_for_input(ref, dt, self.num_comp_rfr) # compute fa mapping
        # this one should be just an execution of the obtained random tree forest
        return self.rfc.predict_proba([np.hstack((freqs,amps))])


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
    plot the freq-amp mapping of a given input against the characterization.
    INPUT:
     - reference generator object
     - sampling time dt of input
     - (optional) the non-linear th. upper bound (used for plotting only)
    '''
    def plot_input_mapping_on_characterization(self, ref, dt, non_linear_threshold) :

        freqs, amps = fa_mapping_for_input(ref, dt, self.num_comp_rfr) # compute fa mapping
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
