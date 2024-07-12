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
 - percentage of time spent with saturated motors
 - percentage of time spent hitting the ground
 - test that was used to sample the point 
   (shape, amplitude_gain and time_scaling)

Note that the class is about the individual points rather than the tests!
E.g. when plotting you don't need to iterate over both tests and points,
but you can just iterate over the points.
'''

date_format    = '%Y%m%d_%H%M%S'

spectrum_amp_threshold = 0.05 # absolute value above which we consider components in spectra
use_peaks_only = True  # use an absolute threshold to select only relevant components of input to check

###################
### LOCAL TYPES ###
###################

# NOTE: shape names are assumed to be strings no longer than 20 characters
faPoint_type = np.dtype([('freq', 'f4'),\
                         ('amp', 'f4'),\
                         ('weight', 'f4'),\
                         ('deg_filtering', 'f4'),\
                         ('deg_non_lin', 'f4'),\
                         ('saturation_perc', 'f4'),\
                         ('hit_ground_perc', 'f4'),\
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

# local function to get colour associated to motors saturation time.
# NOTE: it assumes input in [0,1]
def get_motors_sat_colour(mot_sat) :
    mot_sat_color = [mot_sat,1-mot_sat,0]
    return mot_sat_color

# local function to get colour associated to hit ground time.
# NOTE: it assumes input in [0,1]
def get_hit_ground_colour(hit_ground) :
    hit_ground_color = [hit_ground,1-hit_ground,0]
    return hit_ground_color

class faCharacterization():
    
    # - nlth is the object containing the sinusoidal based upper bounding
    # of the threshold
    def __init__(self, nlth):
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
    def add_test(self, freqs, amps, deg_filtering, deg_non_lin, saturation, hit_ground,\
                 test_shape, test_tScale, test_aScale, only_main_component=False):
        # check that vectors contain info about consistent number of points
        if not(only_main_component) and not(len(freqs)==len(amps) and len(amps)==len(deg_filtering)) :
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
                         saturation,\
                         hit_ground,\
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
                     saturation,\
                     hit_ground,\
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

    '''
    returns only the points belonging to tests associated to a given shape.
    if shape='none' all points of the characterization are returned
    '''
    def get_shape_points(self, shape) :
        if shape=='none' :
            return self.faPoints
        else :
            return np.array([x for x in self.faPoints if (x['shape']==shape) ], dtype=faPoint_type)

    '''
    Function that evaluates the closed loop bandwidth (i.e.
    the frequency threshold above which input signals are not tracked)
    INPUTS
     - shape is the shape of the tests to analyse ('none' uses all tests)
     - non linear threshold is needed to exclude non-linear tests
     - show_plot triggers the plotting of the degree of filtering as function of the frequency
    '''
    def evaluate_shape_bandwidth(self, shape, non_linear_threshold, show_plot=False, title=None) :
        
        # get only points of tests belonging to one shape
        shape_points = self.get_shape_points(shape)
        # do not consider non-linear tests
        lin_points = np.array([x for x in shape_points if x['deg_non_lin']<non_linear_threshold], dtype=faPoint_type)
        # check that points are still ordered (don't think it's really needed)
        if any([lin_points['freq'][i]-lin_points['freq'][i-1]<0 for i in range(1,len(lin_points['freq']))]) :
            print("ERROR faCharacterization:eval_fb -- faPoints of given shape not ordered by frequency")
            exit()
        # print out tests that do not fulfil the MR on increasing filtering over frequencies
        # it feels very unlikely that linear tests do not fulfil the MR by track high frequencies
        # funky_tests= [x for x in lin_points if x['freq']<0.55 and x['deg_filtering']>0.5]
        # for x in funky_tests:
        #     print("funky test : ramp-"+str(x['a_gain'])+"-"+str(x['t_scale']))

        # actual analysis and plotting (if requested)
        print("Evaluate closed-loop bandwidth according to shape "+str(shape))
        print("  max tracked frequency  = "+str(max([x['freq'] for x in lin_points if x['deg_filtering']<0.5])))
        print("  min filtered frequency = "+str(min([x['freq'] for x in lin_points if x['deg_filtering']>0.5])))
        if show_plot :
            if title is None :
                title = 'Tests from shape: '+shape
            else :
                title = title+' ('+shape+')'
            _, axs = plt.subplots(1, 1)
            axs.scatter(lin_points['freq'],lin_points['deg_filtering'], s=2, c='black')
            axs.plot([0.1,2],[0.5,0.5], linestyle='dashed', c='red')
            axs.title.set_text(title)
            axs.set_xlabel('Frequency [Hz]')
            axs.set_ylabel('Degree of Filtering []')
            axs.set_xlim([0.1,2])

    ##########################
    ### PLOTTING FUNCTIONS ###
    ##########################

    '''
    plot sinusoidal-tests-based upper-bounding of non-linear threshold
    NOTE: this is used as a base for the other characterization plotting functions!
    '''
    def plot_nlth_upperbound(self) :

        _, axs = plt.subplots(1, 1)
        # plot frequency limits
        axs.plot([self.nlth.f_min,self.nlth.f_min],[0,self.nlth.get_maximum_amp()], linestyle='dashed', c='black')
        axs.plot([self.nlth.f_max,self.nlth.f_max],[0,self.nlth.get_maximum_amp()], linestyle='dashed', c='black')
        # plot linearity upper bounds as from pre-estimation
        axs.plot(self.nlth.nlth['freq'],self.nlth.nlth['A_min'])
        axs.plot(self.nlth.nlth['freq'],self.nlth.nlth['A_max'])
        # plot aesthetics
        axs.set_xlabel('Frequency [Hz]')
        axs.set_ylabel('Amplitude')
        axs.set_xscale('log')
        axs.set_yscale('log')
        axs.grid()

        return axs # used for adding more elements to the plot

    '''
    plot non-linearity characterization
     - nl_max is the nldeg threshold above which a test is considered too much non-linear
       here it is used for defining the colour gradient when plotting
    '''
    def plot_non_linearity_characterization(self, non_linear_threshold, shape='none', title=None) :

        if title is None:
            title = 'degree of non linearity'
        if not(shape=='none') :
            title = title +' ('+shape+')'
        plot_points = self.get_shape_points(shape) # get only point of the desired shape (if given)
        axs = self.plot_nlth_upperbound()
        axs.title.set_text(title)
        nldg_colours = [get_nldg_colour(x, non_linear_threshold) for x in plot_points['deg_non_lin']]
        axs.scatter(plot_points['freq'], plot_points['amp'], s=2, c=nldg_colours)

        return axs # used for adding more elements to the plot

    '''
    plot filtering characterization
    '''
    def plot_filtering_characterization(self, non_linear_threshold, shape='none') :

        title = 'degree of filtering'
        if not(shape=='none') :
            title = title +' ('+shape+')'
        plot_points = self.get_shape_points(shape) # get only point of the desired shape (if given)
        axs = self.plot_nlth_upperbound()
        axs.title.set_text(title)
        lin_points = np.array([x for x in plot_points if x['deg_non_lin']<non_linear_threshold], dtype=faPoint_type)
        dof_colours = [get_filtering_colour(x['deg_filtering']) for x in lin_points]
        axs.scatter(lin_points['freq'], lin_points['amp'], s=2, c=dof_colours)

        return axs # used for adding more elements to the plot

    '''
    plot motors saturation
    '''
    def plot_motors_saturation_characterization(self, shape='none',title=None) :

        if title is None :
            title = 'motors saturation time percentage'
        if not(shape=='none') :
            title = title +' ('+shape+')'
        plot_points = self.get_shape_points(shape) # get only point of the desired shape (if given)
        axs = self.plot_nlth_upperbound()
        axs.title.set_text(title)
        motor_sat_colours = [get_motors_sat_colour(x) for x in plot_points['saturation_perc']]
        axs.scatter(plot_points['freq'], plot_points['amp'], s=2, c=motor_sat_colours)

        return axs # used for adding more elements to the plot

    '''
    plot hit ground
    '''
    def plot_hit_ground_characterization(self, shape='none') :

        title = 'hit the ground time percentage'
        if not(shape=='none') :
            title = title +' ('+shape+')'
        plot_points = self.get_shape_points(shape) # get only point of the desired shape (if given)
        axs = self.plot_nlth_upperbound()
        axs.title.set_text(title)
        hit_ground_colours = [get_hit_ground_colour(x) for x in plot_points['hit_ground_perc']]
        axs.scatter(plot_points['freq'], plot_points['amp'], s=2, c=hit_ground_colours)

        return axs # used for adding more elements to the plot

    ################################
    ### CSV GENERATION FUNCTIONS ###
    ################################
    '''
    Function to generate csv file with the fa points data
    INPUTS:
     - shapes is needed to iterate over the shapes
     - nonlinear threshold is needed to saturate the dnl and identify non-lin tests
     - linear_only is used to output only the fapoints of tests with linear bh
                   it is needed for frequency analysis (and to not have to filter in latex)
    '''
    def generate_csv(self, shapes, non_linear_threshold, linear_only):
        for s in shapes :
            suffix = '.csv'
            shape_points = self.get_shape_points(s)
            if linear_only :
                shape_points = np.array([x for x in shape_points if x['deg_non_lin']<non_linear_threshold], dtype=faPoint_type)
                suffix = '_lin_only'+suffix
            output_filename = 'faCharacterization_'+s+suffix
            # we don't want the shape field and have to "remove" it manually
            output = shape_points[('freq')]
            output = np.vstack((output,shape_points[('amp')]))
            output = np.vstack((output,shape_points[('weight')]))
            # saturate dnl to non_linear_threshold
            output = np.vstack((output,[min(x,non_linear_threshold) for x in shape_points[('deg_non_lin')]]))
            output = np.vstack((output,shape_points[('saturation_perc')]))
            output = np.vstack((output,shape_points[('hit_ground_perc')]))
            output = np.vstack((output,shape_points[('deg_filtering')]))
            np.savetxt(output_filename, output.transpose() , delimiter=',')

if __name__ == "__main__":
    charact = faCharacterization()

    charact.add_test([1],[1],[0],0,0.5,0.5,'my_shape',2,3)
    charact.add_test([3],[1],[0],1,0.5,0.5,'my_shape',2,3)
    charact.add_test([3],[1],[0],1,0.5,0.5,'my_shape',2,3)
    charact.add_test([4],[2],[0],1,0.5,0.5,'my_shape',2,3)
    charact.add_test([2],[2],[0],1,0.5,0.5,'my_shape',2,3)
    charact.add_test([2.5],[2],[0],1,0.5,0.5,'my_shape',2,3)
    charact.add_test([3],[1],[0],1,0.5,0.5,'my_shape',2,3)

    # test if frequency ordering is working
    freqs = charact.faPoints['freq']
    deltas = [freqs[i]-freqs[i-1] for i in range(1,len(freqs))]
    if any( [ d< -0.001 for d in deltas] ) : 
        print("ERROR: faPoints vector is not ordered!")

    charact.plot_non_linearity_characterization(0.3)
    charact.plot_motors_saturation_characterization()
    plt.show()
