import numpy as np # for structured arrays
import matplotlib.pyplot as plt # for plotting

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

class faCharacterization():
    
    def __init__(self):
        self.faPoints =  np.array([], dtype=faPoint_type) # init main vector containing all the points

    '''
    save object containing current characterization
    '''
    def store() :
        pass

    '''
    open file containing characterization of interest
    '''
    def open() :
        pass

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
    def plot_non_linearity_characterization(self, nlth=0) :
        
        fig, axs = plt.subplots(1, 1)
        if not(nlth==0) : # if sinusoidal based threshold is provided, use it to plot target area
            # plot frequency limits
            axs.plot([nlth.f_min,nlth.f_min],[0,nlth.get_maximum_amp()], linestyle='dashed', c='black')
            axs.plot([nlth.f_max,nlth.f_max],[0,nlth.get_maximum_amp()], linestyle='dashed', c='black')
            # plot linearity upper bounds as from pre-estimation
            axs.plot(nlth.nlth['freq'],nlth.nlth['A_min'])
            axs.plot(nlth.nlth['freq'],nlth.nlth['A_max'])
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
    def plot_filtering_characterization(self, nlth=0) :
        
        fig, axs = plt.subplots(1, 1)
        if not(nlth==0) : # if sinusoidal based threshold is provided, use it to plot target area
            # plot frequency limits
            axs.plot([nlth.f_min,nlth.f_min],[0,nlth.get_maximum_amp()], linestyle='dashed', c='black')
            axs.plot([nlth.f_max,nlth.f_max],[0,nlth.get_maximum_amp()], linestyle='dashed', c='black')
            # plot linearity upper bounds as from pre-estimation
            axs.plot(nlth.nlth['freq'],nlth.nlth['A_min'])
            axs.plot(nlth.nlth['freq'],nlth.nlth['A_max'])
        # plot aesthetics
        axs.set_xscale('log')
        axs.set_yscale('log')
        axs.grid()

        # TODO: exclude points associated to non-linear behaviour
        print("TODO: exclude points associated to non-linear behaviour from filtering plot")
        dof_colours = [get_filtering_colour(x) for x in self.faPoints['deg_filtering']]
        axs.scatter(self.faPoints['freq'], self.faPoints['amp'], s=2, c=dof_colours)

        return axs # used for adding more elements to the plot

    def check_input(self, freqs, amps, nlth=0) :
        axs = self.plot_non_linearity_characterization(nlth=nlth)
        axs.scatter(freqs, amps, s=25, c='black', marker="P")        

if __name__ == "__main__":
    charact = faCharacterization()

    charact.add_test([1],[1],[0],0,'my_shape',2,3)
    charact.add_test([1],[2],[0],1,'my_shape',2,3)
    charact.add_test([2],[1],[0],1,'my_shape',2,3)
    charact.add_test([2],[2],[0],1,'my_shape',2,3)
    charact.plot_characterization()
    plt.show()
