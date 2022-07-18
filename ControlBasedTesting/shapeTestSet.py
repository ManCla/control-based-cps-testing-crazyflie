import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import random as rnd
import matplotlib.pyplot as plt

'''
Class that takes a shape and generates the associated test set for
 a given estimation of the non-linearity threshold.
'''

peak_threshold_percentage = 0.05 # used to identify relevant peaks in input spectrum
scale_factor = 100 # used to scale random integers into floats for test case generation

'''
Class to store the frequency amplitude points of a given test
NOTE: when computing the high/low frequencies/amplitudes we exclude
      the 0Hz frequency because that one simply corresponds to
      the average of the signal.
'''
class faPointsTest(object):
    
    def __init__(self, z_ref_freq_peaks, z_ref_amp_peaks):
        if len(z_ref_freq_peaks)<2 :
            # functions for second highest and lowest peaks will not work
            # for inputs that map too only one point. But this should
            # happen only for sinusoidal inputs that we should not be
            # considering at this stage
            print("WARNING--faPointsTest: there is a non-sinusoidal test exciting only one frequency")
        self.z_ref_freq_peaks = z_ref_freq_peaks
        self.z_ref_amp_peaks  = z_ref_amp_peaks

    # return frequency with largest component in input
    def freq_of_max_amp(self) :
        return self.z_ref_freq_peaks[self.z_ref_amp_peaks[1:].argmax()+1]

    # return second highest amplitude (power at 0Hz excluded)
    def a_Highest(self) :
        return max(self.z_ref_amp_peaks[1:])

'''
Class to generate a test set for a given shape and estimated
non-linearity threshold.
The test-set is a vector of tuples of type:
(t_scale, a_scale, faPointsTest)

TODO: check if you actually need to store the faPointsTest
'''
class shapeTestSet(object):
    
    '''
    INPUTS:
     - test_gen    : class that generates the test cases
     - shape       : shape of this test set
     - nlThreshold : class containing the non-linear threshold
     - dt          : sampling time (needed for absolute values of frequency)
    '''
    def __init__(self, test_gen, shape, nlThreshold, dt):
        self.test_gen = test_gen
        self.shape = shape
        self.nlThreshold = nlThreshold
        self.dt = dt

        # This computation of the coefficient for the minimum and maximum frequency
        # and amplitude coefficients leverages the linearity of the Fourier Transform
        self.faPt11 = self.get_test_coordinates(1,1)
        # Find upper-left point in input space
        self.t_scale_min = nlThreshold.f_min/self.faPt11.freq_of_max_amp()
        self.a_gain_max = nlThreshold.get_maximum_amp()/self.faPt11.a_Highest() #
        # Find lower-right point in input space (a_min=0)
        self.t_scale_max = nlThreshold.f_max/self.faPt11.freq_of_max_amp()

    '''
    Generate the actual test set by sampling uniformly in the rectangular
    range between the points [(t_scale_min,a_gain_max), (t_scale_max,0)]
    '''
    def generate_test_set(self, num_tests):
        # init test case variables
        self.test_set_t_scale = np.zeros((num_tests))
        self.test_set_a_gain  = np.zeros((num_tests))

        # scale up float to integer
        t_min = int(self.t_scale_min*scale_factor)
        t_max = int(self.t_scale_max*scale_factor)+1
        a_min = self.nlThreshold.delta_amp
        for i in range(0,num_tests) :
            # random sampling on integers, then scaled down to get float
            test_t_scale = rnd.randint(t_min,t_max)/scale_factor
            test_f_main  = test_t_scale*self.faPt11.freq_of_max_amp()
            a_max = self.nlThreshold.get_th_at_freq(test_f_main)/self.faPt11.a_Highest()
            if a_min>a_max :
                print("ERROR shapeTestSet: a_min>a_max when generating test set for shape "+self.shape)
            rand_coef = rnd.betavariate(5,3.5)
            test_a_gain  = a_min + rand_coef*(a_max-a_min)
            # store
            self.test_set_t_scale[i] = np.array(test_t_scale)
            self.test_set_a_gain[i]  = np.array(test_a_gain)

    '''
    Given a time scaling coefficient and an amplitude coefficient
    returns the corresponding vector of frequency amplitude 
    coordinates
    '''
    def get_test_coordinates(self, t_scale, a_gain):
        # create vector for one period of input signal
        period = 2 * (self.test_gen.base_period/t_scale) # use two periods otherwise peaks are not peaks
        num_samples = int((period/self.dt)+1)
        time = np.linspace(0, period, num_samples)
        test = self.test_gen(self.shape, a_gain, t_scale) # create ref generator object
        ref = [test.refGen(t+test.settle)[2]-test.offset for t in time]

        # compute fft
        z_fft_freq = fft.fftfreq(num_samples, d=self.dt)
        z_ref_fft  = [abs(x) for x in fft.fft(ref, norm="forward", workers=-1, overwrite_x=True)]
        # spectrum is symmetric
        z_fft_freq = z_fft_freq[:len(z_fft_freq)//2]
        z_ref_fft  = z_ref_fft[:len(z_ref_fft)//2]

        # extract relevant peaks
        peak_threshold = peak_threshold_percentage * max(z_ref_fft[1:])
        ref_peaks_indexes, _  = signal.find_peaks(z_ref_fft, height= peak_threshold)
        ref_peaks_indexes     = np.hstack(([0],ref_peaks_indexes))
        z_ref_freq_peaks = np.array(z_fft_freq)[ref_peaks_indexes]
        z_ref_amp_peaks  = np.array(z_ref_fft)[ref_peaks_indexes]

        return faPointsTest(z_ref_freq_peaks, z_ref_amp_peaks)

    '''
    Given a time scaling coefficient and an amplitude coefficient
    plot the corresponding coordinates in the frequency-amplitude
    plane.
    '''
    def plot_test(self, a_gain, t_scale):
        faPt = self.get_test_coordinates(a_gain,t_scale)

        fig, axs = plt.subplots(1, 1)
        # plot frequency limits
        axs.plot([self.nlThreshold.f_min,self.nlThreshold.f_min],[0,self.nlThreshold.get_maximum_amp()], linestyle='dashed', c='black')
        axs.plot([self.nlThreshold.f_max,self.nlThreshold.f_max],[0,self.nlThreshold.get_maximum_amp()], linestyle='dashed', c='black')
        # plot linearity upper bounds as from pre-estimation
        axs.plot(self.nlThreshold.nlth['freq'],self.nlThreshold.nlth['A_min'])
        axs.plot(self.nlThreshold.nlth['freq'],self.nlThreshold.nlth['A_max'])

        axs.scatter(faPt.z_ref_freq_peaks[1:], faPt.z_ref_amp_peaks[1:], s=10)

        axs.grid()

    '''
    Plot the coordinates in the frequency-amplitude for all the test
    cases in this test set.
    '''
    def plot_test_set(self):
        # init figure for plotting
        fig, axs = plt.subplots(1, 1)
        axs.grid()
        axs.set_xscale('log')
        axs.set_yscale('log')
        axs.title.set_text("Generated Test Set for Shape: "+self.shape)
        axs.set_xlim([self.nlThreshold.f_min/2, 30])
        # axs.set_ylim([0,self.nlThreshold.get_maximum_amp()*2])
        # plot frequency limits
        axs.plot([self.nlThreshold.f_min,self.nlThreshold.f_min],[0,self.nlThreshold.get_maximum_amp()], linestyle='dashed', c='black')
        axs.plot([self.nlThreshold.f_max,self.nlThreshold.f_max],[0,self.nlThreshold.get_maximum_amp()], linestyle='dashed', c='black')
        # plot linearity upper bounds as from pre-estimation
        axs.plot(self.nlThreshold.nlth['freq'],self.nlThreshold.nlth['A_min'])
        axs.plot(self.nlThreshold.nlth['freq'],self.nlThreshold.nlth['A_max'])

        for i in range(0,len(self.test_set_t_scale)) :
            faPt = self.get_test_coordinates(self.test_set_t_scale[i],self.test_set_a_gain[i])
            axs.scatter(faPt.z_ref_freq_peaks[1:5], faPt.z_ref_amp_peaks[1:5], s=5)
