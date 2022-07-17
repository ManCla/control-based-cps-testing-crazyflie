import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import matplotlib.pyplot as plt

'''
Class that takes a shape and generates the associated test set for
 a given estimation of the non-linearity threshold.
'''

peak_threshold_percentage = 0.05

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
            print("WARNING--faPointsTest: there is a non-sinusoidal test\
                                          exciting only one frequency")
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
        faPt11 = self.get_test_coordinates(1,1)
        # Find upper-left point in input space
        self.t_min = nlThreshold.f_min/faPt11.freq_of_max_amp()
        self.a_max = nlThreshold.get_maximum_amp()/faPt11.a_Highest() #
        # Find lower-right point in input space (a_min=0)
        self.t_max = nlThreshold.f_max/faPt11.freq_of_max_amp()

    '''
    Generate the actual test set by sampling uniformly in the rectangular
    range between the points [(t_min,a_max), (t_max,0)]
    '''
    def generate_test_set():
        pass

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
        # axs.set_xlim([0, 50])
        # axs.set_ylim([0,5.1])

    '''
    Plot the coordinates in the frequency-amplitude for all the test
    cases in this test set.
    '''
    def plot_test_set(self):
        pass
