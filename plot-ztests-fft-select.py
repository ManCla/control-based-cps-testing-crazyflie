import os
import matplotlib.pyplot as plt
import numpy as np
from CrazyflieSimulationPython.cfSimulator import ZAnalysis as fdh

directory = "cfdata_nlmax015"

'''
Script to print out tests that satisfy some properties
so that they can be inspected further.
'''

# script parameters
non_lin_threshold = 0.15

if __name__ == "__main__":

    dir_content = os.listdir(directory)
    # filter out directories and look only at files
    dir_content = list(filter(lambda x:os.path.isfile(directory+'/'+x), dir_content))

    ###########################################
    ### Iterate over Test Results and Print ###
    ###########################################

    for file in dir_content:
        # file opening and data extraction
        file_path = directory+'/'+file
        data_storage = fdh()
        data_storage.open(file_path, silent=True)

        ## get coordinates to plot
        freq_coordinates = data_storage.get_z_fft_freq_peaks()
        # freq_coordinates[0] = x_min # show zero frequency to the left of the plot
        ampl_coordinates = data_storage.get_z_ref_fft_peaks()
        nld = min(1,data_storage.get_z_non_linear_degree()) # get dnl

        if  data_storage.test=="ramp" and nld<non_lin_threshold :       # select linear ramp tests
            indexes = (freq_coordinates>0.25) & (freq_coordinates<0.55) # indexes of frequency components of interest
            # select tests with components satisfying 
            if any(np.array(data_storage.get_z_filter_degree())[indexes]>0.5) and \
                data_storage.get_hit_ground()>0.005:
                print(file)
