import os
import matplotlib.pyplot as plt
import numpy as np
from CrazyflieSimulationPython.cfSimulator import ZAnalysis as fdh

directory = "cfdata_nlmax015"

# script parameters
non_lin_threshold = 0.15
stain_non_linear_tests_in_filtering = True
plot_tests_above_wb = True  # tests that have no components below the
                            # bandwidth are interesting but make the
                            # plot look weird. When True all tests are included

# plotting parameters
x_label = "Frequency [Hz]"
y_label = "Amplitude [m]"
x_min   = 0.005 # used as x coordinate for zero frequency component of tests in log scale

'''
This function should implement a gradient from blue to yellow when we are below non_lin_threshold
and from yellow to red when we are above
'''
def get_nld_colour(nld) :
    # nld assumed >1
    if nld<non_lin_threshold  :
        # gradient from green to red
        color = [nld/non_lin_threshold,(non_lin_threshold-nld)/non_lin_threshold,0]
    else :
        color = [1,0,0]
    return color

if __name__ == "__main__":

    #################################
    ### generate plotting objects ###
    ### NON LINEAR BEHAVIOUR      ###
    #################################

    # figure with separated different types of tests
    non_lin_fig_shapes , non_lin_ax_shapes  = plt.subplots(4, 1, sharex=True, sharey=True)
    non_lin_fig_shapes.tight_layout()
    non_lin_fig_shapes.supxlabel(x_label)
    non_lin_fig_shapes.supylabel(y_label)
    for ax in non_lin_ax_shapes : ax.grid(color=fdh.chosen_grid_color, linestyle=fdh.chosen_grid_linestyle, linewidth=fdh.chosen_grid_linewidth)
    non_lin_ax_shapes[0].set_xscale('log')
    non_lin_ax_shapes[0].set_yscale('log')
    non_lin_ax_shapes[0].title.set_text("Non Linear Degree Steps")
    non_lin_ax_shapes[1].title.set_text("Non Linear Degree Ramp")
    non_lin_ax_shapes[2].title.set_text("Non Linear Degree Trapezoidal")
    non_lin_ax_shapes[3].title.set_text("Non Linear Degree Triangular")
    # non_lin_ax_shapes[4].title.set_text("Non Linear Degree Impulse")
    # non_lin_ax_shapes[5].title.set_text("Non Linear Degree Sinus")
    # non_lin_ax_shapes[6].title.set_text("Non Linear Degree User-Defined")

    # figure with all tests together
    non_lin_fig , non_lin_ax = plt.subplots(1, 1)
    non_lin_fig.tight_layout()
    non_lin_ax.grid(color=fdh.chosen_grid_color, linestyle=fdh.chosen_grid_linestyle, linewidth=fdh.chosen_grid_linewidth)
    plt.setp(non_lin_ax, xlabel=x_label, ylabel=y_label)
    non_lin_ax.set_xscale('log')
    non_lin_ax.set_yscale('log')
    non_lin_ax.title.set_text("Non Linear Degree All Shapes Together")

    #################################
    ### generate plotting objects ###
    ### FILTERING ACTION          ###
    #################################
    
    # figure with separated different types of tests
    filter_fig_shapes , filter_ax_shapes  = plt.subplots(4, 1, sharex=True, sharey=True)
    filter_fig_shapes.tight_layout()
    filter_fig_shapes.supxlabel(x_label)
    filter_fig_shapes.supylabel(y_label)
    for ax in filter_ax_shapes : ax.grid(color=fdh.chosen_grid_color, linestyle=fdh.chosen_grid_linestyle, linewidth=fdh.chosen_grid_linewidth)
    filter_ax_shapes[0].set_xscale('log')
    filter_ax_shapes[0].set_yscale('log')
    filter_ax_shapes[0].title.set_text("Filtering Degree Steps")
    filter_ax_shapes[1].title.set_text("Filtering Degree Ramp")
    filter_ax_shapes[2].title.set_text("Filtering Degree Trapezoidal")
    filter_ax_shapes[3].title.set_text("Filtering Degree Triangular")
    # filter_ax_shapes[4].title.set_text("Filtering Degree Impulse")
    # filter_ax_shapes[5].title.set_text("Filtering Degree Sinus")
    # filter_ax_shapes[6].title.set_text("Filtering Degree User-Defined")

    # figure with all tests together
    filter_fig , filter_ax = plt.subplots(1, 1)
    filter_fig.tight_layout()
    filter_ax.grid(color=fdh.chosen_grid_color, linestyle=fdh.chosen_grid_linestyle, linewidth=fdh.chosen_grid_linewidth)
    plt.setp(filter_ax, xlabel=x_label, ylabel=y_label)
    filter_ax.set_xscale('log')
    filter_ax.set_yscale('log')
    filter_ax.title.set_text("Filtering Degree All Shapes Together")

    ##########################################
    ### Iterate over Test Results and Plot ###
    ##########################################

    dir_content = os.listdir(directory)
    # filter out directories and look only at files
    dir_content = list(filter(lambda x:os.path.isfile(directory+'/'+x), dir_content))

    for file in dir_content:
        # file opening and data extraction
        file_path = directory+'/'+file
        data_storage = fdh()
        data_storage.open(file_path, silent=True)

        if data_storage.test=="steps" :
            plot_index = 0
        elif data_storage.test=="ramp" :
            plot_index = 1
        elif data_storage.test=="trapezoidal" :
            plot_index = 2
        elif data_storage.test=="triangular" :
            plot_index = 3
        elif data_storage.test=="impulse" :
            continue
        elif data_storage.test=="sinus" :
             continue
        elif data_storage.test=="ud1" :
            continue
        else :
            print("ERROR: Shape not recognized {}".format(data_storage.test))
            exit()

        ## get coordinates to plot
        freq_coordinates = data_storage.get_z_fft_freq_peaks()
        freq_coordinates[0] = x_min # show zero frequency to the left of the plot
        ampl_coordinates = data_storage.get_z_ref_fft_peaks()
        nld = min(1,data_storage.get_z_non_linear_degree())     # get behaviour

        # plot only tests that have at least one component below the bandwidth
        expected_bandwidth_ub = 0.9
        if plot_tests_above_wb or not((freq_coordinates[1]>expected_bandwidth_ub) and nld<non_lin_threshold) :

            ### NON LINEAR DEGREE
            # note: this is one measure for the whole test
            non_lin_color = [get_nld_colour(nld)] * len(freq_coordinates)
            non_lin_ax_shapes[plot_index].scatter(freq_coordinates, ampl_coordinates, marker='o',s=2,c=non_lin_color)
            non_lin_ax.scatter(freq_coordinates, ampl_coordinates, marker='o',s=2,c=non_lin_color)

            ### FILTERING DEGREE
            # note: this is a measure for each of the peaks of the input
            if nld>non_lin_threshold and stain_non_linear_tests_in_filtering :
                # non linear behaviour is above threshold, stain it
                filter_color = [ [1,0,0] for x in data_storage.get_z_filter_degree()]
            else :
                # colour for degree of filtering
                filter_color = [ [0,1-min(x,1),min(x,1)] for x in data_storage.get_z_filter_degree()]
            filter_ax_shapes[plot_index].scatter(freq_coordinates, ampl_coordinates, marker='o',s=2,c=filter_color)
            filter_ax.scatter(freq_coordinates, ampl_coordinates, marker='o',s=2,c=filter_color)

    plt.show()
