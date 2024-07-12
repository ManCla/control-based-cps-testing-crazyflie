# Control-Based CPS Stress Tesing: Crazyflie case study

This repository is associated to the submission of the paper Stress Testing of Control-Based Cyber-Physical Systems.
It contains the the python scripts, classes and functions that implement the proposed testing apporach.

## Dependancies and Submodules

The drone model can be found in another repository (here inlcuded as a submodule): [Crazyflie model repo](https://github.com/ManCla/crazyflie-simulation-python).
Note that the integration is with the branch [z-test-approach-repo-integration](https://github.com/ManCla/crazyflie-simulation-python/tree/ztest-approach-repo-integration) of the drone model repository and that **the submodule folder needs to be renamed to _CrazyflieSimulationPython_**.

## Repository Structure and Content

The repository contains the following directories

 * _ControlBasedTesting_: contains the classes and functions that  implement the proposed testing approach.
     * _NLthUpperbound.py_:
       Class to store and manage the upper-bound estimation of the amplitudes that cause non-linear behaviour obtained with sinusoidal inputs.
     * _binary\_search\_sinus\_freq.py_:
       Function implementing the binary search along the amplitude axis for a fixed frequency.
     * _faCharacterization.py_:
       Class that contains the frequency amplitude points. The class stores the fa points with the behaviour observed from all of the tests and exposes the relevant methods for using analysing the points.
     * _shapeTestSet.py_:
       Class that manages the testset for a given shape. It generates the test set for a given estimation of the non-linearity threshold and frequency range. It also implements plotting functionalities to visualise the sampling of the frequency-amplitude plane.
* _CrazyflieSimulationPython_: contains the submodule of the drone model.

The repository contains the following scripts

 * _main-crazyflie.py_: main script for the execution of the testing approach
 * _num-periods-eval-crazyflie.py_: script to run the preliminary evaluation of the number of tests needed to be able to compute the degree of non-linearity
 * _plot-faCharact.py_: script to open a stored frequency-amplitude characterisation (given as command line argument) object and plot it (generate same plots as in the paper)
 * _plot-ztests-fft.py_: script that analyses the results of all the tests and plots the resulting characterisation (__note__: this takes some time to run since it has to evaluate each individual test, use _plot-faCharact.py_ to get quickly the same plots as in the paper)
 * _plot.py_: plots the output of a single test given as command line input (in time and amplitude domains)

## Execution of Testing Campaign

To execute the testing campaign for the Crazyflie, just run 
```
python main-crazyflie.py
```
The script stores the intermediate results (the results of each test and the characterisation classes) [pickled](https://docs.python.org/3/library/pickle.html) object.
This allows to interrupt the process anytime and restart it with minimal waste of time.

To plot the set of frequency-amplitude points run
```
python plot-faCharact.py path/to/stored/characterisation
```

To plot the results of a given test run
```
python plot.py path/to/stored/test/results
```

To run the preliminary evaluation of the number  of periods needed for the degree of non-linearity computation run
```
python num-periods-eval-crazyflie.py
```
# Contact

Should you have any questions concerning the code or the paper, feel free to reach out to [me](https://mancla.github.io).
