# Control-Based CPS Stress Tesing: Crazyflie case study

This repository is associated to the submission of the paper Stress Testing of Control-Based Cyber-Physical Systems.
It contains the the python scripts, classes and functions that implement the proposed testing apporach.

## Dependancies and Submodules

The drone model can be found in another repository (here inlcuded as a submodule): [Crazyflie model repo](https://github.com/ManCla/crazyflie-simulation-python).
Note that the integration is with the branch [z-test-approach-repo-integration](https://github.com/ManCla/crazyflie-simulation-python/tree/ztest-approach-repo-integration) of the drone model repository and that **the submodule folder needs to be renamed to _CrazyflieSimulationPython_**.

## Repository Structure and Content

The repository contains the following directories

 * _ControlBasedTesting_: contains the classes and functions that  implement the proposed testing apporach:
     * _NLthUpperbound.py_:
       upper bound degree of non-linearity threshold _class_
     * _binary\_search\_sinus\_freq.py_:
       binary search _function_
     * _faCharacterization.py_:
       frequency-amplitude characterization _class_
     * _shapeTestSet.py_:
       test set _class_


* _CrazyflieSimulationPython_: contains the submodule of the drone model.

The repository contains the followign scripts

 * _main-crazyflie.py_
 * _num-periods-eval-crazyflie.py_
 * _plot-faCharact.py_
 * _plot-ztests-fft.py_
 * _plot.py_

