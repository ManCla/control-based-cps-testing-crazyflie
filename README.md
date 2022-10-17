# Control-Based CPS Stress Tesing: Crazyflie case study

This repository is associated to the submission of the paper Stress Testing of Control-Based Cyber-Physical Systems.
It contains the the python scripts and classes that implement the proposed testing apporach.
The drone model can be found in another repository (here inlcuded as a submodule): [Crazyflie model repo](https://github.com/ManCla/crazyflie-simulation-python). Note that the integration is with the branch [z-test-approach-repo-integration](https://github.com/ManCla/crazyflie-simulation-python/tree/ztest-approach-repo-integration) of the drone model repository.

Package Contents:

 * upper bound degree of non-linearity threshold _class_
 * binary search _function_
 * generate test set _function_
 * frequency-amplitude data _class_
 * frequency-amplitude characterization _class_
