# Python translation of the UNSW QRS detector algorithm

This repo is for work on developing a python translation of the UNSW algorithm.

The starting point is the code [here](https://github.com/floriankri/ecg_detector_assessment/blob/sharon_ho/Algorithm_tester/detectors/unsw/UNSW_QRSDetector.m).

For reference, the original code is available [here](https://doi.org/10.7910/DVN/QTG0EP) under the [CC0 1.0](http://creativecommons.org/publicdomain/zero/1.0) licence.

The aim is to create a python translation of the original Matlab implementation which is as close to the original as possible. Whilst there may well be ways we could improve the algorithm (e.g. in terms of efficiency, accuracy, tidyness of code), I would suggest that we don't attempt to make these improvements at the moment. Keeping the translation as close to the original as possible will help us to verify that our translation is faithful to the original. Exceptions to this include:
* Removing code which is required in Matlab but is redundant in Python
* Achieving PEP 8 compliance
