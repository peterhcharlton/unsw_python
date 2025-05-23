# Python implementation of the UNSW QRS detector algorithm

## Sources

The UNSW QRS detection algorithm was originally developed by Khamis et al. (2016). It was designed for both clinical ECGs and poorer quality telehealth ECGs. The Python implementation provided here is adapted from the original MATLAB implementation by Khamis et al. (available under a CC0 licence). This Python implementation was written by Sharon Yuen Shan Ho, Zixuan Ding, David C. Wong, and Peter H. Charlton, as reported in Ho et al. (2025).

### References

- Khamis, H., Weiss, R., Xie, Y., Chang, C. W., Lovell, N. H., & Redmond, S. J. (2016).
      QRS detection algorithm for telehealth electrocardiogram recordings.
      IEEE Transactions on Biomedical Engineering, 63(7), 1377–1388.
      https://doi.org/10.1109/TBME.2016.2549060

- Khamis, H., Weiss, R., Xie, Y., Chang, C. W., Lovell, N. H., & Redmond, S. J. (2016).
      TELE ECG Database: 250 Telehealth ECG Records (Collected Using Dry Metal Electrodes) with Annotated QRS and Artifact Masks,
      and MATLAB Code for the UNSW Artifact Detection and UNSW QRS Detection Algorithms.
      Harvard Dataverse, https://doi.org/doi:10.7910/DVN/QTG0EP

- Ho, S. et al. (2025).
      Accurate RR-interval extraction from single-lead, telehealth electrocardiogram signals.
      medRxiv 2025.03.10.25323655.
      https://doi.org/10.1101/2025.03.10.25323655

## Development Process

The starting point was the Matlab code [here](https://github.com/floriankri/ecg_detector_assessment/blob/sharon_ho/Algorithm_tester/detectors/unsw/UNSW_QRSDetector.m).

For reference, the original code is available [here](https://doi.org/10.7910/DVN/QTG0EP) under the [CC0 1.0](http://creativecommons.org/publicdomain/zero/1.0) licence.

The aim was to create a python translation of the original Matlab implementation which was as close to the original as possible. Whilst there may well be ways we could improve the algorithm (e.g. in terms of efficiency, accuracy, tidyness of code), we didn't attempt to make these improvements at the moment, in order to verify that our translation is faithful to the original. Exceptions to this included:
* Removing code which is required in Matlab but is redundant in Python
* Not implementing plotting code which was not required to detect QRS complexes

We used the following notebooks to briefly check the implementation:
* unsw_ecg_detector_example.ipynb: used to check the implementation
* compare_functions.ipynb: used to compare this implementation to the matlab implementation
