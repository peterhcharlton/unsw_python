# Python implementation of the UNSW QRS detector algorithm


## Summary

The 'UNSW' QRS detection algorithm was recently found to be one of the highest-performing QRS detection algorithms, showing particularly strong performance on telehealth ECGs, such as those recorded using mobile devices [^4]. The original [^1] [^2] open-source algorithm was provided in MATLAB format. This repo provides a Python translation of the algorithm and examples of how to use it.

## Installation and Requirements

### Requirements

This implementation requires Python 3.6+ and the following packages:
- numpy
- scipy

### Installation

Clone this repository:

```bash
git clone https://github.com/peterhcharlton/unsw_python.git
cd unsw_python
```

Install the required dependencies:

```bash
pip install numpy scipy
```


## Usage

The Python translation provided here can either be used as a standalone tool, or as part of the Neurokit2 package, as follows.

### Using the UNSW QRS detector as a standalone tool

The [using_unsw_as_standalone_tool notebook](using_unsw_as_standalone_tool.ipynb) demonstrates how to use the UNSW QRS detector as a standalone tool.

Briefly, the algorithm can be used with the following commands:
```python
from scipy.datasets import electrocardiogram
rawecg = electrocardiogram() # load sample ECG signal
fs = 360 # in Hz, as specified at in scipy documentation
unswdata = UNSW_QRSDetector(rawecg,fs) # perform QRS detection
qrs_indices = unswdata['qrs'] # extract QRS indices
```

The accompanying notebook demonstrates this process including plotting the results.

### Using the UNSW QRS detector in Neurokit2

The [using_unsw_in_neurokit2 notebook](using_unsw_in_neurokit2.ipynb) demonstrates how to use the UNSW QRS detector as implemented in the [Neurokit2](https://neuropsychology.github.io/NeuroKit/) package.

Briefly, the algorithm can be used by specifying the `"khamis2016"` method when calling the ECG peak detection function.

```python
import neurokit2 as nk

# Simulate ECG signal
sampling_rate = 360
ecg = nk.ecg_simulate(duration=10, sampling_rate=sampling_rate)

# Detect QRS complexes and plot
signals, info = nk.ecg_peaks(ecg, sampling_rate=250, correct_artifacts=True, show=True, method="khamis2016")
```

For more information on using Neurokit2, please refer to the [official documentation](https://neuropsychology.github.io/NeuroKit/).


## Further details

The UNSW QRS detection algorithm was originally developed by Khamis et al. (2016). It was designed for both clinical ECGs and poorer quality telehealth ECGs. The Python implementation provided here is adapted from the original MATLAB implementation by Khamis et al. This Python implementation was written by Sharon Yuen Shan Ho, Zixuan Ding, David C. Wong, and Peter H. Charlton, as reported in Ho et al. (2025) [^3].

### Development Process

The starting point was the Matlab code [here](https://github.com/floriankri/ecg_detector_assessment/blob/sharon_ho/Algorithm_tester/detectors/unsw/UNSW_QRSDetector.m).

For reference, the original code is available [here](https://doi.org/10.7910/DVN/QTG0EP) under the [CC0 1.0](http://creativecommons.org/publicdomain/zero/1.0) licence (see the `UNSW_QRSDetector.m` file).

The aim was to create a python translation of the original Matlab implementation which was as close to the original as possible. Whilst there may well be ways we could improve the algorithm (e.g. in terms of efficiency, accuracy, tidyness of code), we didn't attempt to make these improvements at the moment, in order to verify that our translation is faithful to the original. Exceptions to this included:
* Removing code which is required in Matlab but is redundant in Python
* Not implementing plotting code which was not required to detect QRS complexes

We used the _compare_functions.ipynb_ notebook to compare this implementation to the matlab implementation.


## References

[^1]: Khamis, H. et al. (2016). QRS detection algorithm for telehealth electrocardiogram recordings. IEEE Transactions on Biomedical Engineering, 63(7), 1377–1388. https://doi.org/10.1109/TBME.2016.2549060

[^2]: Khamis, H. et al. (2016). TELE ECG Database: 250 Telehealth ECG Records (Collected Using Dry Metal Electrodes) with Annotated QRS and Artifact Masks, and MATLAB Code for the UNSW Artifact Detection and UNSW QRS Detection Algorithms. Harvard Dataverse, https://doi.org/doi:10.7910/DVN/QTG0EP

[^3]: Ho, S. et al. (2025). Accurate RR-interval extraction from single-lead, telehealth electrocardiogram signals. medRxiv 2025.03.10.25323655. https://doi.org/10.1101/2025.03.10.25323655

[^4]: F. Kristof et al. (2024). QRS detection in single-lead, telehealth electrocardiogram signals: Benchmarking open-source algorithms. PLOS Digital Health, vol. 3, no. 8, p. e0000538. https://doi.org/10.1371/journal.pdig.0000538.
