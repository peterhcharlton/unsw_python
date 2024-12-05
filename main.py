# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:34:47 2024

@author: hssdwo
"""


import numpy as np
import matplotlib.pyplot as plt
from UNSW_QRSDetector import UNSW_QRSDetector

import scipy.io #import packages for example

matlab_file_path = './MIMIC_PERform_1_min_normal.mat'


mat_data = scipy.io.loadmat(matlab_file_path)  # Replace with your file path
# Access the ECG signal and sampling frequency from the 'data' structure
ecg_signal = mat_data['data']['ekg'][0,0]['v'][0][0]  # 'v' contains the ECG signal
fs = mat_data['data']['ekg'][0,0]['fs'][0][0][0][0]  # 'fs' contains the sampling frequency

# Create a time vector for plotting
time = np.arange(len(ecg_signal)) / fs

# Plot the ECG signal
plt.plot(time, ecg_signal)
plt.xlabel('Time (s)')
plt.ylabel('ECG (mV)')
plt.title('ECG Signal')
plt.show()

unswdata = UNSW_QRSDetector(ecg_signal,fs,mask=None,isplot=False)