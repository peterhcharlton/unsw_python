# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:39:17 2024

@author: hssdwo
"""

import numpy as np
from scipy.signal import detrend, butter, filtfilt
import matplotlib.pyplot as plt

from src.preprocessing.clean_signal import *

def turning_points(x, threshold):
    ##
    # turning_points
    # Helper function - finds the locations of peaks and troughs of x 
    # according to the threshold
    #
    # Original version Stephen Redmond
    # Modified by Philip de Chazal 30/5/07
    #

    x = np.append(x, [x[-1] + np.finfo(float).eps, x[-1]])
    
    #tps=1 when at a peak and tps=-1 when at a through, tps=0 elsewhere
    tps = np.concatenate(([0], -np.sign(np.diff(np.sign(np.diff(x)))), [0]))
    tpidx = np.where(tps != 0)[0]
    
    # index of all turning points
    pkth = tps[tpidx]
    
    # start searching for turning point using threshold
    i = 0
    inpeak = 0
    possibleidx = None
    ref = x[0]
    confirmed = np.full(len(pkth), np.nan)
    k = 0
    
    while i < len(tpidx):
        
        # The aim of the following code is to eliminate all the local peaks and
        # troughs. A local peak or trough occurs when the height difference
        # between a peak and trough is less than 'threshold's
        
        # find first pk/tr
        if inpeak == 0 and abs(x[tpidx[i]] - ref) > threshold:
            inpeak = pkth[i]
            possibleidx = tpidx[i]
            ref = x[tpidx[i]]

        # if looking for peak
        if inpeak == 1 and (ref - x[tpidx[i]]) > threshold:
            # peak found when next trough is more then threshold away from
            # current peak
            confirmed[k] = possibleidx * tps[possibleidx]
            k += 1
            # this lower point could be next trough
            possibleidx = tpidx[i]
            ref = x[tpidx[i]]
            inpeak = -1
        elif inpeak == 1 and x[tpidx[i]] > ref:
            possibleidx = tpidx[i]
            ref = x[tpidx[i]]

        # if looking for trough
        if inpeak == -1 and (x[tpidx[i]] - ref) > threshold:
            # trough found when next peak is more then threshold away from
            # current trough
            confirmed[k] = possibleidx * tps[possibleidx]
            k += 1
            # this higher point could be next peak
            possibleidx = tpidx[i]
            ref = x[tpidx[i]]
            inpeak = 1
        elif inpeak == -1 and x[tpidx[i]] < x[possibleidx]:
            possibleidx = tpidx[i]
            ref = x[tpidx[i]]

        i += 1
    
    confirmed = confirmed[~np.isnan(confirmed)].astype(int)
    
    return confirmed

def calculate_rr_interval(qrs, mask, fs):
    rr_list = []
    n_sections = 1

    for k in range(len(qrs) - 1):
        range_vals = np.arange(qrs[k], qrs[k + 1])
        if not np.intersect1d(range_vals, mask).size:
            rr_list.append(qrs[k + 1] - qrs[k])
        else:
            n_sections += 1

    rr_list = np.array(rr_list)
    rr_list = rr_list[rr_list < 5 * fs]
    
    if len(rr_list) > 0:
        m_rr = np.mean(rr_list)
        n_rr = len(rr_list)
    else:
        m_rr = np.nan
        n_rr = 0

    return m_rr, rr_list, n_rr, n_sections

def smashECG(ECG,mask):
    # smashECG
    # Helper function - splits ECG into clean sections according to the mask
    # mask is the sample locations of null signal
    
    expandedMask = np.zeros(len(ECG) + 2, dtype=int)  # Length of ECG + 2
    expandedMask[0] = 1
    expandedMask[-1] = 1
    expandedMask[mask] = 1
    
    startend = np.diff(expandedMask)
    starts = np.where(startend == -1)[0]
    ends = np.where(startend == 1)[0] - 1
    
    smashedECG = []
    for i in range(len(starts)):
        smashedECG.append(ECG[starts[i]:ends[i] + 1])    
    
    return smashedECG



def smashedFFT(smashedECG, fs, M, is_plot):
    Pxx = np.zeros(M)
    a = []
    
    for i in range(len(smashedECG)):
        x = np.array(smashedECG[i]).flatten()  # Flatten to 1D array
        # Number of 2s windows (P) in data rounded up
        P = int(np.ceil(len(x) / (2 * fs)))
        y = np.zeros(2 * fs * P)
        y[:len(x)] = x
        y = y.reshape((P, 2 * fs))
        
        y = y - np.mean(y, axis=1, keepdims=True)  # Center the data
        a.append(len(x))
        z = y
        
        Z = np.zeros(M)
        for j in range(P):
            X = np.fft.fft(z[j, :], M)
            A = np.abs(X)
            if np.sum(A) > 0:
                A = A / np.sum(A)  # Normalize by area
            else:
                A.fill(0)
            
            Z = Z + A
            
            if j == 0 or j == 1:
                if is_plot == 1:
                    n = 4
                    f = np.arange(1, len(A) + 1) * fs / len(A)
                    ax = plt.subplot(n, 2, 2 + j)
                    ax.plot(f, A, 'k-', linewidth=2)
                    ax.set_xlim([0, 20])
                    ax.set_xticks(np.arange(0, 21, 5))
                    ax.set_xticklabels(np.arange(0, 21, 5))
                    ax.set_xlabel('Frequency (Hz)')
                    ax.set_ylabel('Normalized |DFT|')

        Z = Z / P
        Pxx = Pxx + (a[i] * Z)
    
    F = Pxx / np.sum(a)
    
    if is_plot == 1:
        ax = plt.subplot(n, 1, 3)
        ax.plot(f, F, 'k-', linewidth=2)
        ax.set_xlim([0, 20])
        ax.set_xticks(np.arange(0, 21, 5))
        ax.set_xticklabels(np.arange(0, 21, 5))
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('P[k]')
    
    plt.show()
    
    return F

