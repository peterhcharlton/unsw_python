# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:37:21 2024

@author: hssdwo
"""

import datetime  # for runtime assessment
import numpy as np
from scipy.signal import filtfilt, lfilter
from scipy.signal.windows import hamming
from helper import cleansignal, sortfilt1, smashECG, smashedFFT, turning_points, calculate_rr_interval

def UNSW_QRSDetector(rawecg,fs,mask=None,isplot=False):
    
    starttime = datetime.datetime.now()
    
    if fs<50:
        raise Exception('This function requires a sampling rate of at least 50 Hz')
        
    if mask is None:
        finalmask = []
        print('You have not entered a mask. Continuing without.')
    else:
        finalmask = mask
    
    # Clean up Signal - hi pass, then low pass filter
    lphpdata = cleansignal(rawecg,fs)
    
    # Differentiator (to emphasise QRS)
    NumDiff = np.array([1, 0, -1]) / (2 * (1 / fs))
    diffdata = lfilter(NumDiff, [1], lphpdata)
    
    # Sort filter
    top = sortfilt1(lphpdata,round(fs*0.1),100)
    bot = sortfilt1(lphpdata,round(fs*0.1),0)
    envelope = (top-bot)
    envelope[envelope < 0] = 0
    feature = np.abs(diffdata * envelope)
    
    if isplot==1:
        raise Exception('Haven''t coded up this functionality from matlab')
    
    fc=6
    k=((fs/2)*0.0037)/fc
    Nh = round(k*fs)  # Heuristic for Hamming window 3dB point
    b = hamming(Nh, sym=True)
    # Approx fc Hz low pass
    b=b/np.sum(np.abs(b))
    a=[1]
    diffpower1 = np.abs(filtfilt(b, a, feature)) ** 0.5
    
    if isplot==1:
        raise Exception('Haven''t coded up this functionality from matlab')
    
    smashedSig = smashECG(diffpower1, finalmask)
    F = smashedFFT(smashedSig, fs, 2**14, isplot)
    f = np.arange(0, 2**13) * fs / 2**14
    range_indices = np.where((f >= 0.1) & (f < 4))[0]
    max_idx = np.argmax(F[range_indices])
    fftHRfreq = f[range_indices[max_idx]]
    
    # Update smoother feature signal
    HRmin = 1.5  # Hz: 90 BPM  (don't want to go much below this, will miss ectopics otherwise!)
    HRmax = 4.0  # Hz: 240 BPM  (Shouldn't see much above this)
    fc = np.median(2 * np.array([HRmin, fftHRfreq, HRmax]))  # Kills half of second harmonic and all of the rest
    k = ((fs / 2) * 0.0037) / fc
    Nh = round(k * fs)  # Heuristic for Hamming window 3dB point
    b = hamming(Nh, sym=True)
    b = b / np.sum(np.abs(b))
    a = [1]  # Approx fc Hz low pass 
    diffpower2 = np.abs(filtfilt(b, a, feature)) ** 0.5
    
    # Detect QRS points

    # Where isn't masked
    # Find valid indices
    valididx = np.setdiff1d(np.arange(len(diffpower2)), finalmask)
    # Maximum filter to get upper envelope
    Wsort = round(np.median(2 * np.array([fs, fs / fftHRfreq, fs / HRmax])))
    upperenv = sortfilt1(diffpower2, Wsort, 100)  # Morph Open
    upperenv = sortfilt1(upperenv, Wsort, 0)     # Morph Close
    lowerenv = sortfilt1(diffpower2, Wsort, 0)   # Morph Open
    lowerenv = sortfilt1(lowerenv, Wsort, 100)  # Morph Close
    QRSenv = upperenv - lowerenv

    # First pass: high threshold
    sensitivity = 1
    featureHeight = np.median(QRSenv[valididx])
    mainThreshold = 0.2 * featureHeight
    threshold = mainThreshold / sensitivity
    
    if isplot==1:
        raise Exception('Haven''t coded up this functionality from matlab')
    
    tpsidx = turning_points(diffpower2, threshold)
    qrs = tpsidx[tpsidx > 0]
    qrs = np.setdiff1d(qrs, finalmask)
    m_rr, rr_list, n_rr, n_sections = calculate_rr_interval(qrs, finalmask, fs)
    
    # Back-track to find possible missed beats
    # Lower threshold
    sensitivity = 2
    threshold = mainThreshold / sensitivity
    tpsidx = turning_points(diffpower2, threshold)
    qrs2 = tpsidx[tpsidx > 0]

    # Fill in gaps
    newmask = np.union1d(finalmask, qrs)
    newmask = np.array([int(i) for i in newmask]) # convert to integer array
    temp = np.zeros(len(diffpower2) + 2)
    temp[newmask + 1] = 1
    temp = np.concatenate(([1], temp, [1]))
    
    # Fill in sections less than 1.5*mRR seconds;

    for i in range(1, len(temp)):
        if temp[i] - temp[i - 1] == -1:
            start = i
        if temp[i] - temp[i - 1] == 1 and (i - 1) - start < 1.5 * m_rr:
            temp[start:(i - 1)] = 1
    newmask = np.where(temp[1:-1] == 1)[0]

    # Only keep newly found QRS between long beats
    qrs2 = np.setdiff1d(qrs2, newmask)
    # Add them to the old QRS (found using low sensitivity)
    qrs = np.union1d(qrs, qrs2)
    m_rr, rr_list, n_rr, n_sections = calculate_rr_interval(qrs, finalmask, fs)

    # 3rd pass: Highest threshold
    # Back-track to remove possible wrong beats
    sensitivity = 0.5
    threshold = mainThreshold / sensitivity
    tpsidx = turning_points(diffpower2, threshold)
    qrs3 = tpsidx[tpsidx > 0]
    qrs3 = np.setdiff1d(qrs3, finalmask)

    # find indices of RR intervals that are too short
    short_rr_idx = []
    for i in range(1, len(qrs)):
        if qrs[i] - qrs[i - 1] < 0.5 * m_rr:
            short_rr_idx.extend(range(qrs[i - 1], qrs[i]))

    qrs3 = np.intersect1d(qrs3, short_rr_idx)
    qrs = np.setdiff1d(qrs, short_rr_idx)
    qrs = np.union1d(qrs, qrs3)
    m_rr, rr_list, n_rr, n_sections = calculate_rr_interval(qrs, finalmask, fs)

    if isplot == 1:
        raise Exception('Haven''t coded up this functionality from matlab')

    unswdata = {
        'qrs': qrs,
        'RRlist': rr_list,
        'nRR': n_rr,
        'mRR': m_rr,
        'nSections': n_sections,
        'runtime': datetime.datetime.now() - starttime
    }
    
    return unswdata