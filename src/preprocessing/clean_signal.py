import numpy as np
from scipy.signal import detrend, butter, filtfilt

from src.preprocessing.filter import * 

def cleansignal(x,fs):
    # cleansignal
    # Helper function: 
        # baseline removal then high pass (0.7 Hz) filtering 
        # followed by low pass (20 Hz) filtering
    
    x = detrend(x)
    x = x.ravel() #reshape so that it is the same dimension as baseline
    
    # Remove baseline
    baseline = sortfilt1(x,int(round(0.5*fs)),50)
    meddata = x - baseline
    
    if fs>600 or fs<400:
        
        # hpf - used to eliminate dc component or low frequency drift.
        b, a = butter(7, 0.7 / (fs / 2), btype='high')
        hpdata = myfiltfilt(b,a,meddata)

        # low pass linear phase filter
        b, a = butter(7, 20 / (fs / 2), btype='low')
        lphpdata = myfiltfilt(b,a,hpdata)
    
    else: 
        # i.e fs<600 & fs>400
        # load filter coefficients, first FIR HI Pass
        numhpz = np.load('prep/store_coeff.py', allow_pickle=True)['numhpz']

        # hpf - eliminate dc component or low frequency drift
        hpdata = myfiltfilt(numhpz, 1, meddata)

        # low pass linear phase filter
        num, den = butter(8, 20 / (fs / 2), btype='low')
        lphpdata = myfiltfilt(num, den, hpdata)
            
    return lphpdata