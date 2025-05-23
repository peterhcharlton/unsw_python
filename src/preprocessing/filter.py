import numpy as np
from scipy.signal import detrend, butter, filtfilt


def sortfilt1(x,n,p):
    
    N = len(x)
    
    if p>100:
        p=100
    elif p<0:
        p=0
    
    if n % 2 == 0:
        N1 = int((n/2)-1)
        N2 = int((n/2))
    else:
        N1 = int((n-1)/2)
        N2 = int((n-1)/2)
    
    y = np.zeros_like(x)
    
    USE_MEX = 0 # changed from original to force use of python implementation
    if USE_MEX:
        raise Exception('Haven''t included the functionality from Matlab to use the c-code functions')
    else:
        y = np.zeros(N)
        for i in range(1,N+1):
            A = max(1, i-N1)
            B = min(N, i+N2)
            P = 1 + round((p/100)*(B-A))
            #print(i)
            #print(A)
            #print(B)
            #print(P)
            Z = np.sort(x[A-1:B])
            #print(Z[P-1])
            y[i-1] = Z[P-1]
	
    return y

def myfiltfilt(b,a,x):
    
    if len(x) <= 3*max([len(b)-1,len(a)-1]):
        y = np.zeros_like(x)
    else:
        y = filtfilt(b,a,x)
        
    return y

