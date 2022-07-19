import numpy as np

def temporal_mean(phi:np.ndarray):
    
    N = len(phi)
    phi_bar = 0
    
    for i in range(N):
        phi_bar +=  phi[i-1]
    
    phi_bar /= i
    
    return phi_bar