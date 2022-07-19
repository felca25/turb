import numpy as np
from statistical_functions import temporal_mean

def calculate_ordered_moment(phi_prime:np.ndarray, order: int):
    
    variance = temporal_mean(phi_prime ** order)
    
    return variance