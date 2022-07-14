import numpy as np

def temporal_mean(phi:np.ndarray):
    
    N = len(phi)
    phi_bar = 0
    
    for i in range(N):
        phi_bar +=  phi[i-1]
    
    phi_bar /= i
    
    return phi_bar

def calculate_fluctuation(u:np.ndarray, u_bar:float):
    return u - u_bar

def calculate_ordered_moment(phi_prime:np.ndarray, order: int):
    
    phi_prime_bar = temporal_mean(phi_prime)
    variance = phi_prime_bar ** order
    
    return variance

def root_mean_sqrt(phi_prime:np.ndarray):
    
    variance = calculate_ordered_moment(phi_prime, 2)
    
    return np.sqrt(variance)

def calculate_turbulence_intensity(phi_bar:float, phi_prime:np.ndarray):
    
    phi_rms = root_mean_sqrt(phi_prime)
    
    return phi_rms / phi_bar
    
def calculate_dissimetry_coef(phi_prime:np.ndarray):
    
    sigma_3 = calculate_ordered_moment(phi_prime, 3)
    sigma_2 = calculate_ordered_moment(phi_prime, 2)
    
    return sigma_3 / ((sigma_2)**(3/2))

def calculate_flatenning_coef(phi_prime:np.ndarray):
    
    sigma_4 = calculate_ordered_moment(phi_prime, 4)
    sigma_2 = calculate_ordered_moment(phi_prime, 2)
    
    return sigma_4 / sigma_2**2