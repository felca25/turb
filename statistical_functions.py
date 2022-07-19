from statistics import variance
import numpy as np
from matplotlib import pyplot as plt

def temporal_mean(phi:np.ndarray):
    
    N = len(phi)
    phi_bar = 0
    
    for i in range(N):
        phi_bar +=  phi[i-1]
    
    phi_bar /= i
    
    return phi_bar

def statistical_mean(data_list):
    
    spacial_averages = np.zeros(len(data_list[0].u))

    for j in range(len(spacial_averages)):
        spacial_avg = 0.
        for i, data in enumerate(data_list):
            spacial_avg += data.u[j]
            spacial_averages[j] = spacial_avg / len(data_list)

    return spacial_averages
    plt.plot(data_list[0].times, spacial_averages)
    plt.show()

def moving_average(arr, window):
    i = 0
    moving_average = []
    
    while i < len(arr) - window + 1:
        
        window_avg = np.sum(arr[i:i+window])/window
        
        moving_average.append(window_avg)
        
        i += 1
    
    return moving_average

def calculate_fluctuation(u:np.ndarray, u_bar:float):
    
    return u - u_bar


def calculate_ordered_moment(phi_prime:np.ndarray, order: int):
    
    variance = temporal_mean(phi_prime ** order)
    
    return variance



def calculate_kinetic_energy(phi_prime:np.ndarray):
    
    kinetic = 0.5 * calculate_ordered_moment(phi_prime, 2)
    
    return kinetic

def calculate_kinetic_energy(phi_prime:np.ndarray):
    
    variance = calculate_ordered_moment(phi_prime, 2)
    
    return np.sqrt(variance)

def calculate_turbulence_intensity(phi_bar:float, phi_prime:np.ndarray):
    
    phi_rms = calculate_kinetic_energy(phi_prime)
    
    return phi_rms / phi_bar
    
def calculate_dissimetry_coef(phi_prime:np.ndarray):
    
    sigma_3 = calculate_ordered_moment(phi_prime, 3)
    sigma_2 = calculate_ordered_moment(phi_prime, 2)
    
    return sigma_3 / ((sigma_2)**(3/2))

def calculate_flatenning_coef(phi_prime:np.ndarray):
    
    sigma_4 = calculate_ordered_moment(phi_prime, 4)
    sigma_2 = calculate_ordered_moment(phi_prime, 2)
    
    return sigma_4 / sigma_2**2

def pdf(u_t, N=None):
    """Probability Density Function for the values of an
    1D array

    Args:
        u_t (np.ndarray): 1D array
        N (int): numbers of intervals

    Returns:
        tuple: tuple of interval and probabitlity density pair
    """
    N_u = len(u_t)
    if N is None:
        N = N_u
    
    prob = np.zeros(N)
    u = np.sort(u_t)
    x = np.linspace(min(u), max(u), N)
    print('Calculating pdf...')
    pdf = pdf_algorithm(u, x, prob, N, N_u, (1/N))
    print('Done alculating pdf.')
    
    return (x, pdf)

def pdf_algorithm(u, x, pdf, N, N_u, TOL):
    """Main probability density function algorithm
    able to work with numba jit method
    
    This algorithm takes an ordered 1D velocity list
    and calculates the probability of each velocity 
    at each point in time.

    Args:
        u (np.ndarray): velocity 1D array
        x (np.ndarray): array with the intervals
        pdf (np.ndarray): empty result array
        N (np.ndarray): number of intervals to be analysed
        N_u (np.ndarray): number of elements in the velocity array
        TOL (float): a tolerance value to control floating points

    Returns:
        pdf (np.ndarray) : the probability density function array correspondent 
        to the velocity array
    """
    k = 0
    i = 0
    p = 0
    while k < N_u and i < N:
        if u[k] > x[i] - 0.1*TOL and u[k] < x[i+1]+0.1*TOL:
            k += 1
            p += 1
        else:
            pdf[i] = p/N_u
            p = 0 
            i += 1
        
    return pdf

def covariance(a_prime:np.ndarray, b_prime:np.ndarray):
    if len(a_prime) != len(b_prime):
        raise ValueError('Matrices must have the same length')
    
    cov = 0
    for a_p, b_p in zip(a_prime, b_prime):
        cov += a_p * b_p
    cov /= len(a_prime)
    
    return cov

def correlation_coeff(a_prime:np.ndarray, b_prime:np.ndarray):
    
    cov_ab = covariance(a_prime,b_prime)
    a_rms = variance(a_prime)
    b_rms = variance(b_prime)
    corr_coef = cov_ab / (a_rms * b_rms)
    
    return corr_coef
    