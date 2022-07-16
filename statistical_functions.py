import numpy as np

def temporal_mean(phi:np.ndarray):
    
    N = len(phi)
    phi_bar = 0
    
    for i in range(N):
        phi_bar +=  phi[i-1]
    
    phi_bar /= i
    
    return phi_bar

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
    
    phi_prime_bar = temporal_mean(phi_prime ** order)
    variance = phi_prime_bar
    
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

def pdf(u_t, N):
    N_x = len(u_t)
    
    prob = np.zeros(N)
    u = np.sort(u_t)
    x = np.linspace(min(u), max(u), N)
    pdf = pdf_algorithm(u, x, prob, N, N_x, (1/N))
    
    return x, pdf

def pdf_algorithm(u, x, pdf, N, N_x, TOL):
    k = 0
    i = 0
    p = 0
    while k < N_x and i < N:
        print(i, k, p)
        if u[k] > x[i] - 0.1*TOL and u[k] < x[i+1]+0.1*TOL:
            k += 1
            p += 1
        else:
            pdf[i] = p/N_x
            p = 0 
            i += 1
        
    return pdf