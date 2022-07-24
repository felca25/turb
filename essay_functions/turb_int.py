def calculate_turbulence_intensity(u_bar:float, u_prime:np.ndarray):
    
    u_rms = calculate_std_dev(u_prime)
    
    return u_rms / np.abs(u_bar)