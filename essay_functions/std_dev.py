def calculate_std_dev(phi_prime:np.ndarray):
    
    variance = calculate_ordered_moment(phi_prime, 2)
    
    return np.sqrt(variance)