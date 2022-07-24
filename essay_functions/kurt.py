def calculate_flatenning_coef(phi_prime:np.ndarray):
    
    sigma_4 = calculate_ordered_moment(phi_prime, 4)
    sigma_2 = calculate_ordered_moment(phi_prime, 2)
    
    return sigma_4 / (sigma_2**2)