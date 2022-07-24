def calculate_dissimetry_coef(phi_prime:np.ndarray):
    
    sigma_3 = calculate_ordered_moment(phi_prime, 3)
    sigma_2 = calculate_ordered_moment(phi_prime, 2)
    
    return sigma_3 / ((sigma_2)**(3/2))