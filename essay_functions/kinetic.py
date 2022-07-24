def calculate_kinetic_energy(u:np.ndarray):
    
    kinetic = 0.5 * calculate_ordered_moment(u, 2)
    
    return kinetic