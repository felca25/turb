def covariance(a_prime:np.ndarray, b_prime:np.ndarray):
    if len(a_prime) != len(b_prime):
        raise ValueError('Matrices must have the same length')
    
    cov = 0
    for a_p, b_p in zip(a_prime, b_prime):
        cov += a_p * b_p
    cov /= len(a_prime)
    
    return cov