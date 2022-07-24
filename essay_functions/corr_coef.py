def correlation_coeff(a_prime:np.ndarray, b_prime:np.ndarray):
    
    cov_ab = covariance(a_prime,b_prime)
    a_rms = calculate_std_dev(a_prime)
    b_rms = calculate_std_dev(b_prime)
    corr_coef = cov_ab / (a_rms * b_rms)
    
    return corr_coef