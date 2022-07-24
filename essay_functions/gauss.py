def gaussian(x, mu, sigma):
    """
    Gaussian function
    """
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-((x-mu)**2)/(2*sigma**2))

gauss = norm.pdf(x, mu, sigma)