    def __post_init__(self):
        
        self.data_arr.sort(key= lambda x: x.path)
        self.path = "/".join(self.data_arr[0].path.split('/')[:-1])+f'/{self.name}'
        self.times = self.data_arr[0].times
        self.N = len(self.data_arr)
        self.N_u = len(self.data_arr[0].u)
        self.final_time = self.times[-1]
        self.time_step = round(self.times[-1] - self.times[-2], 4)
        self.u = np.array([data.u for data in self.data_arr])
        self.u_bar_s = statistical_mean(self.data_arr)
        self.u_bar_t = temporal_mean(self.u_bar_s)
        self.u_prime = calculate_fluctuation(self.u, self.u_bar_s)
        self.u_prime_bar_s = calculate_fluctuation(self.u_bar_s, self.u_bar_t)
        self.cov = 0.
        self.corr_coef = 0.
        self.positions = [0.]
        self.calculate_properties()
        
    def calculate_properties(self) -> None:
        self.variance = calculate_ordered_moment(self.u_prime, 2)
        self.u_rms = np.sqrt(calculate_ordered_moment(self.u_prime_bar_s, 2))
        self.kinetic_energy = calculate_kinetic_energy(self.u_prime)
        self.turb_int = calculate_turbulence_intensity(self.u_bar_s, self.u_prime_bar_s)
        self.diss_coef = calculate_dissimetry_coef(self.u_prime_bar_s)
        self.flat_coef = calculate_flatenning_coef(self.u_prime_bar_s)
        self.u_x_pdf, self.u_pdf = pdf(self.u_bar_s, 500)
        self.u_prime_x_pdf, self.u_prime_pdf = pdf(self.u_prime_bar_s)
        
    def calculate_cov_corr(self, index_a, index_b):
        
        self.cov = self.covariance(index_a, index_b)
        self.corr_coef = self.correlation(index_a, index_b)
        
        return (self.cov, self.corr_coef)
        
    def covariance(self, index_a:int, index_b:int):

        cov = covariance(self.data_arr[index_a].u_prime,
                                self.data_arr[index_b].u_prime)
        
        return cov
    
    def correlation(self, index_a:int, index_b:int):
        cov = covariance(self.data_arr[index_a].u_prime,
                        self.data_arr[index_b].u_prime)

        corr_coef = cov / (self.data_arr[index_a].u_rms * self.data_arr[index_b].u_rms)
        
        return corr_coef