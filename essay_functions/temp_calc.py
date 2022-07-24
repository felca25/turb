def __post_init__(self) -> None:
        self.N = len(self.u)
        self.final_time = self.times[-1]
        self.time_step = round(self.times[-1] - self.times[-2], 4)
        self.calculate_properties()
        
        if self.name[:-1] == 'lre' or self.name[:-1] == 'hre':
            self.index = self.index
    
    def calculate_properties(self) -> None:
        
        self.u_bar_t = temporal_mean(self.u)
        self.u_prime = calculate_fluctuation(self.u, self.u_bar_t)
        self.kinetic_energy = .5 * calculate_ordered_moment(self.u_prime, 2)
        self.variance = calculate_ordered_moment(self.u_prime, 2)
        self.u_rms = np.sqrt(calculate_ordered_moment(self.u_prime, 2))
        self.turb_int = calculate_turbulence_intensity(self.u_bar_t, self.u_prime)
        self.diss_coef = calculate_dissimetry_coef(self.u_prime)
        self.flat_coef = calculate_flatenning_coef(self.u_prime)
        self.u_x_pdf, self.u_pdf = pdf(self.u, 500)
        self.u_prime_x_pdf, self.u_prime_pdf = pdf(self.u_prime)