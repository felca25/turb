from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd

from statistical_functions import *
        

class VelocityData():
    
    def __init__(self, path_name, M_COND) -> None:
        
        self.get_data(path_name)
        self.id = path_name[-5]
        self.re_type = path_name.split('/')[-1][0:-5]
        
        self.final_time, self.time_step, self.measurements = M_COND

        self.calculate_temporal_mean()

        print(self.u_bar)
    
    def get_data(self, path_name):
        self.times, self.velocity = np.loadtxt(path_name, unpack=True)
        
    def calculate_temporal_mean(self):
        self.u_bar = temporal_mean(self.velocity)
        
    def calculate_fluctuations(self):
        self.u_prime = calculate_fluctuation(self.velocity, self.u_bar)
        
    def calculate_variance(self):
        self.variance = calculate_ordered_moment(self.u_prime, 2)
        
    def calculate_turbulence_intensity(self):
        self.u_rms = calculate_turbulence_intensity(self.u_bar, self.u_prime)
        
        
    def plot_instant_velocity(self):
        
        try:
            os.mkdir(f'{self.re_type}_images')
        except FileExistsError:
            pass
        
        fig = plt.figure(f'plot_{self.re_type}{self.id}')
        ax = plt.axes()
        
        ax.set_title(f'Velocidade instantânea u_{self.id}')
        
        ax.plot(self.times, self.velocity)
        ax.plot(self.times, self.u_bar*np.ones(len(self.times)))
        
        ax.legend(['velocidade instantânea', 'média temporal'])
        
        plt.savefig(f'{self.re_type}_images/instantaneous_velocity_{self.re_type}_{self.id}.png')