from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
import json

from statistical_functions import *
        

class TemporalAnalysis():
    
    def __init__(self, path_name, M_COND, mov_avg=False) -> None:
        
        self.get_data(path_name)
        self.id = path_name[-5]
        self.re_type = path_name.split('/')[-1][0:-5]
        self.mov_avg = mov_avg
        
        self.final_time, self.time_step, self.measurements = M_COND

        print(f'{self.re_type}{self.id}')
        
        if mov_avg:
            
            self.u_bar = self.calculate_moving_avg()
            self.u_prime = self.u[:len(self.u_bar)] - self.u_bar
            
        else:
            
            self.calculate_temporal_mean()
            self.calculate_fluctuations()
            
        self.calculate_properties()
    
    def get_data(self, path_name):
        self.times, self.u = np.loadtxt(path_name, unpack=True)
        
    def calculate_temporal_mean(self):
        self.u_bar = temporal_mean(self.u)
        
    def calculate_moving_avg(self, window=500):
        self.u_mov = moving_average(self.u, window)
        return self.u_mov
        
    def calculate_fluctuations(self):
        self.u_prime = calculate_fluctuation(self.u, self.u_bar)
        
    def calculate_kinetic_energy(self):
        self.kinetic_energy = .5 * calculate_ordered_moment(self.u_prime, 2)
        
    def calculate_variance(self):
        self.variance = calculate_ordered_moment(self.u_prime, 2)
        
    def calculate_rms(self):
        self.u_rms = np.sqrt(calculate_ordered_moment(self.u_prime, 2))
        
    def calculate_turbulence_intensity(self):
        self.turb_int = calculate_turbulence_intensity(self.u_bar, self.u_prime)
        
    def calculate_dissimetry_coef(self):
        self.diss_coef = calculate_dissimetry_coef(self.u_prime)
        
    def calculate_flatenning_coef(self):
        self.flat_coef = calculate_flatenning_coef(self.u_prime)
        
    def calculate_properties(self):
        self.calculate_kinetic_energy()
        self.calculate_variance()
        self.calculate_rms()
        self.calculate_turbulence_intensity()
        self.calculate_dissimetry_coef()
        self.calculate_flatenning_coef()
        print(f'{self.re_type}{self.id} properties')
        # print(f'u_bar = {self.u_bar}')
        print(f'turb_int = {self.turb_int}')
        print(f'variance = {self.variance}')
        print(f'std deviation = {self.u_rms}')
        print(f'dissimetry coefficient = {self.diss_coef}')
        print(f'flatenning coefficient = {self.flat_coef}')
        
    def export_txt(self):
        try:
            os.mkdir('txt_results')
        except FileExistsError:
            pass
        
        with open(f'txt_results/{self.re_type}{self.id}.txt', 'w') as outfile:
            outfile.write(f'{self.re_type.upper()}{self.id} results\n')
            outfile.write(f'u_bar = {self.u_bar}\nvariance = {self.variance}\n'\
                            +f'std_dev = {self.u_rms}\n'\
                                +f'turbulence intensity = {self.turb_int}\n'\
                                    + f'dissimetry coefficient = {self.diss_coef}\n'\
                                        f'flatenning coefficient = {self.flat_coef}\n')
    
    def export_json(self):
        self.times = self.times.tolist()
        self.u = self.u.tolist()
        self.u_prime = self.u_prime.tolist()
        
        try:
            os.mkdir('json_results')
        except FileExistsError:
            pass
        
        with open(f'json_results/{self.re_type}{self.id}.json', 'w') as outfile:
            json.dump(self.__dict__, outfile)
        
    def plot_instant_velocity(self):
        
        try:
            os.mkdir(f'{self.re_type}_images')
        except FileExistsError:
            pass
        
        fig = plt.figure(f'plot_{self.re_type}{self.id}')
        ax = plt.axes()
        
        ax.set_title(f'Velocidade instantânea $u_{self.id}$')
        
        ax.plot(self.times[:len(self.u)], self.u)
        
        if self.mov_avg:
            ax.plot(self.times[:len(self.u_bar)], self.u_bar)
        else:
            ax.plot(self.times, self.u_bar*np.ones(len(self.times)))
            
        ax.set_xlabel('tempo [s]')
        ax.set_ylabel('velocidade [m/s]')
        
        ax.legend(['velocidade instantânea', 'média temporal'])
        
        plt.savefig(f'{self.re_type}_images/instantaneous_velocity_{self.re_type}_{self.id}.png')
    
    def velocity_plot(self, y, description):
        
        path = f'{self.re_type}_{description}_images'
        
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        
        fig = plt.figure(f'plot_{self.re_type}{self.id}')
        ax = plt.axes()
        
        ax.set_title(f'{description} $u_{self.id}$')
        
        ax.plot(self.times[:len(y)], y)
        ax.set_xlabel('tempo [s]')
        ax.set_ylabel(f'{description} [m/s]')
        
        ax.legend([f'{description}'])
        
        plt.savefig(f'{path}/{description}_{self.re_type}_{self.id}.png')
        
        
class SpatialAnalysis(TemporalAnalysis):
    def __init__(self, temporal_list) -> None:
        pass