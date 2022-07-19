from turtle import color
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from statistical_functions import temporal_mean
from test_temporal import TemporalAnalysis


class SpatialAnalysis(TemporalAnalysis):
    
    def __init__(self, temporal_list:list, positions=None, mov_avg=False) -> None:
        self.temporal_list = temporal_list
        self.times = temporal_list[0].times
        self.N = len(temporal_list)
        self.calculate_prop_s()
        pass
    
    def calculate_prop_s(self):
        arr = np.array([data.u_prime for data in self.temporal_list])
        self.u_bar_s = self.calculate_spacial_mean()
        self.corr_coef = np.corrcoef(arr)
        self.cov = np.cov(arr)
        return 1
        
    def calculate_spacial_mean(self, data_arr=None):
        
        if data_arr is None: data_arr = self.temporal_list
            
        if not isinstance(data_arr, (list, np.ndarray, pd.Series)):
            raise TypeError('calculate_spacial_mean() accepts types:'\
                            'list, np.ndarray, pd.Series')
            
        N = len(data_arr[0].u)
        out = np.zeros(N)
        phi_bar = 0
        
        for j in range(N):
            for i, temporal_data in enumerate(self.temporal_list):
                phi_bar += temporal_data.u[j]
            out[j] = phi_bar/temporal_data.final_time
        
        return out
    
    def covariance(self,a_prime:np.ndarray, b_prime:np.ndarray):
        
        if b_prime is None:
            self.cov = np.cov(a_prime)
                
        self.cov = np.sum(a_prime * b_prime)/self.N

        return self.cov

    def correlation_coef(self, a_prime:np.ndarray, b_prime:np.ndarray):
        arr = [arr.u_prime**2 for arr in self.temporal_list]
        self.corr_coef =  np.corrcoef(arr)
        
        return self.corr_coef
    
    def to_data_frame(self, properties):
        
        treated = [list(self.temporal_list[0].times),]
        headers = ['tempos[s]',]
        
        for property, in zip(properties):
            for i, temporal_data, in enumerate(self.temporal_list):
                if property == 'u':
                    treated.append(list(temporal_data.u))
                headers.append(f'{property}_{i+1}')
                
        df = pd.DataFrame(np.transpose(treated), columns=headers)
        
        df['u_bar'] = df.mean(axis=1)
        for i in range(len(self.temporal_list)):   
            df[f'u_prime_{i+1}'] = df['u_bar'] - df[f'u_{i+1}']
        return df
    
    def export_txt(self):
        try:
            os.mkdir('txt_results')
        except FileExistsError:
            pass
        
        with open(f'txt_results/{self.re_type}{self.id}.txt', 'w') as outfile:
            
            outfile.write(f'{self.re_type.upper()}{self.id} results\n')
            outfile.write(f're_type = {self.re_type}\nid = {self.id}\n')
            outfile.write(f'u_bar = {self.u_bar}\nvariance = {self.variance}\n'\
                            f'std_dev = {self.u_rms}\n'\
                                f'turbulence intensity = {self.turb_int}\n'\
                                     f'dissimetry coefficient = {self.diss_coef}\n'\
                                        f'flatenning coefficient = {self.flat_coef}\n')
    
        return super().export_txt()
    
    
    def plot_inst_vel(self, re_type, id):
        
        try:
            os.mkdir(f'{re_type}_images')
        except FileExistsError:
            pass
        
        fig1 = plt.figure('Spacial Instant Velocity Plot')
        ax = plt.axes()
        ax.set_title('Velocidades Instantâneas e Média Estatística no ponto 5')
        
        for data in self.temporal_list:
            ax.plot(data.times, data.u, '-', color='0.8')
            
        ax.plot(self.times, self.u_bar_s)
        
        ax.set_xlabel('Tempo [s]')
        ax.set_ylabel('Velocidade $u$ [m/s]')
        ax.grid(1)
        ax.legend(['velocidade instantânea', 'média temporal'])
        plt.savefig(f'spatial_{re_type}_{id}.png')

