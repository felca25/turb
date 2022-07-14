
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal
from get_data import VelocityData
from statistical_functions import *
from main import MEASUREMENT_CONDITIONS

T_FINAL = 3.2766 # > [s]
TIME_STEP = 0.0002 # > [s]
MEASUREMENTS = 16384 # > [adim]

re_types = ['lre', 'hre']

hre = []
lre = []

for i in range(1,6):
    
    lre_i = VelocityData(f'data/lre/lre{i}.dat', MEASUREMENT_CONDITIONS)
    hre_i = VelocityData(f'data/hre/hre{i}.dat', MEASUREMENT_CONDITIONS)
    
    lre.append(lre_i)
    hre.append(hre_i)
    
    hre_i.calculate_temporal_mean()
    hre_i.calculate_fluctuations()
    hre_i.calculate_variance()
    hre_i.calculate_turbulence_intensity()
    
    print(f'u_rms = {hre_i.u_rms}')
    print(f'variance = {hre_i.variance}')
    u = hre_i.velocity
    
    N  = 1  # Filter order
    Wn = 1 - 1e-6 # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba', fs=1/0.0004)
    hre_i.velocity = signal.filtfilt(B,A, u)
                
    hre_i.calculate_temporal_mean()
    hre_i.calculate_fluctuations()
    hre_i.calculate_variance()
    hre_i.calculate_turbulence_intensity()
    
    print(f'u_rms = {hre_i.u_rms}')
    print(f'variance = {hre_i.variance}')
    
    # u_bar = temporal_mean(hre_i.velocity, hre_i.final_time, hre_i.time_step,
    #                       hre_i.measurements)
    print(hre_i.u_bar)
    
    lre_i.plot_instant_velocity()
    hre_i.plot_instant_velocity()
    
