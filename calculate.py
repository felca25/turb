import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from get_data import TemporalAnalysis
from statistical_functions import *
from main import MEASUREMENT_CONDITIONS

def spacial_averages():
    spacial_averages = np.zeros(len(data_list[0].u))

    for j in range(len(spacial_averages)):
        spacial_avg = 0.
        for i, data in enumerate(data_list):
            spacial_avg += data.u[j]
            spacial_averages[j] = spacial_avg / len(data_list)

    plt.plot(data_list[0].times, spacial_averages)
    plt.show()
    
def export_data(data_list, properties):
    
    treated = [list(data_list[0].times),]
    headers = ['tempos[s]',]
    
    for property in zip(properties):
        for i in range(0,5):
            if property == 'u':
                treated.append(list(data_list[i].u))
            elif property == 'u_bar':
                treated.append(list(data_list[i].u_bar * np.ones(len(data_list[i].times))))
            elif property == 'u_prime':
                treated.append(list(data_list[i].u_prime))
            headers.append(f'{property}_{i}')

    data = pd.DataFrame(np.transpose(treated), columns=headers)
    print('Exporting csv ...')
    pd.DataFrame(data).to_csv('test.csv')
    print('Export complete')
    
    return data

def get_data(rel_path, re_types):
    for j, re_type in enumerate(re_types):
        for i in range(1,6):
            path = f'{rel_path}/{re_type}/{re_type}{i}.dat'
            data = TemporalAnalysis(path, MEASUREMENT_CONDITIONS, mov_avg=0)
            data_list.append(data)
            # hre_i.calculate_moving_avg(1000)
            # hre_i.u_bar = hre_i.u_mov
            data.calculate_properties()
            # hre_i.velocity_plot(hre_i.u_mov, 'moving_average')
            data.plot_instant_velocity()
            data.export_json()
            data.export_txt()
            
    if re_type == 'hre':
        data_list.append(data)
    elif re_type == 'lre':
        lre.append(data)
        
    return 1
    
if __name__ == '__main__':
    T_FINAL = 3.2766 # > [s]
    TIME_STEP = 0.0002 # > [s]
    MEASUREMENTS = 16384 # > [adim]

    rel_path = 'data'
    re_types = ['lre', 'hre']
    properties = ['u', 'u_bar', 'u_prime']

    data_list = []
    lre = []
    
    get_data(rel_path, re_types)
    
    

