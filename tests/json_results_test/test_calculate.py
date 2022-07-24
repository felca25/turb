import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from test_temporal import TemporalAnalysis
from test_spatial import SpatialAnalysis
from statistical_functions import *
from main import MEASUREMENT_CONDITIONS
    
def to_data_frame(data_list, properties):
    
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
    
    return data
    
def export_data(data):
    print('Exporting csv ...')
    data.to_csv('test.csv')
    print('Export complete')
    
    return data

def get_data(rel_path, re_types, positions):
    
    data_dict = {}
    
    for j, re_type in enumerate(re_types):
        data_list = []
        
        if re_type == 'hre' or re_type == 'lre':
            for i in range(1,6):
                path_name = f'{rel_path}/{re_type}/{re_type}{i}.dat'
                times, u_x = np.loadtxt(path_name, unpack=True)
                data = TemporalAnalysis(u_x, times, re_type, i, MEASUREMENT_CONDITIONS, 
                                        positions[i-1], mov_avg=0)
                data.calculate_properties()
                data_list.append(data)
                # hre_i.calculate_moving_avg(1000)
                # hre_i.u_bar = hre_i.u_mov
                # hre_i.velocity_plot(hre_i.u_mov, 'moving_average')
                # data.plot_instant_velocity()
                data.export_json()
                data.export_txt()
        
        elif re_type == 'hrep':
            for i in range(1, 26):
                path_name = f'{rel_path}/hre_prob/{re_type}{i:02d}.dat'
                times, u_x = np.loadtxt(path_name, unpack=True)
                data = TemporalAnalysis(u_x, times, re_type, f'{i:02d}', MEASUREMENT_CONDITIONS, 
                                        positions[4], mov_avg=0)
                data.calculate_properties()
                data_list.append(data)
                # data.plot_instant_velocity()
                data.export_json()
                data.export_txt()
                print(path_name)
            
        elif re_type == 'perfil_jus':
            for i in range(1, 26):
                path_name = f'{rel_path}/{re_type}/PERFILM.W{i:02d}'
                times, u_x = np.loadtxt(path_name, unpack=True)
                data = TemporalAnalysis(u_x, times, re_type, i, MEASUREMENT_CONDITIONS, 
                                        positions[4], mov_avg=0)
                data_list.append(data)
                data.plot_instant_velocity()
                data.export_json()
                data.export_txt()
                print(path_name)
                
        elif re_type == 'perfil_mon':
            for i in range(1, 26):
                path_name = f'{rel_path}/{re_type}/PERFILM.W{i:02d}'
                times, u_x = np.loadtxt(path_name, unpack=True)
                data = TemporalAnalysis(u_x, times, re_type, i, MEASUREMENT_CONDITIONS, 
                                        positions[4], mov_avg=0)
                data_list.append(data)
                # data.plot_instant_velocity()
                data.export_json()
                data.export_txt()
                print(path_name)
        
        data_dict[f'{re_type}'] = data_list
    # if re_type == 'hre':
    #     data_list.append(data)
    # elif re_type == 'lre':
    #     lre.append(data)
        
    return data_dict
    
if __name__ == '__main__':
    T_FINAL = 3.2766 # > [s]
    TIME_STEP = 0.0002 # > [s]
    MEASUREMENTS = 16384 # > [adim]
    M_COND = [T_FINAL, TIME_STEP, MEASUREMENTS]

    rel_path = 'data'
    file_types = ['hrep']
    properties = ['u', 'u_bar', 'u_prime']
    H = 5*1e-2
    positions = [[-1.5*H, 0, .5*H], [H, 0, 0.5*H], [1.5*H, 0, .5*H], [2*H, 0, .5*H], [2.5*H, 0, 1.5*H]]
    
    data = get_data(rel_path, file_types, positions)
    
    for id, file_type, in enumerate(file_types):
        data_s = SpatialAnalysis(data[file_type], M_COND)
        data_s.plot_inst_vel(file_type, id)
        export_data(data_s.to_data_frame(['u']))
        print(data_s.to_data_frame(['u']))
    
