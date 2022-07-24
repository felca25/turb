from dataclasses import dataclass, field
from matplotlib import pyplot as plt
from scipy.stats import norm
import os, os.path
import numpy as np
import pandas as pd
import json
from statistical_functions import *
from data_man import export_data, get_spatial_points

@dataclass
class TemporalData:
    
    path: str
    name: str
    index: str
    u: np.ndarray
    times: np.ndarray
    final_time: float = field(init=0)
    time_step: float = field(init=0)
    N: int = field(init=0)
    u_bar_t: float = field(init=0)
    u_prime: np.ndarray = field(init=0)
    kinetic_energy: float = field(init=0)
    variance: float = field(init=0)
    u_rms: float = field(init=0)
    turb_int: float = field(init=0)
    diss_coef: float = field(init=0)
    flat_coef: float = field(init=0)
    u_x_pdf : np.ndarray = field(init=0)
    u_pdf: np.ndarray = field(init=0)
    u_prime_x_pdf : np.ndarray = field(init=0)
    u_prime_pdf: np.ndarray = field(init=0)
    
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
        self.u_x_pdf, self.u_pdf = pdf(self.u, len(self.u))
        # calculating pdf of u_prime
        print(f'calculating u_prime pdf of {self.name}')
        self.u_prime_x_pdf, self.u_prime_pdf = pdf(self.u_prime, len(self.u))
        
    def save_txt(self):
        try:
            os.mkdir('txt_results')
        except FileExistsError:
            pass
        
        with open(f'txt_results/{self.name}.txt', 'w') as outfile:
            
            outfile.write(f'{self.name.upper()} results\n\n')
            outfile.write(f'u_bar_t = {self.u_bar_t}\n'\
                          f'variance = {self.variance}\n'\
                            f'std_dev = {self.u_rms}\n'\
                                f'turbulence intensity = {self.turb_int}\n'\
                                    f'dissimetry coefficient = {self.diss_coef}\n'\
                                        f'flatenning coefficient = {self.flat_coef}\n')
    
        return 1
    
    def to_list(self)-> None:
        self.times = self.times.tolist()
        self.u = self.u.tolist()
        self.u_prime = self.u_prime.tolist()
        self.u_x_pdf = self.u_x_pdf.tolist()
        self.u_pdf = self.u_pdf.tolist()
        self.u_prime_x_pdf = self.u_prime_x_pdf.tolist()
        self.u_prime_pdf = self.u_prime_pdf.tolist()
        
    def to_array(self) -> None:
        self.times = np.array(self.times)
        self.u = np.array(self.u)
        self.u_prime = np.array(self.u_prime)
        self.u_x_pdf = np.array(self.u_x_pdf)
        self.u_pdf = np.array(self.u_pdf)
        self.u_prime_x_pdf = np.array(self.u_prime_x_pdf)
        self.u_prime_pdf = np.array(self.u_prime_pdf)
    
    def save_json(self):
        self.to_list()
        dir_name = self.path.split('/')[-2]
        
        try:
            os.mkdir(f'json_results/{dir_name}')
        except FileExistsError:
            
            pass
        if self.name == 'PERFILM':
            with open(f'json_results/{dir_name}/{self.name}{self.index}.json', 'w') as outfile:
                json.dump(self.__dict__, outfile)
        else:
            with open(f'json_results/{dir_name}/{self.name}.json', 'w') as outfile:
                json.dump(self.__dict__, outfile)
            
        self.to_array()

        
@dataclass
class TemporalDatas:
    name : str
    data_arr : list
    path : str = field(init=0)
    times : np.ndarray = field(init=0)
    N : int = field(init=0)
    N_u : int = field(init=0)
    final_time : float = field(init=0)
    time_step : np.ndarray = field(init=0)
    u : np.ndarray = field(init=0)
    u_prime : np.ndarray = field(init=0)
    kinetic_energy: np.ndarray = field(init=0)
    variance: np.ndarray = field(init=0)
    u_rms: np.ndarray = field(init=0)
    turb_int: np.ndarray = field(init=0)
    diss_coef: float = field(init=0)
    flat_coef: float = field(init=0)
    u_bar_s : np.ndarray = field(init=0)
    u_prime_bar_s : np.ndarray = field(init=0)
    u_bar_t : float = field(init=0)
    cov : float = field(init=0)
    corr_coef: float = field(init=0)
    positions: list = field(init=0)
    
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
        self.turb_int = calculate_turbulence_intensity(self.u_bar_s, self.u_prime_bar_s)*100
        self.diss_coef = calculate_dissimetry_coef(self.u_prime_bar_s)
        self.flat_coef = calculate_flatenning_coef(self.u_prime_bar_s)
        self.u_x_pdf, self.u_pdf = pdf(self.u_bar_s,len(self.u_bar_s))
        self.u_prime_x_pdf, self.u_prime_pdf = pdf(self.u_prime_bar_s)
        
        if self.data_arr[0].name == 'PERFILM':
            u_t = [self.data_arr[i].u_bar_t for i in range(self.N)]
            u_prime = u_t - self.u_bar_t
            self.u_bar_s = u_t
            self.turb_int = calculate_turbulence_intensity(u_t, u_prime)*100
        
    def calculate_cov_corr(self, index_a, index_b):
        print(self.data_arr[index_a].name)
        print(self.data_arr[index_b].name)
        
        self.cov = self.covariance(index_a, index_b)
        self.corr_coef = self.correlation(index_a, index_b)
        
        return (self.cov, self.corr_coef)
        
    def covariance(self, index_a:int, index_b:int):
        print(self.data_arr[index_a].name)
        print(self.data_arr[index_b].name)
        # cov = np.cov(self.data_arr[index_a].u_prime,
        #                       self.data_arr[index_b].u_prime)
        cov = covariance(self.data_arr[index_a].u_prime,
                                self.data_arr[index_b].u_prime)
        
        return cov
        
    def correlation(self, index_a:int, index_b:int):
        cov = covariance(self.data_arr[index_a].u_prime,
                        self.data_arr[index_b].u_prime)
        # corr_coef = np.corrcoef(self.data_arr[index_a].u_prime,
        #                 self.data_arr[index_b].u_prime)
        corr_coef = cov / (self.data_arr[index_a].u_rms * self.data_arr[index_b].u_rms)
        
        return corr_coef
    
    def save_txt(self):
        try:
            os.mkdir('txt_results')
        except FileExistsError:
            pass
        
        with open(f'txt_results/{self.name}.txt', 'w') as outfile:
            
            outfile.write(f'{self.name.upper()} results\n\n')
            outfile.write(f'u_bar_t = {self.u_bar_t}\n'\
                          f'variance = {self.variance}\n'\
                            f'std_dev = {self.u_rms}\n'\
                                f'turbulence intensity = {self.turb_int}\n'\
                                    f'dissimetry coefficient = {self.diss_coef}\n'\
                                        f'flatenning coefficient = {self.flat_coef}\n')
            
    def to_list(self):
        
        for data in self.data_arr:
            data.to_list()
            
        auxiliary = self.data_arr
        delattr(self, 'data_arr')    
        self.times = self.times.tolist()
        self.u = self.u.tolist()
        self.u_bar_s = self.u_bar_s.tolist()
        self.u_prime = self.u_prime.tolist()
        self.u_prime_bar_s = self.u_prime_bar_s.tolist()
        self.kinetic_energy = self.kinetic_energy.tolist()
        self.variance = self.variance.tolist()
        self.u_rms = self.u_rms.tolist()
        self.turb_int = self.turb_int.tolist()
        self.u_x_pdf = self.u_x_pdf.tolist()
        self.u_pdf = self.u_pdf.tolist()
        self.u_prime_x_pdf = self.u_prime_x_pdf.tolist()
        self.u_prime_pdf = self.u_prime_pdf.tolist()
        
        return auxiliary
        
    def to_array(self):
        
        for data in self.data_arr:
                data.to_array()
                
        self.times = np.array(self.times)
        self.u = np.array(self.u)
        self.u_bar_s = np.array(self.u_bar_s)
        self.u_prime = np.array(self.u_prime)
        self.u_prime_bar_s = np.array(self.u_prime_bar_s)
        self.kinetic_energy = np.array(self.kinetic_energy)
        self.variance = np.array(self.variance)
        self.u_rms = np.array(self.u_rms)
        self.turb_int = np.array(self.turb_int)
        self.u_x_pdf = np.array(self.u_x_pdf)
        self.u_pdf = np.array(self.u_pdf)
        self.u_prime_x_pdf = np.array(self.u_prime_x_pdf)
        self.u_prime_pdf = np.array(self.u_prime_pdf)
            
    def save_json(self):
        
        auxiliary = self.to_list()
        
        try:
            os.mkdir(f'json_results/{self.name}')
        except FileExistsError:
    
            pass
        with open(f'json_results/{self.name}/{self.name}.json', 'w') as outfile:
            json.dump(self.__dict__, outfile)
            
        
        self.data_arr = auxiliary
        self.to_array()

    
    def to_data_frame(self, properties):
        name = self.name
        treated = [list(self.data_arr[0].times),]
        headers = ['tempos[s]',]
        
        if name == 'perfil_mon' or name == 'perfil_jus':
            
            for property, in zip(properties):
                for i, temporal_data, in enumerate(self.data_arr):
                    if property == 'u':
                        treated.append(list(temporal_data.u))
                    headers.append(f'{property}_{i+1}')
                    
            df = pd.DataFrame(np.transpose(treated), columns=headers)
            
            df['u_bar'] = df.mean(axis=1)
            for i in range(len(self.data_arr)):   
                df[f'u_prime_{i+1}'] = df['u_bar'] - df[f'u_{i+1}']
            return df
        
        else:
            for property, in zip(properties):
                for i, temporal_data, in enumerate(self.data_arr):
                    if property == 'u':
                        treated.append(list(temporal_data.u))
                    headers.append(f'{property}_{i+1}')
                    
            df = pd.DataFrame(np.transpose(treated), columns=headers)
            
            df['u_bar'] = df.mean(axis=1)
            for i in range(len(self.data_arr)):   
                df[f'u_prime_{i+1}'] = df['u_bar'] - df[f'u_{i+1}']
            return df
        
    def export_latex_table(self):
        prop = ["u_bar_t", "variance", "u_rms", "turb_int", "kinetic_energy", "diss_coef", "flat_coef"]
        dic = {prop[i]: [f'{self.data_arr[j].__dict__[prop[i]]:.6f}' for j in range(len(self.data_arr))] \
        for i in range(len(prop))}
        
        if self.name != 'hre_prob':
            with open(f'txt_results/latex_{self.name}.txt', 'w') as outfile:
                outfile.write(f'{self.name.upper()} results\n\n')
                outfile.write(f'\\begin{{table}}[htb]\n\
        \\centering\n\
        \\begin{{tabular}}{{l|{"|".join(["c" for i in range(len(self.data_arr))])}}}\n\
            Propriedades & {" & ".join([f"$u_{i+1}$" for i in range(len(self.data_arr))])}\\\ \n\
            \hline\\\ \n')
                for i in range(len(prop)):
                    outfile.write(f'        {prop[i]} & {" & ".join(dic[prop[i]])}\\\ \n')
                outfile.write(f'    \end{{tabular}}\n\
                    \caption{{Tabela de Resultados Para {self.name}}}\n\
        \label{{tab:hre}}\n\\end{{table}}\n')
        # elif self.name == 'hre_prob':
        #     with open(f'txt_results/latex_{self.name}.txt', 'w') as outfile:
        #         outfile.write(f'{self.name.upper()} results\n\n')
        #         outfile.write(f'\\begin{{table}}[htb]\n\
        # \\centering\n\
        # \\begin{{tabular}}{{l|c}}\n\
        #     Propriedades & Valor Estatístico\\\ \n\
        #     \hline\\\ \n')
        #         for i in range(len(prop)):
        #             outfile.write(f'        {prop[i]} & {np.mean(np.float64(dic[prop[i]])):.6f}\\\ \n')
        #         outfile.write(f'    \end{{tabular}}\n\
        #             \caption{{Tabela de Resultados Para {self.name}}}\n\
        # \label{{tab:hre}}\n\\end{{table}}\n')
        
    

def run(FOLDERS) -> None:
    
    for folder in FOLDERS:
        
        FOLDER = f'data/{folder}/'
        paths = [FOLDER+name for name in os.listdir(FOLDER) if os.path.isfile(os.path.join(FOLDER, name))]
        print(paths)
        N = len(paths)

        data_arr = []
        
        for i, path in enumerate(paths):
            splitted_path = path.split('/')
            name = splitted_path[-1][:-4]
            
            if name == 'hre' or name == 'lre':
                index = splitted_path[-1][-5]
            elif name == 'PERFILM':
                index = splitted_path[-1][-2:]
            else:
                index = splitted_path[-1][-6:-4]
            
            t, u = np.loadtxt(path, unpack=True)
            data = TemporalData(path, name, index, u=u, times=t)
            data.save_txt()
            
            if name == 'hre' or name == 'lre':
                index = splitted_path[-1][-5]
            elif name == 'PERFILM':
                index = splitted_path[-1][-2:]
            else:
                index = splitted_path[-1][-6:-4]
            
            data.save_json()
            data_arr.append(data)
                  
        folder = FOLDER.split('/')[-2]
        stat_data = TemporalDatas(name=folder, data_arr=data_arr)
        
        if folder == 'lre'or folder == 'hre':
            
            cov_arr = []
            corr_arr = []
            for n in range(N-1):
                cov, corr = stat_data.calculate_cov_corr(n, n+1)
                print(cov, corr)
                cov_arr.append(cov)
                corr_arr.append(corr)
                
            stat_data.cov = cov_arr
            stat_data.corr_coef = corr_arr
            stat_data.export_latex_table()
            
        elif folder == 'perfil_jus' or folder == 'perfil_mon':
            
            u_bar_arr = []
            for n in range(N):
                u_bar_arr.append(stat_data.data_arr[n].u_bar_t)
                stat_data.u_bar_s = np.array(u_bar_arr)
                
            print(u_bar_arr)
            
            if folder == 'perfil_jus':
                stat_data.positions = get_spatial_points(f'data/pos_jus.txt')
            elif folder == 'perfil_mon':
                stat_data.positions = get_spatial_points(f'data/pos_mon.txt')
            pass
        
        elif folder == 'hre_prob':
            stat_data.export_latex_table()
        
        # print(stat_data.__str__())
        stat_data.save_json()
        # df = stat_data.to_data_frame(['u',])
        # folder = FOLDER.split('/')[-2]
        # export_data(df, f'CSV/{folder}')
        # print(df)
        
        
    return None

if __name__ == '__main__':
    FOLDERS = ('lre', 'hre', 'perfil_jus', 'perfil_mon')
    run(FOLDERS)
    
    
    