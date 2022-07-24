import numpy as np
from data_class import TemporalData, TemporalDatas
from data_man import export_data, get_spatial_points
import os

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
        
        print(stat_data.__str__())
        stat_data.save_json()
        df = stat_data.to_data_frame(['u',])
        folder = FOLDER.split('/')[-2]
        export_data(df, f'CSV/{folder}')
        print(df)
        
        
    return None


if __name__ == '__main__':
    # FOLDERS = ('lre', 'hre', 'hre_prob', 'perfil_mon', 'perfil_jus')
    FOLDERS = ('perfil_mon','perfil_jus')
    run(FOLDERS)