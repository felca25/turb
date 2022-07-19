from main import FOLDERS
from data_class import TemporalData, TemporalDatas
from data_man import export_data
import os

def main() -> None:
    FOLDER = 'data/lre/'
    paths = [FOLDER+name for name in os.listdir(FOLDER) if os.path.isfile(os.path.join(FOLDER, name))]
    print(paths)
    N = len(paths)

    data_arr = []
    
    for i, path in enumerate(paths):
        splitted_path = path.split('/')
        name = splitted_path[-1][:-4]
        index = splitted_path[-1][-5]
        
        t, u = np.loadtxt(path, unpack=True)
        data = TemporalData(path, name, index, u=u, times=t)
        data.save_txt()
        data_arr.append(data)
        
    folder = FOLDER.split('/')[-2]
    stat_data = TemporalDatas(name=folder, data_arr=data_arr)
    cov, corr = stat_data.calculate_cov_corr(0, 1)
    print(cov, corr)
    
    df = stat_data.to_data_frame(['u',])
    folder = FOLDER.split('/')[-2]
    export_data(df, f'CSV/{folder}')
    print(df)
    
    print(stat_data.__str__())
    return None

if __name__ == '__main__':
    main()