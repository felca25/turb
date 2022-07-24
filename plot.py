from unicodedata import name
from matplotlib import pyplot as plt, transforms
from scipy.stats import norm
import os
import json
from statistical_functions import pdf
import numpy as np

def plot_instant_velocity(obj):
    
    dir_name = obj['path'].split("/")[-2]
    save_dir_name = f'images/{dir_name}'
    
    try:
        os.mkdir(save_dir_name)
    except FileExistsError:
        pass
    fig = plt.figure(f'plot_{obj["name"]}')
    ax = plt.axes()
    try:
        ax.set_title(f'Velocidade instantânea u_{obj["index"]}')
        ax.plot(obj["times"][:len(obj["u"])], obj["u"])
    except KeyError:
        if obj["name"] == 'hre_prob':
            ax.set_title(f'Velocidade instantânea u_{obj["name"]}')
            ax.plot(obj["times"], obj["u_bar_s"],)
    

    ax.plot(obj["times"], obj["u_bar_t"]*np.ones(len(obj["times"])), '-r')
        
    ax.set_xlabel('Tempo [s]')
    ax.set_ylabel('Velocidade [m/s]')
    
    ax.legend(['Velocidade instantânea', 'Média temporal'])
    
    plt.savefig(f'{save_dir_name}/instantaneous_velocity_{obj["name"]}.png')
    plt.close(fig)


def plot_pdf(obj):
    dir_name = obj["path"].split("/")[-2]
    save_dir_name = f'images/{dir_name}'
    
    x, pdf_u = obj["u_x_pdf"], obj["u_pdf"]/np.float64(obj["u_rms"])
    mu = obj["u_bar_t"]
    sigma = np.sqrt(obj["variance"])
    gauss = norm.pdf(x, mu, sigma)
    
    fig = plt.figure(f'probability_density_function_{obj["name"]}', [6, 6])
    ax = plt.axes()
    ax.set_title(f'Velocity by Probablility Density Function {obj["name"]}')
    ax.grid(1, 'both')
    
    # ax.plot(x, pdf_u, '-b')
    ax.hist(obj["u"], bins=500, density=True, stacked=True)
    ax.plot(x, gauss, '-r', lw=1.75)
    ax.vlines(obj["u_bar_t"], min(pdf_u), max(gauss), colors='k', linestyles='dashed')
    
    ax.set_xlabel('Velocity $u$ [m/s]')
    ax.set_ylabel('Probability Density Function $P(u)$')
    plt.savefig(f'{save_dir_name}/pdf_{obj["name"]}.png')
    plt.close(fig)
    
def plot_pdf_hrep(obj):
    dir_name = obj['path'].split("/")[-2]
    save_dir_name = f'images/{dir_name}'
    
    try:
        os.mkdir(save_dir_name)
    except FileExistsError:
        pass
        
    x, pdf_u = obj["u_x_pdf"], np.array(obj["u_pdf"])/np.float64(obj["u_rms"])
    mu = obj["u_bar_t"]
    sigma = obj["u_rms"]
    gauss = norm.pdf(x, mu, sigma)
    
    fig = plt.figure(f'probability_density_function_{obj["name"]}', [6, 6])
    ax = plt.axes()
    ax.set_title(f'Velocity by Probablility Density Function {obj["name"]}')
    ax.grid(1, 'both')
    
    # ax.plot(x, pdf_u, '-b')
    plt.hist(obj["u_bar_s"], bins=500, density=True, stacked=True)
    ax.plot(x, gauss, '-r', lw=1.75)
    # ax.vlines(obj["u_bar_t"], min(pdf_u), max(gauss), colors='k', linestyles='dashed')
    
    ax.set_label('Velocity $u$ [m/s]')
    ax.set_ylabel('Probability Density Function $P(u)$ [%]')
    plt.savefig(f'{save_dir_name}/pdf_{obj["name"]}.png')
    plt.close(fig)
    
def plot_pdf_hrep_prime(obj):
    dir_name = obj['path'].split("/")[-2]
    save_dir_name = f'images/{dir_name}'
    
    try:
        os.mkdir(save_dir_name)
    except FileExistsError:
        pass
        
    x, pdf_u = obj["u_x_pdf"], np.array(obj["u_pdf"])
    mu = obj["u_bar_t"]
    sigma = obj["u_rms"]
    gauss = norm.pdf(x, mu, sigma)
    
    fig = plt.figure(f'probability_density_function_{obj["name"]}', [6, 6])
    ax = plt.axes()
    ax.set_title(f'Velocity by Probablility Density Function {obj["name"]}')
    ax.grid(1, 'both')
    
    # ax.plot(x, pdf_u, '-b')
    plt.hist(obj["u_bar_s"], bins=500, density=True, stacked=True)
    ax.plot(x, gauss, '-r', lw=1.75)
    # ax.vlines(obj["u_bar_t"], min(pdf_u), max(gauss), colors='k', linestyles='dashed')
    
    ax.set_label('Velocity $u$ [m/s]')
    ax.set_ylabel('Probability Density Function $P(u)$ [%]')
    plt.savefig(f'{save_dir_name}/pdf_{obj["name"]}.png')
    plt.close(fig)

    
def plot_velocity_distribution(obj):
    dir_name = obj['path'].split("/")[-2]
    save_dir_name = f'images/{dir_name}'
    
    try:
        os.mkdir(save_dir_name)
    except FileExistsError:
        pass
    
    base = plt.gca().transData
    rot = transforms.Affine2D().rotate_deg(90)
        
    fig = plt.figure(f'velocity_distribution_{obj["name"]}')
    ax = plt.axes()
    ax.set_title(f'Distribuição de Velocidade {obj["name"]}')
    ax.grid(1, 'both')
    
    ax.plot(obj["u_bar_s"], obj["positions"], '-b')
    # for position in obj["positions"]:
    #     ax.text(obj["u_bar_s"][position], position, obj["u_bar_s"][position], transform=rot+base)
    # ax.vlines(obj["u_bar_t"], min(obj["positions"]), max(obj["positions"]), colors='k', linestyles='dashed')
    
    ax.set_ylabel('Position in y-axis $P_y$')
    ax.set_xlabel('Velocity $u$ [m/s]')
    plt.savefig(f'{save_dir_name}/velocity_distribution_{obj["name"]}.png')
    plt.close(fig)
    
def plot_turb_int_distribution(obj):
    dir_name = obj['path'].split("/")[-2]
    save_dir_name = f'images/{dir_name}'
    
    try:
        os.mkdir(save_dir_name)
    except FileExistsError:
        pass
    
    base = plt.gca().transData
    rot = transforms.Affine2D().rotate_deg(90)
        
    fig = plt.figure(f'velocity_distribution_{obj["name"]}')
    ax = plt.axes()
    ax.set_title(f'Distribuição de Intensidade de Turbulência {obj["name"]}')
    ax.grid(1, 'both')
    
    ax.plot(obj["turb_int"], obj["positions"], '-b')
    # for position in obj["positions"]:
    #     ax.text(obj["u_bar_s"][position], position, obj["u_bar_s"][position], transform=rot+base)
    # ax.vlines(obj["u_bar_t"], min(obj["positions"]), max(obj["positions"]), colors='k', linestyles='dashed')
    
    ax.set_ylabel('Position in y-axis $P_y$')
    ax.set_xlabel('Turbulence Intensity $I$')
    plt.savefig(f'{save_dir_name}/tub_int_{obj["name"]}.png')
    plt.close(fig)
    
def main():
    try:
        os.mkdir('images')
    except FileExistsError:
        pass
    FOLDERS = ('lre', 'hre', 'hre_prob', 'perfil_mon', 'perfil_jus')
    # FOLDERS = ('perfil_mon', 'perfil_jus')
    for folder in FOLDERS:
        
        FOLDER = f'json_results/{folder}/'
        paths = [FOLDER+name for name in os.listdir(FOLDER) if os.path.isfile(os.path.join(FOLDER, name))]
        for path in paths:
            print(f'{path}')
            try:
                os.mkdir(f'images/{folder}')
            except FileExistsError:
                pass
            with open(path) as data:
                obj = json.load(data)
                
            if ((folder == 'lre' or folder == 'hre') 
                and (path.split("/")[-1] != 'lre.json' and path.split("/")[-1] != 'hre.json')):
                plot_instant_velocity(obj)
                plot_pdf(obj)
                
            elif folder == 'hre_prob':
                if path.split("/")[-1] == 'hre_prob.json':
                    plot_instant_velocity(obj)
                    plot_pdf_hrep(obj)
                    
                else:
                    plot_instant_velocity(obj)
                    plot_pdf(obj)
                    
            elif path.split("/")[-1] == 'perfil_mon.json' or path.split("/")[-1] == 'perfil_jus.json':
                plot_velocity_distribution(obj)
                plot_turb_int_distribution(obj)
    
if __name__ == '__main__':
    main()