from matplotlib import pyplot as plt
from scipy.stats import norm
import os
from statistical_functions import pdf
import numpy as np

def plot_instant_velocity(obj):
    
    dir_name = obj.path.split("/")[-2]
    save_dir_name = f'{dir_name}_images'
    
    try:
        os.mkdir(save_dir_name)
    except FileExistsError:
        pass
    
    fig = plt.figure(f'plot_{obj.name}')
    ax = plt.axes()
    
    ax.set_title(f'Velocidade instantânea $u_{obj.index}$')
    
    ax.plot(obj.times[:len(obj.u)], obj.u)
    
    if obj.mov_avg:
        ax.plot(obj.times[:len(obj.u_bar)], obj.u_bar)
    else:
        ax.plot(obj.times, obj.u_bar*np.ones(len(obj.times)))
        
    ax.set_xlabel('Tempo [s]')
    ax.set_ylabel('Velocidade [m/s]')
    
    ax.legend(['Velocidade instantânea', 'Média temporal'])
    
    plt.savefig(f'{save_dir_name}/instantaneous_velocity_{obj.name}.png')


def plot_pdf(obj):
    
    x, pdf_u = pdf(obj.u,int(len(obj.u)))
    mu = obj.u_bar
    sigma = np.sqrt(obj.variance)
    gauss = norm.pdf(x, mu, sigma)/100
    
    fig = plt.figure(f'probability_density_function_{obj.name}', [6, 6])
    ax = plt.axes()
    ax.set_title(f'Velocity by Probablility Density Function {obj.name}')
    ax.grid(1, 'both')
    
    ax.plot(x, pdf_u, '-b')
    ax.plot(x, gauss, '-r', lw=1.75)
    ax.vlines(obj.u_bar, min(pdf_u), max(gauss), colors='k', linestyles='dashed')
    
    ax.xlabel('Velocity $u$ [m/s]')
    ax.ylabel('Probability Density Function $P(u)$')
    plt.savefig(f'pdf_{obj.name}.png')
    
def plot_pdf_hrep(obj):
        
    x, pdf_u = pdf(obj.u_bar_s,int(len(obj.u_bar_s)))
    mu = obj.u_bar_t
    sigma = np.sqrt(obj.variance)
    gauss = norm.pdf(x, mu, sigma)/100
    
    fig = plt.figure(f'probability_density_function_{obj.name}', [6, 6])
    ax = plt.axes()
    ax.set_title(f'Velocity by Probablility Density Function {obj.name}')
    ax.grid(1, 'both')
    
    ax.plot(x, pdf_u, '-b')
    ax.plot(x, gauss, '-r', lw=1.75)
    ax.vlines(obj.u_bar, min(pdf_u), max(gauss), colors='k', linestyles='dashed')
    
    ax.xlabel('Velocity $u$ [m/s]')
    ax.ylabel('Probability Density Function $P(u)$')
    plt.savefig(f'pdf_{obj.name}.png')
    
def main():
    
    plt.plot([1,2,3,4])