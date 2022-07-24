from scipy.stats import norm
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from test_temporal import TemporalAnalysis
plt.rcParams['font.family'] = 'Times New Roman'
# from numba import njit

def test():
    fig, ax = plt.subplots(1, 1)
    mean, var, skew, kurt = norm.stats(moments='mvsk')
    rv = norm()
    x = np.linspace(norm.ppf(0.01),
                    norm.ppf(0.99), 100)
    ax.plot(x, norm.pdf(x),
        'r-', lw=5, alpha=0.6, label='norm pdf')
    ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    plt.show()

def pdf_1(u_t, times, x, prob, TOL=1e-3):
    i = 0
    
    for i, x in enumerate(x):
        P = 0
        j = 0
        for u, t in zip(u_t, times):
            if u > x - 0.1*TOL and u < x - 0.1*TOL:
                P += t
            else:
                pass
            j += 1
            print(j)
        prob[i] = P / times[-1]
        print(i)
    
    return prob

def pdf(u_t, N):
    N_x = len(u_t)
    
    prob = np.zeros(N)
    u = np.sort(u_t)
    x = np.linspace(min(u), max(u), N)
    pdf = pdf_algorithm(u, x, prob, N, N_x, (1/N))
    
    return x, pdf

def pdf_algorithm(u, x, pdf, N, N_x, TOL):
    k = 0
    i = 0
    p = 0
    while k < N_x and i < N:
        print(i, k, p)
        if u[k] > x[i] - 0.1*TOL and u[k] < x[i+1]+0.1*TOL:
            k += 1
            p += 1
        else:
            pdf[i] = p/N_x
            p = 0 
            i += 1
        
    return pdf
    
def plot_pdf(data):
    x, pdf_u = pdf(data.u,int(len(data.u)/2))
    mu = data.u_bar
    sigma = np.sqrt(data.variance)
    gauss = norm.pdf(x, mu, sigma)/100
    
    plt.figure(f'probability_density_function_{data.re_type}{data.id}', [8, 8])
    plt.title(f'Velocity by Probablility Density Function {data.re_type}{data.id}')
    plt.grid(1, 'both')
    # plt.hist(hre.u, 500)
    # plt.scatter(x, pdf_u, marker='.', linewidths=.1, c='k')
    plt.plot(x, pdf_u, '-b')
    # x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
    plt.plot(x, gauss, '-r', lw=1.75)
    plt.vlines(data.u_bar, min(pdf_u), max(gauss), colors='k', linestyles='dashed')
    plt.xlabel('Velocity $u$ [m/s]')
    plt.ylabel('Probability Density Function $P(u)$')
    plt.show()
    
    


if __name__ == '__main__':
    T_FINAL = 3.2766 # > [s]
    TIME_STEP = 0.0002 # > [s]
    MEASUREMENTS = 16384 # > [adim]
    MEASUREMENT_CONDITIONS = [T_FINAL, TIME_STEP, MEASUREMENTS]
    
    mean, var, skew, kurt = norm.stats(moments='mvsk')
    rv = norm()

    hre = TemporalAnalysis('data/hre_prob/hrep25.dat', MEASUREMENT_CONDITIONS)
    
    # pdf_u = rv.pdf(hre.u)

    # x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), MEASUREMENTS)
    x, pdf_u = pdf(hre.u,int(len(hre.u)/2))
    mu = hre.u_bar
    sigma = np.sqrt(hre.variance)
    gauss = norm.pdf(x, mu, sigma)/100
    
    # plot_pdf = np.zeros(len(pdf_u))
    # curr_max = 0
    # total_max = max(pdf_u)
    # for i in range(len(pdf_u)):
    #     if pdf_u[i] > curr_max and curr_max < total_max:
    #         curr_max = pdf_u[i]
    #         plot_pdf[i] = curr_max
    #     elif pdf_u[i] == curr_max:
    #         break
    # for i in range()
    

    print(x)
    print(pdf_u)
    print(np.sort(hre.u))
    print(np.sum(pdf_u))
    plt.figure(f'probability_density_function_{hre.re_type}{hre.id}', [8, 8])
    plt.title(f'Velocity by Probablility Density Function {hre.re_type}{hre.id}')
    plt.grid(1, 'both')
    # plt.hist(hre.u, 500)
    # plt.scatter(x, pdf_u, marker='.', linewidths=.1, c='k')
    plt.plot(x, pdf_u, '-b')
    # x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
    plt.plot(x, gauss, '-r', lw=1.75)
    plt.vlines(hre.u_bar, min(pdf_u), max(gauss), colors='k', linestyles='dashed')
    plt.xlabel('Velocity $u$ [m/s]')
    plt.ylabel('Probability Density Function $P(u)$')
    plt.show()