import numpy as np

def statistical_mean(data_list):
    
    statistical_mean = np.zeros(len(data_list[0].u))

    for j in range(len(statistical_mean)):
        spacial_avg = 0.
        for i, data in enumerate(data_list):
            spacial_avg += data.u[j]
            statistical_mean[j] = spacial_avg / len(data_list)

    return statistical_mean