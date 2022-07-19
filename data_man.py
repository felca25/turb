from dataclasses import dataclass, field
from statistics import covariance
from matplotlib import pyplot as plt
from scipy.stats import norm
import os, os.path
import numpy as np
import pandas as pd
import json
from statistical_functions import *

def export_data(df, path):
    print('Exporting csv ...')
    df.to_csv(f'{path}.csv')
    print('Export complete')
    
    return df

def get_spatial_points(path):
    points = []
    with open(path, 'r') as f:
        yo = f.readline()[3:-3].split(',')
        yo = float('.'.join(yo))
        
        data = f.readlines()[1:]
        for line in data:
            line = round(float('.'.join(line[14:-3].split(','))) - yo, 1)
            points.append(line)
            
    return points

def main():
    points_mon = get_spatial_points('data/pos_mon.txt')
    points_jus = get_spatial_points('data/pos_jus.txt')
        
if __name__ == '__main__':
    main()
    