import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize


def sample_without_replacement(sample_size, data):
    return data.sample(sample_size, replace = False)

def sample_with_replacement(sample_size, table):
    return table.sample(sample_size, replace = True)

def calculate_percentile(data, percentile):
    sorted_data = sorted(data)
    length_data = len(sorted_data)
    percentile_fraction = percentile / 100 
    index = np.ceil(percentile_fraction * length_data) - 1
    return sorted_data[int(index)]

def calculate_pvalue(observed_test_statistic, simulated_test_statistics):
    how_many_less_than = 0

    for value in simulated_test_statistics:
        if value <= observed_test_statistic:
            how_many_less_than = how_many_less_than + 1
         
    trials = len(simulated_test_statistics)
            
    return how_many_less_than / trials
    
def convert_standard_units(array):
    return (array - np.mean(array)) / np.std(array)

def correlation(table, explanatory_variable, response_variable):
    x_standard = convert_standard_units(table[explanatory_variable])
    y_standard = convert_standard_units(table[response_variable])
    
    return np.mean(x_standard * y_standard)

def slope(table, explanatory_variable, response_variable):
    r = correlation(table, explanatory_variable, response_variable)
    y_sd = np.std(table[response_variable])
    x_sd = np.std(table[explanatory_variable])
    return r * y_sd / x_sd

def intercept(table, explanatory_variable, response_variable):
    y_mean = np.mean(table[response_variable])
    x_mean = np.mean(table[explanatory_variable])
    return y_mean - slope(table, explanatory_variable, response_variable) * x_mean

def fitted_values(table, explanatory_variable, response_variable):
    a = slope(table, explanatory_variable, response_variable)
    b = intercept(table, explanatory_variable, response_variable)
    return a * table[explanatory_variable] + b

def generate_blob(mean, covariance, number):
    coordinates = np.random.randn(2, number)
    coordinates = mean.reshape(2,-1) + np.dot(covariance, coordinates)
    return coordinates

def generate_other_sample():
    mean = np.array([4.5, 73])
    covariance = np.array([[0.3,0],[0,4]])
    number = 100
    coordinates1 = generate_blob(mean, covariance, number)    


    mean = np.array([2.2, 50])
    covariance = np.array([[0.3,0],[0,4]])
    number = 100
    coordinates2 = generate_blob(mean, covariance, number)    

    coordinates = np.empty(coordinates1.shape + (2,))
    coordinates[:,:,0] = coordinates1
    coordinates[:,:,1] = coordinates2
    
    return coordinates

def add_patch(ax, coordinates, color = "green"): 
    ax.add_patch(
        matplotlib.patches.Polygon(coordinates,
            color=color,
            fill=False, hatch = '//'
        )
    )
    
 

       