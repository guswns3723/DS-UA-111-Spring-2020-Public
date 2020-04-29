import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

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
    
def repeat_bootstrap(data, population_median, number_of_samples_from_population): 
    true_or_false = []
    left_right = []

    for rounds in range(number_of_samples_from_population):
        sample_size = 138
        sample = sample_without_replacement(sample_size, data["Delay"])
        sample_median = calculate_percentile(sample, 50)
        sample_median

        replications = 1000
        resample_size = len(sample)

        estimates = []
        for replication in range(replications):
            resample = sample_with_replacement(resample_size, sample)

            guess = calculate_percentile(resample, 50)
            estimates.append(guess)

        percentile_5 = calculate_percentile(estimates, 5)
        percentile_95 = calculate_percentile(estimates, 95)
        left_right.append([percentile_5, percentile_95])

        does_it_lie_in_confidence_interval = percentile_5 <= population_median <= percentile_95
        true_or_false.append(does_it_lie_in_confidence_interval)
        
    return true_or_false, left_right