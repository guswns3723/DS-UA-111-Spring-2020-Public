import numpy as np

def sample_without_replacement(sample_size, choices):
    return np.random.choice(choices, size = sample_size, replace = False)

def sample_with_replacement(sample_size, choices):
    return np.random.choice(choices, size = sample_size, replace = True)

def sample_proportions(sample_size, probabilities):
    return np.random.multinomial(sample_size, probabilities) / sample_size

def calculate_percentile(data, percentile):
    sorted_data = sorted(data)
    length_data = len(sorted_data)
    percentile_fraction = percentile / 100 
    index = np.ceil(percentile_fraction * length_data) - 1
    return sorted_data[int(index)]