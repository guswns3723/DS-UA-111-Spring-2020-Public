import numpy as np

def sample_with_replacement(sample_size, choices):
    return np.random.choice(choices, size = sample_size, replace = True)

def sample_proportions(sample_size, probabilities):
    return np.random.multinomial(sample_size, probabilities) / sample_size