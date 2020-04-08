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

def calculate_pvalue(observed_test_statistic, simulated_test_statistics):
    how_many_less_than = 0

    for value in simulated_test_statistics:
        if value <= observed_test_statistic:
            how_many_less_than = how_many_less_than + 1
         
    trials = len(simulated_test_statistics)
            
    return how_many_less_than / trials

def produce_proportions(df, value_label):
    total = len(df)
    return df[value_label].value_counts() / total

def total_variation_distance(table, value_label, group_label):
    bta_grp = table.groupby(group_label).apply(produce_proportions, value_label)
    
    try:
        bta_grp = bta_grp.stack()
    except: 
        pass
    
    tvd = abs(bta_grp[("Control", 0)] - bta_grp[("Treatment", 0)]) + abs(bta_grp[("Control", 1)] - bta_grp[("Treatment", 1)])
    
    return tvd / 2