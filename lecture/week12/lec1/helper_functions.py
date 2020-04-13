import numpy as np
import matplotlib.pyplot as plt

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

def calculate_delays(amount_delay, simulated_delays):
    output = []
    for amount, simulated in zip(amount_delay, simulated_delays):
        output += int(simulated) * [int(amount)]
        
    return output 

def generate_boxplot(data, lower_cutoff, upper_cutoff, lower_quartile, median, upper_quartile):
    plt.boxplot(data["Delay"], vert = False, showfliers=False)
    plt.title("Flight Delays")

    plt.scatter(lower_cutoff, 1, s = 30, c = "g")
    plt.scatter(upper_cutoff, 1, s = 30, c = "r")

    plt.hlines(y = 1, xmin = lower_cutoff, xmax=lower_quartile, color="green", lw = 3, zorder=10)
    plt.hlines(y = 1, xmin = upper_quartile, xmax=upper_cutoff, color="red", lw = 3, zorder=10)

    plt.annotate("75th\nPercentile",(upper_quartile, 1.2))
    plt.annotate("25th\nPercentile",(lower_quartile, 1.2))
    plt.annotate("Upper\nCutoff",(upper_cutoff, 0.8))
    plt.annotate("Lower\nCutoff",(lower_cutoff, 0.8))
    plt.annotate("50th\nPercentile",(median, 0.8));
    
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