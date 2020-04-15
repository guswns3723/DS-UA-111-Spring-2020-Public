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

def calculate_pvalue(observed_test_statistic, simulated_test_statistics):
    how_many_less_than = 0

    for value in simulated_test_statistics:
        if value <= observed_test_statistic:
            how_many_less_than = how_many_less_than + 1
         
    trials = len(simulated_test_statistics)
            
    return how_many_less_than / trials

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

def plot_standard_deviation_bounds(data):
    for column in ['Birth Weight', 'Gestational Days', 'Maternal Age', 'Maternal Height']:
        mean_column = np.mean(data[column])
        standard_deviation_column = np.std(data[column])

        lower_bound = mean_column - 3 * standard_deviation_column
        upper_bound = mean_column + 3 * standard_deviation_column

        within_three = []
        for weight in data[column]:
            if lower_bound <= weight <= upper_bound:
                within_three.append(1)
            else:
                within_three.append(0)

        fraction_within_three = sum(within_three) / len(within_three)

        percentile_6 = calculate_percentile(data[column], 6)
        percentile_94 = calculate_percentile(data[column], 94)

        plt.hlines(y = 0, xmin = lower_bound, xmax = upper_bound, lw = 10, color = "red", zorder = 5)
        plt.hlines(y = 0, xmin = percentile_6, xmax = percentile_94, lw = 10, color = "green", zorder = 10)

        plt.hist(data[column], density = True, alpha = 0.6, rwidth=0.95)
        plt.title(f'{column} {fraction_within_three:.2f}')
        plt.show();