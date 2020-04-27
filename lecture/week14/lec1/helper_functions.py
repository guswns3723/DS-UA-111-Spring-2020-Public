import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets


def sample_without_replacement(sample_size, data):
    return data.sample(sample_size, replace = False)

def sample_with_replacement(sample_size, choices):
    return np.random.choice(choices, size = sample_size, replace = True)

def calculate_percentile(data, percentile):
    sorted_data = sorted(data)
    length_data = len(sorted_data)
    percentile_fraction = percentile / 100 
    index = np.ceil(percentile_fraction * length_data) - 1
    return sorted_data[int(index)]

def summarize_groups(data, number):
    groups = pd.cut(data["Bitcoin"], number)
    data_grouped = data.groupby(groups).agg({"Bitcoin":np.median, "Ethereum":np.mean})
    data_grouped = data_grouped.rename(columns = {"Bitcoin": "Median of Bitcoin Price", "Ethereum": "Mean of Ethereum Price"}) 
    data_grouped.index.name = "Cryptocurrencies"
    return data_grouped

def plot_standard_deviation_bounds(data, bound):
    within_bound = []
    converted_column = data
    for coverted_score in converted_column:    
        if abs(coverted_score) <= bound:
            within_bound.append(1)
        else:
            within_bound.append(0)

    fraction_within_bound = sum(within_bound) / len(within_bound)

    half_bound = 50 * (1 / (bound**2))                    

    lower_percentile = calculate_percentile(converted_column, half_bound)
    upper_percentile = calculate_percentile(converted_column, 100 - half_bound)

    plt.hist(converted_column, density = True, alpha = 0.6, rwidth=0.95)

    plt.hlines(y = 0, xmin = -bound, xmax = bound, lw = 10, color = "red", zorder = 5)
    plt.hlines(y = 0, xmin = lower_percentile, xmax = upper_percentile, lw = 10, color = "green", zorder = 10)

    plt.title(f'Fraction {fraction_within_bound:.2f} within {bound} Standard Deviations')
    plt.text(2, 0.8, f"Standard deviation bound\npredicts at least {1 -  (half_bound/50):0.2f} of data\nwithin {bound} standard deviations")
    
    plt.show();
        
def plotter(data1, data2, slope, intercept):    
    x = data1["open"]
    y = data2['open']
    plt.scatter(x, y, color='purple', s=30, zorder = 20)

    x_range = np.linspace(0, 20000, 1000)
    plt.plot(x_range, intercept + slope * x_range, "g", lw = 2, zorder = 10)

    residuals = intercept + slope * x - y 
    mse = np.mean(residuals**2)
    plt.title(f"Mean Square Error {np.round(mse)}")

    dates = ["2017-11-28","2015-10-03","2018-01-05"]
    x_state = data1.loc[data1["date"].isin(dates), "open"]
    y_state = data2.loc[data2["date"].isin(dates), "open"]

    x_state = x_state.values
    y_state = y_state.values

    plt.scatter(x_state, y_state, s=75, color='green', zorder = 30)

    for idx, x_position in enumerate(x_state):
        min_ = min(y_state[idx], intercept + slope * x_position)
        max_ = max(y_state[idx], intercept + slope * x_position)
        plt.vlines(x = x_position, ymin = min_, ymax = max_, color = "red", linestyle = "dashed")

    plt.xlim(-1000, 20000)
    plt.ylim(-100, 1400)
    plt.show();
    
    
def regression_widget(data1, data2): 
    interact(
    plotter,
    data1 = fixed(data1),
    data2 = fixed(data2),
    slope=widgets.FloatSlider(min=0.01, max=0.07, step=0.01, value=0.05, msg_throttle=1),
    intercept=widgets.FloatSlider(min=0, max=200, step=120, value=100, msg_throttle=1))