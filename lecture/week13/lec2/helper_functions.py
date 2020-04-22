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

def summarize_groups(data):
    groups = pd.qcut(data["father"], 5)
    data_grouped = data.groupby(groups).agg({"father":np.median, "childHeight":np.mean})
    data_grouped = data_grouped.rename(columns = {"father": "Median Father Height", "childHeight": "Mean Child Height"}) 
    data_grouped.index.name = "Range of Father Heights"
    return data_grouped
    
def r_scatter(r):
    "Generate a scatter plot with a correlation approximately r"
    x = np.random.normal(0, 1, 1000)
    z = np.random.normal(0, 1, 1000)
    y = r*x + (np.sqrt(1-r**2))*z
    plt.scatter(x, y, color='darkblue', s=20)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    
def correlation_widget(): 
    interact(
    r_scatter,
    r=widgets.FloatSlider(min=-1, max=1, step=0.1, value=0, msg_throttle=1))
    
def generate_reading_shoes_table():
    r = 0.7
    x = np.random.normal(0, 1, 1000)
    z = np.random.normal(0, 1, 1000)
    y = r*x + (np.sqrt(1-r**2))*z

    table = pd.DataFrame(data = {"Reading Comprehension" : np.round(100 * (x + 5)), "Shoe Size" :  1.5 * (y + 4)})
    return table

def plotter(data, slope, intercept):
    x = data["Math"]
    y = data['Writing']
    plt.scatter(x, y, color='purple', s=30, zorder = 20)
    
    x_range = np.linspace(425, 625, 1000)
    plt.plot(x_range, intercept + slope * x_range, "g", lw = 2, zorder = 10)
    
    residuals = intercept + slope * x - y 
    mse = np.mean(residuals**2)
    plt.title(f"Mean Square Error {np.round(mse)}")
    
    x_state = data.loc[data["State"].isin(["New York", "Washington", "Texas"]),"Math"].to_list()
    y_state = data.loc[data["State"].isin(["New York", "Washington", "Texas"]),"Writing"].to_list()
    
    plt.scatter(x_state, y_state, s=75, color='green', zorder = 30)
    
    for idx, x_position in enumerate(x_state):
        min_ = min(y_state[idx], intercept + slope * x_position)
        max_ = max(y_state[idx], intercept + slope * x_position)
        plt.vlines(x = x_position, ymin = min_, ymax = max_, color = "red", linestyle = "dashed")
    
    plt.xlim(425, 625)
    plt.ylim(420, 600)
    plt.show();
    
    
def regression_widget(data): 
    interact(
    plotter,
    data = fixed(data),
    slope=widgets.FloatSlider(min=0.75, max=1.25, step=0.01, value=1, msg_throttle=1),
    intercept=widgets.FloatSlider(min=5, max=20, step=1, value=15, msg_throttle=1))