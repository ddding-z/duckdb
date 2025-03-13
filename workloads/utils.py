from collections import Counter
from matplotlib import pyplot as plt
import numpy as np


def get_attribute(onnx_model, attr_name):
    i = 0
    while 1:
        attributes = onnx_model.graph.node[i].attribute
        for attr in attributes:
            if attr.name == attr_name:
                return attr
        i += 1


def plot_value_distribution(data, filename):
    sorted_data = np.sort(data)

    x = np.arange(len(sorted_data))
    plt.scatter(x, sorted_data, marker="o")

    plt.title("Value Distribution")
    plt.ylabel("value")

    plt.savefig(f"model/{filename}.png")
    plt.close()

def plot_hist(data, filename):
    plt.hist(data, bins=20, edgecolor='black')

    plt.title('Histogram of Random Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    plt.savefig(f"model/{filename}_hist.png")
    plt.close()


def percentile_values(data, filename):
    sorted_data = np.sort(data)
    percentiles_1 = np.arange(1, 101)
    percentiles_5 = np.arange(5, 101, 5)

    with open(f"model/{filename}.txt", "w", encoding="utf-8") as f:
        f.write("5~100% (5%)\n")
        for p in percentiles_5:
            value = np.percentile(sorted_data, p)
            f.write(f"{round(value, 3)}\n")
        f.write("\n")
        f.write("1~100% (1%)\n")
        for p in percentiles_1:
            value = np.percentile(sorted_data, p)
            f.write(f"{round(value, 3)}\n")


def value_distribution(data, filename):
    counter = Counter(data)
    with open(f"model/{filename}.txt", "w", encoding="utf-8") as f:
        for item, count in counter.items():
            f.write(f"{item}: {(count/data.shape[0])*100}%\n")
