from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


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
    plt.hist(data, bins=20, edgecolor="black")

    plt.title("Histogram of Random Data")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.savefig(f"model/{filename}_hist.png")
    plt.close()


def percentile_values(data, filename):
    sorted_data = np.sort(data)
    percentiles_1 = np.arange(1, 101)
    percentiles_5 = np.arange(5, 101, 5)

    with open(f"model/{filename}.txt", "w", encoding="utf-8") as f:
        values = [f"{round(np.percentile(sorted_data, p), 3)}" for p in percentiles_1]
        f.write("\n".join(values))
    with open(f"model/predicates.txt", "w", encoding="utf-8") as f:
        values = [f"{round(np.percentile(sorted_data, p), 3)}" for p in percentiles_5]
        f.write("\n".join(values))


def value_distribution(data, filename):
    counter = Counter(data)
    with open(f"model/{filename}.txt", "w", encoding="utf-8") as f:
        for item, count in counter.items():
            f.write(f"{item}: {(count/data.shape[0])*100}%\n")


def plot_feature_importances_v1(model, shape, filename):
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Plotting the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(
        range(shape), importances[indices], color="r", yerr=std[indices], align="center"
    )
    plt.xticks(range(shape), indices)
    plt.xlim([-1, shape])

    plt.savefig(f"model/{filename}_feature_importances.png")
    plt.close()


def plot_feature_importances(model, shape, filename):
    n_features = 0
    features = []
    
    if type(model) is DecisionTreeClassifier or type(model) is DecisionTreeRegressor:
        n_features = model.tree_.n_features
        features = list(model.tree_.feature)
    
    elif type(model) is RandomForestClassifier or type(model) is RandomForestRegressor:
        for tree in model.estimators_:
            n_features = tree.tree_.n_features
            features += list(tree.tree_.feature)
    else:
        raise Exception("???")
    
    xy = dict(Counter(features))
    del xy[-2]

    # Plotting the feature importances of the forest
    plt.figure()
    plt.title("Feature frequency")
    plt.bar(
        xy.keys(), xy.values(), color="r", align="center"
    )
    # plt.xticks(range(shape), indices)
    # plt.xlim([-1, shape])

    plt.savefig(f"model/{filename}_feature_frequency.png")
    plt.close()