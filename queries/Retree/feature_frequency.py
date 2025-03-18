import onnx
import argparse
from collections import defaultdict
from matplotlib import pyplot as plt

def get_attribute(onnx_model, attr_name):
    i = 0
    while 1:
        attributes = onnx_model.graph.node[i].attribute
        for attr in attributes:
            if attr.name == attr_name:
                return attr
        i += 1
        

default_model = 'house_16H_d10_l475_n949_20250302095052_out'

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default=default_model)
args = parser.parse_args()

model_path = args.model
model = onnx.load(model_path + '.onnx')

nodes_featureids = list(get_attribute(model, 'nodes_featureids').ints)
node_modes = list(get_attribute(model, 'nodes_modes').strings)

feature_frequency = defaultdict(int)
for i in range(len(nodes_featureids)):
    if node_modes[i] == b'LEAF':
        continue
    feature_frequency[nodes_featureids[i]] += 1

# print(f'Feature frequency: {feature_frequency}')


plt.figure()
plt.title("Feature frequency")
plt.bar(
    feature_frequency.keys(), feature_frequency.values(), color="r", align="center"
)
plt.savefig(f"{model_path}_feature_frequency.png")
plt.close()
