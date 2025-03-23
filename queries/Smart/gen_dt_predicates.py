import onnx
import pandas as pd
from typing import List
import argparse
from utils import Node, model2tree

# 1. 选择率越小，效果越好 vs 选择率越极端（越大或越小），效果越好
# 2. 扩展的纯 SQL vs SQL + ONNX
# 3. 添加谓词，不修改模型 vs 修改模型
# 4. 只支持决策树 vs 支持决策树 + 随机森林
# 5. 单个模型 vs ML pipeline
# 6. 支持获取模型推理的准确值，不支持获取模型推理的准确值

class Predicate:
    def __init__(
        self,
        feature_id,
        lvalue,
        rvalue
        ):
        self.feature_id: int = feature_id
        self.lvalue: float | None = lvalue
        self.rvalue: float | None = rvalue

def get_attribute(model, attr_name):
    i = 0
    while 1:
        attributes = model.graph.node[i].attribute
        for attr in attributes:
            if attr.name == attr_name:
                return attr
        i += 1        

def get_feature_ids(model):
    input_nodes_featureids = get_attribute(model, 'nodes_featureids').ints
    return list(set(input_nodes_featureids))

def get_feature_names(model) -> List:
    feature_names = []
    for input in model.graph.input:
        feature = input.name
        feature_encoded = []
        for node in model.graph.node:
            if node.op_type == 'OneHotEncoder' and node.input[0] == feature:
                    for attr in node.attribute:
                        if attr.name == "cats_strings":
                            encoded_list = [item.decode('utf-8') for item in attr.strings]
                            feature_encoded.extend(encoded_list)
                        elif attr.name == "cats_int64s":
                            encoded_list = [f"{feature}_{item}" for item in attr.ints]
                            feature_encoded.extend(encoded_list)
        if len(feature_encoded):
            feature_names.extend(feature_encoded)
        else:
            feature_names.append(feature)
    return feature_names

def simplify_predicates_disjunction(predicates) -> 'Predicate | None':
    min_value = float('inf')
    max_value = float('-inf')

    for p in predicates:
        if p == None:
            return None
        # print(f'disjunction_before_feature_id: {p.feature_id}, lvalue: {p.lvalue}, rvalue: {p.rvalue}')
        if p.lvalue is not None and p.lvalue > max_value:
            max_value = p.lvalue
        if p.rvalue is not None and p.rvalue < min_value:
            min_value = p.rvalue
    #     print(f'disjunction_before_feature_id: {p.feature_id}, max_value: {max_value}, min_value: {min_value}')
    # print(f'disjunction_after_feature_id: {predicates[0].feature_id}, lvalue: {max_value}, rvalue: {min_value}\n')
    return Predicate(predicates[0].feature_id, max_value, min_value)

def simplify_predicates_conjunction(predicates) -> 'Predicate | None':
    min_value = float('inf')
    max_value = float('-inf')

    for p in predicates:
        if p == None:
            return None
        # print(f'conjunction_before_feature_id: {p.feature_id}, lvalue: {p.lvalue}, rvalue: {p.rvalue}')
        if p.lvalue is not None and p.lvalue < min_value:
            min_value = p.lvalue
        if p.rvalue is not None and p.rvalue > max_value:
            max_value = p.rvalue
    # print(f'conjunction_after_feature_id: {predicates[0].feature_id}, lvalue: {min_value}, rvalue: {max_value}\n')
    return Predicate(predicates[0].feature_id, min_value, max_value)

def generate_predicate(root:'Node', feature_id, f) -> 'Predicate | None':
    Predicates = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)                
        if node.mode == b'LEAF':
            if (int(f(node.target_weight))):
                predicates = [Predicate(feature_id, float('inf'), float('-inf'))]
                curr = node
                while curr.parent is not None:
                    parent = curr.parent
                    if parent.feature_id == feature_id:
                        if curr.id == parent.left.id:
                            predicates.append(Predicate(feature_id, parent.value, float('-inf')))
                        else:
                            predicates.append(Predicate(feature_id, float('inf'), parent.value))
                    curr = parent
                Predicates.append(simplify_predicates_conjunction(predicates))
    if Predicates:
        return simplify_predicates_disjunction(Predicates)
    else:
        return None

def generate_predicates(input_model, root:'Node', f) -> 'List[Predicate | None]':
    predicates = []
    feature_ids = get_feature_ids(input_model)
    # print(feature_ids)
    for feature_id in feature_ids:
        predicates.append(generate_predicate(root, feature_id, f))
        # break
    return predicates    

parser = argparse.ArgumentParser()
parser.add_argument('--workload', '-w', type=str)
parser.add_argument('--model', '-m', type=str)
parser.add_argument('--threshold', '-t', type=float)
args = parser.parse_args()

workload = args.workload
model_name = args.model
threshold = args.threshold

model_type = 'reg'
if workload == "flights" or  workload == "tpcai-uc08" or workload == 'wine_quality':
    model_type = 'clf'

func = lambda x: x > threshold if model_type == 'reg' else x == threshold
model_path = None

if model_type == "reg": 
    model_path = f'/volumn/Retree_exp/workloads/{workload}/model/{model_name}.onnx'
if model_type == 'clf':
    model_path = f'/volumn/Retree_exp/workloads/{workload}/model/{model_name}_reg.onnx'

model = onnx.load(model_path)
feature_names = get_feature_names(model)
# print(feature_names)

root = model2tree(model, None, 0, None, None, None)
root.parent = None

predicates = generate_predicates(model, root, func)
effective_predicates = []
for p in predicates:
    if p is not None:
        if p.lvalue == float('inf') and p.rvalue == float('-inf'):
            continue
        effective_predicates.append(p)

with open(f"workloads/{workload}/predicates.csv", "a", encoding="utf-8") as f:
    for p in effective_predicates:
        # feature_name,lvalue(<),rvalue(>),predicate
        f.write(f'{feature_names[p.feature_id]},{p.lvalue},{p.rvalue},{threshold}\n')
