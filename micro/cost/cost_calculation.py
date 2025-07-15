import time
from typing import List, Tuple
import onnx
from onnx import helper, utils, checker
import numpy as np
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import onnxruntime as ort
import pandas as pd
import datetime

nprocess = 50

onnx_to_numpy_type = {
    0: None,  # UNDEFINED (未定义，通常不使用)
    1: np.float32,  # FLOAT
    2: np.uint8,  # UINT8
    3: np.int8,  # INT8
    4: np.uint16,  # UINT16
    5: np.int16,  # INT16
    6: np.int32,  # INT32
    7: np.int64,  # INT64
    8: np.str_,  # STRING
    9: np.bool_,  # BOOL
    10: np.float16,  # FLOAT16
    11: np.float64,  # DOUBLE
    12: np.uint32,  # UINT32
    13: np.uint64,  # UINT64
    14: np.complex64,  # COMPLEX64
    15: np.complex128,  # COMPLEX128
    16: np.float16,  # BFLOAT16 (注意：需要 NumPy >= 1.21 并启用支持)
    17: np.float32,  # FLOAT8E4M3FN (目前没有原生 float8 支持，用 float32 模拟)
    18: np.float32,  # FLOAT8E4M3FNUZ (同上)
    19: np.float32,  # FLOAT8E5M2 (同上)
    20: np.float32,  # FLOAT8E5M2FNUZ (同上)
}


# helper functions
def extract_sub_model(model_path, onnx_model) -> str:
    input_names = [input.name for input in onnx_model.graph.input]
    print(input_names)

    output_names = []
    for node in onnx_model.graph.node:
        if (
            node.op_type == "TreeEnsembleClassifier"
            or node.op_type == "TreeEnsembleRegressor"
        ):
            output_names.extend(node.input)
            break
    sub_model_path = model_path.replace(".onnx", "_pre.onnx")
    utils.extract_model(model_path, sub_model_path, input_names, output_names)
    return sub_model_path


def preprocess_walmart(data):
    data.loc[
        (data["Date"] == "2010-02-12")
        | (data["Date"] == "2011-02-11")
        | (data["Date"] == "2012-02-10"),
        "Super_Bowl",
    ] = True
    data.loc[
        (data["Date"] != "2010-02-12")
        & (data["Date"] != "2011-02-11")
        & (data["Date"] != "2012-02-10"),
        "Super_Bowl",
    ] = False

    data.loc[
        (data["Date"] == "2010-09-10")
        | (data["Date"] == "2011-09-09")
        | (data["Date"] == "2012-09-07"),
        "Labor_Day",
    ] = True
    data.loc[
        (data["Date"] != "2010-09-10")
        & (data["Date"] != "2011-09-09")
        & (data["Date"] != "2012-09-07"),
        "Labor_Day",
    ] = False

    data.loc[
        (data["Date"] == "2010-11-26") | (data["Date"] == "2011-11-25"), "Thanksgiving"
    ] = True
    data.loc[
        (data["Date"] != "2010-11-26") & (data["Date"] != "2011-11-25"), "Thanksgiving"
    ] = False

    data.loc[
        (data["Date"] == "2010-12-31") | (data["Date"] == "2011-12-30"), "Christmas"
    ] = True
    data.loc[
        (data["Date"] != "2010-12-31") & (data["Date"] != "2011-12-30"), "Christmas"
    ] = False

    # convert to datetime
    data["Date"] = pd.to_datetime(data["Date"])
    data["week"] = data["Date"].dt.isocalendar().week
    data["month"] = data["Date"].dt.month
    data["year"] = data["Date"].dt.year

    type_group = {"A": 1, "B": 2, "C": 3}  # changing A,B,C to 1-2-3
    data["Type"] = data["Type"].replace(type_group)
    data["Super_Bowl"] = data["Super_Bowl"].astype(bool).astype(int)
    data["Thanksgiving"] = data["Thanksgiving"].astype(bool).astype(int)
    data["Labor_Day"] = data["Labor_Day"].astype(bool).astype(int)
    data["Christmas"] = data["Christmas"].astype(bool).astype(int)
    data["IsHoliday"] = data["IsHoliday"].astype(bool).astype(int)

    return data


def count(data_path):
    X = pd.read_csv(data_path)
    # X = df.iloc[:, :-1]
    print("Input data shape:", X.shape)
    
def generate_inputs(model_path, data_path):
    onnx_model = onnx.load(model_path)
    sub_model_path = extract_sub_model(model_path, onnx_model)

    op = ort.SessionOptions()
    op.intra_op_num_threads = 5
    op.inter_op_num_threads = 10
    ort_session = ort.InferenceSession(
        sub_model_path, sess_options=op, providers=["CPUExecutionProvider"]
    )

    X = pd.read_csv(data_path)
    if "datetime" in X.columns:
        X["hour"] = X.datetime.apply(lambda x: x.split()[1].split(":")[0]).astype("int")
    if "MarkDown1" in X.columns:
        X = preprocess_walmart(X)
    # X = df.iloc[:, :-1]
    print("Input data shape:", X.shape)
    # X = X.astype(np.float32)

    def run_inference(X):
        columns = [input.name for input in ort_session.get_inputs()]
        columns_type_kv = {
            input.name: onnx_to_numpy_type[input.type.tensor_type.elem_type]
            for input in onnx_model.graph.input
        }
        infer_batch = {
            col: np.array(X[col]).astype(columns_type_kv[col]).reshape((-1, 1))
            for col in columns
        }

        outputs = ort_session.run([ort_session.get_outputs()[0].name], infer_batch)
        return outputs[0]

    return run_inference(X)


def get_attribute(onnx_model, attr_name):
    for node in onnx_model.graph.node:
        attributes = node.attribute
        for attr in attributes:
            if attr.name == attr_name:
                return attr
    return None


def has_attribute(onnx_model, attr_name) -> bool:
    for node in onnx_model.graph.node:
        attributes = node.attribute
        for attr in attributes:
            if attr.name == attr_name:
                return True
    return False


class Node:
    def __init__(
        self,
        id,  # 节点id
        feature_id,  # 特征id
        mode,  # 节点类型，LEAF表示叶子节点，BRANCH_LEQ表示非叶子节点
        value,  # 阈值，叶子节点的值为0
        target_id,  # 叶子节点的taget id
        target_weight,  # 叶子节点的权重，即预测值
        samples,  # 节点的样本数
    ):
        self.id: int = id
        self.feature_id: int = feature_id
        self.mode: bytes = mode
        self.value: float = value
        self.target_id: int | None = target_id
        self.target_weight: float | None = target_weight
        self.samples: int = samples

        self.parent: "Node" | None = None
        self.left: "Node" | None = None
        self.right: "Node" | None = None


def get_target_tree_intervals(onnx_model) -> List[Tuple[int, int]]:
    target_tree_roots: List[int] = []
    target_treeids = get_attribute(onnx_model, "target_treeids").ints
    next_tree_id = 0
    for i, tree_id in enumerate(target_treeids):
        if tree_id == next_tree_id:
            next_tree_id += 1
            target_tree_roots.append(i)

    target_tree_intervals: List[Tuple[int, int]] = []
    for i, root in enumerate(target_tree_roots):
        if i == len(target_tree_roots) - 1:
            end = len(target_treeids)
        else:
            end = target_tree_roots[i + 1]
        target_tree_intervals.append((root, end))
    return target_tree_intervals


def get_tree_intervals(onnx_model) -> List[Tuple[int, int]]:
    tree_roots: List[int] = []
    nodes_treeids = get_attribute(onnx_model, "nodes_treeids").ints
    next_tree_id = 0
    for i, tree_id in enumerate(nodes_treeids):
        if tree_id == next_tree_id:
            next_tree_id += 1
            tree_roots.append(i)

    tree_intervals: List[Tuple[int, int]] = []
    for i, root in enumerate(tree_roots):
        if i == len(tree_roots) - 1:
            end = len(nodes_treeids)
        else:
            end = tree_roots[i + 1]
        tree_intervals.append((root, end))
    return tree_intervals


def model2tree_iterative(
    input_model,
    samples_list: "List[int] | None",
    root_node_id,
    parent: "Node | None",
    tree_interval: "Tuple[int, int] | None" = None,
    target_tree_interval: "Tuple[int, int] | None" = None,
) -> "Node":
    if tree_interval is None:
        tree_interval = (0, len(get_attribute(input_model, "nodes_treeids").ints))
    tree_start, tree_end = tree_interval

    if target_tree_interval is None:
        target_tree_interval = (
            0,
            len(get_attribute(input_model, "target_treeids").ints),
        )
    target_tree_start, target_tree_end = target_tree_interval

    nodes_falsenodeids = get_attribute(input_model, "nodes_falsenodeids").ints[
        tree_start:tree_end
    ]
    nodes_truenodeids = get_attribute(input_model, "nodes_truenodeids").ints[
        tree_start:tree_end
    ]
    nodes_featureids = get_attribute(input_model, "nodes_featureids").ints[
        tree_start:tree_end
    ]
    nodes_hitrates = get_attribute(input_model, "nodes_hitrates").floats[
        tree_start:tree_end
    ]
    nodes_modes = get_attribute(input_model, "nodes_modes").strings[tree_start:tree_end]
    nodes_values = get_attribute(input_model, "nodes_values").floats[
        tree_start:tree_end
    ]

    target_nodeids = get_attribute(input_model, "target_nodeids").ints[
        target_tree_start:target_tree_end
    ]
    target_weights = get_attribute(input_model, "target_weights").floats[
        target_tree_start:target_tree_end
    ]

    target_id_map = {nid: idx for idx, nid in enumerate(target_nodeids)}

    stack = [(root_node_id, parent, None)]
    root = None

    while stack:
        node_id, parent_node, child_attr = stack.pop()

        feature_id = nodes_featureids[node_id]
        mode = nodes_modes[node_id]
        value = nodes_values[node_id]
        samples = int(nodes_hitrates[node_id])
        tgt_idx = target_id_map.get(node_id, None)
        tgt_weight = target_weights[tgt_idx] if tgt_idx is not None else None

        node = Node(
            id=node_id,
            feature_id=feature_id,
            mode=mode,
            value=value,
            target_id=tgt_idx,
            target_weight=tgt_weight,
            samples=samples,
        )
        node.parent = parent_node

        if parent_node is not None and child_attr is not None:
            setattr(parent_node, child_attr, node)
        else:
            root = node

        if mode != b"LEAF":
            stack.append((nodes_falsenodeids[node_id], node, "right"))
            stack.append((nodes_truenodeids[node_id], node, "left"))

    return root


def model2trees(input_model, samples_list: "List[int] | None") -> "List[Node]":
    tree_intervals = get_tree_intervals(input_model)
    target_tree_intervals = get_target_tree_intervals(input_model)

    trees = []
    with ProcessPoolExecutor(max_workers=nprocess) as executor:
        future_to_tree = {
            executor.submit(
                model2tree_iterative,
                input_model,
                samples_list,
                0,
                None,
                tree_interval,
                target_tree_intervals[tree_no],
            ): tree_no
            for tree_no, tree_interval in enumerate(tree_intervals)
        }
        for future in as_completed(future_to_tree):
            try:
                tree_root = future.result()
                trees.append(tree_root)
            except Exception as e:
                print(f"Error constructing tree {future_to_tree[future]}: {e}")
    return trees


def clf2reg(input_model: onnx.ModelProto) -> onnx.ModelProto:
    # input model attributes
    # # class_ids: 叶子节点权重对应的类别id
    input_class_ids = get_attribute(input_model, "class_ids").ints
    # # class_nodeids: 叶子节点权重对应的节点id
    input_class_nodeids = get_attribute(input_model, "class_nodeids").ints
    # # class_treeids: 叶子节点权重对应的树id
    input_class_treeids = get_attribute(input_model, "class_treeids").ints
    # # class_weights: 叶子节点权重，即预测值
    input_class_weights = get_attribute(input_model, "class_weights").floats
    # # classlabels_int64s: 类别id
    input_classlabels_int64s = get_attribute(input_model, "classlabels_int64s").ints
    # # nodes_falsenodeids: 右侧分支
    input_nodes_falsenodeids = get_attribute(input_model, "nodes_falsenodeids").ints
    # # nodes_featureids: 特征id
    input_nodes_featureids = get_attribute(input_model, "nodes_featureids").ints
    # # nodes_hitrates
    input_nodes_hitrates = get_attribute(input_model, "nodes_hitrates").floats
    # # nodes_missing_value_tracks_true
    input_nodes_missing_value_tracks_true = get_attribute(
        input_model, "nodes_missing_value_tracks_true"
    ).ints
    # # nodes_modes：节点类型，LEAF表示叶子节点，BRANCH_LEQ表示非叶子节点
    input_nodes_modes = get_attribute(input_model, "nodes_modes").strings
    # # nodes_nodeids
    input_nodes_nodeids = get_attribute(input_model, "nodes_nodeids").ints
    # # nodes_treeids
    input_nodes_treeids = get_attribute(input_model, "nodes_treeids").ints
    # # nodes_truenodeids: 左侧分支
    input_nodes_truenodeids = get_attribute(input_model, "nodes_truenodeids").ints
    # # nodes_values: 阈值，叶子节点的值为0
    input_nodes_values = get_attribute(input_model, "nodes_values").floats
    # # post_transform
    input_post_transform = get_attribute(input_model, "post_transform").s

    n_trees = len(set(input_class_treeids))

    # output model attributes
    # # n_targets
    n_targets = 1

    # # nodes_falsenodeids: 右侧分支
    nodes_falsenodeids = input_nodes_falsenodeids

    # # nodes_featureids: 特征id
    nodes_featureids = input_nodes_featureids

    # # nodes_hitrates
    nodes_hitrates = input_nodes_hitrates

    # # nodes_missing_value_tracks_true
    nodes_missing_value_tracks_true = input_nodes_missing_value_tracks_true

    # # nodes_modes：节点类型，LEAF表示叶子节点，BRANCH_LEQ表示非叶子节点
    nodes_modes = input_nodes_modes

    # # nodes_nodeids
    nodes_nodeids = input_nodes_nodeids

    # # nodes_treeids
    nodes_treeids = input_nodes_treeids

    # # nodes_truenodeids: 左侧分支
    nodes_truenodeids = input_nodes_truenodeids

    # # nodes_values: 阈值，叶子节点的值为0
    nodes_values = input_nodes_values

    # # post_transform
    post_transform = input_post_transform

    stride = len(input_classlabels_int64s)
    if stride == 2:
        stride = 1

    n_leaf = len(input_class_weights) // stride

    # # target_ids
    target_ids = []
    for i in range(n_leaf):
        target_ids.append(input_class_ids[i * stride])

    # # target_nodeids: 叶子节点的id
    target_nodeids = []
    for i in range(n_leaf):
        target_nodeids.append(input_class_nodeids[i * stride])

    # # target_treeids
    target_treeids = []
    for i in range(n_leaf):
        target_treeids.append(input_class_treeids[i * stride])

    # # target_weights: 叶子节点的权重，即预测值
    target_weights = []
    if stride == 1:
        # binary mode: only store positive class weight
        target_weights = [
            1.0 if w > 0.5 / n_trees else 0.0 for w in input_class_weights
        ]
    else:
        for i in range(n_leaf):
            targets = input_class_weights[i * stride : (i + 1) * stride]
            target_weights.append(float(np.argmax(targets)))

    # node
    node = helper.make_node(
        op_type="TreeEnsembleRegressor",
        inputs=[input_model.graph.input[0].name],
        outputs=[input_model.graph.output[0].name],
        name="TreeEnsembleRegressor",
        domain="ai.onnx.ml",
        # attributes
        n_targets=n_targets,
        nodes_falsenodeids=nodes_falsenodeids,
        nodes_featureids=nodes_featureids,
        nodes_hitrates=nodes_hitrates,
        nodes_missing_value_tracks_true=nodes_missing_value_tracks_true,
        nodes_modes=nodes_modes,
        nodes_nodeids=nodes_nodeids,
        nodes_treeids=nodes_treeids,
        nodes_truenodeids=nodes_truenodeids,
        nodes_values=nodes_values,
        post_transform=post_transform,
        target_ids=target_ids,
        target_nodeids=target_nodeids,
        target_treeids=target_treeids,
        target_weights=target_weights,
    )

    # graph
    output = helper.make_tensor_value_info(
        name=input_model.graph.output[0].name,
        elem_type=onnx.TensorProto.FLOAT,
        shape=[None, 1],
    )
    graph = helper.make_graph(
        nodes=[node],
        name=input_model.graph.name,
        initializer=[],
        inputs=input_model.graph.input,
        outputs=[output],
    )

    # model
    output_model = helper.make_model(
        graph=graph,
        opset_imports=input_model.opset_import,
    )
    output_model.ir_version = input_model.ir_version

    onnx.checker.check_model(output_model)

    return output_model


def calculate_leafs(model) -> int:
    node_modes = list(get_attribute(model, "nodes_modes").strings)
    return node_modes.count(b"LEAF")


def calculate_nodes(model) -> int:
    return len(get_attribute(model, "nodes_modes").strings)


def calculate_trees(model) -> int:
    return len(set(get_attribute(model, "target_treeids").ints))


def calculate_depths(root: Node) -> float:
    if not root:
        return 0.0
    queue = deque([(root, 0)])
    total_cost = 0.0
    leafs = 0
    while queue:
        node, depth = queue.popleft()
        if node.left is None and node.right is None:
            total_cost += depth
            leafs += 1
        if node.left:
            queue.append((node.left, depth + 1))
        if node.right:
            queue.append((node.right, depth + 1))
    return total_cost / leafs


def calculate_cost(root: Node, inputs) -> float:
    if not root:
        return 0.0
    cost = 0
    for input in inputs:
        current = root
        while current.mode != b"LEAF":
            feature_value = input[current.feature_id]
            if feature_value <= current.value:
                current = current.left
            else:
                current = current.right
            cost += 1
    return cost


def cost_count(model_path: str, data_path: str, inputs) -> Tuple[int, int, int, float]:
    orginal_model = onnx.load(model_path)
    model = (
        clf2reg(orginal_model)
        if has_attribute(orginal_model, "class_treeids")
        else orginal_model
    )
    nodes = calculate_nodes(model)
    trees = calculate_trees(model)
    leafs = calculate_leafs(model)
    roots = model2trees(model, None)

    total_cost = 0
    with ProcessPoolExecutor(max_workers=nprocess) as executor:
        futures = [executor.submit(calculate_cost, root, inputs) for root in roots]
        for future in as_completed(futures):
            total_cost += future.result()
    return trees, leafs, nodes, total_cost


def run(path, model_type):  # model_type: rf/dt
    df = pd.read_csv(path, dtype={"workload": str, "model": str, "predicate": str})
    for row in df.itertuples():
        # if row.workload == "medical_charges":
        #     continue
        # if row.workload == "tpcai-uc08":
        #     continue
        # if row.workload == "bike_sharing_demand":
        #     continue
        # if row.workload == "walmart_sales":
        #     continue
        model_path = f"/volumn/Retree_exp/workloads/{row.workload}/model/"
        print(row.workload)
        data_path = f"/volumn/Retree_exp/workloads/{row.workload}/data-extension/1G/{row.workload}.csv"
        count(data_path)
        # model_path1 = "/volumn/Retree_exp/workloads/tpcai-uc08/model/tpcai-uc08_d10_l89_n177_20250707071323.onnx"
        # model_names = None
        # if (
        #     row.workload == "flights"
        #     or row.workload == "tpcai-uc08"
        #     or row.workload == "wine_quality"
        # ):
        #     model_names = [
        #         f"{row.model}",
        #         f"{row.model}_reg_pruned{str(row.predicate)}",
        #         f"{row.model}_reg_pruned{str(row.predicate)}_merged",
        #     ]
        # else:
        #     model_names = [
        #         f"{row.model}",
        #         f"{row.model}_pruned{str(row.predicate)}",
        #         f"{row.model}_pruned{str(row.predicate)}_merged",
        #     ]
        # model_names = [
            # f"{row.model}",
            # f"{row.model}_reg_pruned{str(row.predicate)}",
            # f"{row.model}_reg_pruned{str(row.predicate)}_merged",
        # ]
        # inputs = generate_inputs(model_path1, data_path)
        # for model_name in model_names:
        #     path = model_path + model_name + ".onnx"
        #     print(path)
            # trees, leafs, nodes, total_cost = cost_count(path, data_path, inputs)
            # print(row.workload, trees, leafs, nodes, total_cost)


if __name__ == "__main__":
    run("workload_models-dt.csv", "dt")
    run("workload_models-rf.csv", "rf")
