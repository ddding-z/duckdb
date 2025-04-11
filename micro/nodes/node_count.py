import pandas as pd
import onnx


def get_attribute(onnx_model, attr_name):
    i = 0
    while 1:
        attributes = onnx_model.graph.node[i].attribute
        for attr in attributes:
            if attr.name == attr_name:
                return attr
        i += 1

# w/o, prune, merge(hyperplane), merge(node), merge(one boundary) 
def nodes_count(path, model_type):
    df = pd.read_csv(path, dtype={"workload": str, "model": str, "predicate": str})
    for row in df.itertuples():
        model_path = f"/volumn/Retree_exp/workloads/{row.workload}/model"
        model_names = None
        if (
            row.workload == "flights"
            or row.workload == "tpcai-uc08"
            or row.workload == "wine_quality"
        ):
            model_names = [
                f"{row.model}",
                f"{row.model}_reg_pruned{str(row.predicate)}",
                f"{row.model}_reg_pruned{str(row.predicate)}_merged",
                f"{row.model}_reg_pruned{str(row.predicate)}_naive_merged",
                f"{row.model}_reg_pruned{str(row.predicate)}_merged_one_boundary",
            ]
        else:
            model_names = [
                f"{row.model}",
                f"{row.model}_pruned{str(row.predicate)}",
                f"{row.model}_pruned{str(row.predicate)}_merged",
                f"{row.model}_pruned{str(row.predicate)}_naive_merged",
                f"{row.model}_pruned{str(row.predicate)}_merged_one_boundary",
            ]
        nodes_nums = [row.workload, model_type]
        for model_name in model_names:
            model = onnx.load(f"{model_path}/{model_name}.onnx")
            nodes_nums.append(str(len(get_attribute(model, "nodes_modes").strings)))
        with open(f"nodes.csv", "a", encoding="utf-8") as f:
            f.write(",".join(nodes_nums))
            f.write("\n")


def main():
    nodes_count("workload_models-dt.csv", "dt")
    nodes_count("workload_models-rf.csv", "rf")


if __name__ == "__main__":
    main()
