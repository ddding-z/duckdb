import argparse
import time
import onnxruntime as ort
import duckdb
import numpy as np
from duckdb.typing import BIGINT, FLOAT
import re

times = 7
thread_duckdb = 1
thread_ort = 1

parser = argparse.ArgumentParser()
parser.add_argument(
    "--workload",
    "-w",
    type=str,
    default="walmart_sales",
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="tpch-q9_t100_d10_l1024_n2047_20250321151057",
)
parser.add_argument("--scale", "-s", type=str, default="1G")
args = parser.parse_args()

workload = args.workload
model_name = args.model
scale = args.scale

model_path = f"/volumn/Retree_exp/workloads/{workload}/model/{model_name}.onnx"
pattern = "t100"
model_type = "dt"
if re.search(pattern, model_name):
    model_type = "rf"
    thread_duckdb = 4

op = ort.SessionOptions()
op.intra_op_num_threads = thread_ort
session = ort.InferenceSession(
    model_path, sess_options=op, providers=["CPUExecutionProvider"]
)

type_map = {
    "bool": np.int64,
    "int32": np.int64,
    "int64": np.int64,
    "float32": np.float32,
    "float64": np.float32,
    "object": str,
}


def predict(
    Store,
    Dept,
    IsHoliday,
    Fuel_Price,
    MarkDown1,
    MarkDown2,
    MarkDown3,
    Type,
    Size,
    Super_Bowl,
    Labor_Day,
    Thanksgiving,
    Christmas,
    week,
    month,
    year
):
    columns = [input.name for input in session.get_inputs()]

    def predict_wrap(*args):
        infer_batch = {
            elem: np.array(args[i])
            .astype(type_map[args[i].to_numpy().dtype.name])
            .reshape((-1, 1))
            for i, elem in enumerate(columns)
        }
        outputs = session.run([session.get_outputs()[0].name], infer_batch)
        return outputs[0].reshape(-1)

    return predict_wrap(
        Store,
        Dept,
        IsHoliday,
        Fuel_Price,
        MarkDown1,
        MarkDown2,
        MarkDown3,
        Type,
        Size,
        Super_Bowl,
        Labor_Day,
        Thanksgiving,
        Christmas,
        week,
        month,
        year
    )


duckdb.create_function(
    "predict",
    predict,
    [FLOAT] * 16,
    FLOAT,
    type="arrow",
)

load_data = None
query = None
predicates = None

with open("load_data.sql", "r") as file:
    load_data = file.read()
with open("query.sql", "r") as file:
    query = file.read()
with open("predicates.txt", "r") as file:
    predicates = [str(line.strip()) for line in file if line.strip() != ""]

load_data = load_data.replace("?", scale)
duckdb.sql(f"SET threads={thread_duckdb};")
duckdb.sql(load_data)

for predicate in predicates:
    pquery = query.replace("?", predicate)
    timer = []
    for i in range(times):
        start = time.time()
        duckdb.sql(pquery)
        end = time.time()
        timer.append(end - start)
    timer.remove(min(timer))
    timer.remove(max(timer))
    average = sum(timer) / len(timer)
    print(
        f"{workload},{model_name},{model_type},{predicate},{scale},{thread_duckdb},0,{average}"
    )
    with open(f"output.csv", "a", encoding="utf-8") as f:
        f.write(
            f"{workload},{model_name},{model_type},{predicate},{scale},{thread_duckdb},0,{average}\n"
        )
