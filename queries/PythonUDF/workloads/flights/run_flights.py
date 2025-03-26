import argparse
import time
import onnxruntime as ort
import duckdb
import numpy as np
from duckdb.typing import BIGINT, FLOAT, VARCHAR
import re

times = 7

scale = '1G'
predicate = '0'

thread_duckdb = 1
thread_ort = 1

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="flights_t100_d10_l421_n841_20250321151145")
args = parser.parse_args()

model_name = args.model
model_path = f"/volumn/Retree_exp/workloads/flights/model/{model_name}.onnx"
pattern = 't100'
model_type = 'dt'
if re.search(pattern, model_name):
    model_type = 'rf'
    thread_duckdb = 1

with open("load_data.sql", "r") as file:
    load_data = file.read()
with open("query.sql", "r") as file:
    query = file.read()

load_data = load_data.replace('?', scale)
# only use one predicate
query = query.replace('?', predicate)

op = ort.SessionOptions()
op.intra_op_num_threads = thread_ort
session = ort.InferenceSession(
    model_path, sess_options=op, providers=["CPUExecutionProvider"]
)

def predict(slatitude, slongitude, dlatitude, dlongitude, active, sdst, ddst):    
    type_map = {
        "bool": np.int64,
        "int32": np.int64,
        "int64": np.int64,
        "float32": np.float32,
        "float64": np.float32,
        "object": str,
    }
    columns = [input.name for input in session.get_inputs()]

    def predict_wrap(*args):
        infer_batch = {
            elem: np.array(args[i])
            .astype(type_map[args[i].to_numpy().dtype.name])
            .reshape((-1, 1))
            for i, elem in enumerate(columns)
        }
        outputs = session.run([session.get_outputs()[0].name], infer_batch)
        return outputs[0]

    return predict_wrap(slatitude, slongitude, dlatitude, dlongitude, active, sdst, ddst)

duckdb.create_function(
    "predict",
    predict,
    [FLOAT, FLOAT, FLOAT, FLOAT, VARCHAR, VARCHAR, VARCHAR],
    BIGINT,
    type="arrow",
)

duckdb.sql(f"SET threads={thread_duckdb};")
duckdb.sql(load_data)

timer = []
for i in range(times):
    start = time.time()
    duckdb.sql(query)
    end = time.time()
    timer.append(end - start)

print(timer)

timer.remove(min(timer))
timer.remove(max(timer))
average = sum(timer) / len(timer)

with open(f"output.csv", "a", encoding="utf-8") as f:
    # Workload,Model,Model_Type,Predicate,Scale,Thread,Optimized,Average Time(ms)
    f.write(f"flights,{model_name},{model_type},{predicate},{scale},{thread_duckdb},0,{average}\n")
