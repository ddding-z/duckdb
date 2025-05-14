import onnxruntime as ort
import time
import numpy as np
import pandas as pd

batch_size = 2048

# data_path = '/volumn/Retree_exp/workloads/flights/data/flights_1G.csv'
# data = pd.read_csv(data_path)
# features = ['slatitude', 'slongitude', 'dlatitude', 'dlongitude', 'active', 'sdst', 'ddst']
# X = data[features].copy()
# X['batch_id'] = np.arange(len(X)) // batch_size

data_path = '/volumn/Retree_exp/workloads/tpcai-uc08/data/tpcai-uc08-train.csv'
label = "trip_type"

data = pd.read_csv(data_path)
X = data.drop(label, axis=1)
X['batch_id'] = np.arange(len(X)) // batch_size

options = ort.SessionOptions()
options.intra_op_num_threads = 1
# options.inter_op_num_threads = 1
type_map = {
    "bool": np.int64,
    "int32": np.int64,
    "int64": np.int64,
    "float32": np.float32,
    "float64": np.float32,
    "object": str,
}

# cats = 37
# res = []
# for i in range(cats):
#     model_path = f'/volumn/Retree_exp/workloads/tpcai-uc08/model/tpcai-uc08_d10_l89_n177_20250321150856_reg_pruned{i}.000000.onnx'
#     session = ort.InferenceSession(model_path, sess_options=options)
#     columns = [input.name for input in session.get_inputs()]
#     ltotal = []
#     for ii in range(10):
#         for batch_id, batch_df in X.groupby('batch_id'):
#             infer_batch = {
#                 elem: batch_df.iloc[:, j].to_numpy()
#                 .astype(type_map[batch_df.iloc[:, j].to_numpy().dtype.name])
#                 .reshape((-1, 1))
#                 for j, elem in enumerate(columns)
#             }
#             start = time.perf_counter()
#             outputs = session.run([session.get_outputs()[0].name], infer_batch)
#             end = time.perf_counter()
#             ltotal.append(end-start)
#     ltotal.remove(min(ltotal))
#     ltotal.remove(max(ltotal))
#     print(f"{i},{sum(ltotal)/8}")
#     res.append(f"{i},{sum(ltotal)/8}")
    
# with open('tpcai.csv','a') as f:
#     f.write("\n".join(res))

# model_path = f'/volumn/Retree_exp/workloads/flights/model/flights_t100_d10_l421_n841_20250321151145_reg.onnx'
model_path = '/volumn/Retree_exp/workloads/tpcai-uc08/model/tpcai-uc08_d10_l89_n177_20250321150856{0}.onnx'
model_paths = [model_path.format(""), model_path.format("_reg")]
for i, model_path in enumerate(model_paths):
    session = ort.InferenceSession(model_path, sess_options=options)
    columns = [input.name for input in session.get_inputs()]
    ltotal = []
    for ii in range(10):
        for batch_id, batch_df in X.groupby('batch_id'):
            infer_batch = {
                elem: batch_df.iloc[:, j].to_numpy()
                .astype(type_map[batch_df.iloc[:, j].to_numpy().dtype.name])
                .reshape((-1, 1))
                for j, elem in enumerate(columns)
            }
            start = time.perf_counter()
            outputs = session.run([session.get_outputs()[0].name], infer_batch)
            end = time.perf_counter()
            ltotal.append(end-start)
    ltotal.remove(min(ltotal))
    ltotal.remove(max(ltotal))
    print(f"{i},{sum(ltotal)/8}")


     






