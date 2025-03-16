import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
import onnx
import datetime
from skl2onnx import convert_sklearn
from onnxconverter_common import FloatTensorType, Int64TensorType, StringTensorType
import onnxoptimizer
import argparse

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import plot_value_distribution, percentile_values

""" 
hospital:
单表回归任务

mse: 0.9106147416794484
rmsle: 0.2137308344851731
r2: 0.8315769528499954

mse: 1.2762458800283785
rmsle: 0.2689760103075762
r2: 0.7639515261628782

python train_hospital_rf.py -tn 100 -td 10
"""

parser = argparse.ArgumentParser()
parser.add_argument("--tree_num", "-tn", type=int, default=100)
parser.add_argument("--tree_depth", "-td", type=int, default=10)
args = parser.parse_args()

data_name = "hospital"
tree_num = args.tree_num
tree_depth = args.tree_depth
label = "lengthofstay"

# load data
data_path = f"{data_name}.csv"
data = pd.read_csv(data_path)
# data.head(2048).to_csv(f"{data_name}-2048.csv", index=False)

# choose feature: 4 numerical, 4 categorical
numerical = [
    "hematocrit",
    "neutrophils",
    # "sodium",
    "glucose",
    "bloodureanitro",
    # "creatinine",
    "bmi",
    "pulse",
    "respiration",
    # "secondarydiagnosisnonicd9",
]
categorical = [
    "rcount",
    # "gender",
    # "dialysisrenalendstage",
    "asthma",
    # "irondef",
    "pneum",
    # "substancedependence",
    # "psychologicaldisordermajor",
    # "depress",
    # "psychother",
    # "fibrosisandother",
    # "malnutrition",
    "hemo",
]
input_columns = numerical + categorical

# define pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "Regressor",
            RandomForestRegressor(
                n_estimators=tree_num, max_depth=tree_depth, n_jobs=48
            ),
        ),
    ]
)

# define data
X = data.loc[:, input_columns]
y = np.array(data.loc[:, label].values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=42
)

# train
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f"mse: {mean_squared_error(y_test, y_pred)}")
print(f"rmsle: {np.sqrt(mean_squared_log_error(y_test, y_pred))}")
print(f"r2: {r2_score(y_test, y_pred)}")

# define path
model = pipeline.named_steps["Regressor"]
depth = [model.estimators_[i].get_depth() for i in range(tree_num)]
leaves = [model.estimators_[i].get_n_leaves() for i in range(tree_num)]
node_count = [model.estimators_[i].tree_.node_count for i in range(tree_num)]
now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

model_name = f"{data_name}_t{tree_num}_d{sum(depth) // tree_num}_l{sum(leaves) // tree_num}_n{sum(node_count) //tree_num}_{now}"
onnx_path = f"model/{model_name}.onnx"

# save model pred distribution
pred = pipeline.predict(X)
plot_value_distribution(pred, model_name)
percentile_values(pred, model_name)

# convert and save model
# type_map = {
#     "int64": Int64TensorType([None, 1]),
#     "float32": FloatTensorType([None, 1]),
#     "float64": FloatTensorType([None, 1]),
#     "object": StringTensorType([None, 1]),
# }
# init_types = [(elem, type_map[X[elem].dtype.name]) for elem in input_columns]

init_types = [
    ("hematocrit", FloatTensorType(shape=[None, 1])),
    ("neutrophils", FloatTensorType(shape=[None, 1])),
    ("glucose", FloatTensorType(shape=[None, 1])),
    ("bloodureanitro", FloatTensorType(shape=[None, 1])),
    ("bmi", FloatTensorType(shape=[None, 1])),
    ("pulse", FloatTensorType(shape=[None, 1])),
    ("respiration", FloatTensorType(shape=[None, 1])),
    ("rcount", StringTensorType(shape=[None, 1])),
    ("asthma", Int64TensorType(shape=[None, 1])),
    ("pneum", Int64TensorType(shape=[None, 1])),
    ("hemo", Int64TensorType(shape=[None, 1])),
]

model_onnx = convert_sklearn(pipeline, initial_types=init_types)

# optimize model
optimized_model = onnxoptimizer.optimize(model_onnx)
onnx.save_model(optimized_model, onnx_path)
