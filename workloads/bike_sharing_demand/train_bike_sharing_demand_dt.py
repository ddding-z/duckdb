from collections import Counter
import numpy as np
import onnxoptimizer
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
import onnx
import datetime
from skl2onnx import convert_sklearn
from onnxconverter_common import FloatTensorType, Int64TensorType, StringTensorType
import argparse

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import plot_value_distribution, percentile_values

""" 
bike sharing demand:
单表回归任务
mse: 5598.800180021919
rmsle: 0.43906592184116683
r2: 0.8629115975271225

python train_bike_sharing_demand_dt.py -td 10
"""

parser = argparse.ArgumentParser()
parser.add_argument("--tree_depth", "-td", type=int, default=10)
args = parser.parse_args()

data_name = "bike_sharing_demand"
tree_depth = args.tree_depth
label = "count"

# load data
data_path = f"{data_name}.csv"
data = pd.read_csv(data_path)
data["hour"] = data.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")

# choose feature: 4 numerical, 4 categorical
numerical = ["hour", "atemp", "humidity", "windspeed"]
categorical = ["season", "holiday", "workingday", "weather"]
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
        ("Regressor", DecisionTreeRegressor(max_depth=tree_depth)),
    ]
)

# define data
X = data.drop(columns=[label])
y = data[label]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=42
)

# train
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f"mse: {mean_squared_error(y_test, y_pred)}")
print(f"rmsle: {np.sqrt(mean_squared_log_error(y_test, y_pred))}")
print(f"r2_score: {r2_score(y_test, y_pred)}")

# define path
model = pipeline.named_steps["Regressor"]
depth = model.get_depth()
leaves = model.get_n_leaves()
node_count = model.tree_.node_count
now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

model_name = f"{data_name}_d{depth}_l{leaves}_n{node_count}_{now}"
onnx_path = f"model/{model_name}.onnx"
out_path = f"model/{model_name}.txt"

# save model pred distribution
pred = pipeline.predict(X)
plot_value_distribution(pred, model_name)
percentile_values(pred, model_name)

# convert and save model
type_map = {
    "int64": Int64TensorType([None, 1]),
    "float32": FloatTensorType([None, 1]),
    "float64": FloatTensorType([None, 1]),
    "object": StringTensorType([None, 1]),
}
init_types = [(elem, type_map[X[elem].dtype.name]) for elem in input_columns]
model_onnx = convert_sklearn(pipeline, initial_types=init_types)

# optimize model
optimized_model = onnxoptimizer.optimize(model_onnx)
onnx.save_model(optimized_model, onnx_path)