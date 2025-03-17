import argparse
import datetime
import numpy as np
import onnxoptimizer
import pandas as pd
import onnx
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from onnxconverter_common import FloatTensorType, Int64TensorType, StringTensorType

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import plot_feature_importances, value_distribution

""" 
walmart:
多表分类任务
              precision    recall  f1-score   support

           1       0.00      0.00      0.00        12
           2       0.66      0.65      0.66       661
           3       0.55      0.63      0.59       968
           4       0.95      0.02      0.05       747
           5       0.41      0.83      0.54       943
           6       0.90      0.54      0.68       836
           7       1.00      0.06      0.12        49

    accuracy                           0.55      4216
   macro avg       0.64      0.39      0.38      4216
weighted avg       0.68      0.55      0.50      4216

python train_walmart_rf.py -tn 100 -td 10
"""

parser = argparse.ArgumentParser()
parser.add_argument("--tree_num", "-tn", type=int, default=100)
parser.add_argument("--tree_depth", "-td", type=int, default=10)
args = parser.parse_args()

data_name = "walmart"
tree_num = args.tree_num
tree_depth = args.tree_depth
label = "weekly_sales"

path1 = "S_sales.csv"
path2 = "R1_indicators.csv"
path3 = "R2_stores.csv"

# 读取csv表
S_sales = pd.read_csv(path1)
R1_indicators = pd.read_csv(path2)
R2_stores = pd.read_csv(path3)
# 连接3张表
S_sales["purchaseid"] = S_sales["purchaseid"].str.replace("'", "")
S_sales["weekly_sales"] = S_sales["weekly_sales"].str.replace("'", "").astype("int")
data = pd.merge(pd.merge(S_sales, R1_indicators, how="inner"), R2_stores, how="inner")
data["store"] = data["store"].str.replace("'", "").astype("int")
data["dept"] = data["dept"].str.replace("'", "").astype("int")
data["type"] = data["type"].str.replace("'", "").astype("int")

data.dropna(inplace=True)
data.head(2048).to_csv("/volumn/Retree_exp/workloads/walmart/walmart-2048.csv", index=False)
data.to_csv("/volumn/Retree_exp/workloads/walmart/walmart.csv", index=False)

# 2 categorical, 10 numerical
numerical = [
    # "temperature_avg",
    # "temperature_stdev",
    "fuel_price_avg",
    "fuel_price_stdev",
    # "cpi_avg",
    # "cpi_stdev",
    # "unemployment_avg",
    # "unemployment_stdev",
    "holidayfreq",
    "size",
    "type",
    "store",
    "dept"
]
# categorical = [
#     # "dept",
#     "type"
# ]
input_columns = numerical
# input_columns = numerical + categorical


preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical),
        # (
        #     "cat",
        #     OneHotEncoder(handle_unknown="ignore"),
        #     categorical,
        # ),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "Classifier",
            RandomForestClassifier(
                n_estimators=tree_num, max_depth=tree_depth, n_jobs=50
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
print(f"{classification_report(y_test, y_pred)}")

# define path
model = pipeline.named_steps["Classifier"]
depth = [model.estimators_[i].get_depth() for i in range(tree_num)]
leaves = [model.estimators_[i].get_n_leaves() for i in range(tree_num)]
node_count = [model.estimators_[i].tree_.node_count for i in range(tree_num)]
now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

model_name = f"{data_name}_t{tree_num}_d{sum(depth) // tree_num}_l{sum(leaves) // tree_num}_n{sum(node_count) //tree_num}_{now}"
onnx_path = f"model/{model_name}.onnx"

# save model pred distribution
pred = pipeline.predict(X)
value_distribution(pred, model_name)
plot_feature_importances(model, X.shape[1], model_name)

# convert and save model
type_map = {
    # "int64": Int64TensorType([None, 1]),
    "int64": FloatTensorType([None, 1]),
    "float32": FloatTensorType([None, 1]),
    "float64": FloatTensorType([None, 1]),
    "object": StringTensorType([None, 1]),
}
init_types = [(elem, type_map[X[elem].dtype.name]) for elem in input_columns]
model_onnx = convert_sklearn(pipeline, initial_types=init_types)

# optimize model
optimized_model = onnxoptimizer.optimize(model_onnx)
onnx.save_model(optimized_model, onnx_path)
