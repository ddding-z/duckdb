import argparse
import datetime
import numpy as np
import onnxoptimizer
import pandas as pd
import onnx
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from onnxconverter_common import FloatTensorType, Int64TensorType, StringTensorType
from imblearn.over_sampling import SMOTE, ADASYN

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import value_distribution, plot_feature_importances

""" 
nsl-kdd: 多分类
python train_nsl-kdd_rf.py -tn 100 -td 10
"""

parser = argparse.ArgumentParser()
parser.add_argument("--tree_num", "-tn", type=int, default=100)
parser.add_argument("--tree_depth", "-td", type=int, default=10)
args = parser.parse_args()

data_name = "nsl-kdd"
tree_num = args.tree_num
tree_depth = args.tree_depth
label = "label"

path1 = "data/nsl-kdd-train.csv"
path2 = "data/nsl-kdd-test.csv"

# load data
data_train = pd.read_csv(path1)
data_test = pd.read_csv(path2)

labelencoder = LabelEncoder()
data_train[label] = labelencoder.fit_transform(data_train[label])
data_test[label] = labelencoder.fit_transform(data_test[label])

for label, number in zip(labelencoder.classes_, range(len(labelencoder.classes_))):
    print(f"{label}: {number}")
    
X_train = data_train.drop('label', axis=1)
y_train = data_train['label']

X_test = data_test.drop('label', axis=1)
y_test = data_test['label']

# choose feature:
categorical_mask = (X_train.dtypes == object)
categorical_columns = X_train.columns[categorical_mask].tolist()

numerical_mask = (X_train.dtypes != object)
numerical_columns = X_train.columns[numerical_mask].tolist()

input_columns = numerical_columns + categorical_columns

# def label_encoder(data):
#     labelencoder = LabelEncoder()
#     for col in data.columns:
#         data.loc[:,col] = labelencoder.fit_transform(data[col])
#     return data

# X_train[categorical_columns] = label_encoder(X_train[categorical_columns])
# X_test[categorical_columns] = label_encoder(X_test[categorical_columns])

# oversample = ADASYN()
# X_train, y_train = oversample.fit_resample(X_train, y_train)

# define pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
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
pred = pipeline.predict(X_test)
value_distribution(pred, model_name)

# convert and save model
type_map = {
    "int64": Int64TensorType([None, 1]),
    "float32": FloatTensorType([None, 1]),
    "float64": FloatTensorType([None, 1]),
    "object": StringTensorType([None, 1]),
}
init_types = [(elem, type_map[X_train[elem].dtype.name]) for elem in input_columns]
model_onnx = convert_sklearn(pipeline, initial_types=init_types)

# optimize model
optimized_model = onnxoptimizer.optimize(model_onnx)
onnx.save_model(optimized_model, onnx_path)

with open(f"/volumn/Retree_exp/queries/Retree/workloads/workload_models.csv", "a", encoding="utf-8") as f:
    f.write(f"{data_name},{model_name}\n")
