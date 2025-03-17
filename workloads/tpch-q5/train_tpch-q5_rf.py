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

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import value_distribution

""" 
tpch-q5:
多表分类任务
              precision    recall  f1-score   support

           0       0.80      0.96      0.87       513
           1       0.60      0.19      0.29       153

    accuracy                           0.79       666
   macro avg       0.70      0.58      0.58       666
weighted avg       0.75      0.79      0.74       666

python train_tpch-q5_rf.py -tn 100 -td 10
"""

parser = argparse.ArgumentParser()
parser.add_argument("--tree_num", "-tn", type=int, default=100)
parser.add_argument("--tree_depth", "-td", type=int, default=10)
args = parser.parse_args()

data_name = "tpch-q5"
tree_num = args.tree_num
tree_depth = args.tree_depth
label = "o_orderpriority"

path1 = "customer.tbl"
path2 = "orders.tbl"

customer = pd.read_table(path1, names=['c_custkey', 'c_name', 'c_address', 'c_nationkey', 'c_phone', 'c_acctbal', 'c_mktsegment', 
                                       'c_comment'], sep='|', index_col=False, nrows=500000)
orders = pd.read_table(path2, names=['o_orderkey', 'o_custkey', 'o_orderstatus', 'o_totalprice', 'o_orderdate', 'o_orderpriority', 
                                     'o_clerk', 'o_shippriority', 'o_comment'], sep='|', index_col=False)

data = pd.merge(customer, orders, how = 'inner', left_on = 'c_custkey', right_on = 'o_custkey')

select_cols = ['o_shippriority', 'c_acctbal',  'c_mktsegment', 'o_totalprice', 'o_orderpriority']
data = data[select_cols]

data.head(2048).to_csv('/volumn/Retree_exp/workloads/tpch-q5-2048.csv', index=False)

numerical = ['c_acctbal', 'o_totalprice']
categorical = ['c_mktsegment','o_shippriority']
input_columns = numerical + categorical

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical),
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('Classifier', RandomForestClassifier(n_estimators = tree_num, max_depth = tree_depth, n_jobs = 48))
])

X = data.loc[:, input_columns]
# '1-URGENT'->0, '2-HIGH'->1, '3-MEDIUM'->2, '4-NOT SPECIFIED'->3, '5-LOW'->4 
label = np.array(data.loc[:, label])
le = LabelEncoder()
y = le.fit_transform(label)

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



