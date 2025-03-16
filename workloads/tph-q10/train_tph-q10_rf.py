# python train_Q10_rf.py -tn 100 -td 10

import argparse
import datetime
import statistics
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import onnx
from onnxconverter_common import FloatTensorType, Int64TensorType, StringTensorType
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from skl2onnx import convert_sklearn

def get_attribute(onnx_model, attr_name):
    i = 0
    while(1):
        attributes = onnx_model.graph.node[i].attribute
        for attr in attributes:
            if attr.name == attr_name:
                return attr
        i += 1
        
parser = argparse.ArgumentParser()
parser.add_argument('--tree_num', '-tn', type=int, default=100)
parser.add_argument('--tree_depth', '-td', type=int, default=10)
args = parser.parse_args()

data_name = 'tpch-Q10'
tree_num = args.tree_num
tree_depth = args.tree_depth

# 读取数据
path1 = "/volumn/Retree_exp/data/tpch/lineitem.tbl"
path2 = "/volumn/Retree_exp/data/tpch/customer.tbl"
path3 = "/volumn/Retree_exp/data/tpch/orders.tbl"
path4 = "/volumn/Retree_exp/data/tpch/nation.tbl"

lineitem = pd.read_table(path1, names=['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity', 'l_extendedprice', 
                                       'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus', 'l_shipdate', 'l_commitdate', 
                                       'l_receiptdate', 'l_shipinstruct', 'l_shipmode', 'l_comment'], sep='|', index_col=False, nrows=1000000) # nrows=1000000
#print(lineitem.info())
customer = pd.read_table(path2, names=['c_custkey', 'c_name', 'c_address', 'c_nationkey', 'c_phone', 'c_acctbal', 'c_mktsegment', 
                                       'c_comment'], sep='|', index_col=False, nrows=500000) # nrows=500000
#print(customer.info())
orders = pd.read_table(path3, names=['o_orderkey', 'o_custkey', 'o_orderstatus', 'o_totalprice', 'o_orderdate', 'o_orderpriority', 
                                     'o_clerk', 'o_shippriority', 'o_comment'], sep='|', index_col=False)
#print(orders.info())
nation = pd.read_table(path4, names=['n_nationkey', 'n_name', 'n_regionkey', 'n_comment'], sep='|', index_col=False)
#print(nation.info())

# join
#data = pd.merge(customer, orders, how = 'inner', left_on = 'c_custkey', right_on = 'o_custkey')
data = pd.merge(pd.merge(pd.merge(customer, orders, how = 'inner', left_on = 'c_custkey', right_on = 'o_custkey'), 
                         lineitem, how = 'inner', left_on = 'o_orderkey', right_on = 'l_orderkey'), 
                nation, how = 'inner', left_on = 'c_nationkey', right_on = 'n_nationkey')
#print(data.info())
select_cols = ['c_custkey', 'c_acctbal', 'o_orderkey', 'o_orderstatus', 'o_totalprice', 'o_orderpriority', 'o_clerk', 
               'l_linenumber', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus', 'l_shipinstruct', 'l_shipmode',
               'n_nationkey', 'n_regionkey']
data = data[select_cols]
# data.head(2048).to_csv('/volumn/Retree_exp/data/tpch-q10-2048.csv', index=False)
# print(data.info())
# 获取特征和数据预处理

numerical = ['c_acctbal', 'o_totalprice', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax'] # 
categorical = ['o_orderstatus', 'o_orderpriority', 'l_linestatus', 'l_shipinstruct', 'l_shipmode', 'n_nationkey', 'n_regionkey'] # 
input_columns = numerical + categorical

X = data.loc[:, input_columns]

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

# 'A' - > 0, 'N' -> 1, 'R' -> 2
label = np.array(data.loc[:, 'l_returnflag'])
le = LabelEncoder()
y = le.fit_transform(label) 
print(y)
print(le.inverse_transform(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=42)
print('训练集维度:{}\n测试集维度:{}'.format(X_train.shape, X_test.shape))

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f'classification_report: {classification_report(y_test, y_pred)}')

model = pipeline.named_steps['Classifier']
depth = [model.estimators_[i].get_depth() for i in range(tree_num)]
leaves = [model.estimators_[i].get_n_leaves() for i in range(tree_num)]
node_count =[model.estimators_[i].tree_.node_count for i in range(tree_num)]
print('depth:', depth)
print('leaves:', leaves)
print('nodes:', node_count)


now = datetime.datetime.now()
now = now.strftime('%Y%m%d%H%M%S')
model_name = f'{data_name}_t{tree_num}_d{sum(depth) // tree_num}_l{sum(leaves) // tree_num}_n{sum(node_count) //tree_num}_{now}'
onnx_path = f'/volumn/Retree_exp/model/rf/{model_name}.onnx'

# joblib_path = f'/volumn/Retree_exp/model/{model_name}.joblib'
# joblib.dump(model, joblib_path)

type_map = {
    "int64": Int64TensorType([None, 1]),
    "float32": FloatTensorType([None, 1]),
    "float64": FloatTensorType([None, 1]),
    "object": StringTensorType([None, 1])
}
init_types = [(elem, type_map[X[elem].dtype.name]) for elem in input_columns]

model_onnx = convert_sklearn(
    pipeline,
    initial_types = init_types,
    options={'zipmap': False}
)

onnx.save_model(model_onnx, onnx_path)

with open('/volumn/Retree_exp/workloads/modelInfo/model_name.txt', 'a', encoding='utf-8') as f:
    f.write(f'{model_name}\n')
    
with open('/volumn/Retree_exp/workloads/modelInfo/model_leaf_range.txt', 'a', encoding='utf-8') as f:
    f.write(f'{model_name}\n')
    labels = list(set(y))
    labels.sort()
    for label in labels:
        f.write(f'{label}\n')