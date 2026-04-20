import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 加载数据（需提前从Kaggle下载：https://www.kaggle.com/ellipticco/elliptic-data-set）
nodes_df = pd.read_csv('data/elliptic_txs_features.csv', header=None)
edges_df = pd.read_csv('data/elliptic_txs_edgelist.csv')
labels_df = pd.read_csv('data/elliptic_txs_classes.csv')

# 特征矩阵：第1列为交易ID，第2-167列为166维特征
node_features = nodes_df.iloc[:, 2:].values.astype(np.float32)
node_ids = nodes_df.iloc[:, 0].values

# 处理缺失值（用列均值填充）
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
node_features = imputer.fit_transform(node_features)

# 标签映射：'unknown' → -1（忽略），'1'（非法）→ 1，'2'（合法）→ 0
label_map = {'unknown': -1, '1': 1, '2': 0}
labels = labels_df['class'].map(label_map).values