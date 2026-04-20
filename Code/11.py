from sklearn.model_selection import ParameterGrid

# 搜索空间
param_grid = {
    'hidden_channels': [64, 128, 256],
    'num_layers': [2, 3],
    'dropout': [0.2, 0.3, 0.4],
    'lr': [0.001, 0.0005],
    'batch_size': [512, 1024],
    'neighbor_sizes': [[15, 10], [25, 10], [25, 15]]
}

# 根据已有研究，推荐起始配置：
# - hidden_channels: 128
# - num_layers: 2（比特币网络较浅，2层足够捕获有效信息）
# - dropout: 0.3
# - lr: 0.001
# - batch_size: 1024
# - neighbor_sizes: [25, 10]