from torch_geometric.loader import NeighborLoader

# 分层采样配置
# sizes=[25, 10] 表示：
# - 第一层采样25个一阶邻居
# - 第二层为每个一阶邻居采样10个二阶邻居
# - 总采样节点数上限 = batch_size * (25 + 25*10) = batch_size * 275
# 对于度较低的比特币网络，25+10已经足够覆盖大部分节点的邻居范围
neighbor_sizes = [25, 10]  # 每层采样数

# 采样时可以使用不同的聚合函数：
# - mean: 平均聚合，推荐用于平衡度分布
# - max: 最大聚合，对异常特征敏感
# - sum: 和聚合，适合度差异大的场景
# - lstm: LSTM聚合，可捕捉序列信息但计算量大

train_loader = NeighborLoader(
    data,
    num_neighbors=neighbor_sizes,  # 每层采样邻居数
    batch_size=1024,  # 根据GPU显存调整，12GB显存建议512-1024
    input_nodes=data.train_mask,
    shuffle=True,
    drop_last=False
)