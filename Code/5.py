import torch
from torch_geometric.utils import to_undirected

# 构建边索引
# edges_df包含两列：txId1, txId2
node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
edges = edges_df[['txId1', 'txId2']].values
edge_index = []
for u, v in edges:
    if u in node_id_to_idx and v in node_id_to_idx:
        edge_index.append([node_id_to_idx[u], node_id_to_idx[v]])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# 转换为无向图（因为比特币交易流是双向可分析的）
edge_index = to_undirected(edge_index)

# 基于时间步进行划分（防止数据泄漏）
time_steps_array = node_features[:, 0]  # 原始时间步

# 训练集：时间步 1-35（约前70%时间）
# 验证集：时间步 36-42（中间时段）
# 测试集：时间步 43-49（未来数据，用于测试归纳泛化能力）
train_mask = (time_steps_array >= 1) & (time_steps_array <= 35) & (labels != -1)
val_mask = (time_steps_array >= 36) & (time_steps_array <= 42) & (labels != -1)
test_mask = (time_steps_array >= 43) & (labels != -1)

# 确保训练集内包含足够的非法样本
print(f"训练集: 合法={np.sum(labels[train_mask]==0)}, 非法={np.sum(labels[train_mask]==1)}")
print(f"验证集: 合法={np.sum(labels[val_mask]==0)}, 非法={np.sum(labels[val_mask]==1)}")
print(f"测试集: 合法={np.sum(labels[test_mask]==0)}, 非法={np.sum(labels[test_mask]==1)}")

# 创建PyG数据对象
data = Data(
    x=torch.tensor(final_features, dtype=torch.float),
    edge_index=edge_index,
    y=torch.tensor(labels, dtype=torch.long),
    train_mask=torch.tensor(train_mask, dtype=torch.bool),
    val_mask=torch.tensor(val_mask, dtype=torch.bool),
    test_mask=torch.tensor(test_mask, dtype=torch.bool)
)