# 针对类别不平衡的加权采样
from torch_geometric.loader import NeighborLoader

# 自定义采样器：提高非法节点被采样的概率
def create_weighted_loader(data, class_weights, batch_size):
    # 获取训练节点权重
    node_weights = np.ones(data.num_nodes)
    illicit_nodes = torch.where((data.y == 1) & data.train_mask)[0].numpy()
    node_weights[illicit_nodes] = class_weights[1] * 5  # 非法节点采样权重提高
    
    sampler = WeightedRandomSampler(node_weights[data.train_mask], len(data.train_mask))
    
    return NeighborLoader(
        data,
        num_neighbors=[25, 10],
        batch_size=batch_size,
        input_nodes=torch.where(data.train_mask)[0],
        sampler=sampler
    )