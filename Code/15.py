def simple_neighbor_contribution(model, node_idx, x, edge_index, target_class=1, num_hops=2):
    """
    简化版：通过遮挡实验（occlusion）估计每个邻居的重要性。
    虽然计算量稍大（需要对每个邻居进行一次前向），但实现可靠且直观。
    """
    model.eval()
    original_logit = model(x, edge_index)[node_idx, target_class].item()
    
    # 获取k-hop子图
    from torch_geometric.utils import k_hop_subgraph
    subset, edge_index_sub, mapping, _ = k_hop_subgraph(
        node_idx, num_hops=num_hops, edge_index=edge_index, num_nodes=x.size(0)
    )
    # mapping[0]是中心节点在subset中的索引
    
    contributions = {}
    # 计算自身贡献：将节点特征置零
    x_modified = x.clone()
    x_modified[node_idx] = 0
    with torch.no_grad():
        masked_logit = model(x_modified, edge_index)[node_idx, target_class].item()
    self_contrib = original_logit - masked_logit
    contributions[node_idx] = self_contrib
    
    # 计算每个邻居的贡献：逐个邻居特征置零
    for neigh in subset[1:]:  # 跳过中心节点
        x_modified = x.clone()
        x_modified[neigh] = 0
        with torch.no_grad():
            masked_logit = model(x_modified, edge_index)[node_idx, target_class].item()
        contrib = original_logit - masked_logit
        contributions[neigh] = contrib
    
    # 归一化到百分比（正贡献表示促进非法分类，负贡献表示抑制）
    # 通常我们只关心正贡献
    positive_sum = sum(v for v in contributions.values() if v > 0)
    if positive_sum > 0:
        for k in contributions:
            if contributions[k] > 0:
                contributions[k] /= positive_sum
    
    return contributions