import networkx as nx
from torch_geometric.utils import to_networkx

def visualize_subgraph_with_contributions(model, node_idx, x, edge_index, contributions, num_hops=2):
    # 提取子图
    subset, edge_index_sub, _, _ = k_hop_subgraph(node_idx, num_hops, edge_index)
    G = to_networkx(edge_index_sub, to_undirected=True)
    
    # 节点颜色：根据贡献值（红色越深贡献越大）
    node_colors = []
    for n in subset:
        contrib = contributions.get(n.item(), 0)
        # 红色通道强度
        node_colors.append((1.0, 1.0 - contrib, 1.0 - contrib))
    
    # 边颜色：根据两端节点贡献之和
    edge_colors = []
    for u, v in G.edges():
        contrib_sum = contributions.get(u, 0) + contributions.get(v, 0)
        edge_colors.append(contrib_sum)
    
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edge_cmap=plt.cm.Reds, edge_vmin=0, edge_vmax=1, width=2)
    nx.draw_networkx_labels(G, pos, labels={n: str(n.item()) for n in subset}, font_size=8)
    plt.title(f'Neighbor Contribution Heatmap for Node {node_idx}')
    plt.axis('off')
    plt.show()