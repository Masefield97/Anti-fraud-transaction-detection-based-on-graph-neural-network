import matplotlib.pyplot as plt

def plot_contribution_pie(contributions, node_idx, one_hop_neighbors, two_hop_neighbors):
    self_val = contributions.get(node_idx, 0)
    one_hop_val = sum(contributions.get(n, 0) for n in one_hop_neighbors)
    two_hop_val = sum(contributions.get(n, 0) for n in two_hop_neighbors)
    others_val = 1 - (self_val + one_hop_val + two_hop_val)
    
    labels = ['Self', '1-hop neighbors', '2-hop neighbors', 'Others']
    sizes = [self_val, one_hop_val, two_hop_val, others_val]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#95a5a6']
    
    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f'Contribution Decomposition for Node {node_idx}')
    plt.show()