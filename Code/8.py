import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGEIllicitDetector(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # 输入层 -> 第一隐藏层
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='mean'))
        
        # 中间隐藏层
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean'))
        
        # 输出层
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='mean'))
        
        self.dropout = dropout
        self.norm = nn.LayerNorm(hidden_channels)  # 可选项，根据实验效果决定
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        # 根据研究结论：GraphSAGE单独使用Xavier初始化最优
        for conv in self.convs:
            if hasattr(conv, 'lin_l') and conv.lin_l is not None:
                nn.init.xavier_uniform_(conv.lin_l.weight)
            if hasattr(conv, 'lin_r') and conv.lin_r is not None:
                nn.init.xavier_uniform_(conv.lin_r.weight)
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # 非最后一层
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# 模型实例化
# 输入维度：经过时间步独热编码后的维度（49时间步 + 165连续特征 = 214维）
model = GraphSAGEIllicitDetector(
    in_channels=data.x.size(1),  # 实际特征维度
    hidden_channels=128,  # 第一隐藏层
    out_channels=2,  # 二分类
    num_layers=2,
    dropout=0.3
).to(device)