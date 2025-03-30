import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# -------------------------
# Define Simple GCN Model
# -------------------------
class TSGUARD(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TSGUARD, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# -------------------------
# Build Adjacency Matrix (KNN or geographical distance-based)
# -------------------------
def build_edge_index(sensor_positions):
    edge_list = []
    sensor_ids = list(sensor_positions.keys())
    for i in range(len(sensor_ids)):
        for j in range(len(sensor_ids)):
            if i != j:
                edge_list.append([i, j])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index
