from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import torch
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, dropout=0.0):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(768, 768)
        self.conv2 = GCNConv(768, 768)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x

class GAT(torch.nn.Module):
    def __init__(self, dropout=0.0):
        super(GAT, self).__init__()
        self.conv1 = GATConv(768, 768)
        self.conv2 = GATConv(768, 768)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, dropout=0.0):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(768, 768)
        self.conv2 = SAGEConv(768, 768)
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x