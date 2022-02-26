import torch
from sentence_transformers.models import Dense
from torch_geometric.nn import GCNConv

from utils import buildGraph


class XGVAEncoder(torch.nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels):
        super().__init__()
        self.encoder = None

        self.dense1 = Dense(in_channels_1)
        self.dense2 = Dense(in_channels_2)


        self.conv1 = GCNConv(in_channels_1 + in_channels_2, 2 * out_channels)

        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x1, x2, edge_index):
        x1 = self.dence1(x1)
        x2 = self.dence2(x2)
        x = buildGraph(x1, x2)
        concat = torch.cat([x1,x2])
        x = self.conv1(concat, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class Pos2GraphLayer(torch.nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels ):
        super().__init__()

    def froward(self, x1, x2):
        graph, data = buildGraph(torch.cat(x1,x2), dim=1)
