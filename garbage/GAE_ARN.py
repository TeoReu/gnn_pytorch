import argparse
import os.path as osp

import numpy as np
import pandas as pd
import torch
import torch_geometric

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv



def readSCG(address):
    #data = pd.read_csv(address, sep="\t")
    #data = train.copy().drop("Gene", axis=1)
    data = pd.read_csv(address, sep="\t").drop("Gene", axis=1)
    data = torch.FloatTensor(data.values)
    return data


parser = argparse.ArgumentParser()
parser.add_argument('--variational', action='store_true')
parser.add_argument('--linear', action='store_true')
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'CiteSeer', 'PubMed'])
parser.add_argument('--epochs', type=int, default=400)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      split_labels=True, add_negative_train_samples=True),
])
#path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
#dataset = Planetoid(path, args.dataset, transform=transform)
pos = readSCG('/Users/teodorareu/PycharmProjects/gnn_pytorch/AML_data/BM1dem.txt')


data = torch_geometric.data.Data(pos=pos, dtype=torch.float)
edge_index = torch_geometric.nn.knn_graph(data.pos, k=2, loop=False, flow='source_to_target')
print("graph initiated")
data = torch_geometric.data.Data(x=pos, edge_index=edge_index, pos=pos,  dtype=torch.float)

data_split = transform(data)
train_data, val_data, test_data = data_split


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


in_channels, out_channels = data.num_features, 16


model = VGAE( VariationalGCNEncoder(in_channels, out_channels))

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)

    true_samples = torch.normal(torch.zeros(z.size()[0], out_channels), torch.ones(z.size()[0], out_channels))
    distance = mmd(true_samples, z)
    loss = loss + (1 / train_data.num_nodes) * distance

    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


for epoch in range(1, args.epochs + 1):
    loss = train()
    auc, ap = test(test_data)
    print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')