import copy
from random import random

import networkx as nx
import numpy as np
import matplotlib
import pandas as pd
import torch
import torch_geometric
from matplotlib import pyplot as plt
from torch import long
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import knn_graph, RandomLinkSplit
from torch_geometric.nn import VGAE
from torch_geometric.utils import train_test_split_edges

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)



def print_graph(pos):
    data = torch_geometric.data.Data(pos=pos, dtype=torch.LongTensor)
    edge_index = torch_geometric.nn.knn_graph(data.pos, k=20, loop=False, flow='source_to_target')

    new_data = torch_geometric.data.Data(x=pos, edge_index=edge_index, pos=pos, dtype=torch.LongTensor)
    #print(new_data)
    #new_data = new_data.type(torch.LongTensor)
    #g = torch_geometric.utils.to_networkx(new_data, to_undirected=True)
    #nx.draw(g, node_size=10)

    #plt.show()
    return new_data


def readSCG(address):
    #data = pd.read_csv(address, sep="\t")
    #data = train.copy().drop("Gene", axis=1)
    data = pd.read_csv(address, sep="\t").drop("Gene", axis=1)
    data = np.array(data)
    #print(data.drop('METABRIC_ID', axis=1))
    data = torch.from_numpy(data)
    return data


vec = readSCG('AML1012dem.txt').T

print(vec.size())

data = print_graph(vec)


tfs = RandomLinkSplit(is_undirected=True,
                      add_negative_train_samples=True,
                      neg_sampling_ratio=1.0,
                      key = "edge_label", # supervision label
                      disjoint_train_ratio=0,# disjoint mode if > 0
                      # edge_types=None, # for heteroData
                      # rev_edge_types=None, # for heteroData
                      )

train_data, val_data, test_data = tfs(data)
#print()

from torch_geometric.utils import subgraph
train_mask = torch.rand(data.num_nodes) < 0.5
test_mask = ~train_mask

train_data = copy.copy(data)
train_data.edge_index, _ = subgraph(train_mask, data.edge_index, relabel_nodes=True)
train_data.x = data.x[train_mask]

test_data = copy.copy(data)
test_data.edge_index, _ = subgraph(test_mask, data.edge_index, relabel_nodes=True)
test_data.x = data.x[test_mask]

out_channels = 2
#print(data.train_mask)
num_features = data.num_features
epochs = 300


model = VGAE(VariationalGCNEncoder(num_features, out_channels))  # new line
x = data.x
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    print(train_data.x.type())
    print(train_data.edge_index.type())
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.edge_index)

    loss = loss + (1 / data.num_nodes) * model.kl_loss()  # new line
    loss.backward()
    optimizer.step()
    return loss


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        print(x.type())
        print(test_data.edge_index.type())
        z = model.encode(x, test_data.edge_index)
    return model.test(z.type(torch.LongTensor), test_data.edge_index.type(torch.LongTensor))

for epoch in range(1, epochs + 1):
    loss = train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
