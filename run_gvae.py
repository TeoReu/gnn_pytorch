import argparse

import networkx as nx
import numpy as np
import scipy as sp
import torch
from torch_geometric.utils import to_networkx

from misc.dataset import DatasetWhole
from misc.helpers import normalizeRNA
from models.gvae import GCNModelVAE, loss_function
from utils import buildGraph, mask_test_edges, get_roc_score, preprocess_graph
from torch import optim

parser = argparse.ArgumentParser()
# parser.add_argument('--integration', help='Type of integration Clin+mRNA, CNA+mRNA or Clin+CNA', type=str,
#                    required=True, default='Clin+mRNA')
parser.add_argument('--save_model', help='Saves the weights of the model', action='store_true')
parser.add_argument('--fold', help='The fold to train on, if 0 will train on the whole data set', type=str, default='0')
parser.add_argument('--dtype', help='The type of data (Pam50, Pam50C, IC, ER)', type=str, default='W')
parser.add_argument('--beta', help='beta size', type=int, default=0.5)
parser.add_argument('--distance', help='regularization', type=str, default='mmd')
# parser.add_argument('--ls', help='latent dimension size', type=int, required=True)
parser.add_argument('--writedir', help='/PATH/TO/OUTPUT - Default is current dir', type=str, default='')
# parser.add_argument('--k', help='knn', type=int, required=True)
# parser.add_argument('--epochs', help='epochs', type=int, required=True)

if __name__ == "__main__":
    args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = DatasetWhole('W')

# parameters
beta = 1
out_channels = 128
epochs = 1000

# model

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

s1_train = dataset.train['clin']
s2_train = normalizeRNA(dataset.train['rnanp'])

conc_input = torch.cat((torch.from_numpy(s1_train), torch.from_numpy(s2_train)), -1)

data_split, data = buildGraph(conc_input, 100)
train_data, val_data, test_data = data_split

num_features = data.num_features

# print("Using {} dataset".format(args.dataset_str))
G = to_networkx(data, to_undirected=True)

features = data.x
adj = nx.adjacency_matrix(G)

n_nodes, feat_dim = data.num_nodes, data.num_features

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.sparse.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

# Some preprocessing
adj_norm = preprocess_graph(adj)
adj_label = adj_train + sp.sparse.eye(adj_train.shape[0])
# adj_label = sparse_to_tuple(adj_label)
adj_label = torch.FloatTensor(adj_label.toarray())

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

model = GCNModelVAE(feat_dim, 64, 32, 0.5)
optimizer = optim.Adam(model.parameters(), lr=0.001)

hidden_emb = None
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    recovered, mu, logvar = model(features, adj_norm)
    loss = loss_function(preds=recovered, labels=adj_label,
                         mu=mu, logvar=logvar, n_nodes=n_nodes,
                         norm=norm, x=data.x)
    loss.backward()
    cur_loss = loss.item()
    optimizer.step()

    hidden_emb = mu.data.numpy()
    roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
          "val_ap=", "{:.5f}".format(ap_curr),
          )

print("Optimization Finished!")

roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))
