import argparse

import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import VGAE, GAE

from misc.dataset import DatasetWhole
from misc.helpers import normalizeRNA, save_embedding

import os

from models.cncgvae import CNCGVAE
from utils import buildGraph, MMD



parser = argparse.ArgumentParser()
parser.add_argument('--integration', help='Type of integration Clin+mRNA, CNA+mRNA or Clin+CNA', type=str,
                    required=True, default='Clin+mRNA')
parser.add_argument('--save_model', help='Saves the weights of the model', action='store_true')
parser.add_argument('--fold', help='The fold to train on, if 0 will train on the whole data set', type=str, default='0')
parser.add_argument('--dtype', help='The type of data (Pam50, Pam50C, IC, ER)', type=str, default='W')
parser.add_argument('--beta', help='beta size', type=int, default=0.5)
parser.add_argument('--distance', help='regularization', type=str, default='mmd')
parser.add_argument('--ls', help='latent dimension size', type=int, required=True)
parser.add_argument('--writedir', help='/PATH/TO/OUTPUT - Default is current dir', type=str, default='')
parser.add_argument('--k', help='knn', type=int, required=True)
parser.add_argument('--epochs', help='epochs', type=int, required=True)

if __name__ == "__main__":
    args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = DatasetWhole('W')

print('TRAINING on the complete data')

dataset = DatasetWhole('W')

if args.integration == 'Clin+mRNA':  # integrate Clin+mRNA
    s1_train = dataset.train['clin']
    s2_train = normalizeRNA(dataset.train['rnanp'])

elif args.integration == 'Clin+CNA':  # integrate Clin+CNA
    s1_train = dataset.train['clin']
    s2_train = dataset.train['cnanp']

else:
    s1_train = dataset.train['cnanp']  # integrate CNA+mRNA
    s2_train = normalizeRNA(dataset.train['rnanp'])

conc_input = torch.cat((torch.from_numpy(s1_train), torch.from_numpy(s2_train)), -1)

data_split, data = buildGraph(conc_input, args.k, device)

train_data, val_data, test_data = data_split
in_channels, out_channels = data.num_features, args.ls

model = GAE(CNCGVAE(in_channels, out_channels))
model = model.to(device)
train_pos_edge_index = train_data.pos_edge_label_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    if args.distance == 'kl':
        distance = model.kl_loss()
        beta = 1 / data.num_nodes
    else:
        true_samples = torch.normal(torch.zeros(z.size()[0], z.size()[1]), torch.ones(z.size()[0], z.size()[1]))
        distance = MMD(true_samples, z, device=device)
        beta = args.beta
    loss = loss + beta * distance

    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(pos_edge_index, neg_edge_index):
    model.eval()
    z = model.encode(test_data.x, test_data.edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


for epoch in range(1, args.epochs):
    loss = train()
    auc, ap = test(test_data.pos_edge_label_index, test_data.neg_edge_label_index)
    if epoch % 50 == 0:
        print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')

emb_train = model.encode(train_data.x, train_data.edge_index)

if args.writedir == '':
    emb_save_dir = 'results/CNCVAE_' + format(args.integration) + '_integration/cncvae_LS_' + format(
        args.ls) + '_K_' + format(args.k) + '_' + format(args.distance) + '_beta_' + format(args.beta) + '_epochs_' + format(args.epochs)
else:
    emb_save_dir = args.writedir + '/CNCVAE_' + format(args.integration) + '_integration/cncvae_LS_' + format(
        args.ls) + '_K_' + format(args.k) + '_' + format(args.distance) + '_beta_' + format(args.beta) + '_epochs_' + format(args.epochs)
if not os.path.exists(emb_save_dir):
    os.makedirs(emb_save_dir)
emb_save_file = args.dtype + '.npz'
save_embedding(emb_save_dir, emb_save_file, emb_train.detach())

print("Done")
