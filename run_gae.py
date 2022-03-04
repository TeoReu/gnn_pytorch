import argparse

import torch
from torch_geometric.nn import GCNConv, GAE, VGAE

from misc.dataset import DatasetWhole
from misc.helpers import normalizeRNA
from utils import buildGraph



class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 4 * out_channels)
        self.conv12 = GCNConv(4 * out_channels, 2 * out_channels)

        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv12(x, edge_index).relu()
        return self.conv2(x, edge_index)


parser = argparse.ArgumentParser()
#parser.add_argument('--integration', help='Type of integration Clin+mRNA, CNA+mRNA or Clin+CNA', type=str,
#                    required=True, default='Clin+mRNA')
parser.add_argument('--save_model', help='Saves the weights of the model', action='store_true')
parser.add_argument('--fold', help='The fold to train on, if 0 will train on the whole data set', type=str, default='0')
parser.add_argument('--dtype', help='The type of data (Pam50, Pam50C, IC, ER)', type=str, default='W')
parser.add_argument('--beta', help='beta size', type=int, default=0.5)
parser.add_argument('--distance', help='regularization', type=str, default='mmd')
#parser.add_argument('--ls', help='latent dimension size', type=int, required=True)
parser.add_argument('--writedir', help='/PATH/TO/OUTPUT - Default is current dir', type=str, default='')
#parser.add_argument('--k', help='knn', type=int, required=True)
#parser.add_argument('--epochs', help='epochs', type=int, required=True)

if __name__ == "__main__":
    args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = DatasetWhole('W')

# parameters
out_channels = 64
epochs = 1000

# model

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

s1_train = dataset.train['clin']
s2_train = normalizeRNA(dataset.train['rnanp'])

conc_input = torch.cat((torch.from_numpy(s1_train), torch.from_numpy(s2_train)), -1)

data_split, data = buildGraph(conc_input, 20)
train_data, val_data, test_data = data_split

num_features = data.num_features



model = GAE(GCNEncoder(num_features, out_channels))

x = data.x
train_pos_edge_index = train_data.pos_edge_label_index

# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)

    #loss = loss + (10 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


for epoch in range(1, epochs + 1):
    loss = train()

    auc, ap = test(test_data.pos_edge_label_index, test_data.neg_edge_label_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))