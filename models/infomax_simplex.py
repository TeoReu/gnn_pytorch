import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from models.infomax import AvgReadout, Discriminator


class SimCAT(nn.Module):
    def __init__(self, n_in):
        super(SimCAT, self).__init__()
        self.dense = nn.Sequential(nn.Flatten(), nn.Linear(2 * n_in, n_in), nn.ReLU())

    def forward(self, h):
        h_1, h_2 = torch.split(h, int(h.shape[0] / 2), dim=0)
        output = torch.cat((h_1, h_2), dim=1)
        output = self.dense(output)
        return output


class DGIS(nn.Module):
    def __init__(self, n_in, n_h):
        super(DGIS, self).__init__()
        self.gcn = GCNConv(n_in, n_h, bias=True)
        self.cat = SimCAT(n_h)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, edge_index, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, edge_index).relu()
        h_1 = self.cat(h_1)
        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, edge_index).relu()
        h_2 = self.cat(h_2)
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, edge_index, msk):
        h_1 = self.gcn(seq, edge_index).relu()
        h_1 = self.cat(h_1)
        # c = self.read(h_1, msk)

        t = h_1.detach()
        # t = torch.reshape(t, (2, int(t.shape / 2)))
        # t = torch.mean(t, 1, True)

        return t  # , c.detach()
