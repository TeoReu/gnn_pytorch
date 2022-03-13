import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from models.infomax import AvgReadout, Discriminator


class CAT(nn.Module):
    def __init__(self, n_h):
        super(CAT, self).__init__()
        self.dense = nn.Sequential(nn.Flatten(), nn.Linear(2 * n_h, n_h), nn.ReLU())

    def forward(self, h_1, h_2):
        output = torch.cat((h_1, h_2), dim=1)
        output = self.dense(output)
        return output


class DGICAT(nn.Module):
    def __init__(self, n_in1, n_in2, n_h):
        super(DGICAT, self).__init__()
        self.gcn_a = GCNConv(n_in1, n_h, bias=True)
        self.gcn_b = GCNConv(n_in2, n_h, bias=True)

        self.cat = CAT(n_h)

        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1a, seq2a, seq1b, seq2b, edge_index1, edge_index2, msk, samp_bias1, samp_bias2):
        h_1a = self.gcn_a(seq1a, edge_index1).relu()
        h_1b = self.gcn_b(seq1b, edge_index2).relu()

        h_1 = self.cat(h_1a, h_1b)
        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2a = self.gcn_a(seq2a, edge_index1).relu()
        h_2b = self.gcn_b(seq2b, edge_index2).relu()
        h_2 = self.cat(h_2a, h_2b)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq1a, seq1b, edge_index1, edge_index2,msk):
        h_1a = self.gcn_a(seq1a, edge_index1).relu()
        h_1b = self.gcn_b(seq1b, edge_index2).relu()
        h_1 = self.cat(h_1a, h_1b)

        # c = self.read(h_1, msk)

        t = h_1.detach()
        # t = torch.reshape(t, (2, int(t.shape / 2)))
        # t = torch.mean(t, 1, True)

        return t  # , c.detach()
