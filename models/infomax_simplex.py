import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from models.infomax import AvgReadout, Discriminator


class DGIS(nn.Module):
    def __init__(self, n_in, n_h):
        super(DGIS, self).__init__()
        self.gcn = GCNConv(n_in, n_h, bias=True)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, edge_index, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, edge_index).relu()

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, edge_index).relu()

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, edge_index, msk):
        h_1 = self.gcn(seq, edge_index).relu()
        # c = self.read(h_1, msk)

        t = h_1.detach()
        #t = torch.reshape(t, (2, int(t.shape / 2)))
        #t = torch.mean(t, 1, True)

        return t  # , c.detach()
