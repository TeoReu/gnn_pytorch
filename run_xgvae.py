import argparse

import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import VGAE

from misc.dataset import DatasetWhole
from misc.helpers import normalizeRNA

import os

from models.cncgvae import CNCGVAE
from utils import buildGraph, MMD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = DatasetWhole('W')

s1_train = dataset.train['clin']
s2_train = normalizeRNA(dataset.train['cnanp'])
beta = 0.9

in_channels, out_channels = s1_train.x.size()[1], 16

model = VGAE(CNCGVAE(in_channels, out_channels))
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(s1_train, s2_train, train_data.edge_index)
    loss =  model.recon_loss(z, train_data.pos_edge_label_index)
    true_samples = torch.normal(torch.zeros(z.size()[0], z.size()[1]), torch.ones(z.size()[0], z.size()[1]))
    distance = MMD(true_samples, z,device=device)
    loss = loss + beta * distance

    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


for epoch in range(1, 99 + 1):
    loss = train()
    auc, ap = test(test_data)
    print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')