import argparse

import numpy as np
import torch
from torch_geometric import nn
from torch import nn
from misc.dataset import DatasetWhole
from misc.helpers import normalizeRNA
from models.infomax import DGI
from utils import buildGraph, build_simplex

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = DatasetWhole('W')

# parameters
out_channels = 64
epochs = 1000
patience = 100

# model

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

s1_train = dataset.train['clin']
s2_train = normalizeRNA(dataset.train['rnanp'])


_, data1 = buildGraph(torch.from_numpy(s1_train), 20)
_, data2 = buildGraph(torch.from_numpy(s2_train), 20)

data_split, data = build_simplex(data1, data2)

train_data, val_data, test_data = data_split

num_features = data.num_features



model = DGI(num_features, out_channels)
b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0

best = 1e9
best_t = 0

x = data.x
train_pos_edge_index = train_data.pos_edge_label_index

# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
nb_epochs = 1000

for epoch in range(nb_epochs):
    model.train()
    optimizer.zero_grad()
    idx = np.random.permutation(data.num_nodes)
    shuf_fts = x[idx, :]

    lbl_1 = torch.ones(data.num_nodes, 1)
    lbl_2 = torch.zeros(data.num_nodes, 1)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    logits = model(data.x, shuf_fts, train_pos_edge_index, None, None, None)

    loss = b_xent(logits, lbl)

    print('Loss:', loss)


    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'best_dgi.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimizer.step()


print('Loading {}th epoch'.format(best_t))
#model.load_state_dict(torch.load('best_dgi.pkl'))

#embeds, _ = model.embed(data.x, edge_index=train_pos_edge_index)


'''
for _ in range(50):
    log = LogReg(hid_units, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    log.cuda()

    pat_steps = 0
    best_acc = torch.zeros(1)
    best_acc = best_acc.cuda()
    for _ in range(100):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)

        loss.backward()
        opt.step()

    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    accs.append(acc * 100)
    print(acc)
    tot += acc

print('Average accuracy:', tot / 50)

accs = torch.stack(accs)
print(accs.mean())
print(accs.std())

'''