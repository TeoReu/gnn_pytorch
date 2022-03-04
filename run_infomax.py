import argparse
import os

import numpy as np
import torch
from torch_geometric import nn
from torch import nn
from misc.dataset import DatasetWhole
from misc.helpers import normalizeRNA, save_embedding
from models.infomax import DGI
from utils import buildGraph


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

# parameters
out_channels = args.ls
epochs = args.epochs
patience = 1000

# model

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

data_split, data = buildGraph(conc_input, args.k)
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


emb_train = model.embed(train_data.x, train_data.edge_index,None)

if args.writedir == '':
    emb_save_dir = 'results/infomax_' + format(args.integration) + '_integration/infomax_LS_' + format(
        args.ls) + '_K_' + format(args.k) + '_' + format(args.distance) + '_beta_' + format(args.beta) + '_epochs_' + format(args.epochs)
else:
    emb_save_dir = args.writedir + '/infomax_' + format(args.integration) + '_integration/infomax_LS_' + format(
        args.ls) + '_K_' + format(args.k) + '_' + format(args.distance) + '_beta_' + format(args.beta) + '_epochs_' + format(args.epochs)
if not os.path.exists(emb_save_dir):
    os.makedirs(emb_save_dir)
emb_save_file = args.dtype + '.npz'
save_embedding(emb_save_dir, emb_save_file, emb_train.detach())

print("Done")

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