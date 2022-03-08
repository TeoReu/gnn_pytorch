import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from torch_geometric import nn
from torch import nn
from misc.dataset import DatasetWhole, Dataset
from misc.helpers import normalizeRNA, save_embedding
from models.infomax import DGI
from models.infomax_simplex import DGIS
from models.logreg import LogReg
from utils import buildGraph, build_simplex

parser = argparse.ArgumentParser()
parser.add_argument('--integration', help='Type of integration Clin+mRNA, CNA+mRNA or Clin+CNA', type=str,
                    required=True, default='Clin+mRNA')
parser.add_argument('--save_model', help='Saves the weights of the model', action='store_true')
parser.add_argument('--fold', help='The fold to train on, if 0 will train on the whole data set', type=int, default=1)
parser.add_argument('--dtype', help='The type of data (Pam50, Pam50C, IC, ER)', type=str, default='ER')
parser.add_argument('--ls', help='latent dimension size', type=int, required=True)
parser.add_argument('--writedir', help='/PATH/TO/OUTPUT - Default is current dir', type=str, default='')
parser.add_argument('--k', help='knn', type=int, required=True)
parser.add_argument('--epochs', help='epochs', type=int, required=True)
parser.add_argument('--graph_type', help='The type of the graph: simple or simplex', type=str, default='simple')
if __name__ == "__main__":
    args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameters
out_channels = args.ls
epochs = args.epochs
patience = 1000000

# model

# move to GPU (if available)
if args.fold == 0:
    dataset = DatasetWhole('W')

    if args.integration == 'Clin+mRNA':  # integrate Clin+mRNA
        s1_data = dataset.train['clin']
        s2_data = normalizeRNA(dataset.train['rnanp'])

    elif args.integration == 'Clin+CNA':  # integrate Clin+CNA
        s1_data = dataset.train['clin']
        s2_data = dataset.train['cnanp']

    else:
        s1_data = dataset.train['cnanp']  # integrate CNA+mRNA
        s2_data = normalizeRNA(dataset.train['rnanp'])

else:
    print('TRAINING on the fold '+ format(args.fold))

    dataset = Dataset(args.dtype, str(args.fold))

    if (args.integration == 'Clin+mRNA'):  # integrate Clin+mRNA
        s1_data_train = dataset.train['clin']
        s1_data_test = dataset.test['clin']
        s2_data_train = dataset.train['rnanp']
        s2_data_test = dataset.test['rnanp']
    elif (args.integration == 'Clin+CNA'):  # integrate Clin+CNA
        s1_data_train = dataset.train['clin']
        s1_data_test = dataset.test['clin']
        s2_data_train = dataset.train['cnanp']
        s2_data_test = dataset.test['cnanp']


    else:  # integrate CNA+mRNA
        s1_data_train = dataset.train['cnanp']
        s1_data_test = dataset.test['cnanp']
        s2_data_train = dataset.train['rnanp']
        s2_data_test = dataset.test['rnanp']

if(args.graph_type == 'simple'):
    conc_train = torch.cat((torch.from_numpy(s1_data_train), torch.from_numpy(s2_data_train)), -1)
    conc_test = torch.cat((torch.from_numpy(s1_data_test), torch.from_numpy(s2_data_test)), -1)
    train_data = buildGraph(conc_train, args.k)
    test_data = buildGraph(conc_test, args.k)

else:
    train_data_1 = buildGraph(torch.from_numpy(s1_data_train), args.k)
    train_data_2 = buildGraph(torch.from_numpy(s2_data_train), args.k)

    train_data = build_simplex(train_data_1, train_data_2)

    test_data_1 = buildGraph(torch.from_numpy(s1_data_test), args.k)
    test_data_2 = buildGraph(torch.from_numpy(s2_data_test), args.k)

    test_data = build_simplex(test_data_1, test_data_2)




num_features = train_data.num_features
if args.graph_type == 'simple':
    model = DGI(num_features, out_channels)
else:
    model = DGIS(num_features, out_channels)
b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0

best = 1e9
best_t = 0

x = train_data.x

# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#nb_epochs = 100

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    idx = np.random.permutation(train_data.num_nodes)
    shuf_fts = x[idx, :]

    lbl_1 = torch.ones(train_data.num_nodes, 1)
    lbl_2 = torch.zeros(train_data.num_nodes, 1)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    logits = model(train_data.x, shuf_fts, train_data.edge_index, None, None, None)

    loss = b_xent(logits, lbl)

    #print('Loss:', loss)

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

model.load_state_dict(torch.load('best_dgi.pkl'))

# embeds, _ = model.embed(data.x, edge_index=train_pos_edge_index)

emb_train = model.embed(train_data.x, train_data.edge_index, None)
emb_test = model.embed(test_data.x, test_data.edge_index, None)

if args.writedir == '':
    emb_save_dir = 'results/infomax_' + format(args.graph_type) + "_" + format(args.integration) + '_integration/infomax_LS_' + format(
        args.ls) + '_K_' + format(args.k) + '_epochs_' + format(args.epochs)
else:
    emb_save_dir = args.writedir + '/infomax_' + format(args.graph_type) + "_" + format(args.integration) + '_integration/infomax_LS_' + format(
        args.ls) + '_K_' + format(args.k) + '_epochs_' + format(args.epochs)
if not os.path.exists(emb_save_dir):
    os.makedirs(emb_save_dir)
emb_save_file = args.dtype + str(args.fold) + '.npz'
save_embedding(emb_save_dir, emb_save_file, emb_train.detach(), emb_test.detach())

print("Done")
labels_train = dataset.train["ernp"]
labels_test = dataset.test["ernp"]

if args.graph_type == 'simplex':
    labels_train = np.append(labels_train, labels_train)
    labels_test = np.append(labels_test, labels_test)

accsTest = []
accsTrain =[]
tot = 0

for _ in range(2):
    rf = SVC(C=1.5, kernel='rbf', random_state=42, gamma='auto')
    rf.fit(emb_train, labels_train)

    x_p_classes1 = rf.predict(emb_test)
    x_p_classes2 = rf.predict(emb_train)

    accTest = accuracy_score(labels_test, x_p_classes1)
    accTrain = accuracy_score(labels_train, x_p_classes2)

    accsTest.append(accTest)
    accsTrain.append(accTrain)

    tot += accTest

print('Average accuracy:', tot / 2)

accsTest = np.stack(accsTest)
print(accsTest.mean())
print(accsTest.std())

accsTrain = np.stack(accsTrain)
print(accsTrain.mean())
print(accsTrain.std())

