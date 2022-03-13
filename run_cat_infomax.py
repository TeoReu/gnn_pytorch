import argparse
import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from torch import nn
from misc.dataset import DatasetWhole, Dataset
from misc.helpers import normalizeRNA, save_embedding
from models.infomax_cat import DGICAT
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


if(args.graph_type == 'cat'):
    train_data_1 = buildGraph(torch.from_numpy(s1_data_train), args.k)
    train_data_2 = buildGraph(torch.from_numpy(s2_data_train), args.k)

    test_data_1 = buildGraph(torch.from_numpy(s1_data_test), args.k)
    test_data_2 = buildGraph(torch.from_numpy(s2_data_test), args.k)


num_ft_1 = train_data_1.num_features
num_ft_2 = train_data_2.num_features


model = DGICAT(num_ft_1, num_ft_2, out_channels)

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0

best = 1e9
best_t = 0

x1 = train_data_1.x
x2 = train_data_2.x

num_nodes = train_data_1.num_nodes
# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#nb_epochs = 100

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    idx1 = np.random.permutation(train_data_1.num_nodes)
    idx2 = np.random.permutation(train_data_2.num_nodes)

    shuf_fts_1 = x1[idx1, :]
    shuf_fts_2 = x2[idx2, :]

    if (args.graph_type == 'cat'):
        lbl_1 = torch.ones(num_nodes, 1)
        lbl_2 = torch.zeros(num_nodes, 1)
        lbl = torch.cat((lbl_1, lbl_2), 1)
    #elif (args.graph_type == 'simplex'):
    else:
        l_size = int(train_data_1.num_nodes/2)
        lbl_1 = torch.ones(l_size, 1)
        lbl_2 = torch.zeros(l_size, 1)
        lbl = torch.cat((lbl_1, lbl_2), 1)

#    def forward(self, seq1a, seq2a, seq1b, seq2b, edge_index1, edge_index2, msk, samp_bias1, samp_bias2):

    logits = model(x1,  shuf_fts_1, x2, shuf_fts_2, train_data_1.edge_index, train_data_2.edge_index, None, None, None)

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

model.load_state_dict(torch.load('best_dgi.pkl'))

# embeds, _ = model.embed(data.x, edge_index=train_pos_edge_index)

emb_train = model.embed(x1, x2, train_data_1.edge_index, train_data_2.edge_index, None)
emb_test = model.embed(test_data_1.x, test_data_2.x, test_data_1.edge_index, test_data_2.edge_index, None)

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
labels_train = dataset.train["pam50np"]
labels_test = dataset.test["pam50np"]

accsTest = []
accsTrain =[]
tot = 0

rf = MLPClassifier(random_state=2, max_iter=300)

rf.fit(emb_train, labels_train)

x_p_classes1 = rf.predict(emb_test)
x_p_classes2 = rf.predict(emb_train)

accTest = accuracy_score(labels_test, x_p_classes1)
accTrain = accuracy_score(labels_train, x_p_classes2)

accsTest.append(accTest)
accsTrain.append(accTrain)

tot += accTest

print('Average accuracy:', tot)

accsTest = np.stack(accsTest)
print(accsTest.mean())
print(accsTest.std())

accsTrain = np.stack(accsTrain)
print(accsTrain.mean())
print(accsTrain.std())

