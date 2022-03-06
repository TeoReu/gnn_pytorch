import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os

import argparse

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
from misc.dataset import Dataset, DatasetWhole
from misc.helpers import normalizeRNA, save_embedding

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import warnings

warnings.filterwarnings('ignore')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'T', 'TRUE'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'F', 'FALSE'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


train_labels = []
test_labels = []

train_raw = []
test_raw = []

train_cna = []
test_cna = []

train_rna = []
test_rna = []

train_clin = []
test_clin = []

parser = argparse.ArgumentParser()
parser.add_argument('--integration', help='Type of integration Clin+mRNA, CNA+mRNA or Clin+CNA. Default is Clin+mRNA.',
                    type=str,  default='Clin+mRNA')
parser.add_argument('--dtype', help='The type of data (Pam50, Pam50C, IC, ER). Default is ER.', type=str, default='W')
parser.add_argument('--writedir', help='/PATH/TO/OUTPUT - Default is current dir', type=str, default='')
parser.add_argument('--resdir', help='/PATH/TO/EMBEDDINGS - Default is  results/', type=str, default='results/')
parser.add_argument('--numfolds', help='number of folds of CV-analyses, 0 indicates whole data set. Default is 5',
                    type=int, default=5)
parser.add_argument('--model',
                    help='Can be either CNCVAE, XVAE, MMVAE, HVAE or BENCH. BENCH refers to benchmark, i.e. learning a classifier on raw data and PCA transformed data. Default is CNCVAE',
                    type=str, default='infomax')
parser.add_argument('--tSNE',
                    help='generate a tSNE plot (True/False). Default is false. Warning: it is computationaly demanding, and only enabled for whole dataset analyses',
                    type=str2bool, default=False)

parser.add_argument('--NB', help='train a Naive Bayes Classifier. Default True', type=str2bool, default=True)
parser.add_argument('--SVM', help='train a SVM Classifier. Default False', type=str2bool, default=True)
parser.add_argument('--RF', help='train a Random Forest. Default False', type=str2bool, default=True)

if __name__ == "__main__":
    args = parser.parse_args()

dataset = DatasetWhole('W')
clin_train = dataset.train['clin']
print(dataset.train['clin'])

train_labels.append(dataset.train["ernp"])

accsTrain_NB = []
accsTest_NB = []

accsTrain_SVM = []
accsTest_SVM = []

accsTrain_RF = []
accsTest_RF = []

model = 'infomax'
resdir = "results"

model_conf = format(model.lower()) + '_LS_' + format(16) + '_K_' + format(20) \
             + '_kl_'  + '_epochs_' + format(1500)

embed = np.load("OLDresults/infomax_simple_CNA+mRNA_integration/infomax_LS_16_K_20_epochs_10000/W0.npz")

emb_train = embed['emb_train']

random_state = 42
note = ""

savedir = 'results'
with open(
        savedir + '/' + format(args.model) + '_' + format(args.integration) + '_' + format(args.dtype) + '.csv',
        'w') as f:
    for model in [args.model]:
        print("------------------------------------------------", file=f)
        print(model, file=f)
        print("------------------------------------------------", file=f)
        print(
            "model, type_integration, regularization, beta, latent_size, dense_layer_size, NB_Train_ACC, NB_Train_ACC_std, NB_Test_ACC, NB_Test_ACC_std, SVM_Train_ACC, SVM_Train_ACC_std, SVM_Test_ACC, SVM_Test_ACC_std, RF_Train_ACC, RF_Train_ACC_std, RF_Test_ACC, RF_Test_ACC_std ",
            file=f)
    if (np.isnan(emb_train).any()):
        note += "Check *ER  embeding for problems"
        print(resdir + '/' + model_conf + '/' + 'W' +  '.npz is invalid. Consider re-training.')
    else:
        if args.NB:
            nb = GaussianNB()
            nb.fit(emb_train, train_labels[0])

            x_p_classes = nb.predict(emb_train)
            accTrain = accuracy_score(train_labels[0], x_p_classes)
            accsTrain_NB.append(accTrain)
        if args.SVM:
                    svm = SVC(C=1.5, kernel='rbf', random_state=42, gamma='auto')
                    svm.fit(emb_train, train_labels[0])

                    x_p_classes = svm.predict(emb_train)
                    accTrain = accuracy_score(train_labels[0], x_p_classes)
                    accsTrain_SVM.append(accTrain)

        else:
            accsTrain_SVM.append(0.0)

        if args.RF:
            rf = RandomForestClassifier(n_estimators=50, random_state=42,
                                        max_features=.5)
            rf.fit(emb_train, train_labels[0])

            x_p_classes = rf.predict(emb_train)
            accTrain = accuracy_score(train_labels[0], x_p_classes)
            accsTrain_RF.append(accTrain)

        else:
            accsTrain_RF.append(0.0)

    print(
        format(model) + ',' + format(args.integration) + ','

        + ',' + format(np.mean(accsTrain_NB))
        + ',' + format(np.var(accsTrain_NB))


        + ',' + format(np.mean(accsTrain_SVM))
        + ',' + format(np.var(accsTrain_SVM))


        + ',' + format(np.mean(accsTrain_RF))
        + ',' + format(np.var(accsTrain_RF))

        + note, file=f)
    f.flush()
    print('Done.')
    f.close()
