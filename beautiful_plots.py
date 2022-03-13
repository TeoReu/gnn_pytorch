import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from misc.dataset import Dataset
from utils import buildGraph, draw_graph
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt

save_dir = 'beautiful_plots'
dataset = Dataset('ER', str(1))

s1_data_test = dataset.test['rnanp']
s2_data_test = dataset.test['cnanp']
s3_data_test = dataset.test['clin']

train_data_1 = buildGraph(torch.from_numpy(s1_data_test), 5)
train_data_2 = buildGraph(torch.from_numpy(s2_data_test), 5)
train_data_3 = buildGraph(torch.from_numpy(s3_data_test), 5)


def draw_graphs(df, name):
    draw_graph(df, name)

def draw_corr(df, name):
    fig = plt.matshow(pd.DataFrame(preprocessing.normalize(df).T).corr())
    fig.figure.savefig(save_dir + '/' + name + '_corr' + '.png', bbox_inches='tight')


draw_graphs(train_data_1, 'rnanp')
draw_graphs(train_data_2, 'cnanp')
draw_graphs(train_data_3, 'clin')

draw_corr(s1_data_test, 'rnanp')
draw_corr(s2_data_test, 'cnanp')
draw_corr(s3_data_test, 'clin')

