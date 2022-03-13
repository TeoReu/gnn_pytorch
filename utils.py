import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch_geometric
import torch_geometric.transforms as T
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx


def draw_graph(data):
    g = torch_geometric.utils.to_networkx(data)
    # g.add_edges_from(data.edge_index)
    # pretty colours: #db9cc3, #d17c62
    node_color = '#00b4d9'
    nx.draw_spring(g.to_undirected(reciprocal=False, as_view=False), with_labels=False, node_size=10, node_color= node_color)
    plt.savefig("fin_report.png")

def draw_graph(data, s):
    fig = plt.figure()
    save_dir = 'beautiful_plots'

    g = torch_geometric.utils.to_networkx(data)
    # g.add_edges_from(data.edge_index)
    # pretty colours: #db9cc3, #d17c62
    if s == 'rnanp':
        node_color = '#00b4d9'
    elif s == 'clin':
        node_color = '#db9cc3'
    else:
        node_color = '#d17c62'
    nx.draw_spring(g.to_undirected(reciprocal=False, as_view=False), with_labels=False, node_size=10, node_color= node_color)
    fig.savefig(save_dir + '/' + s + '_graph' + '.png')


def load_node_csv(path, encoders=None, **kwargs):
    df = pd.read_csv(path, sep='\t', **kwargs)
    # mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x


class GenresEncoder(object):
    def __call__(self, df):
        genres = df.values
        mapping = {genre: i for i, genre in enumerate(genres)}

        j = 0
        for i in mapping.keys():
            mapping[i] = j
            j += 1

        print(mapping)
        x = torch.zeros(len(df), 1)

        for i, col in enumerate(df.values):
            x[i] = mapping[col]

        print(x)
        return x


class NoEncoding(object):
    def __call__(self, df):
        x = torch.zeros(len(df), 1)

        for i, col in enumerate(df.values):
            x[i] = col

        return x


def encoded_anno(address):
    cell_x = load_node_csv(
        'AML_data/BM1anno.txt', index_col='Cell', encoders={
            'CellType': GenresEncoder(),
            'PredictionRF2': GenresEncoder(),
            'PredictionRefined': GenresEncoder(),
            'CyclingBinary': GenresEncoder(),
            'AlignedToGenome': NoEncoding(),
            'AlignedToTranscriptome': NoEncoding(),
            'TranscriptomeUMIs': NoEncoding(),
        })

    return cell_x


def MMD(x, y, device, kernel="multiscale"):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
        :param device: cpu or cuda
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)


def buildGraph(conc_input, k, device):
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_test=0.2, is_undirected=True,
                          split_labels=True, add_negative_train_samples=True),
    ])

    data = torch_geometric.data.Data(pos=conc_input, dtype=torch.float)

    edge_index = torch_geometric.nn.knn_graph(data.pos, k=k, loop=False, flow='source_to_target')

    data = torch_geometric.data.Data(x=conc_input, edge_index=edge_index, pos=conc_input, dtype=torch.float)

    data_split = transform(data)

    return data_split, data


def buildGraph(conc_input, k):
    transform = T.Compose([
        T.NormalizeFeatures()
    ])

    data = torch_geometric.data.Data(pos=conc_input, dtype=torch.float)

    edge_index = torch_geometric.nn.knn_graph(data.pos, k=k, loop=False, flow='source_to_target')

    data = torch_geometric.data.Data(x=conc_input, edge_index=edge_index, pos=conc_input, dtype=torch.float)

    # data_split = transform(data)

    return data


def build_simplex(g1, g2):
    g1_num_ft = g1.x.shape[1]
    g2_num_ft = g2.x.shape[1]

    g_num_nodes = g1.x.shape[0]

    transform = T.Compose([
        T.NormalizeFeatures(),
    ])

    t = torch.zeros(2 * g_num_nodes, g1_num_ft + g2_num_ft)
    t[:g_num_nodes, :g1_num_ft] = g1.x
    t[g_num_nodes:2 * g_num_nodes, g1_num_ft: g1_num_ft + g2_num_ft] = g2.x
    x = t

    new_edge_g2 = g2.edge_index + g_num_nodes * torch.ones(g2.edge_index.shape)
    print(new_edge_g2.shape)
    arr = np.zeros((2, 2 * g_num_nodes), dtype=int)
    print(arr.shape)

    for j in range(g_num_nodes):
        arr[0][j] = j
        arr[1][j] = j + g_num_nodes

    edge_index = torch.concat((g1.edge_index, new_edge_g2, torch.from_numpy(arr)), 1).type(torch.LongTensor)
    new_graph = torch_geometric.data.Data(x=x, edge_index=edge_index, pos=x)

    new_graph = transform(new_graph)

    return new_graph


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values,


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

# def buildLayeredGraphs(g_1, g_2):
