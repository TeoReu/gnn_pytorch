import pandas as pd
import torch
import torch_geometric
import torch_geometric.transforms as T


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
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=True),
    ])

    data = torch_geometric.data.Data(pos=conc_input, dtype=torch.float)

    edge_index = torch_geometric.nn.knn_graph(data.pos, k=k, loop=False, flow='source_to_target')

    data = torch_geometric.data.Data(x=conc_input, edge_index=edge_index, pos=conc_input, dtype=torch.float)

    data_split = transform(data)

    return data_split, data