import dgl
import torch
import numpy as np
from pathlib import Path
from torch_geometric.datasets import LINKXDataset
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.data import Data
from utils import *

homo_datasets = ['cora', 'citeseer', 'pubmed', 'a-photo', 'a-computer', 'cs', 'phy', 'cora-full']
hetero_datasets = ['texas', 'cornell', 'wisconsin', 'chameleon', 'squirrel', 'actor', 'penn94', 'arxiv-year', 'arating', 'romanempire']


def _data_root(param, subdir=None):
    root = Path(param.get('data_dir', './data')).expanduser()
    if subdir is not None:
        root = root / subdir
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_dgl_dataset(dataset_cls, param, *args, **kwargs):
    """Load a DGL dataset under a clean project-local data directory."""
    root = _data_root(param, 'dgl')
    return dataset_cls(*args, raw_dir=str(root), **kwargs)[0]


def load_data(dataset, param):
    """Load graph data.

    All downloaded files are stored under ``param['data_dir']``. DGL datasets
    are placed in ``data/dgl/``, PyG datasets in ``data/pyg/``, and OGB
    datasets in ``data/ogb/`` by default.
    """
    if dataset == 'cora':
        g = _load_dgl_dataset(dgl.data.CoraGraphDataset, param)
    elif dataset == 'citeseer':
        g = _load_dgl_dataset(dgl.data.CiteseerGraphDataset, param)
    elif dataset == 'pubmed':
        g = _load_dgl_dataset(dgl.data.PubmedGraphDataset, param)
    elif dataset == 'a-photo':
        g = _load_dgl_dataset(dgl.data.AmazonCoBuyPhotoDataset, param)
    elif dataset == 'a-computer':
        g = _load_dgl_dataset(dgl.data.AmazonCoBuyComputerDataset, param)
    elif dataset == 'cs':
        g = _load_dgl_dataset(dgl.data.CoauthorCSDataset, param)
    elif dataset == 'phy':
        g = _load_dgl_dataset(dgl.data.CoauthorPhysicsDataset, param)
    elif dataset == 'cora-full':
        g = _load_dgl_dataset(dgl.data.CoraFullDataset, param)
    elif dataset == 'texas':
        g = _load_dgl_dataset(dgl.data.TexasDataset, param)
    elif dataset == 'cornell':
        g = _load_dgl_dataset(dgl.data.CornellDataset, param)
    elif dataset == 'wisconsin':
        g = _load_dgl_dataset(dgl.data.WisconsinDataset, param)
    elif dataset == 'squirrel':
        g = _load_dgl_dataset(dgl.data.SquirrelDataset, param)
    elif dataset == 'actor':
        g = _load_dgl_dataset(dgl.data.ActorDataset, param)
    elif dataset == 'chameleon':
        g = _load_dgl_dataset(dgl.data.ChameleonDataset, param)
    elif dataset == 'arating':
        g = _load_dgl_dataset(dgl.data.AmazonRatingsDataset, param)
    elif dataset == 'romanempire':
        g = _load_dgl_dataset(dgl.data.RomanEmpireDataset, param)
    elif dataset == 'penn94':
        graph = LINKXDataset(root=str(_data_root(param, 'pyg')), name='penn94')[0]
        g = to_dgl(graph)
    elif dataset == 'arxiv-year':
        data = NodePropPredDataset(name='ogbn-arxiv', root=str(_data_root(param, 'ogb')))
        label = even_quantile_labels(data.graph['node_year'].flatten(), 5, verbose=False)
        data.label = torch.as_tensor(label).reshape(-1, 1)
        graph = Data(
            x=torch.from_numpy(data.graph['node_feat']).float(),
            edge_index=torch.from_numpy(data.graph['edge_index']).long(),
            y=data.label.long().squeeze(1)
        )
        g = to_dgl(graph)
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')

    labels = g.ndata['label']
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, labels


def get_mask(g, param):
    if param['dataset'] in hetero_datasets:
        if param['dataset'] == 'arxiv-year':
            train_mask, val_mask, test_mask = rand_train_test_idx(g.ndata['label'], seed=param.get('seed', 2))
        else:
            split_id = int(param.get('split_id', 0))
            train_mask = g.ndata['train_mask'][:, split_id]
            val_mask = g.ndata['val_mask'][:, split_id]
            test_mask = g.ndata['test_mask'][:, split_id]
    elif param['dataset'] in ['cora', 'citeseer', 'pubmed']:
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
    else:
        labels = g.ndata['label']
        n_class = int(labels.max().item() + 1)
        nrange = torch.arange(labels.shape[0]).to(device)
        train_mask = torch.zeros(labels.shape[0], dtype=torch.bool).to(device)

        for y in range(n_class):
            label_mask = (g.ndata['label'] == y)
            train_mask[nrange[label_mask][torch.randperm(label_mask.sum())[:20]]] = True

        val_mask = ~train_mask
        val_mask[nrange[val_mask][torch.randperm(val_mask.sum())[500:]]] = False
        test_mask = ~(train_mask | val_mask)
        test_mask[nrange[test_mask][torch.randperm(test_mask.sum())[1000:]]] = False

    return train_mask.nonzero()[:, 0], val_mask.nonzero()[:, 0], test_mask.nonzero()[:, 0]


def load_out_t(out_t_dir):
    out_t_dir = Path(out_t_dir)
    return torch.from_numpy(np.load(out_t_dir / 'out.npz')['arr_0'])
