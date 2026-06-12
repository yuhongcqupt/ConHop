import os
import dgl
import torch
import shutil
import random
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_writable(path, overwrite=True):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    elif overwrite:
        shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def idx_split(idx, ratio, seed=0):
    set_seed(seed)
    n = len(idx)
    cut = int(n * ratio)
    idx_idx_shuffle = torch.randperm(n)

    idx1_idx, idx2_idx = idx_idx_shuffle[:cut], idx_idx_shuffle[cut:]
    idx1, idx2 = idx[idx1_idx], idx[idx2_idx]

    return idx1, idx2


def graph_split(idx_train, idx_val, idx_test, rate, seed):
    idx_test_ind, idx_test_tran = idx_split(idx_test, rate, seed)

    idx_obs = torch.cat([idx_train, idx_val, idx_test_tran])
    N1, N2 = idx_train.shape[0], idx_val.shape[0]
    obs_idx_all = torch.arange(idx_obs.shape[0])
    obs_idx_train = obs_idx_all[:N1]
    obs_idx_val = obs_idx_all[N1:N1 + N2]
    obs_idx_test = obs_idx_all[N1 + N2:]

    return obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind


def get_evaluator(dataset):
    def evaluator(out, labels):
        pred = out.argmax(1)
        return pred.eq(labels).float().mean().item()
    return evaluator


def to_dgl(data):
    """Convert a PyG Data or HeteroData object to a DGL graph."""
    import dgl
    from torch_geometric.data import Data, HeteroData

    if isinstance(data, Data):
        if data.edge_index is not None:
            row, col = data.edge_index
        elif 'adj' in data:
            row, col, _ = data.adj.coo()
        elif 'adj_t' in data:
            row, col, _ = data.adj_t.t().coo()
        else:
            row, col = [], []

        g = dgl.graph((row, col), num_nodes=data.num_nodes)

        for attr in data.node_attrs():
            if attr == 'x':
                g.ndata['feat'] = data[attr]
            elif attr == 'y':
                g.ndata['label'] = data[attr]
            else:
                g.ndata[attr] = data[attr]
        for attr in data.edge_attrs():
            if attr in ['edge_index', 'adj_t']:
                continue
            g.edata[attr] = data[attr]

        return g

    if isinstance(data, HeteroData):
        data_dict = {}
        for edge_type, edge_store in data.edge_items():
            if edge_store.get('edge_index') is not None:
                row, col = edge_store.edge_index
            else:
                row, col, _ = edge_store['adj_t'].t().coo()
            data_dict[edge_type] = (row, col)

        g = dgl.heterograph(data_dict)

        for node_type, node_store in data.node_items():
            for attr, value in node_store.items():
                g.nodes[node_type].data[attr] = value

        for edge_type, edge_store in data.edge_items():
            for attr, value in edge_store.items():
                if attr in ['edge_index', 'adj_t']:
                    continue
                g.edges[edge_type].data[attr] = value

        return g

    raise ValueError(f"Invalid data type (got '{type(data)}')")


def even_quantile_labels(vals, nclasses, verbose=True):
    """Partition values into equal-frequency classes by quantiles."""
    label = np.ones(vals.shape[0]) * -1
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.quantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})')
    return label


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True, seed=1234):
    """Randomly split labels into train/validation/test masks."""
    train_idx, test_idx = train_test_split(np.arange(len(label)), train_size=train_prop, random_state=seed)
    val_idx, test_idx = train_test_split(test_idx, train_size=valid_prop / (1 - train_prop), random_state=seed)

    train_mask = torch.zeros(len(label), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(len(label), dtype=torch.bool)
    val_mask[val_idx] = True
    test_mask = torch.zeros(len(label), dtype=torch.bool)
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def get_train_config(config_path, model_name, dataset):
    """Load optional YAML hyperparameter configuration.

    The expected structure is:
        global:
          ...
        dataset_name:
          MLP:
            ...

    Missing dataset/model sections are allowed; in that case only the global
    configuration is returned.
    """
    config_path = Path(config_path)
    with config_path.open('r', encoding='utf-8') as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader) or {}

    global_config = full_config.get('global', {}) or {}
    dataset_config = full_config.get(dataset, {}) or {}
    model_config = dataset_config.get(model_name, {}) or {}

    specific_config = dict(global_config, **model_config)
    specific_config['model_name'] = model_name
    return specific_config
