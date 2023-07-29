import os.path as osp
import re

import torch
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from sklearn.model_selection import StratifiedKFold
from feature_expansion import FeatureExpander
# from image_dataset import ImageDataset
from torch_geometric.datasets import TUDataset
from tu_dataset import TUDatasetExt


def get_dataset(name, feat_str="deg+ak3+reall", root=None):
    if root is None or root == '':
        path = osp.join(osp.expanduser('~'), 'pyG_data', name)
    else:
        path = osp.join(root, name)
    degree = feat_str.find("deg") >= 0
    onehot_maxdeg = re.findall("odeg(\d+)", feat_str)
    onehot_maxdeg = int(onehot_maxdeg[0]) if onehot_maxdeg else None
    k = re.findall("an{0,1}k(\d+)", feat_str)
    k = int(k[0]) if k else 0
    groupd = re.findall("groupd(\d+)", feat_str)
    groupd = int(groupd[0]) if groupd else 0
    remove_edges = re.findall("re(\w+)", feat_str)
    remove_edges = remove_edges[0] if remove_edges else 'none'
    edge_noises_add = re.findall("randa([\d\.]+)", feat_str)
    edge_noises_add = float(edge_noises_add[0]) if edge_noises_add else 0
    edge_noises_delete = re.findall("randd([\d\.]+)", feat_str)
    edge_noises_delete = float(
        edge_noises_delete[0]) if edge_noises_delete else 0
    centrality = feat_str.find("cent") >= 0
    coord = feat_str.find("coord") >= 0

    pre_transform = FeatureExpander(
        degree=degree, onehot_maxdeg=onehot_maxdeg, AK=k,
        centrality=centrality, remove_edges=remove_edges,
        edge_noises_add=edge_noises_add, edge_noises_delete=edge_noises_delete,
        group_degree=groupd).transform

    # print(aug, aug_ratio)
    # if 'MNIST' in name or 'CIFAR' in name:
    #     if name == 'MNIST_SUPERPIXEL':
    #         train_dataset = MNISTSuperpixels(path, True,
    #             pre_transform=pre_transform, transform=T.Cartesian())
    #         test_dataset = MNISTSuperpixels(path, False,
    #             pre_transform=pre_transform, transform=T.Cartesian())
    #     else:
    #         train_dataset = ImageDataset(path, name, True,
    #             pre_transform=pre_transform, coord=coord,
    #             processed_file_prefix="data_%s" % feat_str)
    #         test_dataset = ImageDataset(path, name, False,
    #             pre_transform=pre_transform, coord=coord,
    #             processed_file_prefix="data_%s" % feat_str)
    #     dataset = (train_dataset, test_dataset)
    # else:
    #     dataset = TUDatasetExt(
    #         path, name, pre_transform=pre_transform,
    #         use_node_attr=True, processed_filename="data_%s.pt" % feat_str, aug=aug, aug_ratio=aug_ratio)
    #
    #     dataset.data.edge_attr = None

    # dataset = TUDataset(root, name, transform=pre_transform).shuffle()
    # dataset = TUDataset(root, name, pre_transform=pre_transform)
    print(path)
    # dataset = TUDataset(path, name, use_node_attr=True)
    dataset = TUDataset(path, name, pre_transform=pre_transform, use_node_attr=True)
    # dataset = TUDataset(path, name, use_node_attr=True)
    # dataset = TUDatasetExt(
    #     path, name, pre_transform=pre_transform,
    #     use_node_attr=True, processed_filename="data_%s.pt" % feat_str)

    # dataset.data.edge_attr = None

    return dataset


def k_fold(dataset, folds, epoch_select, n_percents=3):
    n_splits = folds - 2

    if n_percents == 10:
        all_indices = torch.arange(0, len(dataset), 1, dtype=torch.long)
        return [all_indices], [all_indices], [all_indices], [all_indices]

    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices, train_indices_unlabel = [], [], []
    save_test, save_train, save_val, save_train_unlabel = [], [], [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))
        if len(save_test) > 0 and len(list(idx)) < len(save_test[0]):
            save_test.append(list(idx) + [list(idx)[-1]])
        else:
            save_test.append(list(idx))

    if epoch_select == 'test_max':
        val_indices = [test_indices[i] for i in range(folds)]
        save_val = [save_test[i] for i in range(folds)]
        n_splits += 1
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)]
        save_val = [save_test[i - 1] for i in range(folds)]

    skf_semi = StratifiedKFold(n_splits, shuffle=True, random_state=12345)
    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        idx_train_all = train_mask.nonzero(as_tuple=False).view(-1)

        idx_train = []
        for _, idx in skf_semi.split(torch.zeros(idx_train_all.size()[0]), dataset.data.y[idx_train_all]):
            idx_train.append(idx_train_all[idx])
            if len(idx_train) >= n_percents:
                break
        idx_train = torch.concat(idx_train).view(-1)

        train_indices.append(idx_train)
        cur_idx = list(idx_train.cpu().detach().numpy())
        if i > 0 and len(cur_idx) < len(save_train[0]):
            save_train.append(cur_idx + [cur_idx[-1]])
        else:
            save_train.append(cur_idx)

        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        # train_mask[train_indices[i].long()] = 0
        idx_train_unlabel = train_mask.nonzero(as_tuple=False).view(-1)
        train_indices_unlabel.append(idx_train_unlabel)  # idx_train_all, idx_train_unlabel
        cur_idx = list(idx_train_unlabel.cpu().detach().numpy())
        if i > 0 and len(cur_idx) < len(save_train_unlabel[0]):
            save_train_unlabel.append(cur_idx + [cur_idx[-1]])
        else:
            save_train_unlabel.append(cur_idx)

    print("Train:", len(train_indices[i]), "Val:", len(val_indices[i]), "Test:", len(test_indices[i]))

    return train_indices, test_indices, val_indices, train_indices_unlabel

