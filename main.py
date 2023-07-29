import time

import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import remove_self_loops
# from feature_expansion import FeatureExpander
from datasets import get_dataset, k_fold
import numpy as np
import re
import os
import os.path as osp
import random
import torch
import torch_geometric.transforms as T
import sys
import pickle


class MyTransform(object):
    def __call__(self, data):
        # Specify target.
        # data.y = data.y[:, target]
        # print(data.pos)
        # print(data.y)
        # device = data.edge_index.device
        # if not data.x:
        #     data.x = torch.eye(data.num_nodes, dtype=torch.long, device=device)
        return data


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        # print(data.num_nodes)
        # print(edge_index)
        # print(edge_index.shape)
        # time.sleep(10)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

        # print(edge_index)
        # time.sleep(10)

        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


def train(epoch, use_unsup_loss):
    model.train()
    loss_all = 0
    sup_loss_all = 0
    unsup_loss_all = 0
    unsup_sup_loss_all = 0
    criterion = nn.CrossEntropyLoss()

    # print(args.separate_encoder)
    # print(args.use_unsup_loss)
    # i = 0
    # print(len(train_loader))
    # print(len(unsup_train_loader))
    # i = 0

    if use_unsup_loss:
        for data, data2 in zip(train_loader, unsup_train_loader):
            data = data.to(device)
            data2 = data2.to(device)
            # print(i)
            # i+=1
            # if i == 0:
            #     import time
            #     print(data[10])
            #     print(data2[10])
            #     time.sleep(5)
            # i += 1

            optimizer.zero_grad()

            # sup_loss = F.mse_loss(model(data), data.y)
            pred, _, _ = model(data)
            sup_loss = criterion(pred, data.y)
            unsup_loss, _ = model.unsup_loss(data2)

            if separate_encoder:
                unsup_sup_loss = model.unsup_sup_loss(data2)
                loss = sup_loss + unsup_loss * lamda1 + unsup_sup_loss * lamda2
                # if epoch <= 200:
                #     loss = unsup_loss #+ model.unsup_loss1(data2)
                # else:
                #     loss = sup_loss + unsup_loss * lamda1 + unsup_sup_loss * lamda2
            else:
                loss = sup_loss + unsup_loss * lamda1

            loss.backward()

            sup_loss_all += sup_loss.item()
            unsup_loss_all += unsup_loss.item()
            if separate_encoder:
                unsup_sup_loss_all += unsup_sup_loss.item()
            loss_all += loss.item() * data.num_graphs

            optimizer.step()

        if separate_encoder:
            print(sup_loss_all, unsup_loss_all, unsup_sup_loss_all)
        else:
            print(sup_loss_all, unsup_loss_all)
        return loss_all / len(train_loader.dataset)
    else:
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            # sup_loss = F.mse_loss(model(data), data.y)
            pred, _, _ = model(data)
            sup_loss = criterion(pred, data.y)
            loss = sup_loss

            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()

        return loss_all / len(train_loader.dataset)


def valid(loader, val_idx):
    model.eval()
    error = 0
    # correct = 0
    criterion = nn.CrossEntropyLoss()
    outs_s = {}
    outs_u = {}
    i = 0
    for data in loader:
        data = data.to(device)
        idx = val_idx[i].item()
        # error += (model(data) * std - data.y * std).abs().sum().item()  # MAE
        # error += (model(data) - data.y).abs().sum().item()  # MAE
        with torch.no_grad():
            pred, _, out_s = model(data)
            _, out_u = model.unsup_loss(data)
            outs_s[idx] = out_s
            outs_u[idx] = out_u
        error += criterion(pred, data.y).item()
        i += 1
        # with torch.no_grad():
        #     pred = model(data).max(1)[1]
        # correct += pred.eq(data.y.view(-1)).sum().item()
    return error / len(loader.dataset), outs_s, outs_u
    # return correct / len(loader.dataset)

def test(loader, test_idx):
    model.eval()
    correct = 0
    atts = {}
    i = 0
    outs_s = {}
    outs_u = {}
    for data in loader:
        data = data.to(device)
        idx = test_idx[i].item()
        # p = model(data)
        # _, pred = p.max(dim=1)
        # print(pred)
        # print(data.y)
        # import time
        # time.sleep(5)
        # correct += pred.eq(data.y).sum().item()
        with torch.no_grad():
            pred, att, out_s = model(data)
            _, out_u = model.unsup_loss(data)
            atts[idx] = att
            outs_s[idx] = out_s
            outs_u[idx] = out_u
            pred = pred.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        i += 1
    return correct/len(loader.dataset), atts, outs_s, outs_u

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def print_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    sys.stdout.flush()

if __name__ == '__main__':
    # seed_everything()
    torch.multiprocessing.set_sharing_strategy('file_system')
    from model import Net
    from arguments import arg_parse

    args = arg_parse()

    # ============
    # Hyperparameters
    # ============
    # target = args.target
    dim = 128
    epochs = 200
    folds = 10

    batch_size = args.batch_size
    n_factor = args.n_factor
    n_layer = args.n_layer
    lamda1 = args.lamda1
    lamda2 = args.lamda2
    use_unsup_loss = args.use_unsup_loss
    separate_encoder = args.separate_encoder

    print(use_unsup_loss)
    print(separate_encoder)

    # path = osp.join(osp.dirname(osp.realpath(__file__)), '../data', 'TUDataset')
    # dataset = args.dataset

    # transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
    # transform = T.Compose([MyTransform(), Complete()])

    # transform = Complete()

    # feat_str = 'deg+odeg100'
    # degree = feat_str.find("deg") >= 0
    # onehot_maxdeg = re.findall("odeg(\d+)", feat_str)
    # onehot_maxdeg = int(onehot_maxdeg[0]) if onehot_maxdeg else None
    #
    # k = re.findall("an{0,1}k(\d+)", feat_str)
    # k = int(k[0]) if k else 0
    # groupd = re.findall("groupd(\d+)", feat_str)
    # groupd = int(groupd[0]) if groupd else 0
    # remove_edges = re.findall("re(\w+)", feat_str)
    # remove_edges = remove_edges[0] if remove_edges else 'none'
    # edge_noises_add = re.findall("randa([\d\.]+)", feat_str)
    # edge_noises_add = float(edge_noises_add[0]) if edge_noises_add else 0
    # edge_noises_delete = re.findall("randd([\d\.]+)", feat_str)
    # edge_noises_delete = float(
    #     edge_noises_delete[0]) if edge_noises_delete else 0
    # centrality = feat_str.find("cent") >= 0
    # coord = feat_str.find("coord") >= 0
    #
    # pre_transform = FeatureExpander(
    #     degree=degree, onehot_maxdeg=onehot_maxdeg, AK=k,
    #     centrality=centrality, remove_edges=remove_edges,
    #     edge_noises_add=edge_noises_add, edge_noises_delete=edge_noises_delete,
    #     group_degree=groupd).transform

    # dataset = TUDataset(path, dataset, transform=pre_transform).shuffle()
    feat_str = 'deg+odeg100'
    if args.dataset in ['DD','REDDIT-BINARY', 'REDDIT-MULTI-5K']:
        feat_str = 'deg+odeg10'
    dataset = get_dataset(args.dataset, feat_str, root=args.data_root)
    # transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
    # transform = T.Compose([MyTransform(), Complete()])
    # print(osp.join(args.data_root, args.dataset))
    # dataset = TUDataset(osp.join(args.data_root, args.dataset), args.dataset, transform=transform, use_node_attr=True)
    # print(dataset.num_node_features)
    print('---')
    print(feat_str)
    # transform = T.Compose([MyTransform(), Complete()])
    # transform = T.Compose([MyTransform()])
    # dataset = TUDataset(path, args.dataset, transform=transform)
    # dataset = TUDataset(path, args.dataset)

    dataset_size = dataset.len()
    n_class = dataset.num_classes
    print('num_features : {}\n'.format(dataset.num_features))
    print('num_classes : {}\n'.format(n_class))
    print('num_factors : {}\n'.format(n_factor))
    print('num_layers : {}\n'.format(n_layer))

    # Normalize targets to mean = 0 and std = 1.
    # mean = dataset.data.y[:, target].mean().item()
    # std = dataset.data.y[:, target].std().item()
    # dataset.data.y[:, target] = (dataset.data.y[:, target] - mean) / std

    # print(type(dataset[0]))
    # print(type(dataset.data.x)) #tensor
    # print(type(dataset.data.y)) #tensor

    # Split datasets.
    # valid_size = int(dataset_size*0.1)
    # test_size = int(dataset_size*0.1)
    # train_size = int(dataset_size*args.train_ratio)
    # test_dataset = dataset[:test_size]
    # val_dataset = dataset[test_size:test_size+valid_size]
    # train_dataset = dataset[test_size+valid_size:test_size+valid_size+train_size]

    # print(test_dataset[0])
    # print(val_dataset[0])
    # print(train_dataset[0])
    # print(valid_size)

    # test_loader = DataLoader(test_dataset, batch_size=batch_size)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #
    # if use_unsup_loss:
    #     unsup_train_dataset = dataset[test_size+valid_size:]
    #     unsup_train_loader = DataLoader(unsup_train_dataset, batch_size=batch_size, shuffle=True)
    #
    #     print(len(train_dataset), len(val_dataset), len(test_dataset), len(unsup_train_dataset))
    # else:
    #     print(len(train_dataset), len(val_dataset), len(test_dataset))

    test_accs = []
    print('n_percents : {}\n'.format(args.n_percents))
    # print('n_percents:'+str(args.n_percents))
    for fold, (train_idx, test_idx, val_idx, train_idx_unlabel) in enumerate(
            zip(*k_fold(dataset, folds, args.epoch_select, n_percents=int(args.n_percents)))):
        # train_idx[train_idx < 0] = train_idx[0]
        # train_idx[train_idx >= len(dataset)] = train_idx[0]
        # test_idx[test_idx < 0] = test_idx[0]
        # test_idx[test_idx >= len(dataset)] = test_idx[0]
        # val_idx[val_idx < 0] = val_idx[0]
        # val_idx[val_idx >= len(dataset)] = val_idx[0]

        print(fold)
        # if fold != 0:
        #     continue
        # if fold < 8:
        #     continue

        # else:
        #     print(train_idx)
        #     print(test_idx)
        #     print(val_idx)
        #     continue
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]
        train_dataset_unlabel = dataset[train_idx_unlabel]

        print(len(train_dataset))
        print(len(train_dataset_unlabel))

        # train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=64)
        # val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=64)
        # test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=64)
        # unsup_train_loader = DataLoader(train_dataset_unlabel, batch_size*4, shuffle=True, num_workers=64)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=96)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=96)
        # test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=96)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=96)
        unsup_train_loader = DataLoader(train_dataset_unlabel, batch_size * 4, shuffle=True, num_workers=96)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(dataset.num_features)
        model = Net(dataset.num_features, dim, n_factor, n_layer, n_class, use_unsup_loss, separate_encoder).to(device)
        # if fold == 0:
        #     print_weights(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.95, patience=5, min_lr=0.000001)

        # val_error = valid(val_loader)
        # test_acc = test(test_loader)
        # print('Epoch: {:03d}, Validation Error: {:.4f}, Test ACC: {:.4f},'.format(0, val_error, test_acc))

        # best_val_error = None
        # best_test_acc = None
        best_test_acc = 0
        for epoch in range(epochs):
            lr = scheduler.optimizer.param_groups[0]['lr']
            loss = train(epoch, use_unsup_loss)
            val_error, outs_s_val, outs_u_val = valid(val_loader, val_idx)
            scheduler.step(val_error)
            test_acc, atts, outs_s_test, outs_u_test = test(test_loader, test_idx)

            # if best_test_acc is None or test_acc >= best_test_acc:
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), 'models/' + args.dataset + '_fold' + str(fold) + 'us.pt')
                pickle.dump(atts, open('models/atts_'+str(fold)+'.pkl', 'wb'))
                pickle.dump(test_idx, open('models/test_idx_'+str(fold)+'.pkl', 'wb'))
                pickle.dump(outs_s_test, open('models/outs_s_test_'+str(fold)+'.pkl', 'wb'))
                pickle.dump(outs_u_test, open('models/outs_u_test_'+str(fold)+'.pkl', 'wb'))
                pickle.dump(outs_s_val, open('models/outs_s_val_' + str(fold) + '.pkl', 'wb'))
                pickle.dump(outs_u_val, open('models/outs_u_val_' + str(fold) + '.pkl', 'wb'))

            # if best_val_error is None or val_error <= best_val_error:
            #     test_error = test(test_loader)
            #     best_val_error = val_error

            # best_val_error = val_error

            print('Epoch: {:03d}, LR: {:4f}, Loss: {:.4f}, Validation Loss: {:.4f}, '
                  'Test ACC: {:.4f},'.format(epoch, lr, loss, val_error, test_acc))

        test_accs.append(best_test_acc)
        print(test_accs)
            # test_accs.append(test_acc)

    # test_acc = torch.tensor(test_accs)
    # test_acc = test_acc.view(10, epochs)
    # _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
    # selected_epoch = selected_epoch.repeat(10)
    #
    # test_acc = test_acc[torch.arange(10, dtype=torch.long), selected_epoch]
    # test_acc_mean = test_acc.mean().item()
    # test_acc_std = test_acc.std().item()
    # print(test_acc_mean, test_acc_std)
    # print(test_accs)
    # # print(sum(test_accs) / len(test_accs))
    print(np.mean(test_accs))
    print(np.std(test_accs, ddof=1))

        # with open('supervised.log', 'a+') as f:
        #     f.write('{},{},{},{},{},{},{},{}\n'.format(args.train_ratio, use_unsup_loss, separate_encoder,
        #                                                    args.lamda1, args.lamda2, args.weight_decay, val_error, test_acc))