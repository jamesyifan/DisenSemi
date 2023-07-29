import time

import torch.nn
import torch.nn as nn
import torch_geometric.nn

from torch.nn import GRU

from torch_geometric.nn import Set2Set, MeanAggregation, GraphConv, GCNConv, SAGEConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool

# from torch_geometric.nn.conv import MessagePassing

from infomax import *


# class Encoder(torch.nn.Module):
#     def __init__(self, num_features, dim):
#         super(Encoder, self).__init__()
#         self.lin0 = torch.nn.Linear(num_features, dim)
#
#         nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
#         self.conv = NNConv(dim, dim, nn, aggr='mean', root_weight=False)
#         self.gru = GRU(dim, dim)
#
#         self.set2set = Set2Set(dim, processing_steps=3)
#         # self.lin1 = torch.nn.Linear(2 * dim, dim)
#         # self.lin2 = torch.nn.Linear(dim, 1)
#
#     def forward(self, data):
#         out = F.relu(self.lin0(data.x))
#         h = out.unsqueeze(0)
#
#         feat_map = []
#         for i in range(3):
#             m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
#             out, h = self.gru(m.unsqueeze(0), h)
#             out = out.squeeze(0)
#             # print(out.shape) : [num_node x dim]
#             feat_map.append(out)
#
#         out = self.set2set(out, data.batch)
#         return out, feat_map[-1]

class DisentangleEncoder(torch.nn.Module):
    def __init__(self, num_features, dim, n_factor, n_layer):
        super(DisentangleEncoder, self).__init__()
        self.n_factor = n_factor
        self.n_layer = n_layer
        self.n_dim = dim // self.n_factor

        # self.linear = torch.nn.Linear(num_features, self.n_dim)

        # self.att = torch.nn.ModuleList()
        # for factor_i in range(self.n_factor):
        #     self.att.append(torch.nn.Linear(self.n_dim*2, 1))

        # nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        # self.conv = NNConv(self.n_dim, self.n_dim, nn, aggr='mean', root_weight=False)

        self.linears = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.grus = torch.nn.ModuleList()
        self.set2sets = torch.nn.ModuleList()
        # self.linears_out = torch.nn.ModuleList()
        self.meanaggr = MeanAggregation()
        # self.l_projs = torch.nn.ModuleList()
        for factor_i in range(self.n_factor):
            self.linears.append(torch.nn.Linear(num_features, self.n_dim))
            # self.linears_out.append(torch.nn.Linear(2*self.n_dim, self.n_dim))
            for layer_j in range(self.n_layer):
                # self.convs.append(SAGEConv(self.n_dim, self.n_dim))
                # self.convs.append(GATConv(self.n_dim, self.n_dim))
                # self.convs.append(GINConv(nn=nn.Linear(self.n_dim, self.n_dim)))
                # self.convs.append(GCNConv(self.n_dim, self.n_dim))
                self.convs.append(GraphConv(self.n_dim, self.n_dim))

            # self.convs.append(GraphConv(self.n_dim, self.n_dim))
            self.grus.append(GRU(self.n_dim, self.n_dim))
            self.set2sets.append(Set2Set(self.n_dim, processing_steps=3))
            # self.l_projs.append(torch.nn.Linear(3*self.n_dim, 3*self.n_dim))
        # self.conv = GraphConv(self.n_dim, self.n_dim)

        # self.conv = GCNConv(self.n_dim, self.n_dim)
        # self.conv = SAGEConv(self.n_dim, self.n_dim)

        # self.gru = GRU(self.n_dim, self.n_dim)
        # self.set2set = Set2Set(self.n_dim, processing_steps=3)

    def forward(self, data, att):
        #
        # src, dst = data.edge_index
        # out_src = out[src]
        # out_dst = out[dst]
        # att_i_list = []
        feat_map_list = []
        out_list = []
        for factor_i in range(self.n_factor):
            # att_i = self.att[factor_i](torch.cat([out_src, out_dst], dim=-1))
            # att_i = torch.sigmoid(att_i)
            att_i = att[factor_i]
            # out = self.linear(data.x)
            out = self.linears[factor_i](data.x)
            h = out.unsqueeze(0)
            feat_map = []
            for i in range(self.n_layer):
                # print(out.shape)
                # print(data.edge_index.shape)
                # print(att_i.shape)
                # time.sleep(20)
                m = F.relu(self.convs[self.n_layer*factor_i+i](out, data.edge_index, att_i))
                # m = F.relu(self.convs[3*factor_i+i](out, data.edge_index))
                # m = F.relu(self.convs[factor_i](out, data.edge_index, att_i))
                # m = F.relu(self.conv(out, data.edge_index, att_i))
                # m = F.relu(self.conv(out, data.edge_index))
                out, h = self.grus[factor_i](m.unsqueeze(0), h)
                # out = self.convs[factor_i](out, data.edge_index, att_i)
                # out, h = self.gru(m.unsqueeze(0), h)
                out = out.squeeze(0)
                # out = F.dropout(out, 0.15, training=self.training)
                # print(out.shape) : [num_node x dim]
                # if i == 2:
                #     out = F.dropout(out, 0.3, training=self.training)
                feat_map.append(out)
            if self.n_layer == 0:
                feat_map_list.append(out)
            else:
                feat_map_list.append(feat_map[-1])
            # out = self.set2sets[factor_i](out, data.batch)
            # out = self.set2sets[factor_i](self.l_projs[factor_i](torch.cat(feat_map, dim=-1)), data.batch)
            # out = self.set2sets[factor_i](torch, data.batch)

            # out = self.meanaggr(out, data.batch)
            out = global_mean_pool(out, data.batch)
            # out = self.set2set(out, data.batch)
            # out = self.linears_out[factor_i](out)

            # feat_map_list.append(feat_map[-1])

            # feat_map_list.append(self.l_projs[factor_i](torch.cat(feat_map, dim=-1)))
            out_list.append(out)
        # out_list = torch.cat(out_list, dim=-1)
        return out_list, feat_map_list
            # att_i_list.append(att_i)
        # data.edge_attr = torch.stack([att_i_list], dim=-1)

# class PriorDiscriminator(nn.Module):
# def __init__(self, input_dim):
# super().__init__()
# self.l0 = nn.Linear(input_dim, input_dim)
# self.l1 = nn.Linear(input_dim, input_dim)
# self.l2 = nn.Linear(input_dim, 1)

# def forward(self, x):
# h = F.relu(self.l0(x))
# h = F.relu(self.l1(h))
# return torch.sigmoid(self.l2(h))

class FF(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class Net(torch.nn.Module):
    def __init__(self, num_features, dim, n_factor, n_layer, n_class, use_unsup_loss=False, separate_encoder=False):
        super(Net, self).__init__()

        self.n_dim = dim // n_factor
        self.n_factor = n_factor
        self.separate_encoder = separate_encoder

        self.local = True
        # self.prior = False

        # self.lin0 = torch.nn.Linear(num_features, dim)
        self.lin0s = torch.nn.ModuleList()
        self.att = torch.nn.ModuleList()
        for factor_i in range(self.n_factor):
            self.lin0s.append(torch.nn.Linear(num_features, dim))
            self.att.append(torch.nn.Linear(2*dim, 1))

        # self.encoder = Encoder(num_features, dim)
        self.encoder = DisentangleEncoder(num_features, dim, n_factor, n_layer)
        if separate_encoder:
            self.unsup_encoder = DisentangleEncoder(num_features, dim, n_factor, n_layer)
            self.ff1 = torch.nn.ModuleList()
            self.ff2 = torch.nn.ModuleList()
            for factor_i in range(n_factor):
                # self.ff1.append(FF(2*self.n_dim, self.n_dim))
                # self.ff2.append(FF(2*self.n_dim, self.n_dim))
                self.ff1.append(FF(self.n_dim, self.n_dim))
                self.ff2.append(FF(self.n_dim, self.n_dim))
            # self.ff1 = FF(2 * dim, dim)
            # self.ff2 = FF(2 * dim, dim)

        # self.fc1 = torch.nn.Linear(2 * dim, dim)
        # self.center_v = torch.rand((n_factor, 2*self.n_dim), requires_grad=True).cuda()
        self.center_v = torch.rand((n_factor, self.n_dim), requires_grad=True).cuda()

        # self.fc1 = torch.nn.Linear(2*dim, dim)
        self.fc1 = torch.nn.Linear(dim, dim)
        self.fc2 = torch.nn.Linear(dim, n_class)

        # self.fc1 = torch.nn.Linear(2*self.n_dim, self.n_dim)
        # self.fc2 = torch.nn.Linear(self.n_dim, n_class)

        self.local_d = torch.nn.ModuleList()
        self.global_d = torch.nn.ModuleList()
        if use_unsup_loss:
            for factor_i in range(n_factor):
                self.local_d.append(FF(self.n_dim, self.n_dim))
                # self.global_d.append(FF(2*self.n_dim, self.n_dim))
                self.global_d.append(FF(self.n_dim, self.n_dim))
            # self.local_d = FF(dim, dim)
            # self.global_d = FF(2 * dim, dim)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.n_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def disentangle_graph(self, data):
        # out = self.lin0(data.x)
        src, dst = data.edge_index
        # out_src = out[src]
        # out_dst = out[dst]
        att = []
        for factor_i in range(self.n_factor):
            # att_i = self.att[factor_i](torch.cat([out_src, out_dst], dim=-1))
            att_i = self.att[factor_i](torch.cat([self.lin0s[factor_i](data.x[src]),
                                                  self.lin0s[factor_i](data.x[dst])], dim=-1))
            # if self.n_factor == 1:
            #     att_i = torch.ones_like(att_i)
            # else:
            att_i = torch.sigmoid(6.0 * att_i)
            # att_i = torch.sigmoid(att_i)
            att.append(att_i.squeeze(-1))
        return att

    def forward(self, data):
        att = self.disentangle_graph(data)
        out_, M = self.encoder(data, att)

        out = torch.cat(out_, dim=-1)

        # out = torch.stack(out, dim=1)
        # T_c = 0.2
        # ck = F.normalize(self.center_v, dim=-1)
        # p_k_x_ = torch.einsum('bkd,kd->bk', F.normalize(out, dim=-1), ck)
        # p_k_x = F.softmax(p_k_x_ / T_c, dim=-1)
        # out = torch.einsum('bkd,bk->bd', out, p_k_x)

        out = F.relu(self.fc1(out))
        pred = self.fc2(out)
        # print(out.shape)
        # time.sleep(10)
        return pred, att, out_

    def unsup_loss(self, data):
        att = self.disentangle_graph(data)
        if self.separate_encoder:
            y, M = self.unsup_encoder(data, att)
            # y, M = self.encoder(data, att)
        else:
            y, M = self.encoder(data, att)

        g_enc = []
        l_enc = []
        for factor_i in range(self.n_factor):
            g_enc.append(self.global_d[factor_i](y[factor_i]))
            l_enc.append(self.local_d[factor_i](M[factor_i]))

        measure = 'JSD'
        if self.local:
            loss = local_global_loss_(l_enc, g_enc, self.n_factor, data.edge_index, data.batch, measure)
        return loss, y

    # def unsup_loss1(self, data):
    #     att = self.disentangle_graph(data)
    #     y, M = self.encoder(data, att)
    #     g_enc = []
    #     l_enc = []
    #     for factor_i in range(self.n_factor):
    #         g_enc.append(self.global_d[factor_i](y[factor_i]))
    #         l_enc.append(self.local_d[factor_i](M[factor_i]))
    #
    #     measure = 'JSD'
    #     if self.local:
    #         loss = local_global_loss_(l_enc, g_enc, self.n_factor, data.edge_index, data.batch, measure)
    #     return loss

    def unsup_sup_loss(self, data):
        att = self.disentangle_graph(data)
        y, M = self.encoder(data, att)
        y_, M_ = self.unsup_encoder(data, att)

        g_enc = []
        g_enc1 = []
        for factor_i in range(self.n_factor):
            g_enc.append(self.ff1[factor_i](y[factor_i]))
            g_enc1.append(self.ff2[factor_i](y_[factor_i]))

        y = torch.stack(y, dim=1)

        measure = 'JSD'
        T_c = 0.2
        ck = F.normalize(self.center_v, dim=-1)
        p_k_x_ = torch.einsum('bkd,kd->bk', F.normalize(y, dim=-1), ck)
        p_k_x = F.softmax(p_k_x_ / T_c, dim=-1)

        #variant1
        # p_k_x_ = torch.ones_like(p_k_x_).cuda()
        # p_k_x = F.softmax(p_k_x_ / T_c, dim=-1)

        #variant2
        # dist = torch.distributions.uniform.Uniform(0,1)
        # expanded_dist = dist.expand(torch.Size([p_k_x.shape[0]]))
        # p_k_x_ = expanded_dist.sample(torch.Size([p_k_x.shape[1]])).T.cuda()
        # p_k_x = F.softmax(p_k_x_ / T_c, dim=-1)

        loss = global_global_loss_(g_enc, g_enc1, self.n_factor, p_k_x, data.edge_index, data.batch, measure)

        return loss
