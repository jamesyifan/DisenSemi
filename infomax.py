import torch
import math
import torch.nn.functional as F
# from cortex_DIM.functions.gan_losses import get_positive_expectation, get_negative_expectation

def log_sum_exp(x, axis=None):
    """Log sum exp function
    Args:
        x: Input.
        axis: Axis over which to perform sum.
    Returns:
        torch.Tensor: log sum exp
    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y

def random_permute(X):
    """Randomly permutes a tensor.
    Args:
        X: Input tensor.
    Returns:
        torch.Tensor
    """
    X = X.transpose(1, 2)
    b = torch.rand((X.size(0), X.size(1))).cuda()
    idx = b.sort(0)[1]
    adx = torch.range(0, X.size(1) - 1).long()
    X = X[idx, adx[None, :]].transpose(1, 2)
    return X

def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        # raise_measure_error(measure)
        print(measure + 'Error')

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        # raise_measure_error(measure)
        print(measure + 'Error')

    if average:
        return Eq.mean()
    else:
        return Eq

def local_global_loss_(l_enc, g_enc, n_factor, edge_index, batch, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc[0].shape[0]
    num_nodes = l_enc[0].shape[0]
    # num_factor = len(g_enc)

    pos_mask = torch.zeros((num_nodes, num_graphs)).cuda()
    neg_mask = torch.ones((num_nodes, num_graphs)).cuda()
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    total_loss = 0.0
    # T = 0.2

    for factor_i in range(n_factor):
        res = torch.mm(l_enc[factor_i], g_enc[factor_i].t())
        E_pos = get_positive_expectation(res * pos_mask, measure, average=False)
        E_pos = (E_pos * pos_mask).sum() / pos_mask.sum()
        # E_pos = torch.exp(res/T)
        # E_pos = (E_pos * pos_mask).sum(dim=1)
        E_neg = get_negative_expectation(res * neg_mask, measure, average=False)
        E_neg = (E_neg * neg_mask).sum() / neg_mask.sum()
        # E_neg = torch.exp(res/T)
        # E_neg = (E_neg * neg_mask).sum(dim=1)
        total_loss += E_neg - E_pos


        # total_loss += - torch.log(E_pos/E_neg).mean()

    # disentangle loss

    # g_enc = torch.stack(g_enc, dim=1)
    # out = torch.matmul(g_enc, g_enc.transpose(1, 2))
    # deno = torch.max(out, dim=-1)[0]
    # deno.masked_fill_(deno == 0, 1)
    # out = out / deno.unsqueeze(-2)
    # i_m = torch.eye(g_enc.shape[-2]).unsqueeze(0).cuda()
    # orthLoss = torch.abs(out - i_m).view(g_enc.shape[0], -1).sum(-1).mean()
    # total_loss += orthLoss

    T = 0.2
    g_enc = torch.stack(g_enc, dim=1)
    B, K, d = g_enc.size()
    g_enc_abs = g_enc.norm(dim=-1)
    sim_matrix = torch.einsum('bid,bjd->bij', g_enc, g_enc) / (1e-8 + torch.einsum('bi,bj->bij', g_enc_abs, g_enc_abs))
    sim_matrix = torch.exp(sim_matrix / T)
    deno = F.normalize(sim_matrix, p=1, dim=-1)
    i_m = torch.eye(K).unsqueeze(0).cuda()
    orthLoss = torch.abs(deno - i_m).view(B, -1).sum(-1).mean()
    total_loss += orthLoss

    # E_pos_disen = get_positive_expectation(res_disen * pos_mask_disen, measure, average=False)
    # E_pos_disen = (E_pos_disen * pos_mask_disen).sum() / pos_mask_disen.sum()
    # E_neg_disen = get_negative_expectation(res_disen * neg_mask_disen, measure, average=False)
    # E_neg_disen = (E_neg_disen * neg_mask_disen).sum() / neg_mask_disen.sum()

    # total_loss += E_neg_disen - E_pos_disen

    return total_loss

# def global_global_loss_(g_enc, g_enc1, edge_index, batch, measure):
#     '''
#     Args:
#         g: Global features
#         g1: Global features.
#         measure: Type of f-divergence. For use with mode `fd`
#         mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
#     Returns:
#         torch.Tensor: Loss.
#     '''
#     num_graphs = g_enc.shape[0][0]
#
#     pos_mask = torch.eye(num_graphs).cuda()
#     neg_mask = 1 - pos_mask
#
#     res = torch.mm(g_enc, g_enc1.t())
#
#     E_pos = get_positive_expectation(res * pos_mask, measure, average=False)
#     E_pos = (E_pos * pos_mask).sum() / pos_mask.sum()
#     E_neg = get_negative_expectation(res * neg_mask, measure, average=False)
#     E_neg = (E_neg * neg_mask).sum() / neg_mask.sum()
#
#     return E_neg - E_pos

def global_global_loss_(g_enc, g_enc1, n_factor, p_k_x, edge_index, batch, measure):
    g_enc = torch.stack(g_enc, dim=1)
    g_enc1 = torch.stack(g_enc1, dim=1)
    # print(g_enc.size)
    B, K, d = g_enc.size()

    g_enc_abs = g_enc.norm(dim=-1)
    g_enc1_abs = g_enc1.norm(dim=-1)

    # g_enc = torch.reshape(g_enc, (B*K, d))
    # g_enc1 = torch.reshape(g_enc1, (B*K, d))
    # g_enc_abs = torch.squeeze(torch.reshape(g_enc_abs, (B*K, 1)), 1)
    # g_enc1_abs = torch.squeeze(torch.reshape(g_enc1_abs, (B*K, 1)), 1)

    # T = 0.2
    # sim_matrix = torch.einsum('ik,jk->ij', g_enc, g_enc1) / (1e-8 + torch.einsum('i,j->ij', g_enc_abs, g_enc1_abs))
    # sim_matrix = torch.exp(sim_matrix / T)
    # pos_sim = sim_matrix[range(B * K), range(B * K)]
    # score = pos_sim / (sim_matrix.sum(dim=-1) - pos_sim)
    # p_y_xk = score.view(B, K)

    # q_k = torch.einsum('bk,bk->bk', p_k_x, p_y_xk)
    # q_k = F.normalize(q_k, p=1, dim=-1)
    # elbo = q_k * (torch.log(p_k_x) + torch.log(p_y_xk) - torch.log(q_k))
    # loss = - elbo.view(-1).mean()

    T = 0.2
    sim_matrix = torch.einsum('ikd,jkd->ijk', g_enc, g_enc1) / (1e-8 + torch.einsum('ik,jk->ijk', g_enc_abs, g_enc1_abs))
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(B), range(B), :]
    score = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    p_y_xk = score.view(B, K)

    q_k = torch.einsum('bk,bk->bk', p_k_x, p_y_xk)
    q_k = F.normalize(q_k, p=1, dim=-1)
    elbo = q_k * (torch.log(p_k_x) + torch.log(p_y_xk) - torch.log(q_k))
    loss = - elbo.view(-1).mean()
    return loss


