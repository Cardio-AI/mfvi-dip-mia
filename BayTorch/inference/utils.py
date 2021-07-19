import torch
import torch.nn as nn
from torch.nn.utils import prune
from torch.nn.functional import softplus

import matplotlib.pyplot as plt
import numpy as np

from ..modules import LinearRT, LinearLRT, Conv2dRT, Conv2dLRT

def uncert_regression_gal(img_list: torch.Tensor, reduction: str = 'mean'):
    img_list = torch.cat(img_list, dim=0)
    mean = img_list[:,:-1].mean(dim=0, keepdim=True)
    ale = img_list[:,-1:].mean(dim=0, keepdim=True)
    epi = torch.var(img_list[:,:-1], dim=0, keepdim=True)
    #if epi.shape[1] == 3:
    epi = epi.mean(dim=1, keepdim=True)
    uncert = ale + epi
    if reduction == 'mean':
        return ale.mean().item(), epi.mean().item(), uncert.mean().item()
    elif reduction == 'sum':
        return ale.sum().item(), epi.sum().item(), uncert.sum().item()
    else:
        return ale.detach(), epi.detach(), uncert.detach()

def uncert_classification_kwon(p_hat, var='sum'):
    p_mean = torch.mean(p_hat, dim=0)
    ale = torch.mean(p_hat*(1-p_hat), dim=0)
    epi = torch.mean(p_hat**2, dim=0) - p_mean**2
    if var == 'sum':
        ale = torch.sum(ale, dim=1)
        epi = torch.sum(epi, dim=1)
    elif var == 'top':
        ale = aleatoric[torch.argmax(p_mean)]
        epi = epistemic[torch.argmax(p_mean)]
    uncert = ale + epi
    return p_mean, uncert, ale, epi

def accuracy(inputs, target):
    _, max_indices = torch.max(inputs.data, 1)
    acc = (max_indices == target).sum().float() / max_indices.size(0)
    return acc.item()

def get_beta(batch_idx, m, beta_type, epoch, num_epochs, warmup_epochs=0):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    if epoch < warmup_epochs:
        beta /= warmup_epochs - epoch
    return beta

class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        super(ThresholdPruning, self).__init__()
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        import pdb;pdb.set_trace()
        return torch.abs(tensor) > self.threshold

class L1UnstructuredFFGOnTheFly(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, W, amount):
        super(L1UnstructuredFFGOnTheFly, self).__init__()
        self.amount = amount

    def compute_mask(self, tensor, default_mask):
        mu = tensor[:int(0.5 * tensor.size()[0])].detach().cpu().numpy()
        rho = tensor[int(0.5 * tensor.size()[0]):].detach().cpu().numpy()
        snrs = np.abs(mu) / np.log(1 + np.exp(rho))

        kth = int(self.amount * len(snrs))
        idx = self.smallest_N_indices(snrs, kth)
        mask = torch.ones(len(snrs)).to(tensor.device)
        mask[idx.flatten()] = 0.
        return torch.cat((mask, mask))

    @staticmethod
    def smallest_N_indices(array, N):
        idx = array.ravel().argsort()[:N]
        return np.stack(np.unravel_index(idx, array.shape)).T

def prune_weights_ffg_on_the_fly(net, amount):
    l1_prune = lambda w, amount: prune.global_unstructured(w, pruning_method=L1UnstructuredFFGOnTheFly, amount=amount, W=w)
    w_to_prune = ['weight', 'bias']

    for w in w_to_prune:
        _w_to_prune = [(m, w) for m in net.modules() if isinstance(m, (LinearRT, LinearLRT, Conv2dRT, Conv2dLRT))]
        l1_prune(_w_to_prune, amount)

class L1UnstructuredFFG(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, W, amount):
        super(L1UnstructuredFFG, self).__init__()
        masks = []
        snrs = np.array([])
        for w in W:
            if w[1][0] == 'W':
                mu, rho = w[0].W_mu, w[0].W_rho
            elif w[1][0] == 'b':
                mu, rho = w[0].bias_mu, w[0].bias_rho
            snr = torch.abs(mu) / softplus(rho)
            snr_np = snr.detach().cpu().numpy().flatten()
            snrs = np.hstack((snrs, np.log(snr_np)))

        kth = int(amount * len(snrs))
        idx = self.smallest_N_indices(snrs, kth)
        self.mask = torch.ones(len(snrs)).to(mu.device)
        self.mask[idx.flatten()] = 0.

        # self.mask = mask.type(torch.ByteTensor)
        #     kth = int(amount * np.array(snr_np.shape).prod())
        #     idx = self.smallest_N_indices(snr_np, kth)
        #     mask = torch.ones(mu.size()).type(mu.dtype)
        #     if isinstance(w[0], (Conv2dRT, Conv2dLRT)):
        #         mask[idx[:,0], idx[:,1], idx[:,2], idx[:,3]] = 0.
        #     else:
        #         mask[idx[:,0], mask[:,1]] = 0.
        #     masks.append(mask.flatten())
        # self.mask = torch.cat(masks).to(mu.device)

    def compute_mask(self, tensor, default_mask):
        return self.mask

    @staticmethod
    def smallest_N_indices(array, N):
        idx = array.ravel().argsort()[:N]
        return np.stack(np.unravel_index(idx, array.shape)).T

def prune_weights(net, mode='threshold', thresh=0., amount=0.):
    thresh_prune = lambda w, thresh: prune.global_unstructured(w, pruning_method=ThresholdPruning, threshold=thresh)
    L1_prune = lambda w, amount: prune.global_unstructured(w, pruning_method=prune.L1Unstructured, amount=amount)
    w_to_prune = ['weight', 'bias']

    for w in w_to_prune:
        _w_to_prune = [(m, w) for m in net.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
        if mode == 'threshold':
            thresh_prune(_w_to_prune, thresh)
        elif mode == 'percentage':
            L1_prune(_w_to_prune, amount)

def prune_weights_ffg(net, mode='percentage', thresh=0., amount=0.):
    thresh_prune = lambda w, thresh: prune.global_unstructured(w, pruning_method=ThresholdPruning, threshold=thresh)
    L1_prune = lambda w, amount: prune.global_unstructured(w, pruning_method=L1UnstructuredFFG, amount=amount, W=w)
    w_to_prune = ['W_mu', 'W_rho', 'bias_mu', 'bias_rho']

    for w in w_to_prune:
        _w_to_prune = [(m, w) for m in net.modules() if isinstance(m, (LinearRT, LinearLRT, Conv2dRT, Conv2dLRT))]
        if mode == 'threshold':
            thresh_prune(_w_to_prune, thresh)
        elif mode == 'percentage':
            L1_prune(_w_to_prune, amount)

def norm_grad(net):
    pass
