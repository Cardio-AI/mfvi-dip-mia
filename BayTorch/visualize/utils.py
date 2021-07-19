import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import Conv2dRT, Conv2dLRT, LinearRT, LinearLRT

def get_params(net):
    # _net = copy.deepcopy(net)
    params = [torch.flatten(p.detach()) for p in net.parameters() if p.requires_grad]
    params = torch.cat(params).cpu().numpy()
    return params

def get_params_mi(net):
    # _net = nn.Sequential(net._modules.copy())
    # _net.load_state_dict(net.state_dict().copy())

    mus = []
    sigmas = []
    for module in net.modules():
        if isinstance(module, (Conv2dRT, Conv2dLRT, LinearRT, LinearLRT)):
            mus.append(torch.flatten(module.W_mu.detach()))
            sigmas.append(torch.flatten(F.softplus(module.W_rho.detach())))
            mus.append(torch.flatten(module.bias_mu.detach()))
            sigmas.append(torch.flatten(F.softplus(module.bias_rho.detach())))

    return torch.cat(mus).cpu().numpy(), torch.cat(sigmas).cpu().numpy()
