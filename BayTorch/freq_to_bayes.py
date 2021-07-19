import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Conv2dRT, Conv2dLRT, Conv3dRT, Conv3dLRT, LinearRT, LinearLRT, MCDropout

class MeanFieldVI(nn.Module):

    def __init__(self,
                 net,
                 prior=None,
                 posteriors=None,
                 beta=1.,
                 kl_type='reverse',
                 reparam='local',
                 replace_layers='all',
                 device=torch.device('cpu')):

        super(MeanFieldVI, self).__init__()
        self.net = net
        self.device = device

        if reparam == 'local':
            self._conv3d = Conv3dLRT
            self._conv2d = Conv2dLRT
            self._linear = LinearLRT
        else:
            self._conv3d = Conv3dRT
            self._conv2d = Conv2dRT
            self._linear = LinearRT

        assert replace_layers in ['up', 'down', 'all', 'none']
        self._replace_layers = replace_layers

        if self._replace_layers == 'all':
            self._replace_layers = ''

        self._replace_deterministic_modules(self.net, prior, posteriors, kl_type)
        self.net = net.to(device)

    def forward(self, x):
        return self.net(x)

    def kl(self):
        kl = torch.FloatTensor([0.0]).to(self.device)
        for layer in self.modules():
            if hasattr(layer, '_kl'):
                kl += layer._kl
        return kl

    def _replace_deterministic_modules(self, module, prior, posteriors, kl_type):
        for key, _module in module._modules.items():
            if len(_module._modules):
                self._replace_deterministic_modules(_module, prior, posteriors, kl_type)
            else:
                if self._replace_layers in key:
                    if isinstance(_module, nn.Linear):
                        layer = self._linear(
                            _module.in_features,
                            _module.out_features,
                            torch.is_tensor(_module.bias))
                        module._modules[key] = layer
                    elif isinstance(_module, nn.Conv2d):
                        layer = self._conv2d(
                            in_channels=_module.in_channels,
                            out_channels=_module.out_channels,
                            kernel_size=_module.kernel_size,
                            bias=torch.is_tensor(_module.bias),
                            stride=_module.stride,
                            padding=_module.padding,
                            dilation=_module.dilation,
                            groups=_module.groups,
                            prior=prior,
                            posteriors=posteriors,
                            kl_type=kl_type)
                        module._modules[key] = layer
                    elif isinstance(_module, nn.Conv3d):
                        layer = self._conv3d(
                            in_channels=_module.in_channels,
                            out_channels=_module.out_channels,
                            kernel_size=_module.kernel_size,
                            bias=torch.is_tensor(_module.bias),
                            stride=_module.stride,
                            padding=_module.padding,
                            dilation=_module.dilation,
                            groups=_module.groups,
                            prior=prior,
                            posteriors=posteriors,
                            kl_type=kl_type)
                        module._modules[key] = layer

class MCDropoutVI(nn.Module):

    def __init__(self, net, dropout_type='1d',
                 dropout_p=0.5, deterministic_output=False,
                 output_dip_drop=False):

        super(MCDropoutVI, self).__init__()
        self.net = net
        self.dropout_type = dropout_type
        self.dropout_p = dropout_p

        self._replace_deterministic_modules(self.net)
        # self.deterministic_output = deterministic_output
        if deterministic_output:
            self._make_last_layer_deterministic(self.net)
        if not output_dip_drop:
            self._dip_make_output_deterministic(self.net)

    def forward(self, x):
        return self.net(x)

    def _replace_deterministic_modules(self, module):
        for key, _module in module._modules.items():
            if len(_module._modules):
                self._replace_deterministic_modules(_module)
            else:
                if isinstance(_module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                    module._modules[key] =  MCDropout(_module, self.dropout_type, self.dropout_p)

    def _make_last_layer_deterministic(self, module):
        for i, (key, layer) in enumerate(module._modules.items()):
            if i == len(module._modules) - 1:
                if isinstance(layer, MCDropout):
                    module._modules[key] = layer.layer
                elif len(layer._modules):
                    self._make_last_layer_deterministic(layer)

    ##############################################
    # only for dip -> has to be moved
    def _dip_make_output_deterministic(self, module):
        for i, (key, layer) in enumerate(module._modules.items()):
            if type(layer) == nn.Sequential:
                for name, m in layer._modules.items():
                    if type(m) == MCDropout:
                        layer._modules[name] = m.layer
    ##############################################
