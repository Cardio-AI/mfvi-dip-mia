import torch
from torch.nn.functional import softplus

from .module import VIModule

class RTLayer(VIModule):

    def __init__(self,
                 layer_fn,
                 weight_size,
                 bias_size=None,
                 prior=None,
                 posteriors=None,
                 kl_type='reverse',
                 _version='old',
                 **kwargs):

        super(RTLayer, self).__init__(layer_fn=layer_fn,
                                      weight_size=weight_size,
                                      bias_size=bias_size,
                                      prior=prior,
                                      posteriors=posteriors,
                                      kl_type=kl_type)
        self.kwargs = kwargs

    def forward(self, x):
        if self.training:
            weight = self.rsample(self.W_mu, softplus(self.W_rho))
            if self.bias_mu is not None:
                bias = self.rsample(self.bias_mu, softplus(self.bias_rho))
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu

        return self.layer_fn(x, weight, bias, **self.kwargs)

class LRTLayer(VIModule):

    def __init__(self,
                 layer_fn,
                 weight_size,
                 bias_size=None,
                 prior=None,
                 posteriors=None,
                 kl_type='reverse',
                 **kwargs):

        super(LRTLayer, self).__init__(layer_fn=layer_fn,
                                       weight_size=weight_size,
                                       bias_size=bias_size,
                                       prior=prior,
                                       posteriors=posteriors,
                                       kl_type=kl_type)
        self.kwargs = kwargs

    def forward(self, x):
        act_mu = self.layer_fn(x, self.W_mu, self.bias_mu, **self.kwargs)

        if self.training:
            self.W_sigma = softplus(self.W_rho)

            if self.bias_mu is not None:
                bias_var = softplus(self.bias_rho) ** 2
            else:
                bias_var = None

            act_std = torch.sqrt(1e-16 + self.layer_fn(x**2, self.W_sigma**2, bias_var, **self.kwargs))
            return self.rsample(act_mu, act_std)
        else:
            return act_mu
