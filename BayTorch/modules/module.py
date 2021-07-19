import torch
from torch.nn import Parameter, Module
from torch.nn.functional import softplus
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

from ..distributions import mc_kl_divergence, MixtureNormal

class VIModule(Module):

    def __init__(self,
                 layer_fn,
                 weight_size,
                 bias_size=None,
                 prior=None,
                 posteriors=None,
                 kl_type='reverse'):

        super(VIModule, self).__init__()
        # dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.layer_fn = layer_fn

        if prior is None:
            prior = {'mu': 0, 'sigma': 0.1}

        if posteriors is None:
            posteriors = {
                'mu': (0, 0.1),
                'rho': (-3., 0.1)
            }

        if 'pi' in list(prior.keys()):
            self._kl_divergence = mc_kl_divergence
            self.prior = MixtureNormal(torch.tensor(prior['mu']), torch.tensor(prior["sigma"] + 1e-6),
                                       torch.tensor(prior['pi']))
        else:
            self._kl_divergence = kl_divergence
            self.prior = Normal(torch.tensor(prior['mu']), torch.tensor(prior["sigma"] + 1e-6))

        self.kl_type = kl_type

        self.posterior_mu_initial = posteriors['mu']
        self.posterior_rho_initial = posteriors['rho']

        self.W_mu = Parameter(torch.empty(weight_size))
        self.W_rho = Parameter(torch.empty(weight_size))
        if bias_size is not None:
            self.bias_mu = Parameter(torch.empty(bias_size))
            self.bias_rho = Parameter(torch.empty(bias_size))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.bias_mu is not None:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    @property
    def _kl(self):
        self.prior.loc = self.prior.loc.to(self.W_mu.device)
        self.prior.scale = self.prior.scale.to(self.W_mu.device)
        if isinstance(self.prior, MixtureNormal):
            self.prior.pi = self.prior.pi.to(self.W_mu.device)
        kl = self.kl_divergence(Normal(self.W_mu, softplus(self.W_rho)), self.prior, self.kl_type).sum()
        if self.bias_mu is not None:
            kl += self.kl_divergence(Normal(self.bias_mu, softplus(self.bias_rho)), self.prior, self.kl_type).sum()

        return kl

    def kl_divergence(self, p, q, kl_type='reverse'):
        if kl_type == 'reverse':
            return self._kl_divergence(q, p)
        else:
            return self._kl_divergence(p, q)

    @staticmethod
    def rsample(mu, sigma):
        eps = torch.randn_like(mu)
        return mu + eps * sigma
