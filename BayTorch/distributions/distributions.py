import torch
from torch.nn.functional import softplus
from torch.distributions.normal import Normal
import math

class MixtureNormal():
    def __init__(self, loc, scale, pi):
        assert(len(loc) == len(pi))
        assert(len(scale) == len(pi))

        self.loc = torch.tensor(loc)
        self.scale = torch.tensor(scale)
        self.pi = torch.tensor(pi)

        self.dists = [Normal(loc, scale) for loc, scale in zip(self.loc, self.scale)]

    def rsample(self):
        x = torch.rand(1)
        rsample = 0
        for pi, dist in zip(self.pi, self.dists):
            rsample += pi * torch.exp(dist.log_prob(dist.cdf(x)))
        return rsample

    def log_prob(self, x):
        pdf = 0
        for pi, dist in zip(self.pi, self.dists):
            pdf += pi.to(x.device) * torch.exp(dist.log_prob(x))
        return torch.log(pdf)

def mc_kl_divergence(p, q, n_samples=1):
    kl = 0
    for _ in range(n_samples):
        sample = p.rsample()
        kl += p.log_prob(sample) - q.log_prob(sample)
    return kl / n_samples
