import torch
import torch.nn as nn
import torch.nn.functional as F

def Gaussian_dropout3d(x, p, layer):
    #mu = F.conv2d(x, layer.weight.data)
    mu = layer(x)
    sigma = F.conv3d(x**2, layer.weight.data**2, stride=layer.stride, padding=layer.padding)
    sigma = (p / (1 - p) * sigma).sqrt()
    eps = torch.randn_like(mu)
    return mu + sigma * eps

def Gaussian_dropout2d(x, p, layer):
    #mu = F.conv2d(x, layer.weight.data)
    mu = layer(x)
    sigma = F.conv2d(x**2, layer.weight.data**2, stride=layer.stride, padding=layer.padding)
    sigma = (p / (1 - p) * sigma).sqrt()
    eps = torch.randn_like(mu)
    return mu + sigma * eps

def Gaussian_dropout(x, p, layer):
    mu = F.linear(x, layer.weight.data)
    sigma = F.linear(x**2, layer.weight.data**2)
    sigma = (p / (1 - p) * sigma).sqrt()
    eps = torch.randn_like(mu)
    return mu + sigma * eps

class MCDropout(nn.Module):

    def __init__(self, layer, dropout_type='adaptive', p=0.5):

        super(MCDropout, self).__init__()
        self.layer = layer
        self.p = p

        if dropout_type == 'adaptive':
            if type(self.layer) == nn.Conv2d:
                self.dropout_type = '2d'
            elif type(self.layer) == nn.Conv3d:
                self.dropout_type = '3d'
            else:
                self.dropout_type = '1d'
        elif dropout_type == 'gaussian_adaptive':
            if type(self.layer) == nn.Conv2d:
                self.dropout_type = 'g2d'
            elif type(self.layer) == nn.Conv3d:
                self.dropout_type = 'g3d'
            else:
                self.dropout_type = 'g1d'
        else:
            self.dropout_type = dropout_type

    def forward(self, x):
        if self.dropout_type == '1d':
            x = self.layer(x)
            return F.dropout(x, self.p, training=True)
        elif self.dropout_type == '2d':
            x = self.layer(x)
            return F.dropout2d(x, self.p, training=True)
        elif self.dropout_type == '3d':
            x = self.layer(x)
            return F.dropout3d(x, self.p, training=True)
        elif self.dropout_type == 'g1d':
            return Gaussian_dropout(x, self.p, self.layer)
        elif self.dropout_type == 'g2d':
            return Gaussian_dropout2d(x, self.p, self.layer)
        elif self.dropout_type == 'g3d':
            return Gaussian_dropout3d(x, self.p, self.layer)

    def __repr__(self):
        return "%s(\n\tlayer: %s,\n\tdropout_type: %s,\n\tdropout_p: %.1f" % (self.__class__.__name__, self.layer.__repr__(), self.dropout_type, self.p)
