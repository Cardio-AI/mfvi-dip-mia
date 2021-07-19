from torch.nn.functional import conv2d, conv3d
from torch.nn.modules.utils import _pair, _triple

from .reparam_layers import RTLayer, LRTLayer

class Conv2dRT(RTLayer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior=None,
                 posteriors=None,
                 kl_type='reverse'):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)

        weight_size = (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])

        bias_size = (out_channels) if bias else None

        super(Conv2dRT, self).__init__(layer_fn=conv2d,
                                        weight_size=weight_size,
                                        bias_size=bias_size,
                                        prior=prior,
                                        posteriors=posteriors,
                                        kl_type=kl_type,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups)

class Conv3dRT(RTLayer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior=None,
                 posteriors=None,
                 kl_type='reverse'):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)

        weight_size = (out_channels, in_channels,
                       self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])

        bias_size = (out_channels) if bias else None

        super(Conv3dRT, self).__init__(layer_fn=conv3d,
                                        weight_size=weight_size,
                                        bias_size=bias_size,
                                        prior=prior,
                                        posteriors=posteriors,
                                        kl_type=kl_type,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups)

class Conv2dLRT(LRTLayer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior=None,
                 posteriors=None,
                 kl_type='reverse'):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)

        weight_size = (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])

        bias_size = (out_channels) if bias else None

        super(Conv2dLRT, self).__init__(layer_fn=conv2d,
                                        weight_size=weight_size,
                                        bias_size=bias_size,
                                        prior=prior,
                                        posteriors=posteriors,
                                        kl_type=kl_type,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups)

class Conv3dLRT(LRTLayer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior=None,
                 posteriors=None,
                 kl_type='reverse'):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)

        weight_size = (out_channels, in_channels,
                       self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])

        bias_size = (out_channels) if bias else None

        super(Conv3dLRT, self).__init__(layer_fn=conv3d,
                                        weight_size=weight_size,
                                        bias_size=bias_size,
                                        prior=prior,
                                        posteriors=posteriors,
                                        kl_type=kl_type,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups)
