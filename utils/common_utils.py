import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import numpy as np
import io
from collections import OrderedDict
import matplotlib.pyplot as plt


def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2),
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped


def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params


def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()


def plot_image_grid(images_np, nrow =8, factor=1, interpolation='lanczos'):
    """Draws images in a grid

    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)

    plt.figure(figsize=(len(images_np) + factor, 12 + factor))

    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)

    return grid


def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img


def normalize(fm):
    fm -= fm.min()
    fm /= fm.max()
    return fm


def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size.

    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10, library='torch', data_format='channels_first'):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        if data_format == 'channels_first':
            shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        elif data_format == 'channels_last':
            shape = [1, spatial_size[0], spatial_size[1], input_depth]

        if library == 'torch':
            net_input = torch.zeros(shape)

            fill_noise(net_input, noise_type)

        # elif library == 'tensorflow':
        #     if noise_type == 'u':
        #         net_input = tf.random.uniform(shape)
        #     elif noise_type == 'n':
        #         net_input = tf.random.normal(shape)
        net_input *= var

    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])

        if type == 'torch':
            net_input =  np_to_torch(meshgrid)
        # elif type == 'tensorflow':
        #     net_input = np_to_tf(meshgrid)
    else:
        assert False

    return net_input


def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_pil = Image.open(buf)
    # img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    # buf.close()
    # img = pil_to_np(img_pil)
    # img = cv2.imdecode(img_arr, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # remove 4th channel (transparency)
    img = np.array(img_pil.getdata()).astype(np.float32)[...,:3].reshape( (img_pil.size[0],img_pil.size[1],3))

    return img


def swap_channels(img_np):
    img_np = np.moveaxis(img_np, 0, 1)
    img_np = np.moveaxis(img_np, 1, 2)
    return img_np


def rename_modules(sequential, string, number):
    _modules = OrderedDict()
    for key in sequential._modules.keys():
        module = sequential._modules[key]
        if len(key) == 1:
            _module_name = '{}_{}_{}'.format(module._get_name(), string, number)
            # now it's only equipped for two same modules
            if _module_name not in _modules.keys():
                _modules[_module_name] = module
            else:
                _modules[_module_name + '_1'] = module
        else:
            _modules[key] = module
    sequential._modules = _modules
    return number + 1


def init_normal(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.normal_(m.weight, mean=0, std=0.1)


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def optimize(optimizer, closure, num_iter):

    for j in range(num_iter):
        optimizer.zero_grad()
        loss = closure()
        optimizer.step()


def optimize_lr(optimizer, closure, num_iter):
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9996)
    for j in range(num_iter):
        optimizer.zero_grad()
        loss = closure()
        optimizer.step()

        if scheduler.get_last_lr()[0] > 1e-8:
            scheduler.step()

        if j % 100 == 0:
            print("lr =", scheduler.get_last_lr()[0])


def peak_signal_noise_ratio(image_true: torch.Tensor, image_test: torch.Tensor):
    """Compute PSNR on GPU.
    We always assume float images with a max. value of 1.
    Args:
        image_true: ground truth image, same shape as test image
        image_test: test image
    """
    err = torch.nn.functional.mse_loss(image_true, image_test)
    return (10 * torch.log10(1 / err)).item()


def structural_similarity(image_true: torch.Tensor, image_test: torch.Tensor,
                          window_size: int = 11, size_average: bool = True, sigma: float = 1.5) -> torch.Tensor:
    """Compute SSIM on GPU.
    Taken from https://github.com/Po-Hsun-Su/pytorch-ssim
    Args:
        image_true: ground truth image, same shape as test image
        image_test: test image
        window_size:
        size_average:
        sigma:
    """

    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    gauss /= gauss.sum()

    (_, channel, _, _) = image_true.size()
    # window = create_window(window_size, channel)

    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    if image_true.is_cuda:
        window = window.cuda(image_true.get_device())
    window = window.type_as(image_true)

    mu1 = F.conv2d(image_true, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(image_test, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(image_true*image_true, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(image_test*image_test, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(image_true*image_test, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()
