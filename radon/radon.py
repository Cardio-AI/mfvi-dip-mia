import torch


class FastRadonTransform(torch.nn.Module):
    """
    Calculates the radon transform of an image given specified
    projection angles. This is a generator that returns a closure.

    Parameters
    ----------
    image : Tensor
        Input image of shape (B, C, H, W).
    theta : Tensor, optional
        Projection angles (in degrees) of shape (T,). If `None`, the value is set to
        torch.arange(180).

    Returns
    -------
    radon_image : Tensor
        Radon transform (sinogram) of shape (B, C, T, W).
    """

    def __init__(self, image_size, theta=None):
        super().__init__()
        assert image_size[-2] == image_size[-1]

        if theta is None:
            theta = torch.deg2rad(torch.arange(180.))
        else:
            theta = torch.deg2rad(theta)

        ts = torch.sin(theta)
        tc = torch.cos(theta)
        z = torch.zeros_like(tc)

        trans = torch.stack([tc, -ts, z, ts, tc, z]).permute(1, 0).reshape(theta.size(0), 2, 3)
        grid = torch.nn.functional.affine_grid(trans,
                                               (theta.size(0), image_size[1], image_size[2], image_size[3]),
                                               align_corners=False)

        self.register_buffer("theta", theta)
        self.register_buffer("ts", ts)
        self.register_buffer("tc", tc)
        self.register_buffer("z", z)
        self.register_buffer("trans", trans)
        self.register_buffer("grid", grid)

    def forward(self, image):
        img_r = torch.nn.functional.grid_sample(
            image.expand(self.theta.size(0), -1, -1, -1),
            self.grid,
            mode='bilinear', padding_mode='zeros', align_corners=False)
        radon_image = img_r.sum(2, keepdims=True).permute(1, 2, 0, 3)

        return radon_image
