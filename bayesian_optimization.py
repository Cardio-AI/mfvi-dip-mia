# !/usr/bin/env python
# coding: utf-8

# Max-Heinrich Laves
# Institute of Medical Technology and Intelligent Systems
# Hamburg University of Technology, Germany
# 2021

# std lib
import time
from typing import Dict, List, Tuple
from pathlib import Path
import itertools
from PIL import Image

# third party
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
sns.set()
import numpy as np
from tqdm import tqdm
from skimage.feature import peak_local_max
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim
import torch.autograd as autograd
from torch.distributions import constraints, transform_to
import gpytorch

# own
from models import get_net
from utils.denoising_utils import get_noisy_image_gaussian
from utils.bayesian_utils import gaussian_nll
from utils.common_utils import crop_image, get_image, pil_to_np, np_to_pil, plot_image_grid, \
    get_noise, get_params, np_to_torch, peak_signal_noise_ratio, structural_similarity
from BayTorch.freq_to_bayes import MeanFieldVI

torch.manual_seed(0)
np.random.seed(0)

def run_den(
        img: int = 0,
        num_iter: int = 5000,
        lr: float = 3e-4,
        beta: float = 4e-6,  # lambda in the paper
        tau: float = 0.01,
        input_depth: int = 16,
        device: torch.device = torch.device('cpu'),
        index: int = 0,
        seed: int = 42,
        show_every: int = 100,
        plot: bool = True,
        save: bool = True,
        save_path: str = '../logs'
) -> float:
    imsize = (256, 256)

    if img == 0:
        fname = 'data/NORMAL-4951060-8.png'
        imsize = (256, 256)
    elif img == 1:
        fname = 'data/BACTERIA-1351146-0006.png'
        imsize = (256, 256)
    elif img == 2:
        fname = 'data/081_HC.png'
        imsize = (256, 256)
    elif img == 3:
        fname = 'data/CNV-9997680-30.png'
        imsize = (256, 256)
    elif img == 4:
        fname = 'data/VIRUS-9815549-0001.png'
        imsize = (256, 256)
    else:
        assert False

    if fname == 'data/NORMAL-4951060-8.jpeg':

        # Add Gaussian noise to simulate speckle
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        p_sigma = 0.1
        img_noisy_pil, img_noisy_np = get_noisy_image_gaussian(img_np, p_sigma)

    elif fname in ['data/BACTERIA-1351146-0006.png', 'data/VIRUS-9815549-0001.png']:

        # Add Poisson noise to simulate low dose X-ray
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        # img_noisy_pil, img_noisy_np = get_noisy_image_poisson(img_np, p_lambda)
        # for lam > 20, poisson can be approximated with Gaussian
        p_sigma = 0.1
        img_noisy_pil, img_noisy_np = get_noisy_image_gaussian(img_np, p_sigma)

    elif fname == 'data/081_HC.png':

        # Add Gaussian noise to simulate speckle
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        p_sigma = 0.1
        img_noisy_pil, img_noisy_np = get_noisy_image_gaussian(img_np, p_sigma)

    elif fname == 'data/CNV-9997680-30.png':

        # Add Gaussian noise to simulate speckle
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        p_sigma = 0.1
        img_noisy_pil, img_noisy_np = get_noisy_image_gaussian(img_np, p_sigma)

    else:
        assert False

    NET_TYPE = 'skip'

    # TODO: must be changed for inp, sr
    skip_n33d = [16, 32, 64, 128, 128]
    skip_n33u = [16, 32, 64, 128, 128]
    skip_n11 = 4
    num_scales = 5
    upsample_mode = 'bilinear'
    pad = 'reflection'

    net = get_net(input_depth, NET_TYPE, pad,
                  skip_n33d=skip_n33d,
                  skip_n33u=skip_n33u,
                  skip_n11=skip_n11,
                  num_scales=num_scales,
                  n_channels=2,
                  upsample_mode=upsample_mode).to(device)

    psnr = run(
        net=net,
        img_np=img_np,
        img_pil=img_pil,
        img_corrupted_np=img_noisy_np,
        imsize=imsize,
        problem="noisy",
        num_iter=num_iter,
        lr=lr,
        beta=beta,
        tau=tau,
        input_depth=input_depth,
        device=device,
        index=index,
        seed=seed,
        show_every=show_every,
        plot=plot,
        save=save,
        save_path=save_path
    )

    return psnr



def run(
        # img: int = 0,
        net: nn.Module,
        img_np: np.ndarray,
        img_pil: Image,
        img_corrupted_np: np.ndarray,
        imsize: Tuple[int] = (256, 256),
        problem: str = 'noisy',
        num_iter: int = 5000,
        lr: float = 3e-4,
        beta: float = 4e-6,  # lambda in the paper
        tau: float = 0.01,
        input_depth: int = 16,
        device: torch.device = torch.device('cpu'),
        index: int = 0,
        seed: int = 42,
        show_every: int = 100,
        plot: bool = True,
        save: bool = True,
        save_path: str = '../logs',
) -> float:
    timestamp = str(time.time())
    Path(f'{save_path}/{timestamp}').mkdir(parents=True, exist_ok=False)

    with open(f'{save_path}/{timestamp}/locals.txt', 'w') as f:
        for key, val in locals().items():
            print(key, '=', val, file=f)

    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = True

    if plot:
        q = plot_image_grid([img_np, img_corrupted_np], 4, 6)
        out_pil = np_to_pil(q)
        out_pil.save(f'{save_path}/{timestamp}/input.png', 'PNG')

    INPUT = 'noise'
    # pad = 'reflection'
    OPT_OVER = 'net'  # 'net,input'

    reg_noise_std = 1. / 10.
    LR = lr

    num_iter += 1

    exp_weight = 0.99

    mse = torch.nn.MSELoss()

    # TODO: must be changed for SR and inp
    img_torch = np_to_torch(img_np).to(device)
    img_corrupted_torch = np_to_torch(img_corrupted_np).to(device)

    MSE_CORRUPTED = {}
    MSE_GT = {}
    UNCERTS_EPI = {}
    UNCERTS_ALE = {}
    PSNRS = {}
    SSIMS = {}

    figsize = 4

    ## MFVI
    weight_decay = 0

    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).to(device).detach()

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    out_avg = None

    mc_iter = 25
    mc_ring_buffer_epi = torch.zeros((mc_iter,) + imsize)  # saves the last mc_iter reconstructions
    mc_ring_buffer_ale = torch.zeros((mc_iter,) + imsize)  # saves the last mc_iter reconstructions

    # net = get_net(input_depth, NET_TYPE, pad,
    #               skip_n33d=skip_n33d,
    #               skip_n33u=skip_n33u,
    #               skip_n11=skip_n11,
    #               num_scales=num_scales,
    #               n_channels=2,
    #               upsample_mode=upsample_mode).to(device)

    prior = {'mu': 0.0,
             'sigma': np.sqrt(tau) * 1.0}

    net = MeanFieldVI(net,
                      prior=prior,
                      beta=beta,
                      replace_layers='all',
                      device=device,
                      reparam='')

    # TODO: this must be changed
    mse_corrupted = np.zeros((num_iter))
    mse_gt = np.zeros((num_iter))
    uncerts_epi = np.zeros((num_iter // show_every + 1, 1) + imsize)
    uncerts_ale = np.zeros((num_iter // show_every + 1, 1) + imsize)
    psnrs = np.zeros((num_iter, 3))
    ssims = np.zeros((num_iter, 3))

    img_mean = 0
    sample_count = 0
    psnr_corrupted_last = 0

    parameters = get_params(OPT_OVER, net, net_input)
    optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=weight_decay)

    pbar = tqdm(range(num_iter), miniters=num_iter // show_every, position=index)
    for i in pbar:
        optimizer.zero_grad()

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)

        nll = gaussian_nll(out[:, :1], out[:, 1:], img_corrupted_torch)
        kl = net.kl()
        loss = nll + beta * kl
        loss.backward()
        optimizer.step()

        out[:, 1:] = torch.exp(-out[:, 1:])  # aleatoric uncertainty

        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        with torch.no_grad():
            mse_corrupted[i] = mse(out_avg[:, :1], img_corrupted_torch).item()
            mse_gt[i] = mse(out_avg[:, :1], img_torch).item()

            _out = out.detach()[:, :1].clip(0, 1)
            _out_avg = out_avg.detach()[:, :1].clip(0, 1)
            _out_ale = out.detach()[:, 1:].clip(0, 1)

            mc_ring_buffer_epi[i % mc_iter] = _out[0]
            mc_ring_buffer_ale[i % mc_iter] = _out_ale[0]

            psnr_corrupted = peak_signal_noise_ratio(img_corrupted_torch, _out)
            psnr_gt = peak_signal_noise_ratio(img_torch, _out)
            psnr_gt_sm = peak_signal_noise_ratio(img_torch, _out_avg)
            ssim_corrupted = structural_similarity(img_corrupted_torch, _out)
            ssim_gt = structural_similarity(img_torch, _out)
            ssim_gt_sm = structural_similarity(img_torch, _out_avg)

        psnrs[i] = [psnr_corrupted, psnr_gt, psnr_gt_sm]
        ssims[i] = [ssim_corrupted, ssim_gt, ssim_gt_sm]

        if i % show_every == 0:
            pbar.set_description(f'MSE: {mse_corrupted[i].item():.4f} | PSNR_{problem}: {psnr_corrupted:7.4f} \
| PSRN_gt: {psnr_gt:7.4f} PSNR_gt_sm: {psnr_gt_sm:7.4f}')

            _out_var = torch.var(mc_ring_buffer_epi, dim=0)
            _out_ale = torch.mean(mc_ring_buffer_ale, dim=0)
            uncerts_epi[i // show_every] = _out_var.cpu().numpy()
            uncerts_ale[i // show_every] = _out_ale.cpu().numpy()

            if plot:
                fig, ax0 = plt.subplots()
                ax0.plot(range(len(mse_corrupted[:i])), mse_corrupted[:i])
                ax0.plot(range(len(mse_gt[:i])), mse_gt[:i])
                ax0.set_title('MSE MFVI')
                ax0.set_xlabel('iteration')
                ax0.set_ylabel('mse')
                ax0.set_ylim(0, 0.03)
                ax0.grid(True)

                ax1 = ax0.twinx()
                ax1.plot(range(len(psnrs[:i])), psnrs[:i, 2], 'g')
                ax1.set_ylabel('psnr gt sm')

                fig.tight_layout()
                fig.savefig(f'{save_path}/{timestamp}/loss_mfvi.png')
                plt.close('all')

    MSE_CORRUPTED['mfvi'] = mse_corrupted
    MSE_GT['mfvi'] = mse_gt
    UNCERTS_EPI['mfvi'] = uncerts_epi
    UNCERTS_ALE['mfvi'] = uncerts_ale
    PSNRS['mfvi'] = psnrs
    SSIMS['mfvi'] = ssims

    ## END

    file = open(f'{save_path}/{timestamp}/locals.txt', 'a')

    if plot:
        fig, ax = plt.subplots(1, 1)
        for key, loss in MSE_CORRUPTED.items():
            ax.plot(range(len(loss)), loss, label=key)
            ax.set_title(f'MSE {problem}')
            ax.set_xlabel('iteration')
            ax.set_ylabel('mse loss')
            ax.set_ylim(0, 0.03)
            ax.grid(True)
            ax.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path}/{timestamp}/mse_{problem}.png')

        fig, ax = plt.subplots(1, 1)
        for key, loss in MSE_GT.items():
            ax.plot(range(len(loss)), loss, label=key)
            ax.set_title('MSE GT')
            ax.set_xlabel('iteration')
            ax.set_ylabel('mse loss')
            ax.set_ylim(0, 0.01)
            ax.grid(True)
            ax.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path}/{timestamp}/mse_gt.png')

        fig, axs = plt.subplots(1, 3, constrained_layout=True)
        labels = [f"psnr_{problem}", "psnr_gt", "psnr_gt_sm"]
        for key, psnr in PSNRS.items():
            psnr = np.array(psnr)
            print(f"{key} PSNR_max: {np.max(psnr)}", file=file)
            for i in range(psnr.shape[1]):
                axs[i].plot(range(psnr.shape[0]), psnr[:, i], label=key)
                axs[i].set_title(labels[i])
                axs[i].set_xlabel('iteration')
                axs[i].set_ylabel('psnr')
                axs[i].legend()
        plt.savefig(f'{save_path}/{timestamp}/psnrs.png')

        fig, axs = plt.subplots(1, 3, constrained_layout=True)
        labels = [f"ssim_{problem}", "ssim_gt", "ssim_gt_sm"]
        for key, ssim in SSIMS.items():
            ssim = np.array(ssim)
            print(f"{key} SSIM_max: {np.max(ssim)}", file=file)
            for i in range(ssim.shape[1]):
                axs[i].plot(range(ssim.shape[0]), ssim[:, i], label=key)
                axs[i].set_title(labels[i])
                axs[i].set_xlabel('iteration')
                axs[i].set_ylabel('ssim')
                axs[i].legend()
        plt.savefig(f'{save_path}/{timestamp}/ssims.png')

    file.close()

    # save stuff for plotting
    if save:
        np.savez(f"{save_path}/{timestamp}/save.npz",
                 noisy_img=img_corrupted_np, mse_noisy=MSE_CORRUPTED, mse_gt=MSE_GT,
                 uncerts=UNCERTS_EPI, uncerts_ale=UNCERTS_ALE, psnrs=PSNRS)

    plt.close('all')

    return PSNRS['mfvi'][-1, 2]


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(
            prior=gpytorch.priors.NormalPrior(15., 4.)
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

        self.covar_module.base_kernel.lengthscale = 3e-1

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp(X_train, Y_train, iter_max=1000):
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_prior=gpytorch.priors.GammaPrior(concentration=0.01, rate=100.0)
    ).double()
    # likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
    #    noise=torch.ones(X_train.shape[0])*1e-4
    #    ).double().to(device)
    gp = ExactGPModel(
        X_train,
        Y_train,
        likelihood).double().to(X_train.device)
    gp.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer_gp = torch.optim.Adam(gp.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

    for i in range(iter_max):
        # Zero gradients from previous iteration
        optimizer_gp.zero_grad()
        # Output from model
        output = gp(X_train)
        # Calc loss and backprop gradients
        loss = -mll(output, Y_train)
        loss.backward()
        if i % 100 == 0:
            print(f'Iter {i + 1:4d}/{iter_max} - Loss: {loss.item():.4f}   \
lengthscale: {gp.covar_module.base_kernel.lengthscale.item():.3f}   \
noise: {gp.likelihood.noise[0].item():.4f}')
        optimizer_gp.step()

    gp.eval()
    likelihood.eval()
    return gp, likelihood


def expected_improvement(gp: gpytorch.models.ExactGP,
                         X: torch.Tensor,
                         X_train: torch.Tensor
                         ) -> torch.Tensor:
    '''
    Computes the EI at points for the parameter space based on
    cost samples using a Gaussian process surrogate model.
    Args:
        model: surrogate GP model
        params_space: Parameter space at which EI shall be computed (m x d).
        params_samples: already evaluated parameters (n x d)
    Returns:
        Expected improvements for paramter space.
    '''
    pred = gp(X)
    pred_sample = gp(X_train)

    mu, sigma = pred.mean, pred.variance.clamp_min(1e-9).sqrt()
    mu_sample = pred_sample.mean

    sigma = sigma.reshape(-1, 1)

    imp = mu - mu_sample.max()
    u = imp.reshape(-1, 1) / sigma
    normal = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u))
    ucdf = normal.cdf(u)
    updf = torch.exp(normal.log_prob(u))
    ei = sigma * (updf + u * ucdf)

    return ei.clamp_min(0)


def upper_confidence_bound(gp, X, kappa=2):
    pred = gp(X)
    return pred.mean + kappa * pred.variance.sqrt()


def acquisition_fun(gp, X, X_train, acq_fn, *args):
    assert acq_fn in ['ei', 'ucb']
    gp.eval()
    if acq_fn == 'ei':
        return expected_improvement(gp, X, X_train)
    elif acq_fn == 'ucb':
        return upper_confidence_bound(gp, X, *args)


def find_candidates(gp, X_, samples, acq_fn='ei'):
    with torch.no_grad():
        acq = acquisition_fun(gp, X_, samples, acq_fn)

    acq = acq.cpu().numpy().reshape(100, 100)
    peaks = peak_local_max(acq, min_distance=5, threshold_rel=0.1, num_peaks=4)
    global_max = np.array(np.unravel_index(np.argmax(acq, axis=None), acq.shape)).reshape(1, -1)
    peaks = np.append(peaks, global_max, axis=0)
    peaks = np.unique(peaks, axis=0)
    peaks = np.ravel_multi_index(peaks.transpose(), acq.shape)

    X_init = X_[peaks]

    constraint = constraints.interval(0, 1)
    candidates = []
    expected_improvement = []

    for i in range(len(X_init[:4])):
        unconstrained_X_init = transform_to(constraint).inv(X_init[i].unsqueeze(0))
        unconstrained_X = unconstrained_X_init.clone().detach().requires_grad_(True)
        minimizer = torch.optim.LBFGS([unconstrained_X], line_search_fn='strong_wolfe')

        def closure():
            minimizer.zero_grad()
            x = transform_to(constraint)(unconstrained_X)
            y = -acquisition_fun(gp, x, samples, acq_fn)
            autograd.backward(unconstrained_X, autograd.grad(y, unconstrained_X))
            return y

        minimizer.step(closure)
        X = transform_to(constraint)(unconstrained_X)

        expected_improvement.append(acquisition_fun(gp, X, samples, acq_fn).item())
        candidates.append(X.detach().cpu())

    return candidates, expected_improvement, acq


def normalize_X(X_unnorm, beta_logbounds, tau_logbounds):
    X_norm = X_unnorm.clone().log10()
    X_norm[:, 0] -= beta_logbounds[0]
    X_norm[:, 0] /= (beta_logbounds[1] - beta_logbounds[0])

    X_norm[:, 1] -= tau_logbounds[0]
    X_norm[:, 1] /= (tau_logbounds[1] - tau_logbounds[0])

    return X_norm


def unnormalize_X(X_norm, beta_logbounds, tau_logbounds):
    X_unnorm = X_norm.clone()
    X_unnorm[:, 0] *= (beta_logbounds[1] - beta_logbounds[0])
    X_unnorm[:, 0] += beta_logbounds[0]

    X_unnorm[:, 1] *= (tau_logbounds[1] - tau_logbounds[0])
    X_unnorm[:, 1] += tau_logbounds[0]

    return torch.pow(10, X_unnorm)


def f(idx, queue, candidate, device, params):
    res = run_den(beta=candidate[0], tau=candidate[1], index=idx, device=device, **params)
              # img=1, seed=1, num_iter=50, lr=2e-3, input_depth=16, save=True, save_path='./bo_logs')
    queue.put((candidate, res))


def bo(
        bo_params: Dict[str, List[float]],
        run_params: Dict,
        bo_out_path: str = './bo_results',
) -> None:

    mp.set_start_method('spawn')

    # bo_out_path = './bo_results'
    Path(bo_out_path).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    device_list = [
        torch.device("cuda:0"),
        # torch.device("cuda:1")
    ]

    X = []
    Y = []

    p1_logbounds, p2_logbounds = [v["logbounds"] for p, v in bo_params.items()]

    X_lr = torch.logspace(*p1_logbounds, 100, dtype=torch.double).to(device)
    X_wd = torch.logspace(*p2_logbounds, 100, dtype=torch.double).to(device)
    XX_lr, XX_wd = torch.meshgrid(X_lr, X_wd)
    X_ = torch.stack([XX_lr.reshape(-1), XX_wd.reshape(-1)]).transpose(1, 0)

    candidates = list(itertools.product(*[v["candidates"] for p, v in bo_params.items()]))

    for runs_num in range(100):

        plt.close('all')

        queue = mp.Queue()
        processes = []
        for i, (candidate, dev) in enumerate(zip(candidates, itertools.cycle(device_list))):
            p = mp.Process(target=f, args=(i, queue, candidate, dev, run_params))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        y_run = []
        candidates_run = []
        while not queue.empty():
            candidate, res = queue.get()
            candidates_run.append(candidate)
            y_run.append(res)

        print()
        print("beta      tau       psnr")
        for c, y in zip(candidates_run, y_run):
            print(f"{c[0]:.6f}  {c[1]:.6f}  {y:.6f}")

        X += candidates_run
        Y += y_run

        X_train = torch.stack([
            torch.DoubleTensor(np.array(X)[:, 0]),
            torch.DoubleTensor(np.array(X)[:, 1])
        ]).transpose(1, 0).to(device)
        X_train = normalize_X(X_train, p1_logbounds, p2_logbounds)

        Y_train = torch.DoubleTensor(np.array(Y)).to(device)

        gp, likelihood = train_gp(X_train, Y_train)

        with torch.no_grad():
            X_test = torch.stack([
                X_[:, 0],
                X_[:, 1]
            ]).transpose(1, 0)
            X_test = normalize_X(X_test, p1_logbounds, p2_logbounds)
            candidates, exp_imp, acq = find_candidates(gp, X_test, X_train)

            candidates = torch.cat(candidates).cpu()
            candidates = torch.unique(candidates, dim=0)
            candidates = unnormalize_X(candidates, p1_logbounds, p2_logbounds).numpy()

            pred = gp(X_test)
            # acq = upper_confidence_bound(gp, X_test)

        fig1, ax1 = plt.subplots()
        ln11 = ax1.contourf(XX_lr.cpu().numpy(), XX_wd.cpu().numpy(),
                            pred.mean.cpu().reshape(100, 100).numpy())
        ln12 = ax1.plot(np.array(X)[:, 0], np.array(X)[:, 1], 'g.', label='observed')
        ax1.set_title(f"{runs_num} mean acc")
        fig1.colorbar(ln11, ax=ax1)
        ax1.set_xlabel('beta')
        ax1.set_ylabel('tau')
        # ax1.set_xlim(np.power(10, beta_logbounds))
        # ax1.set_ylim(np.power(10, tau_logbounds))
        ax1.loglog()
        fig1.tight_layout()
        fig1.savefig(f'{bo_out_path}/{runs_num}_fig1.pdf', bbox_inches='tight')
        fig1.show()

        fig2, ax2 = plt.subplots()
        ln21 = ax2.contourf(XX_lr.cpu().numpy(), XX_wd.cpu().numpy(),
                            (pred.confidence_region()[1].detach().cpu().reshape(100, 100) \
                             - pred.confidence_region()[0].detach().cpu().reshape(100, 100)).numpy(),
                            )
        ln22 = ax2.plot(np.array(X)[:, 0], np.array(X)[:, 1], 'g.', label='observed')
        ax2.set_title(f"{runs_num} uncertainty")
        fig2.colorbar(ln21, ax=ax2)
        ax2.set_xlabel('beta')
        ax2.set_ylabel('tau')
        # ax2.set_xlim(np.power(10, beta_logbounds))
        # ax2.set_ylim(np.power(10, tau_logbounds))
        ax2.loglog()
        fig2.tight_layout()
        fig2.savefig(f'{bo_out_path}/{runs_num}_fig2.pdf', bbox_inches='tight')
        fig2.show()

        fig3, ax3 = plt.subplots()
        ln31 = ax3.contourf(XX_lr.cpu().numpy(), XX_wd.cpu().numpy(),
                            acq.reshape(100, 100))
        ln32 = ax3.plot(candidates[:, 0], candidates[:, 1], 'g.', label='candidates')
        ax3.set_title(f"{runs_num} acq_fun")
        ax3.set_xlabel('beta')
        ax3.set_ylabel('tau')
        # ax3.set_xlim(np.power(10, beta_logbounds))
        # ax3.set_ylim(np.power(10, tau_logbounds))
        ax3.loglog()
        fig3.colorbar(ln31, ax=ax3)
        fig3.tight_layout()
        fig3.savefig(f'{bo_out_path}/{runs_num}_fig3.pdf', bbox_inches='tight')
        fig3.show()

        fig4, ax4 = plt.subplots(subplot_kw={"projection": "3d"})
        ln41 = ax4.plot_surface(XX_lr.log10().cpu().numpy(),
                                XX_wd.log10().cpu().numpy(),
                                acq.reshape(100, 100),
                                cmap=cm.jet,
                                linewidth=0, antialiased=False)
        ax4.plot(np.log10(candidates[:, 0]), np.log10(candidates[:, 1]), exp_imp, 'gx')
        ax4.set_title(f"{runs_num} acq_fun")
        fig4.tight_layout()
        fig4.savefig(f'{bo_out_path}/{runs_num}_fig4.pdf', bbox_inches='tight')

        fig4.show()

        np.savez(
            f"{bo_out_path}/{runs_num}_fig_data.npz",
            XX_lr=XX_lr.cpu().numpy(), XX_wd=XX_wd.cpu().numpy(),
            pred=pred.mean.cpu().reshape(100, 100).numpy(),
            observed_X=np.array(X),
            observed_Y=np.array(Y),
            expected_improvement=np.array(exp_imp),
            confidence=pred.confidence_region()[1].detach().cpu().reshape(100, 100) \
                       - pred.confidence_region()[0].detach().cpu().reshape(100, 100),
            acq=acq.reshape(100, 100),
            candidates=candidates
        )

if __name__ == '__main__':
    import argparse
    from collections import OrderedDict
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_config", type=str, default="./bo.json")
    args = parser.parse_args()

    filter_nans = lambda d: {k: v for k, v in d.items() if v is not np.nan}
    config = pd.read_json(args.path_config).to_dict(into=OrderedDict)
    bo_params = filter_nans(config["bo_params"])
    run_params = filter_nans(config["run_params"])

    bo(bo_params=bo_params, run_params=run_params)
