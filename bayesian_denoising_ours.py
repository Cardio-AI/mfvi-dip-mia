#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Max-Heinrich Laves
# Institute of Medical Technology and Intelligent Systems
# Hamburg University of Technology, Germany
# 2021

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from models import get_net
import torch
import torch.optim
from utils.denoising_utils import get_noisy_image_gaussian
from utils.bayesian_utils import gaussian_nll
from utils.common_utils import init_normal, crop_image, get_image, pil_to_np, np_to_pil,                                plot_image_grid, get_noise, get_params, optimize, np_to_torch, torch_to_np,                                peak_signal_noise_ratio, structural_similarity
import time
from tqdm import tqdm
from skimage.exposure import rescale_intensity
from BayTorch.freq_to_bayes import MeanFieldVI


# In[7]:


def main(
    img: int=0,
    num_iter: int=50000,
    lr: float=3e-4,
    beta: float=4e-6,
    tau: float=0.01,
    input_depth: int=16,
    gpu: int=0,
    seed: int=42,
    show_every: int=100,
    plot: bool=True,
    save: bool=True,
    save_path: str='../logs',
):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = False

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    imsize = (256, 256)

    timestamp = int(time.time())
    os.makedirs(f'{save_path}/{timestamp}')

    # denoising
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
        #img_noisy_pil, img_noisy_np = get_noisy_image_poisson(img_np, p_lambda)
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

    if plot:
        q = plot_image_grid([img_np, img_noisy_np], 4, 6)
        out_pil = np_to_pil(q)
        out_pil.save(f'{save_path}/{timestamp}/input.png', 'PNG')

    INPUT = 'noise'
    pad = 'reflection'
    OPT_OVER = 'net' # 'net,input'

    reg_noise_std = 1./10.
    LR = lr
    
    num_iter += 1

    exp_weight = 0.99

    mse = torch.nn.MSELoss()

    img_torch = np_to_torch(img_np).type(dtype)
    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

    MSE_NOISY = {}
    MSE_GT = {}
    RECONS = {}
    UNCERTS_EPI = {}
    UNCERTS_ALE = {}
    PSNRS = {}
    SSIMS = {}

    figsize = 4

    NET_TYPE = 'skip'

    skip_n33d = [16, 32, 64, 128, 128]
    skip_n33u = [16, 32, 64, 128, 128]
    skip_n11 = 4
    num_scales = 5
    upsample_mode = 'bilinear'

    ## MFVI

    weight_decay = 0

    dropout_mode_down = 'None'
    dropout_p_down = 0.0
    dropout_mode_up = 'None'
    dropout_p_up = dropout_p_down
    dropout_mode_skip = 'None'
    dropout_p_skip = dropout_p_down
    dropout_mode_output = 'None'
    dropout_p_output = dropout_p_down

    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    out_avg = None

    mc_iter = 25
    mc_ring_buffer_epi = torch.zeros((mc_iter,) + imsize)  # saves the last mc_iter reconstructions
    mc_ring_buffer_ale = torch.zeros((mc_iter,) + imsize)  # saves the last mc_iter reconstructions

    net = get_net(input_depth, NET_TYPE, pad,
                  skip_n33d=skip_n33d,
                  skip_n33u=skip_n33u,
                  skip_n11=skip_n11,
                  num_scales=num_scales,
                  n_channels=2,
                  upsample_mode=upsample_mode,
                  dropout_mode_down=dropout_mode_down,
                  dropout_p_down=dropout_p_down,
                  dropout_mode_up=dropout_mode_up,
                  dropout_p_up=dropout_p_up,
                  dropout_mode_skip=dropout_mode_skip,
                  dropout_p_skip=dropout_p_skip,
                  dropout_mode_output=dropout_mode_output,
                  dropout_p_output=dropout_p_output).type(dtype)
    
    prior = {'mu': 0.0,
             'sigma': np.sqrt(tau)*1.0}
    
    net = MeanFieldVI(net,
                      prior=prior,
                      beta=beta,
                      replace_layers='all',
                      device=device,
                      reparam='')

    mse_noisy = np.zeros((num_iter))
    mse_gt = np.zeros((num_iter))
    recons = np.zeros((num_iter//show_every+1, 1)+imsize)
    uncerts_epi = np.zeros((num_iter//show_every+1, 1)+imsize)
    uncerts_ale = np.zeros((num_iter//show_every+1, 1)+imsize)
    psnrs = np.zeros((num_iter, 3))
    ssims = np.zeros((num_iter, 3))

    img_mean = 0
    sample_count = 0
    psnr_noisy_last = 0

    parameters = get_params(OPT_OVER, net, net_input)
    optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=weight_decay)
    
    pbar = tqdm(range(num_iter), miniters=num_iter//show_every)
    for i in pbar:
        optimizer.zero_grad()
        
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)

        nll = gaussian_nll(out[:,:1], out[:,1:], img_noisy_torch)
        kl = net.kl()
        loss = nll + beta*kl
        loss.backward()
        optimizer.step()

        out[:,1:] = torch.exp(-out[:,1:])  # aleatoric uncertainty

        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        with torch.no_grad():
            mse_noisy[i] = mse(out_avg[:,:1], img_noisy_torch).item()
            mse_gt[i] = mse(out_avg[:,:1], img_torch).item()

            _out = out.detach()[:,:1].clip(0, 1)
            _out_avg = out_avg.detach()[:,:1].clip(0, 1)
            _out_ale = out.detach()[:,1:].clip(0, 1)

            mc_ring_buffer_epi[i % mc_iter] = _out[0]
            mc_ring_buffer_ale[i % mc_iter] = _out_ale[0]

            psnr_noisy = peak_signal_noise_ratio(img_noisy_torch, _out)
            psnr_gt    = peak_signal_noise_ratio(img_torch, _out)
            psnr_gt_sm = peak_signal_noise_ratio(img_torch, _out_avg)
            ssim_noisy = structural_similarity(img_noisy_torch, _out)
            ssim_gt    = structural_similarity(img_torch, _out)
            ssim_gt_sm = structural_similarity(img_torch, _out_avg)

        psnrs[i] = [psnr_noisy, psnr_gt, psnr_gt_sm]
        ssims[i] = [ssim_noisy, ssim_gt, ssim_gt_sm]        

        if i % show_every == 0:
            pbar.set_description(f'MSE: {mse_noisy[i].item():.4f} | PSNR_noisy: {psnr_noisy:7.4f} | PSRN_gt: {psnr_gt:7.4f} PSNR_gt_sm: {psnr_gt_sm:7.4f}')
            
            recons[i//show_every] = _out_avg.cpu().numpy()[0]

            _out_var = torch.var(mc_ring_buffer_epi, dim=0)
            _out_ale = torch.mean(mc_ring_buffer_ale, dim=0)
            uncerts_epi[i//show_every] = _out_var.cpu().numpy()
            uncerts_ale[i//show_every] = _out_ale.cpu().numpy()
            
            if plot:
                fig, ax0 = plt.subplots()
                ax0.plot(range(len(mse_noisy[:i])), mse_noisy[:i])
                ax0.plot(range(len(mse_gt[:i])), mse_gt[:i])
                ax0.set_title('MSE MFVI')
                ax0.set_xlabel('iteration')
                ax0.set_ylabel('mse')
                ax0.set_ylim(0, 0.03)
                ax0.grid(True)

                ax1 = ax0.twinx()
                ax1.plot(range(len(psnrs[:i])), psnrs[:i,2], 'g')
                ax1.set_ylabel('psnr gt sm')

                fig.tight_layout()
                fig.savefig(f'{save_path}/{timestamp}/loss_mfvi.png')
                plt.close('all')


    MSE_NOISY['mfvi'] = mse_noisy
    MSE_GT['mfvi'] = mse_gt
    RECONS['mfvi'] = recons
    UNCERTS_EPI['mfvi'] = uncerts_epi
    UNCERTS_ALE['mfvi'] = uncerts_ale
    PSNRS['mfvi'] = psnrs
    SSIMS['mfvi'] = ssims

    if plot:
        to_plot = [img_np] + [np.clip(img, 0, 1) for img in RECONS['mfvi']]
        q = plot_image_grid(to_plot, factor=13)
        out_pil = np_to_pil(q)
        out_pil.save(f'{save_path}/{timestamp}/mfvi_recons.png', 'PNG')
        
        to_plot = [img_np] + [rescale_intensity(img, out_range=(0, 1)) for img in UNCERTS_EPI['mfvi']]
        q = plot_image_grid(to_plot, factor=13)
        out_pil = np_to_pil(q)
        out_pil.save(f'{save_path}/{timestamp}/uncert_epi_mfvi.png', 'PNG')
        
        to_plot = [img_np] + [rescale_intensity(img, out_range=(0, 1)) for img in UNCERTS_ALE['mfvi']]
        q = plot_image_grid(to_plot, factor=13)
        out_pil = np_to_pil(q)
        out_pil.save(f'{save_path}/{timestamp}/uncert_ale_mfvi.png', 'PNG')

    ## END
        
    if plot:  
        fig, ax = plt.subplots(1, 1)
        for key, loss in MSE_NOISY.items():
            ax.plot(range(len(loss)), loss, label=key)
            ax.set_title('MSE noisy')
            ax.set_xlabel('iteration')
            ax.set_ylabel('mse loss')
            ax.set_ylim(0,0.03)
            ax.grid(True)
            ax.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path}/{timestamp}/mse_noisy.png')
        
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
        labels = ["psnr_noisy", "psnr_gt", "psnr_gt_sm"]
        for key, psnr in PSNRS.items():
            psnr = np.array(psnr)
            print(f"{key} PSNR_max: {np.max(psnr)}")
            for i in range(psnr.shape[1]):
                axs[i].plot(range(psnr.shape[0]), psnr[:,i], label=key)
                axs[i].set_title(labels[i])
                axs[i].set_xlabel('iteration')
                axs[i].set_ylabel('psnr')
                axs[i].legend()
        plt.savefig(f'{save_path}/{timestamp}/psnrs.png')
        
        fig, axs = plt.subplots(1, 3, constrained_layout=True)
        labels = ["ssim_noisy", "ssim_gt", "ssim_gt_sm"]
        for key, ssim in SSIMS.items():
            ssim = np.array(ssim)
            print(f"{key} SSIM_max: {np.max(ssim)}")
            for i in range(ssim.shape[1]):
                axs[i].plot(range(ssim.shape[0]), ssim[:,i], label=key)
                axs[i].set_title(labels[i])
                axs[i].set_xlabel('iteration')
                axs[i].set_ylabel('ssim')
                axs[i].legend()
        plt.savefig(f'{save_path}/{timestamp}/ssims.png')

    # save stuff for plotting
    if save:
        np.savez(f"{save_path}/{timestamp}/save.npz",
                 noisy_img=img_noisy_np, mse_noisy=MSE_NOISY, mse_gt=MSE_GT, recons=RECONS,
                 uncerts=UNCERTS_EPI, uncerts_ale=UNCERTS_ALE, psnrs=PSNRS)
        print(f"Saved results to {save_path}/{timestamp}/save.npz")

    plt.close('all')


# In[8]:


main(beta=1e-4, tau=0.007, img=4, seed=1, num_iter=50000, lr=2e-3, gpu=0, save_path='/opt/laves/logs')
main(beta=4e-6, tau=0.1,   img=4, seed=1, num_iter=50000, lr=2e-3, gpu=0, save_path='/opt/laves/logs')
main(beta=4e-8, tau=0.1,   img=4, seed=1, num_iter=50000, lr=2e-3, gpu=0, save_path='/opt/laves/logs')


# In[ ]:




