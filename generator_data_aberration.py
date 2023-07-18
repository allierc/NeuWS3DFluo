# Copyright (c) 2023 
# Brandon Y. Feng, University of Maryland, College Park and Rice University. All rights reserved

import os, time, imageio, tqdm, argparse
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

import torch
print(f"Using PyTorch Version: {torch.__version__}")
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

import torch.nn.functional as F
from torch.fft import fft2, fftshift
from networks import *
from utils import *
from dataset import *
import matplotlib.pyplot as plt

from tifffile import imwrite
from tifffile import imread
import torch.nn.functional as f
from torch.fft import fft2, fftshift

from solve_data_3D import bpmPytorch

# python3 ./recon_exp_data.py --static_phase True --num_t 100 --data_dir ./DATA_DIR/static_objects_static_aberrations/dog_esophagus_0.5diffuser/Zernike_SLM_data  --scene_name dog_esophagus_0.5diffuser --phs_layers 4 --num_epochs 1000 --save_per_fram

# python ./recon_exp_data.py --dynamic_scene --num_t 100 --data_dir ./DATA_DIR/NeuWS_experimental_data-selected/dynamic_objects_dynamic_aberrations/owlStamp_onionSkin/Zernike_SLM_data --scene_name owlStamp_onionSkin --phs_layers 4 --num_epochs 1000 --save_per_frame

DEVICE = 'cuda'

if __name__ == "__main__":

    PSF_size = 256
    num_polynomials = 48


    model_config = {'nm': 1.33,
                    'na': 1.0556,
                    'dx': 0.1154,
                    'dz': 0.1,
                    'lbda': 0.5320,
                    'volume': [256, 256, 256],
                    'padding': False,
                    'dtype': torch.float32,
                    'device': 'cuda:0',
                    'bAber': False,
                    'bFit': False,
                    'num_feats': 4,
                    'out_path': "./Recons3D/"}

    bpm = bpmPytorch(model_config)

    z_in = torch.abs(torch.randn(256,num_polynomials,device=DEVICE))
    z_in = f.normalize(z_in, p=1, dim=1)
    z_in = z_in.repeat(256, 256, 1, 1).permute(2,3,0,1)
    basis = compute_zernike_basis(num_polynomials=num_polynomials,field_res=(PSF_size, PSF_size)).unsqueeze(0).repeat(256, 1, 1, 1)
    basis = basis.to(DEVICE)
    out=basis*z_in
    out=torch.sum(out, dim=1)*torch.pi

    out_cpx = torch.zeros(256, 256, 256, dtype=torch.cfloat)
    t = fftshift(bpm.pupil)

    for k in range(256):
        tt =  torch.exp(torch.mul(torch.abs(t)*out[k, :, :], 20j))
        tt=fftshift(tt)
        out_cpx[k, :, :]=tt

    torch.save(out_cpx,'./Pics_input/aberration.pt')
    imwrite(f'./Pics_input/aberration.tif', torch.angle(out_cpx).detach().cpu().numpy())

    nmean = torch.mean(out, axis=0)
    nstd = torch.std(out, axis=0)

    plt.ion()
    plt.imshow(out[10, :, :].detach().cpu().numpy())
    plt.colorbar()

    plt.ion()
    plt.imshow(torch.angle(out_cpx[20, :, :]).detach().cpu().numpy())
    plt.colorbar()

    plt.imshow(torch.angle(out_cpx[10, :, :]).detach().cpu().numpy())
    plt.imshow(out[10, :, :].detach().cpu().numpy())
    plt.imshow(torch.abs(bpm.pupil).detach().cpu().numpy())
    plt.imshow(torch.angle(bpm.pupil).detach().cpu().numpy())
    t = fftshift(bpm.pupil)
    tt= t*torch.exp(torch.mul(out[10, :, :],1j))
    plt.imshow(torch.angle(tt).detach().cpu().numpy())

    out=torch.angle(out_cpx)


