import math
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import tifffile
import torch.nn as nn
import torch.nn.functional as nf
import torch
from torch.fft import fft2, fftshift, irfftn, rfftn, ifftshift
from tqdm import tqdm
from timeit import default_timer as timer
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import io
import json
from tifffile import imwrite
from tifffile import imread
import logging
from utils import compute_zernike_basis, fft_2xPad_Conv2D
from solve_data_3D import bpmPytorch

if __name__ == '__main__':

    print('Init ...')

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

    coeff_RI = 0
    Niter = 100

    torch.cuda.empty_cache()
    bpm = bpmPytorch(model_config)

    new_obj = np.zeros((bpm.Nz, bpm.Nx, bpm.Ny))
    tmp=imread(f'./Pics_input/target.tif')
    new_obj=tmp
    # new_obj[:, int(bpm.Nx / 4):int(bpm.Nx * 3 / 4), int(bpm.Ny / 4):int(bpm.Ny * 3 / 4)] = tmp
    new_obj = np.moveaxis(new_obj, 0, -1)
    bpm.fluo = torch.tensor(new_obj, device=bpm.device, dtype=bpm.dtype, requires_grad=False)

    new_obj = np.zeros((bpm.Nz, bpm.Nx, bpm.Ny))
    tmp=imread(f'./Pics_input/dn.tif')
    new_obj=tmp
    # new_obj[:, int(bpm.Nx / 4):int(bpm.Nx * 3 / 4), int(bpm.Ny / 4):int(bpm.Ny * 3 / 4)] = tmp
    new_obj = np.moveaxis(new_obj, 0, -1)
    bpm.dn0 = torch.tensor(new_obj, device=bpm.device, dtype=bpm.dtype, requires_grad=False)

    bpm.dn0_layers = bpm.dn0.unbind(dim=2)
    bpm.fluo_layers = bpm.fluo.unbind(dim=2)
    bpm.aber_layers = bpm.aber.unbind(dim=0)

    Istack = np.zeros([bpm.Nx, bpm.Ny, bpm.Nz])

    for plane in tqdm(range(0, bpm.Nz)):
        I = torch.tensor(np.zeros((bpm.Nx, bpm.Ny)), device=bpm.device, dtype=bpm.dtype, requires_grad=False)
        phiL = torch.rand([bpm.Nx, bpm.Ny, 1000], dtype=torch.float32, requires_grad=False,device='cuda:0') * 2 * np.pi
        for w in range(0, Niter):
            zoi = np.random.randint(1000 - bpm.Nz)
            with torch.no_grad():
                I = I + bpm(plane=int(plane), phi=phiL[:, :, zoi:zoi + bpm.Nz ])

        Istack[:, :, plane] = I.detach().cpu().numpy() / Niter

        # imwrite(f'./Recons3D/fluo_LSM_RIx{np.round(coeff_RI)/100}_{plane}.tif',
        #         np.moveaxis(np.array(I.cpu()), -1, 0) / Niter * 5 )

    tmp = np.moveaxis(Istack, -1, 0)
    imwrite(f'./Pics_input/input_aberration_null.tif', tmp )

    t=torch.load('./Pics_input/aberration.pt')
    bpm.aber=t.to('cuda:0')
    bpm.aber_layers = bpm.aber.unbind(dim=0)

    bpm.bAber = True

    Istack = np.zeros([bpm.Nx, bpm.Ny, bpm.Nz])

    for plane in tqdm(range(0, bpm.Nz)):
        I = torch.tensor(np.zeros((bpm.Nx, bpm.Ny)), device=bpm.device, dtype=bpm.dtype, requires_grad=False)
        phiL = torch.rand([bpm.Nx, bpm.Ny, 1000], dtype=torch.float32, requires_grad=False,device='cuda:0') * 2 * np.pi
        for w in range(0, Niter):
            zoi = np.random.randint(1000 - bpm.Nz)
            with torch.no_grad():
                I = I + bpm(plane=int(plane), phi=phiL[:, :, zoi:zoi + bpm.Nz ])
        Istack[:, :, plane] = I.detach().cpu().numpy() / Niter

        # imwrite(f'./Recons3D/fluo_LSM_RIx{np.round(coeff_RI)/100}_{plane}.tif',
        #         np.moveaxis(np.array(I.cpu()), -1, 0) / Niter * 5 )

    tmp = np.moveaxis(Istack, -1, 0)
    imwrite(f'./Pics_input/input_aberration.tif', tmp )

    plt.ion()
    plt.imshow(I.detach().cpu().numpy())








